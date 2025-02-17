import cv2
import time
import sys
import pyttsx3
import numpy as np
import traceback
from ultralytics import YOLO
import threading
from queue import Queue, Empty, Full
from concurrent.futures import ThreadPoolExecutor
import torch
import os
import gc
import torch.backends.cudnn as cudnn
from pathlib import Path
import torch.nn as nn

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_alert_time = 0

# Configuration constants (move these to the top, before any function definitions)
DETECTION_CONF_THRESHOLD = 0.4  # Higher threshold for fewer detections
MIN_DETECTION_SIZE = 15  # Smaller minimum size
FRAME_SKIP = 1  # Process every frame
ALERT_COOLDOWN = 3  # Seconds between alerts
OBJECT_ALERT_COOLDOWN = 3  # Cooldown for object detection alerts
MAX_OBJECTS_PER_FRAME = 5  # Limit concurrent detections
HISTORY_FRAMES = 5  # Number of frames to keep in history
SMOOTHING_ALPHA = 0.6  # Smoothing factor (0-1)
FRAME_WIDTH = 416  # Smaller frame size
FRAME_HEIGHT = 416
BATCH_SIZE = 1

# Add new configuration constants after existing ones
MOTION_THRESHOLD = 25
MOTION_MIN_AREA = 500
MOTION_BLUR_SIZE = (21, 21)
BACKGROUND_HISTORY = 500
BACKGROUND_THRESHOLD = 16
LEARNING_RATE = 0.001

# Add new configuration constants
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame
MAX_QUEUE_SIZE = 3  # Smaller queue for less memory usage
ADAPTIVE_RESOLUTION = True
RESOLUTION_SCALES = [(640, 480), (480, 360), (320, 240)]
TARGET_FPS = 30
PERFORMANCE_WINDOW = 30  # frames

# Update model configurations at the top to use standard YOLO model
YOLO_MODEL_NAME = 'models/yolov9.pt'  # Update path to model directory
DETECTION_CONF_THRESHOLD = 0.25  # Lower threshold for more detections
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100  # Increase max detections
INPUT_SIZE = 640  # YOLOv9 default input size

# Add new camera constants
CAMERA_INDEX = 0  # Default camera
CAMERA_API = cv2.CAP_DSHOW  # Windows DirectShow API
CAMERA_RETRY_DELAY = 2  # Seconds between retries
CAMERA_INIT_TIMEOUT = 10  # Seconds to try initializing

# Initialize frame processing queue and thread pool
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
result_queue = Queue(maxsize=MAX_QUEUE_SIZE)
thread_pool = ThreadPoolExecutor(max_workers=2)

class PerformanceMonitor:
    def __init__(self, window_size=PERFORMANCE_WINDOW):
        self.process_times = []
        self.window_size = window_size
        self.current_resolution_index = 0
        
    def add_process_time(self, process_time):
        self.process_times.append(process_time)
        if len(self.process_times) > self.window_size:
            self.process_times.pop(0)
    
    def get_average_time(self):
        return sum(self.process_times) / len(self.process_times) if self.process_times else 0
    
    def should_adjust_resolution(self):
        avg_time = self.get_average_time()
        return avg_time > (1.0 / TARGET_FPS)

    def adjust_resolution(self, frame):
        """Dynamically adjust frame resolution based on performance"""
        if not self.should_adjust_resolution():
            return frame
            
        if self.current_resolution_index < len(RESOLUTION_SCALES) - 1:
            self.current_resolution_index += 1
            new_size = RESOLUTION_SCALES[self.current_resolution_index]
            return cv2.resize(frame, new_size)
        return frame

performance_monitor = PerformanceMonitor()

def process_frame_worker():
    try:
        while True:
            frame = frame_queue.get(timeout=1.0)
            if frame is None:
                break
                
            # Skip if queue is getting full
            if result_queue.qsize() >= MAX_QUEUE_SIZE - 1:
                continue
                
            start_time = time.time()
            
            # Adjust resolution based on performance
            frame = performance_monitor.adjust_resolution(frame)
            
            # Process frame
            processed_frame = detect_objects(frame)
            
            # Monitor performance
            process_time = time.time() - start_time
            performance_monitor.add_process_time(process_time)
            
            result_queue.put(processed_frame)
            
            # Cleanup
            del frame
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"Worker error: {e}")
    finally:
        # Cleanup resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Start worker thread
worker_thread = threading.Thread(target=process_frame_worker, daemon=True)
worker_thread.start()

# Initialize video capture with fallback
def init_camera():
    """Improved camera initialization with better error handling"""
    print("Initializing camera...")
    
    # Try different camera APIs
    apis = [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF]
    
    for api in apis:
        try:
            cap = cv2.VideoCapture(CAMERA_INDEX, api)
            if not cap.isOpened():
                continue
                
            # Configure camera
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify camera is working
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"Camera initialized successfully using {api}")
                print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                return cap
            else:
                cap.release()
        except Exception as e:
            print(f"Failed to initialize camera with API {api}: {e}")
            continue
    
    return None

def reset_camera(cap):
    """Reset camera if it stops responding"""
    if cap is not None:
        cap.release()
    time.sleep(CAMERA_RETRY_DELAY)
    return init_camera()

# Initialize camera
cap = init_camera()
if cap is None:
    print("Error: Could not initialize any camera")
    sys.exit(1)

# Load cascades
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
except Exception as e:
    print(f"Error loading cascades: {e}")
    sys.exit(1)

# Update YOLO initialization to use standard model
def init_yolo():
    try:
        print(f"Loading YOLOv9 model: {YOLO_MODEL_NAME}...")
        
        # Check if model exists
        model_path = Path(YOLO_MODEL_NAME)
        if not model_path.exists():
            # Try to download the model
            from download_model import download_yolov9
            model_path = download_yolov9()
            if model_path is None:
                raise FileNotFoundError("Failed to download YOLOv9 model")
        
        # Clone YOLOv9 repository if not exists
        yolov9_path = Path("yolov9")
        if not yolov9_path.exists():
            import subprocess
            subprocess.run(["git", "clone", "https://github.com/WongKinYiu/yolov9.git"])
        
        # Add YOLOv9 to Python path
        import sys
        sys.path.append(str(yolov9_path))
        
        # Load model using torch.hub
        model = torch.hub.load('WongKinYiu/yolov9', 'custom', str(model_path), trust_repo=True)
        
        # Configure model parameters
        model.conf = DETECTION_CONF_THRESHOLD
        model.iou = IOU_THRESHOLD
        model.max_det = MAX_DETECTIONS
        
        if torch.cuda.is_available():
            model.cuda()
            print("Using GPU for inference")
        else:
            model.cpu()
            print("Using CPU for inference")
        
        model.eval()
        print("YOLOv9 model loaded successfully")
        return model
        
    except Exception as e:
        print(f"Error loading YOLOv9 model: {e}")
        traceback.print_exc()
        return None

# Initialize YOLO
yolo_model = init_yolo()

# Change the initial detection state to True
detection = True
last_object_alert_time = 0
OBJECT_ALERT_COOLDOWN = 3

# Add after the global variables
alerted_objects = set()
last_reset_time = time.time()
ALERT_RESET_INTERVAL = 30  # Reset alert tracking every 30 seconds

# Add after the global variables
def speak_alert(message):
    global last_alert_time
    current_time = time.time()
    
    try:
        # Check cooldown
        if current_time - last_alert_time >= ALERT_COOLDOWN:
            engine.say(message)
            engine.runAndWait()
            last_alert_time = current_time
    except Exception as e:
        print(f"Error playing audio alert: {e}")
        # Fallback to console output
        print(f"Alert: {message}")

# UI elements
button_x, button_y = 10, 50
button_w, button_h = 200, 40
exit_button_x = button_x + button_w + 20
exit_button_w = 100
exit_button_clicked = False

# Add new configuration constants
DETECTION_CONF_THRESHOLD = 0.4  # Higher threshold for fewer detections
MIN_DETECTION_SIZE = 30  # Increased minimum size for more reliable detections
FRAME_SKIP = 2  # Skip every other frame for better processing
ALERT_COOLDOWN = 3  # Seconds between alerts for the same object class
MAX_OBJECTS_PER_FRAME = 5  # Limit concurrent detections

# Add new tracking variables
detection_history = {}  # Store recent detections
HISTORY_FRAMES = 5  # Number of frames to keep in history
SMOOTHING_ALPHA = 0.6  # Smoothing factor (0-1)

class TrackedObject:
    def __init__(self, label, box):
        self.label = label
        self.positions = [box]  # List of recent positions
        self.smoothed_box = box
        self.last_seen = time.time()

    def update(self, new_box):
        self.positions.append(new_box)
        if len(self.positions) > HISTORY_FRAMES:
            self.positions.pop(0)
        self.smooth_position()
        self.last_seen = time.time()

    def smooth_position(self):
        if not self.positions:
            return
        current = self.positions[-1]
        if not hasattr(self, 'smoothed_box'):
            self.smoothed_box = current
            return

        # Apply exponential smoothing
        self.smoothed_box = tuple(
            int(SMOOTHING_ALPHA * c + (1 - SMOOTHING_ALPHA) * s)
            for c, s in zip(current, self.smoothed_box)
        )

# Update the tracked_objects dictionary to be properly scoped
tracked_objects = {}

def draw_button(frame):
    # Add detection status with better visibility
    status_color = (0, 255, 0) if detection else (0, 0, 255)
    cv2.putText(frame, f'Detection: {detection}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    cv2.rectangle(frame, (exit_button_x, button_y), 
                 (exit_button_x + exit_button_w, button_y + button_h), 
                 (0, 0, 255), -1)
    cv2.putText(frame, "Exit", (exit_button_x + 20, button_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def mouse_callback(event, x, y, flags, param):
    global exit_button_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if exit_button_x <= x <= exit_button_x + exit_button_w and button_y <= y <= button_y + button_h:
            exit_button_clicked = True

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Camera", mouse_callback)

# Add list of common objects to monitor
COMMON_OBJECTS = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'laptop', 'mouse', 'keyboard', 'cell phone', 'book', 'bottle', 'cup',
    'chair', 'tv', 'laptop', 'remote', 'keyboard', 'cell phone', 'backpack', 'umbrella'
}

# Add after the camera initialization
background_subtractor = cv2.createBackgroundSubtractorKNN(
    history=BACKGROUND_HISTORY,
    dist2Threshold=BACKGROUND_THRESHOLD,
    detectShadows=False
)

def detect_motion(frame):
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(frame, MOTION_BLUR_SIZE, 0)
    
    # Apply background subtraction
    fg_mask = background_subtractor.apply(blurred, learningRate=LEARNING_RATE)
    
    # Remove noise
    fg_mask = cv2.erode(fg_mask, None, iterations=2)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_regions = []
    for contour in contours:
        if cv2.contourArea(contour) > MOTION_MIN_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            motion_regions.append((x, y, x+w, y+h))
    
    return motion_regions

# Update detect_objects function
def detect_objects(frame):
    """Updated object detection for YOLOv9"""
    if frame is None or yolo_model is None:
        return frame

    try:
        # Preprocess frame
        frame_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        if torch.cuda.is_available():
            img = img.cuda()
        
        # Run inference
        with torch.no_grad():
            predictions = yolo_model(img)
        
        # Process results
        processed_frame = draw_detections(frame, predictions)
        return processed_frame
        
    except Exception as e:
        print(f"Error in detect_objects: {e}")
        traceback.print_exc()
        return frame

def draw_detections(frame, predictions):
    """Updated detection visualization for YOLOv9"""
    try:
        if predictions is None or len(predictions.pred) == 0:
            return frame
            
        # Process detections
        for det in predictions.pred[0]:
            if len(det) >= 6:  # YOLOv9 format: [x1, y1, x2, y2, conf, cls]
                x1, y1, x2, y2, conf, cls = det[:6]
                
                if conf < DETECTION_CONF_THRESHOLD:
                    continue
                    
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Get class label
                label = predictions.names[int(cls)]
                
                # Draw box and label
                color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label_text = f'{label} {conf:.2f}'
                cv2.putText(frame, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                print(f"Detected {label} with confidence {conf:.2f}")
        
        return frame
        
    except Exception as e:
        print(f"Error in draw_detections: {e}")
        return frame

def update_tracking(frame, label, xyxy, conf, current_time):
    """Update object tracking and draw annotations"""
    global tracked_objects, last_object_alert_time, alerted_objects
    
    try:
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Update tracked objects
        if label in tracked_objects:
            tracked_objects[label].update((x1, y1, x2, y2))
        else:
            tracked_objects[label] = TrackedObject(label, (x1, y1, x2, y2))
        
        # Draw annotations
        draw_box_and_label(frame, tracked_objects[label], conf)
        
        # Handle alerts
        handle_object_alert(label, current_time)
        
    except Exception as e:
        print(f"Error in update_tracking: {e}")

# Update the main loop to use frame skipping
frame_count = 0
consecutive_failures = 0
MAX_FAILURES = 5

# Add after frame_count initialization
last_fps_time = time.time()
fps = 0

# Add frame display function
def display_frame(frame, processed=False):
    try:
        if frame is not None:
            if detection:
                # Apply face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.05, 8, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            draw_button(frame)
            
            # Add FPS counter
            cv2.putText(frame, f"FPS: {fps}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Camera", frame)
            cv2.resizeWindow("Camera", 800, 600)
    except Exception as e:
        print(f"Error displaying frame: {e}")

def draw_box_and_label(frame, tracked_object, conf):
    """Draw bounding box and label for detected object"""
    try:
        box = tracked_object.smoothed_box
        label = tracked_object.label
        
        # Draw box
        color = (0, 255, 0) if conf > 0.6 else (0, 255, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Draw label
        label_text = f'{label} {conf:.2f}'
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (box[0], box[1] - label_size[1] - 10),
                     (box[0] + label_size[0], box[1]), color, -1)
        cv2.putText(frame, label_text, (box[0], box[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                   
    except Exception as e:
        print(f"Error in draw_box_and_label: {e}")

def handle_object_alert(label, current_time):
    """Handle alerts for detected objects"""
    global last_object_alert_time, alerted_objects
    
    try:
        if (label not in alerted_objects and 
            current_time - last_object_alert_time >= OBJECT_ALERT_COOLDOWN):
            alert_message = f"Detected {label}"
            speak_alert(alert_message)
            alerted_objects.add(label)
            last_object_alert_time = current_time
            
    except Exception as e:
        print(f"Error in handle_object_alert: {e}")

# Move cleanup_project function definition before the main loop
def cleanup_project():
    """Improved cleanup function"""
    try:
        # Stop worker thread
        frame_queue.put(None)
        worker_thread.join(timeout=1.0)
        
        # Release camera
        if cap is not None:
            cap.release()
        
        # Clear queues
        while not frame_queue.empty():
            frame_queue.get()
        while not result_queue.empty():
            result_queue.get()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear other resources
        cv2.destroyAllWindows()
        gc.collect()
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        traceback.print_exc()

while True:
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_failures += 1
            print(f"Camera read failed ({consecutive_failures}/{MAX_FAILURES})")
            
            if consecutive_failures >= MAX_FAILURES:
                print("Attempting to reset camera...")
                cap = reset_camera(cap)
                if cap is None:
                    print("Could not reinitialize camera. Exiting...")
                    break
                consecutive_failures = 0
            continue
        
        consecutive_failures = 0
        frame_count += 1

        # Skip processing if frame is empty or too small
        if frame.size == 0 or frame.shape[0] < 10 or frame.shape[1] < 10:
            continue

        # Process frame
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            try:
                frame_queue.put_nowait(frame.copy())
            except Full:
                # Skip if queue is full
                pass

        # Display frame
        try:
            processed_frame = result_queue.get_nowait()
            display_frame(processed_frame, processed=True)
        except Empty:
            display_frame(frame, processed=False)
        
        # Memory management
        if frame_count % 30 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ...rest of the main loop...

        if detection:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.05, 8, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        draw_button(frame)

        # Calculate and display FPS
        if time.time() - last_fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_fps_time = time.time()
        
        cv2.putText(frame, f"FPS: {fps}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if exit_button_clicked:
            cleanup_project()
            break

        cv2.imshow("Camera", frame)
        cv2.resizeWindow("Camera", 800, 600)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cleanup_project()
            break
        elif key == ord('d'):
            detection = not detection
            print(f"Detection {'enabled' if detection else 'disabled'}")

        # Memory management
        if frame_count % 30 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in main loop: {e}")
        traceback.print_exc()
        time.sleep(0.1)  # Short delay on error

# Cleanup
try:
    if cap is not None:
        cap.release()
    frame_queue.put(None)
    worker_thread.join(timeout=1.0)
    thread_pool.shutdown(wait=False)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Error during cleanup: {e}")

# Add cleanup function at the end of the file
def cleanup_project():
    """Improved cleanup function"""
    try:
        # Stop worker thread
        frame_queue.put(None)
        worker_thread.join(timeout=1.0)
        
        # Release camera
        if cap is not None:
            cap.release()
        
        # Clear queues
        while not frame_queue.empty():
            frame_queue.get()
        while not result_queue.empty():
            result_queue.get()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear other resources
        cv2.destroyAllWindows()
        gc.collect()
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        traceback.print_exc()
