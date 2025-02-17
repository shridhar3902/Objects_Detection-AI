import os
from ultralytics import YOLO
import yaml
import torch

# Training configuration
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'patience': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'workers': 8,
    'project': 'security_camera_model',
    'name': 'custom_yolov9'
}

# Enhanced object classes to detect
CUSTOM_CLASSES = {
    # Personal items
    'wallet', 'keys', 'phone', 'laptop', 'tablet', 'headphones', 'watch',
    'glasses', 'bag', 'backpack', 'purse',
    
    # Office items
    'document', 'folder', 'pen', 'pencil', 'notebook', 'printer',
    'scanner', 'stapler', 'calculator', 'desk',
    
    # Electronic devices
    'charger', 'power_bank', 'usb_drive', 'hard_drive', 'camera',
    'monitor', 'keyboard', 'mouse', 'speaker', 'microphone',
    
    # Security-related
    'id_card', 'badge', 'key_card', 'cash', 'credit_card',
    'passport', 'license', 'access_card',
    
    # Suspicious items
    'unknown_device', 'unattended_bag', 'covered_face',
    
    # Additional detailed items
    'coffee_cup', 'water_bottle', 'medication', 'face_mask',
    'hand_sanitizer', 'tissues'
}

def create_dataset_yaml():
    """Create YAML configuration for custom dataset"""
    dataset_config = {
        'path': './datasets',  # Dataset root directory
        'train': 'train/images',  # Train images
        'val': 'valid/images',    # Validation images
        'test': 'test/images',    # Test images
        'names': list(CUSTOM_CLASSES)
    }
    
    os.makedirs('./datasets', exist_ok=True)
    
    with open('./datasets/custom.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    return './datasets/custom.yaml'

def train_model():
    """Train YOLOv9 model with custom configuration"""
    try:
        # Load base YOLOv9 model
        model = YOLO('yolov9-c.pt')
        
        # Create dataset configuration
        dataset_yaml = create_dataset_yaml()
        
        # Configure model parameters
        model.overrides['data'] = dataset_yaml
        model.overrides['epochs'] = TRAINING_CONFIG['epochs']
        model.overrides['batch'] = TRAINING_CONFIG['batch_size']
        model.overrides['imgsz'] = TRAINING_CONFIG['img_size']
        model.overrides['patience'] = TRAINING_CONFIG['patience']
        model.overrides['device'] = TRAINING_CONFIG['device']
        model.overrides['workers'] = TRAINING_CONFIG['workers']
        
        # Train the model
        results = model.train(
            data=dataset_yaml,
            epochs=TRAINING_CONFIG['epochs'],
            imgsz=TRAINING_CONFIG['img_size'],
            batch=TRAINING_CONFIG['batch_size'],
            name=TRAINING_CONFIG['name'],
            project=TRAINING_CONFIG['project'],
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            lr0=0.001,
            weight_decay=0.0005,
            warmup_epochs=3,
            close_mosaic=10,
            augment=True,
            mixup=0.1,
            copy_paste=0.1,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4
        )
        
        # Save the trained model
        model.save('custom_yolov9.pt')
        print("Model training completed and saved as custom_yolov9.pt")
        
        return results
    except Exception as e:
        print(f"Error during training: {e}")
        return None

if __name__ == "__main__":
    train_model()
