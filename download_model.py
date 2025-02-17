import os
import requests
from pathlib import Path
import torch

def download_yolov9():
    """Download YOLOv9 model"""
    print("Setting up YOLOv9...")
    
    # Create models directory if it doesn't exist
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # YOLOv9 model path
    model_path = model_dir / "yolov9.pt"
    
    if not model_path.exists():
        print("Downloading YOLOv9 model...")
        # YOLOv9 model URL
        url = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print(f"Model downloaded successfully to {model_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
    else:
        print("Model already exists")
    
    return str(model_path)

if __name__ == "__main__":
    download_yolov9()
