import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# Configuration
CONFIG = {
    "model_name": "efficientnet_b0",
    "img_size": 224,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_path": "models/blood_cancer_model.pth"
}

# Validation transforms (same as training)
val_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model(model_name, num_classes=1):
    """Load model architecture"""
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)  # Don't load pretrained weights for inference
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:  # MobileNetV3
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    return model.to(CONFIG["device"])

def predict_blood_cell(img_path, model_path=None, threshold=0.5):
    """
    Predict if a blood cell image shows cancer indicators
    
    Args:
        img_path: Path to the image file
        model_path: Path to trained model (uses default if None)
        threshold: Classification threshold (default 0.5)
    """
    if model_path is None:
        model_path = CONFIG["model_path"]
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    # Check if image exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")
    
    # Load Model
    model = get_model(CONFIG["model_name"])
    model.load_state_dict(torch.load(model_path, map_location=CONFIG["device"]))
    model.eval()
    
    # Prep Image
    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Cannot load image {img_path}: {str(e)}")
    
    img_tensor = val_transforms(image).unsqueeze(0).to(CONFIG["device"])
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
        
    # Determine result
    # During training: cancer=0, non_cancer=1
    # So probability closer to 0 = cancer, closer to 1 = non_cancer
    label = "CANCER INDICATED" if prob < threshold else "NON-CANCER (NORMAL)"
    confidence = prob if prob < threshold else 1 - prob
    
    # Print results
    print("=" * 50)
    print("BLOOD CELL ANALYSIS REPORT")
    print("=" * 50)
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Result: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Probability Score: {prob:.4f}")
    print(f"Threshold: {threshold}")
    print("=" * 50)
    print("IMPORTANT MEDICAL DISCLAIMER:")
    print("This is a research tool for cell pattern analysis only.")
    print("NOT a clinical diagnosis. Consult a hematopathologist.")
    print("Do not use this tool for medical decision making.")
    print("=" * 50)
    
    return {
        "label": label,
        "confidence": confidence,
        "probability": prob,
        "threshold": threshold
    }

def batch_predict(image_dir, model_path=None, threshold=0.5):
    """
    Predict all images in a directory
    
    Args:
        image_dir: Directory containing images
        model_path: Path to trained model
        threshold: Classification threshold
    """
    if model_path is None:
        model_path = CONFIG["model_path"]
    
    # Load model once
    model = get_model(CONFIG["model_name"])
    model.load_state_dict(torch.load(model_path, map_location=CONFIG["device"]))
    model.eval()
    
    results = []
    
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    print(f"Processing images in: {image_dir}")
    print("-" * 50)
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(supported_formats):
            img_path = os.path.join(image_dir, filename)
            try:
                result = predict_blood_cell(img_path, model_path, threshold)
                result["filename"] = filename
                results.append(result)
                print()  # Add spacing between results
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Summary
    if results:
        cancer_count = sum(1 for r in results if "CANCER" in r["label"])
        total_count = len(results)
        
        print("=" * 50)
        print("BATCH ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total images processed: {total_count}")
        print(f"Cancer indicated: {cancer_count} ({cancer_count/total_count:.1%})")
        print(f"Non-cancer: {total_count - cancer_count} ({(total_count-cancer_count)/total_count:.1%})")
        print("=" * 50)
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image: python inference.py path/to/image.jpg")
        print("  Batch directory: python inference.py path/to/directory/ --batch")
        print("  Custom threshold: python inference.py path/to/image.jpg --threshold 0.3")
        sys.exit(1)
    
    img_path = sys.argv[1]
    threshold = 0.5
    batch_mode = False
    
    # Parse arguments
    for i, arg in enumerate(sys.argv):
        if arg == "--threshold" and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])
        elif arg == "--batch":
            batch_mode = True
    
    try:
        if batch_mode:
            batch_predict(img_path, threshold=threshold)
        else:
            predict_blood_cell(img_path, threshold=threshold)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
