import os
import sys
import subprocess

def setup_environment():
    """Setup the cancer classification environment"""
    print("Setting up Blood Cancer Classification Environment...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or python_version.minor < 8:
        print(" Python 3.8+ required")
        return False
    
    # Install requirements
    print("\n Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(" Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f" Failed to install requirements: {e}")
        return False
    
    # Create sample dataset structure
    print("\n Creating sample dataset structure...")
    try:
        subprocess.check_call([sys.executable, "src/prepare_dataset.py", "--sample", "--samples", "50"])
        print("✅ Sample dataset structure created")
    except subprocess.CalledProcessError as e:
        print(f" Failed to create sample structure: {e}")
        return False
    
    # Validate setup
    print("\n Validating setup...")
    try:
        import torch
        import torchvision
        import sklearn
        print(f" PyTorch: {torch.__version__}")
        print(f" TorchVision: {torchvision.__version__}")
        print(f" Scikit-learn: {sklearn.__version__}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" Device available: {device}")
        
    except ImportError as e:
        print(f" Import error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print(" Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your blood cell images to data/cancer/ and data/non_cancer/")
    print("2. Run: python src/train.py")
    print("3. Run: python src/inference.py path/to/test/image.jpg")
    print("\nOr download C-NMC dataset:")
    print("python src/prepare_dataset.py --source /path/to/C-NMC-dataset")
    
    return True

if __name__ == "__main__":
    success = setup_environment()
    if not success:
        print("\n Setup failed. Please check the errors above.")
        sys.exit(1)
