import os
import shutil
import pandas as pd
from pathlib import Path
import argparse

def prepare_c_nmc_dataset(source_dir, target_dir="data"):
    """
    Prepare C-NMC 2019 dataset for training
    
    Args:
        source_dir: Path to C-NMC dataset directory
        target_dir: Target directory for organized data
    """
    target_dir = Path(target_dir)
    cancer_dir = target_dir / "cancer"
    non_cancer_dir = target_dir / "non_cancer"
    
    # Create target directories
    cancer_dir.mkdir(parents=True, exist_ok=True)
    non_cancer_dir.mkdir(parents=True, exist_ok=True)
    
    source_dir = Path(source_dir)
    
    # Check if source directory exists
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} not found")
    
    print(f"Preparing dataset from: {source_dir}")
    print(f"Target directory: {target_dir}")
    
    # Process training data (fold_0, fold_1, fold_2)
    cancer_count = 0
    non_cancer_count = 0
    
    for fold in ["fold_0", "fold_1", "fold_2"]:
        fold_dir = source_dir / "C-NMC_training_data" / fold
        
        if not fold_dir.exists():
            print(f"Warning: {fold_dir} not found, skipping...")
            continue
        
        print(f"Processing {fold}...")
        
        # Process ALL (cancer) images
        all_dir = fold_dir / "all"
        if all_dir.exists():
            for img_file in all_dir.glob("*.bmp"):
                target_path = cancer_dir / f"cancer_{cancer_count:06d}.bmp"
                shutil.copy2(img_file, target_path)
                cancer_count += 1
        
        # Process HEM (normal) images
        hem_dir = fold_dir / "hem"
        if hem_dir.exists():
            for img_file in hem_dir.glob("*.bmp"):
                target_path = non_cancer_dir / f"normal_{non_cancer_count:06d}.bmp"
                shutil.copy2(img_file, target_path)
                non_cancer_count += 1
    
    print(f"\nDataset preparation completed!")
    print(f"Cancer images: {cancer_count}")
    print(f"Non-cancer images: {non_cancer_count}")
    print(f"Total images: {cancer_count + non_cancer_count}")
    
    # Only create info file if we have images
    if cancer_count + non_cancer_count > 0:
        # Create dataset info file
        info = {
            "cancer_images": cancer_count,
            "non_cancer_images": non_cancer_count,
            "total_images": cancer_count + non_cancer_count,
            "cancer_percentage": cancer_count / (cancer_count + non_cancer_count) * 100,
            "non_cancer_percentage": non_cancer_count / (cancer_count + non_cancer_count) * 100
        }
        
        info_file = target_dir / "dataset_info.txt"
        with open(info_file, 'w') as f:
            f.write("Blood Cancer Dataset Information\n")
            f.write("=" * 40 + "\n")
            for key, value in info.items():
                if isinstance(value, float):
                    f.write(f"{key.replace('_', ' ').title()}: {value:.2f}%\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        print(f"Dataset info saved to: {info_file}")
    else:
        print("No images found. Please check the dataset structure.")
    
    return info if cancer_count + non_cancer_count > 0 else None

def create_sample_dataset(target_dir="data", samples_per_class=100):
    """
    Create a small sample dataset for testing
    
    Args:
        target_dir: Target directory for sample data
        samples_per_class: Number of samples per class
    """
    target_dir = Path(target_dir)
    cancer_dir = target_dir / "cancer"
    non_cancer_dir = target_dir / "non_cancer"
    
    # Create directories
    cancer_dir.mkdir(parents=True, exist_ok=True)
    non_cancer_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample dataset structure...")
    print("Note: This creates empty directories. You'll need to add actual images.")
    print(f"Expected {samples_per_class} images per class")
    
    # Create placeholder files (you should replace these with real images)
    for i in range(samples_per_class):
        # Create placeholder cancer images
        placeholder = cancer_dir / f"cancer_{i:06d}.bmp"
        placeholder.touch()
        
        # Create placeholder normal images
        placeholder = non_cancer_dir / f"normal_{i:06d}.bmp"
        placeholder.touch()
    
    print(f"Sample structure created in: {target_dir}")
    print("Please replace placeholder files with actual blood cell images")

def validate_dataset(data_dir="data"):
    """
    Validate the dataset structure and count images
    
    Args:
        data_dir: Directory containing cancer/non_cancer subdirectories
    """
    data_dir = Path(data_dir)
    cancer_dir = data_dir / "cancer"
    non_cancer_dir = data_dir / "non_cancer"
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return False
    
    if not cancer_dir.exists():
        print(f"Error: Cancer directory {cancer_dir} does not exist")
        return False
    
    if not non_cancer_dir.exists():
        print(f"Error: Non-cancer directory {non_cancer_dir} does not exist")
        return False
    
    # Count images
    cancer_images = list(cancer_dir.glob("*.bmp")) + list(cancer_dir.glob("*.jpg")) + list(cancer_dir.glob("*.png"))
    non_cancer_images = list(non_cancer_dir.glob("*.bmp")) + list(non_cancer_dir.glob("*.jpg")) + list(non_cancer_dir.glob("*.png"))
    
    print("Dataset Validation Report")
    print("=" * 30)
    print(f"Cancer images: {len(cancer_images)}")
    print(f"Non-cancer images: {len(non_cancer_images)}")
    print(f"Total images: {len(cancer_images) + len(non_cancer_images)}")
    
    if len(cancer_images) == 0 or len(non_cancer_images) == 0:
        print("Warning: One or both classes have no images!")
        return False
    
    # Check for class imbalance
    total = len(cancer_images) + len(non_cancer_images)
    cancer_pct = len(cancer_images) / total * 100
    non_cancer_pct = len(non_cancer_images) / total * 100
    
    print(f"Cancer: {cancer_pct:.1f}%")
    print(f"Non-cancer: {non_cancer_pct:.1f}%")
    
    if abs(cancer_pct - 50) > 20:
        print("Warning: Significant class imbalance detected!")
        print("Consider using class weights or data augmentation.")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare blood cancer dataset")
    parser.add_argument("--source", type=str, help="Source directory for C-NMC dataset")
    parser.add_argument("--target", type=str, default="data", help="Target directory")
    parser.add_argument("--sample", action="store_true", help="Create sample dataset structure")
    parser.add_argument("--samples", type=int, default=100, help="Samples per class for sample dataset")
    parser.add_argument("--validate", action="store_true", help="Validate existing dataset")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_dataset(args.target)
    elif args.sample:
        create_sample_dataset(args.target, args.samples)
    elif args.source:
        prepare_c_nmc_dataset(args.source, args.target)
    else:
        print("Usage examples:")
        print("  Prepare C-NMC dataset: python prepare_dataset.py --source /path/to/C-NMC")
        print("  Create sample structure: python prepare_dataset.py --sample")
        print("  Validate dataset: python prepare_dataset.py --validate")
