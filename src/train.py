import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
import os
import random
from PIL import Image
import shutil
import time
from datetime import timedelta

# 1. Reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# 2. Hyperparameters
CONFIG = {
    "model_name": "efficientnet_b0", # Alternative: "mobilenet_v3_small"
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 15,
    "img_size": 224,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "data_dir": "data",
    "model_save_path": "models/blood_cancer_model.pth",
    "checkpoint_dir": "checkpoints",
    "save_every": 1,  # Save checkpoint every N epochs
    "colab_mode": False  # Set to True when running in Colab
}

# 3. Data Transformations
train_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Checkpoint Functions
def save_checkpoint(model, optimizer, epoch, loss, acc, checkpoint_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'config': CONFIG
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    acc = checkpoint['accuracy']
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, loss: {loss:.4f}, acc: {acc:.4f}")
    return epoch, loss, acc

# 5. Model Definition (Transfer Learning)
def get_model(model_name, num_classes=1):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else: # Lightweight for CPU
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    return model.to(CONFIG["device"])

# 5. Training Loop
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_num, total_epochs):
    """Train one epoch with detailed progress tracking"""
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()
    
    # Calculate total batches for progress bar
    total_batches = len(loader)
    print(f"\nTraining Epoch {epoch_num}/{total_epochs}")
    print(f"Total batches: {total_batches}")
    print("=" * 60)
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        batch_start_time = time.time()
        
        inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Detailed batch progress
        batch_time = time.time() - batch_start_time
        current_loss = loss.item()
        avg_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
        
        # Progress bar and ETA
        progress = (batch_idx + 1) / total_batches * 100
        elapsed = time.time() - epoch_start_time
        eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
        
        print(f"Batch {batch_idx+1:3d}/{total_batches} | "
              f"Progress: {progress:5.1f}% | "
              f"Loss: {current_loss:.4f} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Time: {batch_time:.2f}s | "
              f"ETA: {str(timedelta(seconds=int(eta)))}")
    
    epoch_time = time.time() - epoch_start_time
    final_loss = running_loss / len(loader.dataset)
    print(f"\nEpoch {epoch_num} completed in {str(timedelta(seconds=int(epoch_time)))}")
    print(f"Final Epoch Loss: {final_loss:.4f}")
    
    return final_loss

def evaluate(model, loader, criterion, device):
    """Evaluate model with detailed progress tracking"""
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    running_loss = 0.0
    eval_start_time = time.time()
    
    total_batches = len(loader)
    print(f"\nEvaluating Model")
    print(f"Total validation batches: {total_batches}")
    print("=" * 60)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            batch_start_time = time.time()
            
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            running_loss += loss.item() * inputs.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            
            # Progress tracking
            batch_time = time.time() - batch_start_time
            progress = (batch_idx + 1) / total_batches * 100
            elapsed = time.time() - eval_start_time
            eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
            
            print(f"Val Batch {batch_idx+1:3d}/{total_batches} | "
                  f"Progress: {progress:5.1f}% | "
                  f"Loss: {loss.item():.4f} | "
                  f"Time: {batch_time:.2f}s | "
                  f"ETA: {str(timedelta(seconds=int(eta)))}")
    
    eval_time = time.time() - eval_start_time
    avg_loss = running_loss / len(loader.dataset)
    print(f"\nEvaluation completed in {str(timedelta(seconds=int(eval_time)))}")
    print(f"Average Validation Loss: {avg_loss:.4f}")
    
    return np.array(y_true), np.array(y_pred), np.array(y_probs)

def prepare_dataset(data_dir):
    """Prepare dataset from cancer/non_cancer folders"""
    cancer_dir = os.path.join(data_dir, "cancer")
    non_cancer_dir = os.path.join(data_dir, "non_cancer")
    
    if not os.path.exists(cancer_dir) or not os.path.exists(non_cancer_dir):
        raise FileNotFoundError("Both 'cancer' and 'non_cancer' subdirectories must exist")
    
    # Create full dataset with transforms, explicitly specifying classes
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    
    # Filter out any unwanted directories and ensure only cancer/non_cancer
    valid_samples = []
    for sample_path, label in full_dataset.samples:
        class_name = full_dataset.classes[label]
        if class_name in ['cancer', 'non_cancer']:
            # Remap labels: cancer=0, non_cancer=1
            new_label = 0 if class_name == 'cancer' else 1
            valid_samples.append((sample_path, new_label))
    
    full_dataset.samples = valid_samples
    full_dataset.classes = ['cancer', 'non_cancer']
    full_dataset.class_to_idx = {'cancer': 0, 'non_cancer': 1}
    
    print(f"Dataset classes: {full_dataset.classes}")
    print(f"Class to index mapping: {full_dataset.class_to_idx}")
    print(f"Total samples: {len(full_dataset.samples)}")
    
    # Get class weights to handle imbalance
    class_counts = [0, 0]
    for _, label in full_dataset.samples:
        if 0 <= label < len(class_counts):
            class_counts[label] += 1
        else:
            print(f"Warning: Invalid label {label} found")
    
    print(f"Class counts: {class_counts}")
    total_samples = sum(class_counts)
    class_weights = [total_samples / (2 * count) for count in class_counts]
    sample_weights = [class_weights[label] for _, label in full_dataset.samples]
    
    # Split dataset
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        stratify=[label for _, label in full_dataset.samples],
        random_state=42
    )
    
    train_sampler = WeightedRandomSampler(
        [sample_weights[i] for i in train_idx], 
        len(train_idx)
    )
    
    # Create validation dataset with validation transforms
    val_dataset_full = datasets.ImageFolder(data_dir, transform=val_transforms)
    
    # Filter validation dataset the same way
    valid_val_samples = []
    for sample_path, label in val_dataset_full.samples:
        class_name = val_dataset_full.classes[label]
        if class_name in ['cancer', 'non_cancer']:
            new_label = 0 if class_name == 'cancer' else 1
            valid_val_samples.append((sample_path, new_label))
    
    val_dataset_full.samples = valid_val_samples
    val_dataset_full.classes = ['cancer', 'non_cancer']
    val_dataset_full.class_to_idx = {'cancer': 0, 'non_cancer': 1}
    
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_idx)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    
    return train_dataset, val_dataset, train_sampler

def train_model():
    print("Starting Blood Cancer Classification Training...")
    print(f"Using device: {CONFIG['device']}")
    print(f"Model: {CONFIG['model_name']}")
    
    # Prepare data
    train_dataset, val_dataset, train_sampler = prepare_dataset(CONFIG["data_dir"])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"],
        sampler=train_sampler,
        num_workers=0,  # Changed from 4 to 0 for CPU
        pin_memory=False  # Changed from True to False for CPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,  # Changed from 4 to 0 for CPU
        pin_memory=False  # Changed from True to False for CPU
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Calculate training time estimation
    total_samples = len(train_dataset)
    batches_per_epoch = len(train_loader)
    estimated_time_per_batch = 0.5  # seconds, will be updated after first batch
    estimated_total_time = batches_per_epoch * estimated_time_per_batch * CONFIG["epochs"]
    
    print(f"\nTRAINING TIME ESTIMATION:")
    print(f"   Total samples: {total_samples}")
    print(f"   Batches per epoch: {batches_per_epoch}")
    print(f"   Total epochs: {CONFIG['epochs']}")
    print(f"   Estimated total time: {str(timedelta(seconds=int(estimated_total_time)))}")
    print(f"   Estimated completion: {(time.time() + estimated_total_time):.0f}")
    print("=" * 80)
    
    # Model and training setup
    model = get_model(CONFIG["model_name"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    best_val_auc = 0
    start_epoch = 0
    training_start_time = time.time()
    
    # Create checkpoint directory
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    
    # Load last checkpoint if exists
    last_checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "last_checkpoint.pth")
    if os.path.exists(last_checkpoint_path):
        start_epoch, _, _ = load_checkpoint(model, optimizer, last_checkpoint_path)
    
    print(f"\nSTARTING TRAINING FROM EPOCH {start_epoch + 1}")
    print("=" * 80)
    
    for epoch in range(start_epoch, CONFIG["epochs"]):
        epoch_start_time = time.time()
        
        # Training with detailed progress
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG["device"], epoch+1, CONFIG["epochs"])
        
        # Evaluation with detailed progress
        y_true, y_pred, y_probs = evaluate(model, val_loader, criterion, CONFIG["device"])
        
        val_auc = roc_auc_score(y_true, y_probs)
        val_loss = criterion(torch.tensor(y_probs), torch.tensor(y_true).float()).item()
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        remaining_epochs = CONFIG["epochs"] - (epoch + 1)
        estimated_remaining = epoch_time * remaining_epochs
        
        print(f"\nEPOCH {epoch+1}/{CONFIG['epochs']} SUMMARY:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val AUC: {val_auc:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Epoch Time: {str(timedelta(seconds=int(epoch_time)))}")
        print(f"   Total Elapsed: {str(timedelta(seconds=int(total_elapsed)))}")
        print(f"   Est. Remaining: {str(timedelta(seconds=int(estimated_remaining)))}")
        print(f"   Best AUC so far: {best_val_auc:.4f}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % CONFIG["save_every"] == 0:
            checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch+1, val_loss, val_auc, checkpoint_path)
        
        # Always save last checkpoint
        save_checkpoint(model, optimizer, epoch+1, val_loss, val_auc, last_checkpoint_path)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"   NEW BEST MODEL! AUC: {val_auc:.4f}")
        
        # Print detailed metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\nDETAILED CLASSIFICATION REPORT:")
            print(classification_report(y_true, y_pred, target_names=["Cancer", "Non-Cancer"]))
        
        print("=" * 80)
    
    total_training_time = time.time() - training_start_time
    print(f"\nTRAINING COMPLETED!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {CONFIG['model_save_path']}")
    print(f"Total training time: {str(timedelta(seconds=int(total_training_time)))}")
    print(f"Checkpoints saved to: {CONFIG['checkpoint_dir']}")

if __name__ == "__main__":
    train_model()
