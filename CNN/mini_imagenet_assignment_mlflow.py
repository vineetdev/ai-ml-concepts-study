"""
Assignment: Module 4 (Deep Learning) - Image Classification on Mini-ImageNet
==============================================================================

This notebook implements:
- Part A: Systematic CNN architecture experimentation (50 marks)
- Part B: Occlusion sensitivity analysis (50 marks)

Dataset: Mini-ImageNet (33 classes, 84x84 images)
- Training: 33 classes × 400 images = 13,200 images
- Validation: 33 classes × 100 images = 3,300 images
- Test: 33 classes × 100 images = 3,300 images

Framework: PyTorch
Experiment Tracking: MLflow
Hardware: CPU
"""

# ============================================================================
# CELL 1: Import Required Libraries
# ============================================================================
"""
Importing all necessary libraries for:
- Deep learning (PyTorch)
- Data handling (torchvision, PIL)
- Numerical operations (numpy)
- Visualization (matplotlib)
- Progress tracking (tqdm)
- Results storage (pandas)
- Experiment tracking (MLflow)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# MLflow for experiment tracking
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"MLflow version: {mlflow.__version__}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# ============================================================================
# CELL 1B: Initialize MLflow Experiment Tracking
# ============================================================================
"""
MLflow Setup:
- Creates an experiment to track all runs
- Logs parameters, metrics, and artifacts
- Provides UI for comparing experiments

Benefits:
- Compare all 10 configurations easily
- Track metrics over time
- Store model artifacts
- Reproducible experiments

To view results: Run 'mlflow ui' in terminal, then open http://localhost:5000
"""

# Set up MLflow experiment
EXPERIMENT_NAME = "Mini-ImageNet-CNN-Classification"
mlflow.set_experiment(EXPERIMENT_NAME)

# Set tracking URI (local directory)
MLFLOW_TRACKING_URI = "./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("="*70)
print("MLFLOW EXPERIMENT TRACKING INITIALIZED")
print("="*70)
print(f"Experiment name: {EXPERIMENT_NAME}")
print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"\nTo view experiments, run in terminal:")
print(f"  mlflow ui")
print(f"Then open: http://localhost:5000")
print("="*70)


# ============================================================================
# CELL 2: Download and Extract Dataset
# ============================================================================
"""
Automatically download and extract the Mini-ImageNet dataset.

The dataset will be downloaded from:
https://owncloud.iitd.ac.in/nextcloud/index.php/s/W3dNgKKHBQo4eAN

Steps:
1. Download the dataset (if not already downloaded)
2. Extract the archive
3. Set up paths to train/val/test folders
"""

import urllib.request
import zipfile
import shutil
from pathlib import Path

# Dataset configuration
DATASET_URL = "https://owncloud.iitd.ac.in/nextcloud/index.php/s/W3dNgKKHBQo4eAN/download"
DATASET_DIR = "./mini_imagenet_dataset"  # Local directory to store dataset
DATASET_ZIP = "./mini_imagenet.zip"  # Temporary zip file

def download_dataset(url, save_path):
    """
    Download dataset from URL with progress indication
    """
    print(f"Downloading dataset from: {url}")
    print("This may take several minutes depending on your internet connection...")
    
    def reporthook(count, block_size, total_size):
        """Progress bar for download"""
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownload progress: {percent}% ", end='')
    
    try:
        urllib.request.urlretrieve(url, save_path, reporthook=reporthook)
        print("\n✓ Download completed!")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

def extract_dataset(zip_path, extract_to):
    """
    Extract zip file to specified directory
    """
    print(f"\nExtracting dataset to: {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✓ Extraction completed!")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

# Check if dataset already exists
if os.path.exists(DATASET_DIR):
    print(f"Dataset directory '{DATASET_DIR}' already exists.")
    print("Skipping download. Delete the directory if you want to re-download.")
else:
    print("="*70)
    print("DATASET DOWNLOAD AND SETUP")
    print("="*70)
    
    # Download dataset
    if download_dataset(DATASET_URL, DATASET_ZIP):
        # Extract dataset
        if extract_dataset(DATASET_ZIP, DATASET_DIR):
            # Clean up zip file
            print("\nCleaning up...")
            os.remove(DATASET_ZIP)
            print("✓ Temporary files removed")
        else:
            print("Please check the zip file and try again.")
    else:
        print("\n⚠️ Dataset download failed!")
        print("Please manually download from:")
        print("https://owncloud.iitd.ac.in/nextcloud/index.php/s/W3dNgKKHBQo4eAN")
        print(f"And extract to: {DATASET_DIR}")

# Find the actual data directory (handle nested folders)
print("\n" + "="*70)
print("LOCATING DATASET FOLDERS")
print("="*70)

# Search for train/val/test folders
def find_data_folders(root_dir):
    """
    Recursively search for train, val, test folders
    Returns the parent directory containing these folders
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'train' in dirnames and 'val' in dirnames and 'test' in dirnames:
            return dirpath
    return None

DATA_PATH = find_data_folders(DATASET_DIR)

if DATA_PATH is None:
    # If not found in standard structure, check if extracted directly
    if os.path.exists(os.path.join(DATASET_DIR, 'train')):
        DATA_PATH = DATASET_DIR
    else:
        print("⚠️ Could not find train/val/test folders!")
        print("Please check the extracted dataset structure.")
        DATA_PATH = DATASET_DIR

print(f"Dataset base path: {DATA_PATH}")

# Set paths for train, validation, and test sets
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
VAL_PATH = os.path.join(DATA_PATH, 'val')
TEST_PATH = os.path.join(DATA_PATH, 'test')

print("\nDataset paths configured:")
print(f"  Training data: {TRAIN_PATH}")
print(f"  Validation data: {VAL_PATH}")
print(f"  Test data: {TEST_PATH}")

# Verify paths exist
if all([os.path.exists(p) for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]]):
    print("\n✓ All dataset folders found successfully!")
else:
    print("\n⚠️ Warning: Some dataset folders are missing!")
    print("Please verify the dataset structure.")


# ============================================================================
# CELL 3: Explore and Understand the Dataset
# ============================================================================
"""
This cell performs exploratory data analysis to understand:
- Number of classes
- Number of images per class
- Image dimensions and format
- Sample visualization
"""

def explore_dataset(data_path, split_name):
    """
    Explore dataset structure and statistics
    
    Args:
        data_path: Path to the dataset split (train/val/test)
        split_name: Name of the split for display
    """
    print(f"\n{'='*60}")
    print(f"Exploring {split_name} Dataset")
    print(f"{'='*60}")
    
    # Check if path exists
    if not os.path.exists(data_path):
        print(f"⚠️ Warning: Path {data_path} does not exist!")
        print("Please update DATA_PATH in Cell 2")
        return None
    
    # Get class folders
    classes = sorted([d for d in os.listdir(data_path) 
                     if os.path.isdir(os.path.join(data_path, d))])
    
    print(f"Number of classes: {len(classes)}")
    print(f"Class names (first 5): {classes[:5]}")
    
    # Count images per class
    class_counts = {}
    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        num_images = len([f for f in os.listdir(cls_path) 
                         if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))])
        class_counts[cls] = num_images
    
    print(f"\nImages per class:")
    print(f"  Min: {min(class_counts.values())}")
    print(f"  Max: {max(class_counts.values())}")
    print(f"  Mean: {np.mean(list(class_counts.values())):.2f}")
    print(f"  Total images: {sum(class_counts.values())}")
    
    # Load a sample image to check dimensions
    first_class = classes[0]
    first_class_path = os.path.join(data_path, first_class)
    sample_img_name = os.listdir(first_class_path)[0]
    sample_img_path = os.path.join(first_class_path, sample_img_name)
    
    sample_img = Image.open(sample_img_path)
    print(f"\nSample image properties:")
    print(f"  Size: {sample_img.size}")
    print(f"  Mode: {sample_img.mode}")
    print(f"  Format: {sample_img.format}")
    
    return classes

# Explore all splits
train_classes = explore_dataset(TRAIN_PATH, "Training")
val_classes = explore_dataset(VAL_PATH, "Validation")
test_classes = explore_dataset(TEST_PATH, "Test")


# ============================================================================
# CELL 4: Visualize Sample Images from Dataset
# ============================================================================
"""
Visualize sample images from different classes to understand:
- Image quality and content
- Class diversity
- Any data issues
"""

def visualize_samples(data_path, classes, num_classes=5, images_per_class=3):
    """
    Display sample images from multiple classes
    
    Args:
        data_path: Path to dataset split
        classes: List of class names
        num_classes: Number of classes to show
        images_per_class: Number of images per class to display
    """
    if not os.path.exists(data_path):
        print("Dataset path not found. Please update DATA_PATH.")
        return
    
    fig, axes = plt.subplots(num_classes, images_per_class, 
                            figsize=(12, num_classes*2.5))
    fig.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold')
    
    for i, cls in enumerate(classes[:num_classes]):
        cls_path = os.path.join(data_path, cls)
        img_files = [f for f in os.listdir(cls_path) 
                    if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
        
        for j in range(images_per_class):
            img_path = os.path.join(cls_path, img_files[j])
            img = Image.open(img_path)
            
            ax = axes[i, j] if num_classes > 1 else axes[j]
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(f'Class: {cls}', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Visualize samples from training set
if train_classes:
    visualize_samples(TRAIN_PATH, train_classes)


# ============================================================================
# CELL 5: Define Data Transformations
# ============================================================================
"""
Data transformations for:
1. Training: Data augmentation to improve generalization
   - Random horizontal flip
   - Random rotation
   - Normalization
   
2. Validation/Test: Only normalization (no augmentation)

Why normalize?
- Neural networks work better with normalized inputs (mean=0, std=1)
- Helps with gradient flow and faster convergence
"""

# Mean and std for Mini-ImageNet (calculated from dataset)
# Using ImageNet statistics as approximation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Training transformations with data augmentation
train_transform = transforms.Compose([
    transforms.Resize((84, 84)),  # Ensure consistent size
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
    transforms.RandomRotation(degrees=15),  # Rotate up to ±15 degrees
    transforms.ToTensor(),  # Convert PIL Image to tensor [0, 1]
    transforms.Normalize(mean=MEAN, std=STD)  # Normalize to mean=0, std=1
])

# Validation/Test transformations (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

print("Data transformations defined:")
print("\nTraining transforms (with augmentation):")
print("  - Resize to 84×84")
print("  - Random horizontal flip (p=0.5)")
print("  - Random rotation (±15°)")
print("  - Convert to tensor")
print("  - Normalize")

print("\nTest/Val transforms (no augmentation):")
print("  - Resize to 84×84")
print("  - Convert to tensor")
print("  - Normalize")


# ============================================================================
# CELL 6: Create PyTorch Datasets and DataLoaders
# ============================================================================
"""
DataLoader wraps a Dataset and provides:
- Batching: Groups images together for efficient processing
- Shuffling: Randomizes order each epoch (for training)
- Parallel loading: num_workers for faster data loading

Batch size:
- Larger batch = faster training but more memory
- Smaller batch = better generalization but slower
- For CPU: 32-64 is reasonable
"""

BATCH_SIZE = 32  # Adjust based on your CPU memory

def create_dataloaders(train_path, val_path, test_path, batch_size=32):
    """
    Create PyTorch datasets and dataloaders
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    # Check if paths exist
    if not all([os.path.exists(p) for p in [train_path, val_path, test_path]]):
        print("⚠️ Dataset paths not found! Please update DATA_PATH in Cell 2")
        return None, None, None, 0
    
    # Create datasets using ImageFolder
    # ImageFolder automatically assigns labels based on folder names
    train_dataset = ImageFolder(root=train_path, transform=train_transform)
    val_dataset = ImageFolder(root=val_path, transform=test_transform)
    test_dataset = ImageFolder(root=test_path, transform=test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=2,  # Parallel data loading (adjust based on CPU cores)
        pin_memory=False  # Set to True if using GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=2
    )
    
    num_classes = len(train_dataset.classes)
    
    print(f"Datasets created successfully!")
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    print(f"  Number of classes: {num_classes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches per epoch: {len(train_loader)}")
    
    return train_loader, val_loader, test_loader, num_classes, train_dataset.classes

# Create dataloaders
train_loader, val_loader, test_loader, num_classes, class_names = create_dataloaders(
    TRAIN_PATH, VAL_PATH, TEST_PATH, BATCH_SIZE
)


# ============================================================================
# CELL 7: Define CNN Model Architecture (Flexible)
# ============================================================================
"""
Creating a flexible CNN class that allows us to experiment with:
1. Number of convolutional layers
2. Number of filters in each layer
3. Pooling strategies
4. Number of fully connected layers
5. Stride values

CNN Architecture Components:
- Conv Layer: Extracts features using learnable filters
- ReLU: Non-linear activation (helps learn complex patterns)
- MaxPool: Reduces spatial dimensions (downsampling)
- Fully Connected: Combines features for classification
- Dropout: Prevents overfitting by randomly dropping neurons
"""

class FlexibleCNN(nn.Module):
    def __init__(self, num_classes, config):
        """
        Flexible CNN architecture
        
        Args:
            num_classes: Number of output classes
            config: Dictionary with architecture parameters
                - conv_layers: List of filter counts [64, 128, 256]
                - kernel_size: Conv kernel size (default 3)
                - stride: Conv stride (default 1)
                - pool_size: MaxPool kernel size (default 2)
                - pool_stride: MaxPool stride (default 2)
                - fc_layers: List of FC layer sizes [512, 256]
                - dropout: Dropout probability (default 0.5)
        """
        super(FlexibleCNN, self).__init__()
        
        self.config = config
        conv_filters = config.get('conv_layers', [64, 128, 256])
        kernel_size = config.get('kernel_size', 3)
        conv_stride = config.get('stride', 1)
        pool_size = config.get('pool_size', 2)
        pool_stride = config.get('pool_stride', 2)
        fc_sizes = config.get('fc_layers', [512])
        dropout_prob = config.get('dropout', 0.5)
        
        # Build convolutional layers dynamically
        self.conv_layers = nn.ModuleList()
        in_channels = 3  # RGB images
        
        for out_channels in conv_filters:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 
                             kernel_size=kernel_size, 
                             stride=conv_stride, 
                             padding=kernel_size//2),
                    nn.BatchNorm2d(out_channels),  # Normalize activations
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
                )
            )
            in_channels = out_channels
        
        # Calculate size after conv layers
        # Starting size: 84x84
        # After each conv+pool: size = (size) / pool_stride
        size_after_conv = 84
        for _ in conv_filters:
            size_after_conv = size_after_conv // pool_stride
        
        flattened_size = conv_filters[-1] * size_after_conv * size_after_conv
        
        # Build fully connected layers dynamically
        self.fc_layers = nn.ModuleList()
        in_features = flattened_size
        
        for fc_size in fc_sizes:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, fc_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_prob)
                )
            )
            in_features = fc_size
        
        # Final classification layer
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor [batch_size, 3, 84, 84]
        
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # [batch_size, flattened_features]
        
        # Pass through fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # Final classification
        x = self.classifier(x)
        
        return x

print("Flexible CNN architecture defined!")
print("\nExample configuration:")
example_config = {
    'conv_layers': [64, 128, 256],
    'kernel_size': 3,
    'stride': 1,
    'pool_size': 2,
    'pool_stride': 2,
    'fc_layers': [512, 256],
    'dropout': 0.5
}
print(example_config)


# ============================================================================
# CELL 8: Training and Evaluation Functions
# ============================================================================
"""
Define functions for:
1. Training: Update model weights using backpropagation
2. Evaluation: Test model performance without updating weights
3. Metrics: Calculate accuracy

Training Process:
1. Forward pass: Compute predictions
2. Calculate loss: How wrong are predictions?
3. Backward pass: Compute gradients
4. Update weights: Adjust parameters to reduce loss
"""

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()  # Set model to training mode (enables dropout, etc.)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar for training
    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        
        # Calculate statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on validation/test set
    
    Returns:
        Average loss and accuracy
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Don't compute gradients (saves memory and time)
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, model_name="Model", config=None, 
                mlflow_run=None):
    """
    Complete training loop with validation and MLflow tracking
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: CPU or GPU
        model_name: Name for display
        config: Configuration dictionary
        mlflow_run: Active MLflow run (optional)
    
    Returns:
        Dictionary with training history
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Log metrics to MLflow (per epoch)
        if mlflow_run:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✓ New best validation accuracy!")
            
            # Log best model to MLflow
            if mlflow_run:
                mlflow.log_metric("best_val_acc", best_val_acc)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Log training time to MLflow
    if mlflow_run:
        mlflow.log_metric("training_time_seconds", total_time)
    
    return history, best_val_acc

print("Training and evaluation functions defined (with MLflow integration)!")


# ============================================================================
# CELL 9: PART A - Define 10 Experimental Configurations
# ============================================================================
"""
Systematic experimentation with 10 configurations:

Strategy: Start with baseline and vary ONE parameter at a time
1. Baseline (simple)
2-4. Vary number of conv layers
5-6. Vary number of filters
7. Vary pooling strategy
8-9. Vary FC layers
10. Best combination

This systematic approach helps isolate the effect of each parameter.
"""

# Configuration dictionary for all experiments
# Each configuration is numbered and has a descriptive name

configurations = {
    # CONFIG 1: BASELINE - Simple starting point
    'config_1': {
        'name': 'Baseline (2 Conv, 1 FC)',
        'conv_layers': [32, 64],
        'kernel_size': 3,
        'stride': 1,
        'pool_size': 2,
        'pool_stride': 2,
        'fc_layers': [128],
        'dropout': 0.5,
        'epochs': 5,  # Quick experiments for CPU
        'lr': 0.001
    },
    
    # CONFIG 2: Increase conv layers depth
    'config_2': {
        'name': 'Deeper (3 Conv layers)',
        'conv_layers': [32, 64, 128],
        'kernel_size': 3,
        'stride': 1,
        'pool_size': 2,
        'pool_stride': 2,
        'fc_layers': [128],
        'dropout': 0.5,
        'epochs': 5,
        'lr': 0.001
    },
    
    # CONFIG 3: Even deeper network
    'config_3': {
        'name': 'Very Deep (4 Conv layers)',
        'conv_layers': [32, 64, 128, 256],
        'kernel_size': 3,
        'stride': 1,
        'pool_size': 2,
        'pool_stride': 2,
        'fc_layers': [128],
        'dropout': 0.5,
        'epochs': 5,
        'lr': 0.001
    },
    
    # CONFIG 4: Test shallow network
    'config_4': {
        'name': 'Shallow (1 Conv layer only)',
        'conv_layers': [64],
        'kernel_size': 3,
        'stride': 1,
        'pool_size': 2,
        'pool_stride': 2,
        'fc_layers': [128],
        'dropout': 0.5,
        'epochs': 5,
        'lr': 0.001
    },
    
    # CONFIG 5: Increase filter capacity
    'config_5': {
        'name': 'More Filters (2 Conv, wider)',
        'conv_layers': [64, 128],  # Double the filters from baseline
        'kernel_size': 3,
        'stride': 1,
        'pool_size': 2,
        'pool_stride': 2,
        'fc_layers': [128],
        'dropout': 0.5,
        'epochs': 5,
        'lr': 0.001
    },
    
    # CONFIG 6: Very wide network
    'config_6': {
        'name': 'Very Wide Filters',
        'conv_layers': [128, 256],
        'kernel_size': 3,
        'stride': 1,
        'pool_size': 2,
        'pool_stride': 2,
        'fc_layers': [128],
        'dropout': 0.5,
        'epochs': 5,
        'lr': 0.001
    },
    
    # CONFIG 7: Different pooling strategy
    'config_7': {
        'name': 'No Pooling After Each Conv',
        'conv_layers': [32, 64, 128],
        'kernel_size': 3,
        'stride': 2,  # Use stride instead of pooling
        'pool_size': 1,  # No pooling
        'pool_stride': 1,
        'fc_layers': [128],
        'dropout': 0.5,
        'epochs': 5,
        'lr': 0.001
    },
    
    # CONFIG 8: Larger FC layers
    'config_8': {
        'name': 'Larger FC Layers',
        'conv_layers': [32, 64],
        'kernel_size': 3,
        'stride': 1,
        'pool_size': 2,
        'pool_stride': 2,
        'fc_layers': [512, 256],  # Two larger FC layers
        'dropout': 0.5,
        'epochs': 5,
        'lr': 0.001
    },
    
    # CONFIG 9: More FC layers
    'config_9': {
        'name': 'Multiple FC Layers',
        'conv_layers': [32, 64],
        'kernel_size': 3,
        'stride': 1,
        'pool_size': 2,
        'pool_stride': 2,
        'fc_layers': [256, 128, 64],  # Three FC layers
        'dropout': 0.5,
        'epochs': 5,
        'lr': 0.001
    },
    
    # CONFIG 10: Best combination (moderate complexity)
    'config_10': {
        'name': 'Balanced Architecture',
        'conv_layers': [64, 128, 256],  # Good depth
        'kernel_size': 3,
        'stride': 1,
        'pool_size': 2,
        'pool_stride': 2,
        'fc_layers': [512, 256],  # Good capacity
        'dropout': 0.5,
        'epochs': 8,  # Train longer for best model
        'lr': 0.001
    }
}

print("10 Experimental Configurations Defined:")
print("="*60)
for config_id, config in configurations.items():
    print(f"\n{config_id.upper()}: {config['name']}")
    print(f"  Conv layers: {config['conv_layers']}")
    print(f"  FC layers: {config['fc_layers']}")
    print(f"  Stride: {config['stride']}, Pooling: {config['pool_size']}")
    print(f"  Epochs: {config['epochs']}")


# ============================================================================
# CELL 10: PART A - Run All Experiments with MLflow Tracking
# ============================================================================
"""
Run all 10 configurations and collect results with MLflow tracking.

For each configuration:
1. Start MLflow run
2. Log parameters (architecture, hyperparameters)
3. Train model and log metrics
4. Test on test set
5. Log final results and artifacts
6. End MLflow run

This will take time on CPU. Estimated: 5-10 minutes per config
Total time: ~50-100 minutes for all 10 configs

MLflow will track:
- Parameters: conv_layers, fc_layers, stride, epochs, lr, etc.
- Metrics: train_loss, train_acc, val_loss, val_acc (per epoch)
- Final metrics: test_acc, best_val_acc, training_time
- Artifacts: training curves, model checkpoints
"""

# Storage for all results
all_results = []
all_histories = {}
device = torch.device('cpu')

# Check if dataloaders were created successfully
if train_loader is None:
    print("⚠️ Please update DATA_PATH and run previous cells first!")
else:
    print("Starting experiments with MLflow tracking...")
    print(f"Training on: {device}")
    print(f"Total configurations: {len(configurations)}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}\n")
    
    for config_id, config in configurations.items():
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT: {config_id.upper()}")
        print(f"# {config['name']}")
        print(f"{'#'*70}")
        
        # Start MLflow run for this configuration
        with mlflow.start_run(run_name=f"{config_id}_{config['name']}"):
            
            # Log configuration parameters
            mlflow.log_param("config_id", config_id)
            mlflow.log_param("config_name", config['name'])
            mlflow.log_param("conv_layers", str(config['conv_layers']))
            mlflow.log_param("num_conv_layers", len(config['conv_layers']))
            mlflow.log_param("fc_layers", str(config['fc_layers']))
            mlflow.log_param("num_fc_layers", len(config['fc_layers']))
            mlflow.log_param("kernel_size", config['kernel_size'])
            mlflow.log_param("stride", config['stride'])
            mlflow.log_param("pool_size", config['pool_size'])
            mlflow.log_param("pool_stride", config['pool_stride'])
            mlflow.log_param("dropout", config['dropout'])
            mlflow.log_param("epochs", config['epochs'])
            mlflow.log_param("learning_rate", config['lr'])
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("optimizer", "Adam")
            
            # Create model
            model = FlexibleCNN(num_classes=num_classes, config=config).to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            mlflow.log_param("total_parameters", total_params)
            mlflow.log_param("trainable_parameters", trainable_params)
            
            print(f"\nModel Architecture:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
            # Define loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            
            # Train model with MLflow tracking
            history, best_val_acc = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=config['epochs'],
                device=device,
                model_name=config['name'],
                config=config,
                mlflow_run=True  # Enable MLflow logging
            )
            
            # Evaluate on test set
            print("\nEvaluating on test set...")
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            
            # Log final test metrics to MLflow
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_acc", test_acc)
            mlflow.log_metric("final_train_acc", history['train_acc'][-1])
            mlflow.log_metric("final_val_acc", history['val_acc'][-1])
            
            # Calculate overfitting metric
            overfitting_gap = history['train_acc'][-1] - history['val_acc'][-1]
            mlflow.log_metric("overfitting_gap", overfitting_gap)
            
            # Create and log training curves
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Accuracy plot
            axes[0].plot(history['train_acc'], label='Train Acc', marker='o')
            axes[0].plot(history['val_acc'], label='Val Acc', marker='s')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy (%)')
            axes[0].set_title(f'{config["name"]} - Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Loss plot
            axes[1].plot(history['train_loss'], label='Train Loss', marker='o')
            axes[1].plot(history['val_loss'], label='Val Loss', marker='s')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title(f'{config["name"]} - Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save and log plot to MLflow
            plot_path = f'training_curves_{config_id}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(plot_path)
            plt.close()
            
            # Log model to MLflow (optional, can be large)
            # mlflow.pytorch.log_model(model, "model")
            
            # Store results
            result = {
                'Config ID': config_id,
                'Name': config['name'],
                'Conv Layers': str(config['conv_layers']),
                'FC Layers': str(config['fc_layers']),
                'Stride': config['stride'],
                'Pool Size': config['pool_size'],
                'Epochs': config['epochs'],
                'Total Params': total_params,
                'Best Val Acc': f"{best_val_acc:.2f}%",
                'Test Acc': f"{test_acc:.2f}%",
                'Final Train Acc': f"{history['train_acc'][-1]:.2f}%",
                'Final Val Acc': f"{history['val_acc'][-1]:.2f}%",
                'Overfitting Gap': f"{overfitting_gap:.2f}%"
            }
            
            all_results.append(result)
            all_histories[config_id] = history
            
            # Log result summary as tag
            mlflow.set_tag("status", "completed")
            mlflow.set_tag("test_accuracy", f"{test_acc:.2f}%")
            
            print(f"\n✓ {config_id} completed and logged to MLflow!")
            print(f"  Run ID: {mlflow.active_run().info.run_id}")
            print("-" * 70)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\n🎯 To view all results in MLflow UI:")
    print(f"   1. Open terminal in this directory")
    print(f"   2. Run: mlflow ui")
    print(f"   3. Open browser: http://localhost:5000")
    print(f"   4. Select experiment: {EXPERIMENT_NAME}")
    print(f"\nYou can compare all 10 configurations side-by-side!")


# ============================================================================
# CELL 11: PART A - Results Table and MLflow Comparison
# ============================================================================
"""
Display all experimental results in a formatted table.
This table will be included in your report.

MLflow provides additional features:
- Interactive comparison of all runs
- Metric plots over epochs
- Parameter correlation analysis
- Model versioning
"""

# Create DataFrame for better visualization
results_df = pd.DataFrame(all_results)

print("\n" + "="*100)
print("PART A: EXPERIMENTAL RESULTS TABLE")
print("="*100)
print()

# Display full table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(results_df.to_string(index=False))
print()

# Save to CSV for report
results_df.to_csv('part_a_results.csv', index=False)
print("✓ Results saved to 'part_a_results.csv'")

# Log results table to MLflow as artifact
try:
    # Create a parent run to store summary artifacts
    with mlflow.start_run(run_name="Part_A_Summary"):
        mlflow.log_artifact('part_a_results.csv')
        
        # Log summary statistics
        test_accs = [float(r['Test Acc'].replace('%', '')) for r in all_results]
        mlflow.log_metric("summary_best_test_acc", max(test_accs))
        mlflow.log_metric("summary_avg_test_acc", np.mean(test_accs))
        mlflow.log_metric("summary_worst_test_acc", min(test_accs))
        mlflow.log_metric("summary_std_test_acc", np.std(test_accs))
        
        print("✓ Results logged to MLflow")
except Exception as e:
    print(f"Note: Could not log to MLflow: {e}")

print("\n" + "="*100)
print("📊 MLFLOW ANALYSIS TIPS:")
print("="*100)
print("""
In MLflow UI (http://localhost:5000), you can:

1. COMPARE RUNS:
   • Click 'Compare' checkbox for multiple runs
   • View side-by-side parameter and metric comparison
   • Identify which parameters correlate with better performance

2. VISUALIZE METRICS:
   • Click on any run to see detailed metrics
   • View training/validation curves over epochs
   • Compare learning dynamics across configurations

3. FILTER AND SORT:
   • Sort by test_acc to find best models
   • Filter by parameters (e.g., num_conv_layers = 3)
   • Search for specific configurations

4. EXPORT DATA:
   • Download CSV of all runs
   • Export plots and artifacts
   • Share experiment results with team

5. ANALYZE CORRELATIONS:
   • Use MLflow's parallel coordinates plot
   • See which hyperparameters matter most
   • Identify optimal parameter ranges
""")


# ============================================================================
# CELL 12: PART A - Visualize Training Curves
# ============================================================================
"""
Plot training and validation curves for all configurations.
This helps understand:
- Convergence behavior
- Overfitting (train acc >> val acc)
- Learning dynamics
"""

def plot_training_curves(histories, configurations, metric='acc'):
    """
    Plot training curves for all configurations
    
    Args:
        histories: Dictionary of training histories
        configurations: Configuration dictionary
        metric: 'acc' or 'loss'
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, (config_id, history) in enumerate(histories.items()):
        ax = axes[idx]
        
        if metric == 'acc':
            ax.plot(history['train_acc'], label='Train Acc', marker='o')
            ax.plot(history['val_acc'], label='Val Acc', marker='s')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f"{configurations[config_id]['name']}\n(Best Val: {max(history['val_acc']):.2f}%)")
        else:
            ax.plot(history['train_loss'], label='Train Loss', marker='o')
            ax.plot(history['val_loss'], label='Val Loss', marker='s')
            ax.set_ylabel('Loss')
            ax.set_title(f"{configurations[config_id]['name']}")
        
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'part_a_training_curves_{metric}.png', dpi=150, bbox_inches='tight')
    plt.show()

# Plot accuracy curves
if all_histories:
    plot_training_curves(all_histories, configurations, metric='acc')
    plot_training_curves(all_histories, configurations, metric='loss')
    print("✓ Training curves saved")


# ============================================================================
# CELL 13: PART A - Performance Analysis and Comparison
# ============================================================================
"""
Analyze results to understand:
1. Which parameters had the biggest impact?
2. Where does performance plateau?
3. Signs of overfitting or underfitting?
"""

if all_results:
    # Extract test accuracies (remove % sign and convert to float)
    test_accs = [float(r['Test Acc'].replace('%', '')) for r in all_results]
    config_names = [r['Name'] for r in all_results]
    
    # Create bar chart comparing test accuracies
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(config_names)), test_accs, color='steelblue', alpha=0.8)
    plt.xlabel('Configuration', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Test Accuracy Comparison Across All Configurations', 
              fontsize=14, fontweight='bold')
    plt.xticks(range(len(config_names)), 
               [f"C{i+1}" for i in range(len(config_names))], 
               rotation=0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{test_accs[idx]:.2f}%',
                ha='center', va='bottom', fontsize=9)
    
    # Add legend showing config names
    legend_labels = [f"C{i+1}: {name}" for i, name in enumerate(config_names)]
    plt.figtext(0.5, -0.15, '\n'.join(legend_labels[:5]), 
                ha='center', fontsize=8, family='monospace')
    plt.figtext(0.5, -0.30, '\n'.join(legend_labels[5:]), 
                ha='center', fontsize=8, family='monospace')
    
    plt.tight_layout()
    plt.savefig('part_a_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Find best and worst
    best_idx = test_accs.index(max(test_accs))
    worst_idx = test_accs.index(min(test_accs))
    
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    print(f"\n🏆 BEST Configuration:")
    print(f"   {config_names[best_idx]}")
    print(f"   Test Accuracy: {test_accs[best_idx]:.2f}%")
    print(f"   Config: {all_results[best_idx]['Conv Layers']}")
    print(f"   FC Layers: {all_results[best_idx]['FC Layers']}")
    
    print(f"\n📉 WORST Configuration:")
    print(f"   {config_names[worst_idx]}")
    print(f"   Test Accuracy: {test_accs[worst_idx]:.2f}%")
    
    print(f"\n📊 Statistics:")
    print(f"   Mean Test Accuracy: {np.mean(test_accs):.2f}%")
    print(f"   Std Dev: {np.std(test_accs):.2f}%")
    print(f"   Range: {max(test_accs) - min(test_accs):.2f}%")


# ============================================================================
# CELL 14: PART A - Analyze Misclassified Images
# ============================================================================
"""
Look at some misclassified images from the best model to understand:
- What types of errors does the model make?
- Are there confusing classes?
- Is there a pattern in misclassifications?
"""

def get_misclassified_images(model, data_loader, device, class_names, num_samples=20):
    """
    Find misclassified images
    
    Returns:
        List of (image, true_label, predicted_label, confidence) tuples
    """
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Find misclassified
            incorrect = predicted.ne(labels)
            incorrect_indices = incorrect.nonzero(as_tuple=True)[0]
            
            for idx in incorrect_indices:
                if len(misclassified) >= num_samples:
                    return misclassified
                
                img = images[idx].cpu()
                true_label = labels[idx].item()
                pred_label = predicted[idx].item()
                confidence = probabilities[idx, pred_label].item()
                
                misclassified.append((img, true_label, pred_label, confidence))
    
    return misclassified


def visualize_misclassified(misclassified, class_names, num_display=12):
    """
    Display misclassified images with predictions
    """
    if not misclassified:
        print("No misclassified images found!")
        return
    
    num_display = min(num_display, len(misclassified))
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    fig.suptitle('Misclassified Images from Best Model', 
                 fontsize=16, fontweight='bold')
    
    for idx in range(num_display):
        img, true_label, pred_label, confidence = misclassified[idx]
        
        # Denormalize image for display
        img = img.numpy().transpose(1, 2, 0)
        mean = np.array(MEAN)
        std = np.array(STD)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(
            f'True: {class_names[true_label]}\n'
            f'Pred: {class_names[pred_label]}\n'
            f'Conf: {confidence:.2f}',
            fontsize=9,
            color='red'
        )
    
    # Hide unused subplots
    for idx in range(num_display, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('part_a_misclassified.png', dpi=150, bbox_inches='tight')
    plt.show()


# Find best model configuration
if all_results:
    test_accs_numeric = [float(r['Test Acc'].replace('%', '')) for r in all_results]
    best_config_idx = test_accs_numeric.index(max(test_accs_numeric))
    best_config_id = all_results[best_config_idx]['Config ID']
    best_config = configurations[best_config_id]
    
    print(f"\nAnalyzing misclassifications from: {best_config['name']}")
    print("Creating best model again...")
    
    # Recreate and retrain best model
    best_model = FlexibleCNN(num_classes=num_classes, config=best_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_config['lr'])
    
    # Quick retrain (or load saved weights if you saved them)
    print("Retraining best model for misclassification analysis...")
    train_model(best_model, train_loader, val_loader, criterion, optimizer,
                best_config['epochs'], device, "Best Model (Retrain)")
    
    # Get misclassified images
    print("\nFinding misclassified images...")
    misclassified = get_misclassified_images(best_model, test_loader, device, 
                                            class_names, num_samples=20)
    
    print(f"Found {len(misclassified)} misclassified images")
    
    # Visualize
    visualize_misclassified(misclassified, class_names, num_display=12)
    
    # Analyze patterns
    print("\n" + "="*70)
    print("MISCLASSIFICATION ANALYSIS")
    print("="*70)
    
    if misclassified:
        # Count confusion pairs
        confusion_pairs = {}
        for _, true_label, pred_label, _ in misclassified:
            pair = (class_names[true_label], class_names[pred_label])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        
        print("\nMost common confusions:")
        for idx, ((true_cls, pred_cls), count) in enumerate(sorted_pairs[:5], 1):
            print(f"  {idx}. {true_cls} → {pred_cls}: {count} times")
        
        print("\n💡 Observations:")
        print("   - Look for visually similar classes being confused")
        print("   - Check if low confidence predictions (< 0.5) indicate uncertainty")
        print("   - Identify if errors are due to image quality or inherent similarity")


# ============================================================================
# CELL 15: PART A - SUMMARY
# ============================================================================
"""
═══════════════════════════════════════════════════════════════════════════
                            PART A SUMMARY
═══════════════════════════════════════════════════════════════════════════
"""

print("\n" + "="*100)
print(" " * 35 + "PART A - SUMMARY")
print("="*100)

if all_results:
    print("\n📊 EXPERIMENTAL RESULTS:")
    print("-" * 100)
    
    # Summary statistics
    test_accs_vals = [float(r['Test Acc'].replace('%', '')) for r in all_results]
    
    print(f"\n1. PERFORMANCE METRICS:")
    print(f"   • Total configurations tested: {len(all_results)}")
    print(f"   • Best test accuracy: {max(test_accs_vals):.2f}%")
    print(f"   • Worst test accuracy: {min(test_accs_vals):.2f}%")
    print(f"   • Average test accuracy: {np.mean(test_accs_vals):.2f}%")
    print(f"   • Performance range: {max(test_accs_vals) - min(test_accs_vals):.2f}%")
    
    print(f"\n2. KEY FINDINGS:")
    print(f"   • Optimal conv layer depth: Based on configs 1-4 comparison")
    print(f"   • Optimal filter capacity: Based on configs 5-6 comparison")
    print(f"   • Pooling vs stride impact: Config 7 analysis")
    print(f"   • FC layer contribution: Configs 8-9 analysis")
    
    # Find where performance plateaus
    print(f"\n3. PERFORMANCE TRENDS:")
    
    # Group by conv depth
    depth_perf = {
        '1 conv': [r for r in all_results if '[64]' in r['Conv Layers']],
        '2 conv': [r for r in all_results if r['Conv Layers'].count(',') == 1],
        '3 conv': [r for r in all_results if r['Conv Layers'].count(',') == 2],
        '4 conv': [r for r in all_results if r['Conv Layers'].count(',') == 3]
    }
    
    print("   Conv Layer Depth Analysis:")
    for depth, configs in depth_perf.items():
        if configs:
            accs = [float(c['Test Acc'].replace('%', '')) for c in configs]
            print(f"      • {depth}: Avg = {np.mean(accs):.2f}% (n={len(accs)})")
    
    print(f"\n4. OVERFITTING ANALYSIS:")
    for result in all_results:
        train_acc = float(result['Final Train Acc'].replace('%', ''))
        val_acc = float(result['Final Val Acc'].replace('%', ''))
        gap = train_acc - val_acc
        
        if gap > 10:
            print(f"   ⚠️  {result['Name']}: Train-Val gap = {gap:.2f}% (possible overfitting)")
    
    print(f"\n5. EFFICIENCY:")
    print("   Parameter count vs Performance:")
    for result in all_results[:3]:  # Show top 3
        print(f"   • {result['Name']}: {result['Total Params']:,} params → {result['Test Acc']}")
    
    print(f"\n6. RECOMMENDATIONS FOR PART B:")
    best_config_name = all_results[best_config_idx]['Name']
    best_test_acc = all_results[best_config_idx]['Test Acc']
    print(f"   ✓ Using: {best_config_name}")
    print(f"   ✓ Test Accuracy: {best_test_acc}")
    print(f"   ✓ This model will be used for occlusion sensitivity analysis")
    
    print("\n" + "="*100)
    print("✓ PART A COMPLETED - Proceed to PART B for model interpretability")
    print("="*100)

else:
    print("\n⚠️  No results available. Please run the experiments in Cell 10 first.")


# ============================================================================
# CELL 16: PART B - Occlusion Sensitivity Setup
# ============================================================================
"""
═══════════════════════════════════════════════════════════════════════════
                            PART B: OCCLUSION SENSITIVITY
═══════════════════════════════════════════════════════════════════════════

Goal: Understand what parts of the image the model focuses on

Method:
1. Take an image
2. Systematically occlude (cover with gray) different regions
3. Measure how classification confidence drops
4. Create a heatmap showing which regions are most important

If confidence drops significantly when a region is occluded,
that region is important for classification.
"""

def apply_occlusion(image, center_i, center_j, window_size):
    """
    Apply gray occlusion patch to image
    
    Args:
        image: Input tensor [C, H, W]
        center_i: Row center of occlusion window
        center_j: Column center of occlusion window
        window_size: Size of square occlusion window (N×N)
    
    Returns:
        Occluded image
    """
    occluded = image.clone()
    
    # Calculate window boundaries
    half_window = window_size // 2
    
    i_start = max(0, center_i - half_window)
    i_end = min(image.shape[1], center_i + half_window)
    j_start = max(0, center_j - half_window)
    j_end = min(image.shape[2], center_j + half_window)
    
    # Apply gray occlusion (0.5 in normalized space)
    # This corresponds to mid-gray after normalization
    occluded[:, i_start:i_end, j_start:j_end] = 0.0
    
    return occluded


def occlusion_sensitivity(model, image, true_class, window_size, stride, device):
    """
    Perform occlusion sensitivity analysis on a single image
    
    Args:
        model: Trained CNN model
        image: Input image tensor [C, H, W]
        true_class: Ground truth class index
        window_size: Size of occlusion window (N×N)
        stride: Step size for sliding window
        device: CPU or GPU
    
    Returns:
        confidence_map: 2D array of confidence values
        original_confidence: Confidence before occlusion
    """
    model.eval()
    
    _, height, width = image.shape
    
    # Get original prediction confidence
    with torch.no_grad():
        original_output = model(image.unsqueeze(0).to(device))
        original_prob = F.softmax(original_output, dim=1)
        original_confidence = original_prob[0, true_class].item()
    
    # Initialize confidence map
    confidence_map = np.zeros((height, width))
    
    # Slide occlusion window across image
    print(f"Running occlusion analysis (window={window_size}, stride={stride})...")
    
    positions = []
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            positions.append((i, j))
    
    for i, j in tqdm(positions, desc="Occlusion", leave=False):
        # Apply occlusion
        occluded_img = apply_occlusion(image, i, j, window_size)
        
        # Get prediction
        with torch.no_grad():
            output = model(occluded_img.unsqueeze(0).to(device))
            prob = F.softmax(output, dim=1)
            confidence = prob[0, true_class].item()
        
        # Store confidence at this position
        # Fill the entire window region with this confidence value
        half_window = window_size // 2
        i_start = max(0, i - half_window)
        i_end = min(height, i + half_window)
        j_start = max(0, j - half_window)
        j_end = min(width, j + half_window)
        
        confidence_map[i_start:i_end, j_start:j_end] = confidence
    
    return confidence_map, original_confidence


def visualize_occlusion_result(image, confidence_map, original_confidence, 
                               true_class, predicted_class, class_names):
    """
    Visualize occlusion sensitivity results
    
    Shows:
    1. Original image
    2. Confidence heatmap
    3. Overlay
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Denormalize image for display
    img_display = image.numpy().transpose(1, 2, 0)
    mean = np.array(MEAN)
    std = np.array(STD)
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    
    # 1. Original image
    axes[0].imshow(img_display)
    axes[0].set_title(f'Original Image\nTrue: {class_names[true_class]}\n'
                     f'Pred: {class_names[predicted_class]}\n'
                     f'Confidence: {original_confidence:.3f}',
                     fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Confidence heatmap
    im = axes[1].imshow(confidence_map, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f'Occlusion Sensitivity Heatmap\n'
                     f'(Red = High importance, Blue = Low importance)',
                     fontsize=11, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Overlay
    axes[2].imshow(img_display)
    axes[2].imshow(confidence_map, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title('Overlay\n(Important regions highlighted)',
                     fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    return fig


# Parameters for occlusion
WINDOW_SIZE = 10  # Size of occlusion patch (N×N)
STRIDE = 4  # Step size for sliding window (smaller = more detailed but slower)

print("\n" + "="*70)
print("PART B: OCCLUSION SENSITIVITY ANALYSIS")
print("="*70)
print(f"\nParameters:")
print(f"  Occlusion window size: {WINDOW_SIZE}×{WINDOW_SIZE} pixels")
print(f"  Stride: {STRIDE} pixels")
print(f"  Number of images to analyze: 10")
print(f"\nInterpretation:")
print(f"  • RED regions: Model confidence drops when occluded (important for classification)")
print(f"  • BLUE regions: Model confidence stable when occluded (less important)")
print(f"  • This reveals if model focuses on object or background/context")


# ============================================================================
# CELL 17: PART B - Select Test Images for Analysis
# ============================================================================
"""
Select 10 diverse test images for occlusion analysis:
- Correctly classified with high confidence
- Some from different classes
- Preferably images where object is clearly visible
"""

def select_test_images_for_occlusion(model, test_loader, device, class_names, 
                                    num_images=10):
    """
    Select diverse test images for occlusion analysis
    
    Strategy:
    - Get correctly classified images
    - Select high confidence predictions
    - Diverse classes
    
    Returns:
        List of (image, true_label, predicted_label, confidence) tuples
    """
    model.eval()
    selected_images = []
    selected_classes = set()
    
    with torch.no_grad():
        for images, labels in test_loader:
            if len(selected_images) >= num_images:
                break
            
            images_dev, labels_dev = images.to(device), labels.to(device)
            outputs = model(images_dev)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted = probabilities.max(1)
            
            # Find correctly classified with high confidence
            correct = predicted.eq(labels_dev)
            
            for idx in range(len(images)):
                if len(selected_images) >= num_images:
                    break
                
                if correct[idx]:  # Correctly classified
                    true_label = labels[idx].item()
                    pred_label = predicted[idx].item()
                    confidence = confidences[idx].item()
                    
                    # Prefer diverse classes and high confidence
                    if (confidence > 0.7 and 
                        (true_label not in selected_classes or len(selected_images) < 5)):
                        
                        selected_images.append((
                            images[idx],
                            true_label,
                            pred_label,
                            confidence
                        ))
                        selected_classes.add(true_label)
    
    return selected_images


# Select images using best model from Part A
if all_results:
    print("\nSelecting 10 test images for occlusion analysis...")
    print("Criteria: Correctly classified, high confidence, diverse classes")
    
    selected_test_images = select_test_images_for_occlusion(
        best_model, test_loader, device, class_names, num_images=10
    )
    
    print(f"\n✓ Selected {len(selected_test_images)} images")
    print("\nSelected images:")
    for idx, (img, true_label, pred_label, conf) in enumerate(selected_test_images, 1):
        print(f"  {idx}. Class: {class_names[true_label]}, Confidence: {conf:.3f}")


# ============================================================================
# CELL 18: PART B - Run Occlusion Analysis on Selected Images
# ============================================================================
"""
Perform occlusion sensitivity analysis on all 10 selected images.

WARNING: This is computationally intensive!
For each image:
- Number of positions = (84/stride) × (84/stride) = (84/4)² = 441 forward passes
- Total: 10 images × 441 passes = 4,410 forward passes

Estimated time on CPU: 10-20 minutes for all 10 images
"""

occlusion_results = []

if selected_test_images:
    print("\n" + "="*70)
    print("RUNNING OCCLUSION SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"\nThis will take approximately 10-20 minutes on CPU...")
    print(f"Processing {len(selected_test_images)} images...\n")
    
    for idx, (image, true_label, pred_label, orig_conf) in enumerate(selected_test_images, 1):
        print(f"\n{'─'*70}")
        print(f"Image {idx}/10: {class_names[true_label]} (Confidence: {orig_conf:.3f})")
        print(f"{'─'*70}")
        
        # Perform occlusion analysis
        confidence_map, original_confidence = occlusion_sensitivity(
            model=best_model,
            image=image,
            true_class=true_label,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            device=device
        )
        
        # Visualize results
        fig = visualize_occlusion_result(
            image=image,
            confidence_map=confidence_map,
            original_confidence=original_confidence,
            true_class=true_label,
            predicted_class=pred_label,
            class_names=class_names
        )
        
        # Save figure
        fig.savefig(f'part_b_occlusion_image_{idx}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Store results
        occlusion_results.append({
            'image_idx': idx,
            'true_class': class_names[true_label],
            'confidence_map': confidence_map,
            'original_confidence': original_confidence,
            'min_confidence': confidence_map.min(),
            'max_confidence': confidence_map.max(),
            'confidence_drop': original_confidence - confidence_map.min()
        })
        
        print(f"  Original confidence: {original_confidence:.3f}")
        print(f"  Min confidence (max drop): {confidence_map.min():.3f}")
        print(f"  Max confidence drop: {original_confidence - confidence_map.min():.3f}")
        print(f"  ✓ Saved: part_b_occlusion_image_{idx}.png")
    
    print("\n" + "="*70)
    print("✓ OCCLUSION ANALYSIS COMPLETED")
    print("="*70)


# ============================================================================
# CELL 19: PART B - Analysis and Observations
# ============================================================================
"""
Analyze occlusion results to answer:
1. Does the model focus on the actual object?
2. Or does it use contextual/background cues?
3. Are there specific regions consistently important?
4. Does behavior vary across classes?
"""

print("\n" + "="*70)
print("OCCLUSION SENSITIVITY ANALYSIS - DETAILED OBSERVATIONS")
print("="*70)

if occlusion_results:
    print("\n📊 QUANTITATIVE ANALYSIS:")
    print("-" * 70)
    
    for result in occlusion_results:
        print(f"\nImage {result['image_idx']}: {result['true_class']}")
        print(f"  Original confidence: {result['original_confidence']:.3f}")
        print(f"  Minimum confidence after occlusion: {result['min_confidence']:.3f}")
        print(f"  Maximum confidence drop: {result['confidence_drop']:.3f}")
        
        # Interpretation
        if result['confidence_drop'] > 0.5:
            print(f"  → Strong localization: Model heavily relies on specific regions")
        elif result['confidence_drop'] > 0.3:
            print(f"  → Moderate localization: Some regions more important than others")
        else:
            print(f"  → Weak localization: Model may use distributed features or context")
    
    # Overall statistics
    avg_drop = np.mean([r['confidence_drop'] for r in occlusion_results])
    max_drop = max([r['confidence_drop'] for r in occlusion_results])
    min_drop = min([r['confidence_drop'] for r in occlusion_results])
    
    print("\n" + "─"*70)
    print("OVERALL STATISTICS:")
    print(f"  Average confidence drop: {avg_drop:.3f}")
    print(f"  Maximum confidence drop: {max_drop:.3f}")
    print(f"  Minimum confidence drop: {min_drop:.3f}")
    
    print("\n💡 KEY OBSERVATIONS:")
    print("-" * 70)
    print("""
    Based on the heatmaps, observe:
    
    1. OBJECT LOCALIZATION:
       • RED/HOT regions indicate where the model "looks" for classification
       • If red regions align with actual object → good object detection
       • If red regions are scattered/background → model uses context cues
    
    2. SPATIAL ATTENTION:
       • Concentrated hotspots → model focuses on specific features (eyes, edges, etc.)
       • Distributed attention → model uses holistic image information
    
    3. ROBUSTNESS:
       • Small confidence drops → model is robust to occlusion
       • Large confidence drops → model heavily depends on specific regions
    
    4. CLASS-SPECIFIC BEHAVIOR:
       • Some classes may show clearer localization than others
       • Compare patterns across different object categories
    
    5. CONTEXT vs OBJECT:
       • If background occlusion affects confidence → uses context
       • If only object occlusion affects confidence → proper object recognition
    """)
    
    print("\n📝 INTERPRETATION GUIDE:")
    print("-" * 70)
    print("For your report, comment on:")
    print("  1. Do heatmaps align with actual object locations?")
    print("  2. Does the model use object features or contextual cues?")
    print("  3. Are there specific discriminative parts (e.g., animal faces, wheels)?")
    print("  4. How robust is the model to partial occlusion?")
    print("  5. Do different classes show different attention patterns?")


# ============================================================================
# CELL 20: PART B - SUMMARY
# ============================================================================
"""
═══════════════════════════════════════════════════════════════════════════
                            PART B SUMMARY
═══════════════════════════════════════════════════════════════════════════
"""

print("\n" + "="*100)
print(" " * 35 + "PART B - SUMMARY")
print("="*100)

if occlusion_results:
    print("\n📊 OCCLUSION SENSITIVITY ANALYSIS SUMMARY:")
    print("-" * 100)
    
    print(f"\n1. EXPERIMENT SETUP:")
    print(f"   • Model used: {best_config['name']}")
    print(f"   • Number of images analyzed: {len(occlusion_results)}")
    print(f"   • Occlusion window size: {WINDOW_SIZE}×{WINDOW_SIZE} pixels")
    print(f"   • Stride: {STRIDE} pixels")
    print(f"   • Analysis positions per image: {(84//STRIDE) * (84//STRIDE)}")
    
    print(f"\n2. QUANTITATIVE FINDINGS:")
    avg_original_conf = np.mean([r['original_confidence'] for r in occlusion_results])
    avg_min_conf = np.mean([r['min_confidence'] for r in occlusion_results])
    avg_drop = np.mean([r['confidence_drop'] for r in occlusion_results])
    
    print(f"   • Average original confidence: {avg_original_conf:.3f}")
    print(f"   • Average minimum confidence: {avg_min_conf:.3f}")
    print(f"   • Average confidence drop: {avg_drop:.3f}")
    
    # Classify localization strength
    if avg_drop > 0.5:
        localization = "STRONG - Model shows clear spatial attention"
    elif avg_drop > 0.3:
        localization = "MODERATE - Model uses some spatial features"
    else:
        localization = "WEAK - Model may rely on distributed/contextual features"
    
    print(f"   • Localization strength: {localization}")
    
    print(f"\n3. QUALITATIVE OBSERVATIONS:")
    print(f"   Based on visual inspection of heatmaps:")
    print(f"   • Check if model focuses on object vs background")
    print(f"   • Identify discriminative parts (edges, textures, shapes)")
    print(f"   • Note any unexpected attention patterns")
    print(f"   • Compare attention across different classes")
    
    print(f"\n4. ANSWERS TO KEY QUESTIONS:")
    print(f"   ❓ Has the model learned object location?")
    print(f"      → Examine if RED regions align with actual objects")
    print(f"   ")
    print(f"   ❓ Does model use contextual cues?")
    print(f"      → Check if background regions affect confidence")
    print(f"   ")
    print(f"   ❓ Is the model robust?")
    print(f"      → Average drop of {avg_drop:.3f} indicates robustness level")
    
    print(f"\n5. VISUALIZATION OUTPUTS:")
    print(f"   Generated files:")
    for i in range(1, len(occlusion_results) + 1):
        print(f"   • part_b_occlusion_image_{i}.png")
    
    print("\n" + "="*100)
    print("✓ PART B COMPLETED")
    print("="*100)

else:
    print("\n⚠️  No occlusion results available. Please run Cell 18 first.")


# ============================================================================
# CELL 21: FINAL SUMMARY - Complete Assignment Overview
# ============================================================================
"""
═══════════════════════════════════════════════════════════════════════════
                        FINAL ASSIGNMENT SUMMARY
═══════════════════════════════════════════════════════════════════════════
"""

print("\n\n")
print("╔" + "="*98 + "╗")
print("║" + " "*30 + "ASSIGNMENT COMPLETION SUMMARY" + " "*39 + "║")
print("╚" + "="*98 + "╝")

print("\n📋 ASSIGNMENT: Module 4 (Deep Learning) - Mini-ImageNet Classification")
print("━" * 100)

# Dataset Summary
print("\n📊 DATASET:")
print("   • Dataset: Mini-ImageNet (33 classes)")
print("   • Training images: 13,200 (33 × 400)")
print("   • Validation images: 3,300 (33 × 100)")
print("   • Test images: 3,300 (33 × 100)")
print("   • Image size: 84×84 pixels")
print("   • Framework: PyTorch")
print("   • Experiment Tracking: MLflow")

# Part A Summary
print("\n" + "─"*100)
print("PART A: CNN ARCHITECTURE EXPERIMENTATION (50 marks)")
print("─"*100)

if all_results:
    print(f"\n✓ Experiments Completed: {len(all_results)}/10 configurations")
    print(f"\n   Parameters Varied:")
    print(f"   • Number of convolutional layers: 1-4 layers")
    print(f"   • Number of filters: 32-256 filters per layer")
    print(f"   • Pooling strategies: MaxPool vs Stride")
    print(f"   • Fully connected layers: 1-3 layers")
    print(f"   • Training epochs: 5-8 epochs")
    
    test_accs_final = [float(r['Test Acc'].replace('%', '')) for r in all_results]
    best_idx_final = test_accs_final.index(max(test_accs_final))
    
    print(f"\n   🏆 Best Configuration:")
    print(f"      Name: {all_results[best_idx_final]['Name']}")
    print(f"      Test Accuracy: {all_results[best_idx_final]['Test Acc']}")
    print(f"      Architecture: {all_results[best_idx_final]['Conv Layers']}")
    
    print(f"\n   📈 Performance Analysis:")
    print(f"      Highest accuracy: {max(test_accs_final):.2f}%")
    print(f"      Lowest accuracy: {min(test_accs_final):.2f}%")
    print(f"      Average accuracy: {np.mean(test_accs_final):.2f}%")
    print(f"      Performance range: {max(test_accs_final) - min(test_accs_final):.2f}%")
    
    print(f"\n   📁 Deliverables:")
    print(f"      • part_a_results.csv - Complete results table")
    print(f"      • part_a_training_curves_acc.png - Accuracy curves")
    print(f"      • part_a_training_curves_loss.png - Loss curves")
    print(f"      • part_a_comparison.png - Performance comparison")
    print(f"      • part_a_misclassified.png - Error analysis")
    print(f"      • MLflow runs - All experiments tracked")

# Part B Summary
print("\n" + "─"*100)
print("PART B: OCCLUSION SENSITIVITY ANALYSIS (50 marks)")
print("─"*100)

if occlusion_results:
    print(f"\n✓ Analysis Completed: {len(occlusion_results)}/10 test images")
    print(f"\n   Methodology:")
    print(f"   • Occlusion window: {WINDOW_SIZE}×{WINDOW_SIZE} pixels")
    print(f"   • Stride: {STRIDE} pixels")
    print(f"   • Forward passes per image: ~{(84//STRIDE)**2}")
    print(f"   • Total forward passes: ~{len(occlusion_results) * (84//STRIDE)**2:,}")
    
    avg_drop_final = np.mean([r['confidence_drop'] for r in occlusion_results])
    
    print(f"\n   🔍 Key Findings:")
    print(f"      Average confidence drop: {avg_drop_final:.3f}")
    print(f"      Model localization: {'Strong' if avg_drop_final > 0.5 else 'Moderate' if avg_drop_final > 0.3 else 'Weak'}")
    print(f"      Object-focused: Determined by heatmap analysis")
    
    print(f"\n   📁 Deliverables:")
    print(f"      • 10 occlusion heatmap visualizations (part_b_occlusion_image_*.png)")
    print(f"      • Detailed interpretation and observations")

# MLflow Summary
print("\n" + "─"*100)
print("🔬 MLFLOW EXPERIMENT TRACKING")
print("─"*100)
print(f"""
MLflow has tracked all experiments with:
   • {len(configurations)} experimental runs (Part A)
   • All hyperparameters logged
   • Training metrics per epoch
   • Final test results
   • Training curves as artifacts
   • Model configurations

Access MLflow UI:
   1. Open terminal: cd to notebook directory
   2. Run command: mlflow ui
   3. Open browser: http://localhost:5000
   4. Experiment: {EXPERIMENT_NAME}

MLflow Benefits:
   ✓ Compare all 10 configurations side-by-side
   ✓ Interactive metric visualization
   ✓ Parameter importance analysis
   ✓ Reproducible experiments
   ✓ Export results and plots
""")

# Submission Checklist
print("\n" + "─"*100)
print("📦 SUBMISSION CHECKLIST")
print("─"*100)

print("""
✓ Report Requirements:
  [ ] Part A results table (from part_a_results.csv)
  [ ] Part A training curves and comparisons
  [ ] Part A error analysis with misclassified images
  [ ] Part A summary: Effect of each parameter
  [ ] Part B occlusion sensitivity heatmaps (10 images)
  [ ] Part B interpretation: Does model learn object location?
  [ ] Part B observations: Context vs object features
  [ ] Overall conclusions and insights
  [ ] MLflow screenshot (optional): Comparison view

✓ Code Requirements:
  [ ] This Jupyter notebook (.ipynb file)
  [ ] All cells executed with outputs visible
  [ ] Comments and documentation included
  [ ] No code screenshots in report (only figures/results)
  [ ] MLflow runs (mlruns/ directory - optional)

✓ Generated Files:
  [ ] part_a_results.csv
  [ ] part_a_training_curves_acc.png
  [ ] part_a_training_curves_loss.png
  [ ] part_a_comparison.png
  [ ] part_a_misclassified.png
  [ ] part_b_occlusion_image_1.png through part_b_occlusion_image_10.png
  [ ] training_curves_config_*.png (from MLflow)
""")

# Final Notes
print("\n" + "─"*100)
print("📝 IMPORTANT NOTES FOR REPORT")
print("─"*100)

print("""
1. PART A ANALYSIS:
   • Create a table showing all configurations and their performance
   • Discuss which parameter had the biggest impact on accuracy
   • Explain where performance plateaus (diminishing returns)
   • Identify signs of overfitting (train-val gap)
   • Comment on misclassified images - why did model fail?
   • Include MLflow comparison screenshots if helpful

2. PART B ANALYSIS:
   • Include all 10 occlusion heatmap visualizations
   • For each image, comment on:
     - Does heatmap align with actual object location?
     - Does model use object features or background context?
     - What specific regions are most discriminative?
   • Overall conclusion: Has the model truly learned object recognition?

3. MLFLOW INSIGHTS (BONUS):
   • Show parameter correlation analysis
   • Demonstrate metric comparison across runs
   • Highlight experiment reproducibility
   • Discuss how MLflow aids hyperparameter tuning

4. REPORT STRUCTURE:
   • Introduction: Dataset and methodology
   • Part A: Experiments, results table, analysis
   • Part B: Occlusion method, results, interpretation
   • Conclusion: Key insights and findings
   • References: [1] Zeiler & Fergus 2014

5. ACADEMIC INTEGRITY:
   ⚠️  WARNING: Plagiarism will result in high penalty
   • Write analysis in your own words
   • Do not copy code from others
   • Properly cite any external resources
   • MLflow runs serve as proof of your work
""")

print("\n" + "="*100)
print("🎉 ASSIGNMENT COMPLETED SUCCESSFULLY!")
print("="*100)
print("\nNext Steps:")
print("  1. Review all generated visualizations")
print("  2. Explore MLflow UI for deeper insights")
print("  3. Prepare your written report with analysis")
print("  4. Include this notebook (.ipynb) in submission")
print("  5. (Optional) Include mlruns/ folder for reproducibility")
print("  6. Double-check all requirements before submitting")
print("\nMLflow Command:")
print("  mlflow ui")
print("  Then open: http://localhost:5000")
print("\nGood luck with your report! 🚀")
print("="*100)


# ============================================================================
# ADDITIONAL HELPER CELL: Save Best Model (Optional)
# ============================================================================
"""
Optional: Save the best model for future use
"""

# Uncomment to save the best model
# if all_results:
#     model_save_path = 'best_model.pth'
#     torch.save({
#         'model_state_dict': best_model.state_dict(),
#         'config': best_config,
#         'test_accuracy': max(test_accs_final),
#         'class_names': class_names
#     }, model_save_path)
#     print(f"✓ Best model saved to {model_save_path}")


# ============================================================================
# ADDITIONAL HELPER CELL: Load Saved Model (Optional)
# ============================================================================
"""
Optional: Load a previously saved model
"""

# Uncomment to load a saved model
# def load_saved_model(path, num_classes):
#     checkpoint = torch.load(path, map_location=device)
#     model = FlexibleCNN(num_classes=num_classes, config=checkpoint['config'])
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     print(f"✓ Model loaded from {path}")
#     print(f"  Test accuracy: {checkpoint['test_accuracy']:.2f}%")
#     return model, checkpoint['config']
#
# # Usage:
# # loaded_model, loaded_config = load_saved_model('best_model.pth', num_classes)


# ============================================================================
# END OF NOTEBOOK
# ============================================================================
"""
This notebook implements the complete Deep Learning assignment:
- Part A: Systematic CNN architecture experimentation
- Part B: Occlusion sensitivity analysis for model interpretability

All results, visualizations, and analyses are ready for your report.

Created for: Assignment Module 4 (Deep Learning)
Dataset: Mini-ImageNet (33 classes, 84×84 images)
Framework: PyTorch
Total Marks: 100 (Part A: 50, Part B: 50)

Author: [Your Name]
Date: [Current Date]
"""

print("\n" + "="*100)
print("END OF NOTEBOOK - All cells completed successfully! ✓")
print("="*100)