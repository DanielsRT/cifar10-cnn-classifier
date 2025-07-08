"""
PyTorch Image Recognition Training Script
-----------------------------------------

This script trains a convolutional neural network on the CIFAR-10 dataset.
It handles:
- Data loading and augmentation
- Model initialization
- Training loop with validation
- Loss tracking and metrics calculation
- Model checkpointing
- Learning rate scheduling
- Training visualization

Usage:
python train.py [--epochs 20] [--batch_size 64] [--lr 0.001] [--device cuda]
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import initialize_model
from utils import load_data, save_model, plot_metrics, calculate_accuracy

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Image Recognition Training')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Input batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: "cuda", "cpu", or "auto" (default: auto)')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save models (default: models)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Dataset directory (default: ./data)')
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation (default: False)')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create directories
    for d in [args.model_dir, args.results_dir]:
        os.makedirs(d, exists_ok=True)

    # Load data with optional augmentation
    train_loader, test_loader, class_names = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment=args.augment
    )
    print(f"Loaded CIFAR-10 dataset with {len(class_names)} classes")
    print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Initialize model
    model = initialize_model(device=device, num_classes=len(class_names))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max', # Monitor validation accuracy
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Training variables
    best_val_accuracy = 0.0
    train_losses, train_accuracies, val_accuracies = [], [], []