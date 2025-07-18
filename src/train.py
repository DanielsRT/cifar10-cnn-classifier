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
from utils import load_data, save_model, plot_metrics, validate_model

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
        os.makedirs(d, exist_ok=True)

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
    )

    # Training variables
    best_val_accuracy = 0.0
    train_losses, train_accuracies, val_accuracies = [], [], []

    # Training loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        # Per-epoch metrics
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        # Mini-batch training
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                batch_acc = 100 * running_correct / total_samples
                print(f"Epoch [{epoch}/{args.epochs}] | Batch [{batch_idx}/{len(train_loader)}] "
                      f"| Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}%")
                
        # Calculate epoch training metrics
        epoch_loss = running_loss / total_samples
        epoch_accuracy = 100 * running_correct / total_samples
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validation phase
        val_accuracy = validate_model(model, test_loader, device)
        val_accuracies.append(val_accuracy)

        # Update learning rate
        scheduler.step(val_accuracy)

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_path = os.path.join(args.model_dir, 'best_model.pth')
            save_model(model, model_path)
            print(f"Saved new best model with accuracy: {best_val_accuracy:.2f}%")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch}/{args.epochs}] completed in {epoch_time:.1f}s ")
        print(f"Training Loss: {epoch_loss:.4f} | Training Acc: {epoch_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")
        print("-" * 70)

    # Save final model
    final_model_path = os.path.join(args.model_dir, 'final_model.pth')
    save_model(model, final_model_path)

    # Plot and save training metrics
    plot_metrics(train_losses, train_accuracies, val_accuracies, save_path=os.path.join(args.results_dir, 'training_metrics.png'))

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Models saved to: {args.model_dir}")
    print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
