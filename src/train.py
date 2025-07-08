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

