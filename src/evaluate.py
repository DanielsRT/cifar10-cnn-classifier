"""
PyTorch Model Evaluation Script
-------------------------------

This script evaluates a trained image recognition model on the test set.
It provides:
- Overall accuracy
- Per-class metrics (precision, recall, F1-score)
- Confusion matrix visualization
- Sample misclassified images
- Inference time benchmarking

Usage:
python evaluate.py --model_path models/best_model.pth [--batch_size 128] [--device cuda]
"""

import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from model import initialize_model
from utils import load_data, load_model, evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Model Evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Evaluation batch size (default: 128)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: "cuda", "cpu", or "auto" (default: auto)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save evaluation results (default: results)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Dataset directory (default: ./data)')
    parser.add_argument('--num_misclassified', type=int, default=16,
                        help='Number of misclassified samples to display (default: 16)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Load data
    _, test_loader, class_names = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment=False  # No augmentation for evaluation
    )
    num_classes = len(class_names)

    # Initialize model architecture
    model = initialize_model(device=device, num_classes=num_classes)

    # Load trained weights
    model = load_model(model, args.model_path)
    print(f"Loaded model from {args.model_path}")

    # Evaluate model
    print("\nEvaluating model on test set...")
    test_start = time.time()
    all_labels, all_preds, misclassified_samples = evaluate_model(
        model,
        test_loader,
        device,
        args.num_misclassified
    )
    test_time = time.time() - test_start

    # Calculate overall accuracy
    accuracy = np.sum(all_labels == all_preds) / len(all_labels)
    print(f"\nEvaluation completed in {test_time:.2f} seconds")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    