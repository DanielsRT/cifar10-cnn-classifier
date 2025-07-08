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
from sklearn.metrics import classification_report
from model import initialize_model
from utils import load_data, load_model, evaluate_model, plot_confusion_matrix, plot_misclassified_samples, benchmark_inference

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

    # Generate and save classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )
    print("\nClassification Report:\n", report)

    # Save report to file
    report_path = os.path.join(args.results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Generate and plot confusion matrix
    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        save_path=os.path.join(args.results_dir, 'confusion_matrix.png')
    )

    # Plot misclassified samples
    if misclassified_samples:
        plot_misclassified_samples(
            misclassified_samples,
            class_names,
            save_path=os.path.join(args.results_dir, 'misclassified_samples.png')
        )

    # Benchmark inference speed
    benchmark_inference(model, device, input_size=(3, 32, 32), num_runs=100)

    print("\nEvaluation complete. Results saved to", args.results_dir)

if __name__ == "__main__":
    main()