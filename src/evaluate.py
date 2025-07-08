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


    