"""
PyTorch Image Prediction Script
-------------------------------

This script makes predictions on images using a trained model.
Features:
- Single image or batch directory processing
- Top-K predictions with confidence scores
- Image display with prediction overlay
- CSV export of results
- GPU acceleration support

Usage:
python predict.py --model models/best_model.pth --input test_image.jpg
python predict.py --model models/best_model.pth --input images/ --output predictions.csv
"""

import argparse
import os
import csv
import time
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from model import initialize_model
from utils import load_model, load_class_names

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Image Prediction')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path for batch results')
    parser.add_argument('--class_names', type=str, default=None,
                        help='Comma-separated class names or path to text file')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top predictions to show (default: 3)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: "cuda", "cpu", or "auto" (default: auto)')
    parser.add_argument('--conf_threshold', type=float, default=0.01,
                        help='Confidence threshold to show predictions (default: 0.01)')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization images with predictions')
    parser.add_argument('--vis_dir', type=str, default='predictions',
                        help='Directory to save visualization images (default: predictions)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load class names
    class_names = load_class_names(args.class_names)
    num_classes = len(class_names)

    # Initialize model
    model = initialize_model(device=device, num_classes=num_classes)

    # Load trained weights
    model = load_model(model, args.model)

    