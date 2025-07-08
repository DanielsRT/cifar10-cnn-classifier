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
from utils import load_model, load_class_names, load_and_preprocess_image, predict_image, save_prediction_vis, save_results_csv

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

    # Prepare input paths
    input_paths = []
    if os.path.isfile(args.input):
        input_paths = [args.input]
    elif os.path.isdir(args.input):
        input_paths = [os.path.join(args.input, f)
                       for f in os.listdir(args.input)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    else:
        raise ValueError(f"Input path not found: {args.input}")
    
    # Create ouput directories if needed
    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)

    # Process images
    all_results = []
    total_time = 0

    for img_path in input_paths:
        start_time = time.time()

        # Load and preprocess image
        image, input_tensor = load_and_preprocess_image(img_path, device)

        # Make predictions
        pred_class, pred_idx, confidences, top_classes = predict_image(
            model,
            input_tensor,
            class_names,
            top_k=args.top_k,
            conf_threshold=args.conf_threshold
        )

        # Calculate processing time
        proc_time = time.time() - start_time
        total_time += proc_time

        # Prepare result
        result = {
            'image' : os.path.basename(img_path),
            'prediction' : pred_class,
            'confidence' : confidences[0],
            'time_ms' : proc_time * 1000,
            'top_predictions' : ', '.join([f"{c} ({p:.2f})" for c, p in zip(top_classes, confidences)])
        }
        all_results.append(result)

        # Print result
        print(f"\nImage: {os.path.basename(img_path)}")
        print(f"  Top prediction: {pred_class} ({confidences[0]:.2f})")
        print(f"  Processing time: {proc_time * 1000:.1f} ms")
        print(f"  Top {args.top_k} predictions:")
        for i, (cls, conf) in enumerate(zip(top_classes, confidences)):
            print(f"    {i+1}. {cls}: ({conf:.2%})")

        # Save visualization
        if args.save_vis:
            vis_path = os.path.join(args.vis_dir, f"pred_{os.path.basename(img_path)}")
            save_prediction_vis(image, pred_class, confidences[0], top_classes, confidences, vis_path)
            print(f"  Visualiation saved to: {vis_path}")

    # Print summary
    avg_time = total_time / len(input_paths) * 1000
    print(f"\nProcessed {len(input_paths)} images in {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time:.1f} ms")

    # Save CSV results
    if args.output:
        save_results_csv(all_results, args.output)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()