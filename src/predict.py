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

