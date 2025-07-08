# CIFAR-10 Image Recognition

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch project for image recognition that implements a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. This project provides a complete pipeline from data loading and model training to evaluation and prediction

## Features

- **End-to-End Pipeline**: Data loading → preprocessing → training → evaluation → prediction
- **Modular Design**: Clean separation of model, training, utils, and prediction code
- **Data Augmentation**: Random flips, rotations, crops, and color jittering
- **Performance Tracking**: Real-time metrics and visualization
- **Detailed Evaluation**: Confusion matrix, classification report, error analysis
- **Prediction Interface**: Supports single images and batch processing with visualization
- **GPU Acceleration**: Automatic CUDA detection

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- NumPy
- Matplotlib
- Pillow
- scikit-learn

## Installation

```bash
# Clone repository
git clone https://github.com/DanielsRT/cifar10-cnn-classifier.git
cd cifar10-cnn-classifier

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended):
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## Usage

### Training

```bash
python src/train.py --epochs 20 --batch_size 64 --lr 0.001 --augment
```

### Evaluation

```bash
python src/evaluate.py --model_path models/best_model.pth
```

### Prediction

```bash
# Single image
python src/predict.py --model models/best_model.pth --input test_image.jpg

# Batch processing
python src/predict.py --model models/best_model.pth \
                     --input images/ \
                     --output predictions.csv \
                     --save_vis
```

## Project Structure

```
cifar10-cnn-classifier/
├── data/                   # Auto-downloaded datasets
├── models/                 # Saved model checkpoints
├── results/                # Evaluation outputs
├── predictions/            # Prediction visualizations
├── src/
│   ├── model.py            # CNN architecture (SimpleCNN)
│   ├── train.py            # Training with LR scheduling
│   ├── evaluate.py         # Metrics & error analysis
│   ├── predict.py          # Inference interface
│   └── utils.py            # Helpers (data loading, visualization)
├── requirements.txt        # Dependencies
└── README.md
```

## Resources
 - [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)
 - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
