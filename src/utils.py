import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os
from PIL import Image
from torchvision import transforms

def load_data(batch_size=64, data_dir='./data'):
    """
    Load and transform CIFAR-10 dataset
    Returns train_loader, test_loader, class_names
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load training dataset
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True,
        download=True, 
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size,
        shuffle=True
    )

    # Download and load test dataset
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False,
        download=True, 
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_set.classes
    return train_loader, test_loader, class_names

def save_model(model, path='models/cifar_model.pth'):
    """
    Save model to specified path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path='models/cifar_model.pth'):
    """
    Load model weights from file
    """
    model.load_state_dict(torch.load(path))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {path}")
    return model
