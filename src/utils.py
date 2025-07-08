import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os
import time
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix

def load_data(batch_size=64, data_dir='./data', augment=False):
    """
    Load and transform CIFAR-10 dataset with optional data augmentation
    
    Args:
        batch_size (int): Number of samples per batch
        data_dir (str): Directory to store/download dataset
        augment (bool): Enable data augmentation for training set
        
    Returns:
        train_loader, test_loader, class_names
    """
    # Define base transformations (applied to both train and test)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2470, 0.2435, 0.2616))
    ])
    
    # Training transformations - add augmentation if enabled
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2470, 0.2435, 0.2616))
        ])
        print("Using data augmentation for training set")
    else:
        train_transform = base_transform
    
    # Test transformations - no augmentation
    test_transform = base_transform

    # Download and load training dataset
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True,
        download=True, 
        transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Download and load test dataset
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False,
        download=True, 
        transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    class_names = train_set.classes
    return train_loader, test_loader, class_names

def imshow(img):
    """
    Display a PyTorch tensor as an image
    """
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def show_sample_images(data_loader, class_names, num_images=8):
    """
    Display grid of sample images with labels
    """
    # Get a batch of images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Create grid of images
    img_grid = torchvision.utils.make_grid(images[:num_images])
    imshow(img_grid)
    
    # Print labels
    print(' '.join(f'{class_names[labels[j]]:10s}' for j in range(num_images)))

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
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {path}")
    return model

def plot_metrics(train_losses, train_accs, val_accs, save_path=None):
    """
    Plot training/validation metrics and optionally save to file
    
    Args:
        train_losses (list): Training loss values per epoch
        train_accs (list): Training accuracy values per epoch (%)
        val_accs (list): Validation accuracy values per epoch (%)
        save_path (str): Optional path to save the plot (e.g., 'results/training_metrics.png')
    """
    plt.figure(figsize=(12, 5))
    
    # Create two subplots side by side
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-o', linewidth=2, markersize=6, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'g-s', linewidth=2, markersize=6, label='Training Accuracy')
    plt.plot(val_accs, 'r-^', linewidth=2, markersize=6, label='Validation Accuracy')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy Metrics', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Add overall title
    plt.suptitle('Training Performance Metrics', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to file if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    
    # Display the plot
    plt.show()
    
    # Close the figure to free memory
    plt.close()

def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path=None):
    """Generate and visualize confusion matrix"""
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Normalize by true labels
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title='Normalized Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, 
                    f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", 
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_misclassified_samples(samples, class_names, save_path=None, num_cols=4):
    """Display grid of misclassified samples"""
    num_samples = len(samples)
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    fig.suptitle('Misclassified Samples', fontsize=16)
    
    for i, (img, true_label, pred_label) in enumerate(samples):
        ax = axes[i // num_cols, i % num_cols] if num_rows > 1 else axes[i]
        
        # Unnormalize and display image
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        
        # Set title with true/pred labels
        true_name = class_names[true_label]
        pred_name = class_names[pred_label]
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, num_rows * num_cols):
        ax = axes[i // num_cols, i % num_cols] if num_rows > 1 else axes[i]
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Misclassified samples plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def benchmark_inference(model, device, input_size=(3, 32, 32), num_runs=100):
    """Benchmark model inference speed"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Warm up GPU
    print("\nBenchmarking inference speed...")
    for _ in range(10):
        _ = model(dummy_input)
    
    # Time inference
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end.record()
        torch.cuda.synchronize()
        avg_time = start.elapsed_time(end) / num_runs
    else:
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(dummy_input)
        avg_time = (time.time() - start_time) * 1000 / num_runs
    
    print(f"Average inference time: {avg_time:.2f} ms")
    
    # Calculate theoretical FPS
    fps = 1000 / avg_time if avg_time > 0 else float('inf')
    print(f"Theoretical FPS: {fps:.2f} (batch size 1)")
    
    return avg_time

def calculate_accuracy(outputs, labels):
    """
    Calculate classification accuracy
    """
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

def predict_image(model, image_path, class_names, transform=None):
    """
    Make prediction on a single image
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
    
    return class_names[class_idx]

def validate_model(model, test_loader, device):
    """Evaluate model on validation set."""
    model.eval() # Set model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate accuracy
            total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = 100 * total_correct / total_samples
    return accuracy

def evaluate_model(model, test_loader, device, num_misclassified=16):
    """Evaluate model and collect predictions"""
    model.eval()  # Set model to evaluation mode
    
    # Initialize collections
    all_labels = []
    all_preds = []
    misclassified_samples = []  # Store (image, true_label, pred_label)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Collect predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            # Collect misclassified samples
            mis_mask = (preds != labels)
            mis_images = images[mis_mask]
            mis_labels = labels[mis_mask]
            mis_preds = preds[mis_mask]
            
            for i in range(min(len(mis_images), num_misclassified - len(misclassified_samples))):
                img = mis_images[i].cpu()
                true_label = mis_labels[i].item()
                pred_label = mis_preds[i].item()
                misclassified_samples.append((img, true_label, pred_label))
    
    return np.array(all_labels), np.array(all_preds), misclassified_samples