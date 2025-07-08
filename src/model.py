import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for image classification
    on the CIFAR-10 dataset (32x32 RGB images, 10 classes)
    
    Architecture:
    - Input: 3x32x32 (RGB images)
    - Conv1: 32 filters, 3x3 kernel
    - Conv2: 64 filters, 3x3 kernel
    - Conv3: 128 filters, 3x3 kernel
    - Fully connected layers with dropout
    
    Total params: ~1.2 million
    """
    def __init__(self, num_classes=10):
        """
        Initialize the CNN layers
        
        Args:
            num_classes (int): Number of output classes
        """
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer (shared)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # After 3 pooling layers: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        """
        Forward pass of the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten the features for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def initialize_model(device='cpu', num_classes=10):
    """
    Create and initialize a CNN model
    
    Args:
        device (str): Device to load the model on ('cuda' or 'cpu')
        num_classes (int): Number of output classes
        
    Returns:
        SimpleCNN: Initialized model instance
    """
    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)
    
    # Print model summary
    print(f"Model initialized on {device} device")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

if __name__ == "__main__":
    # Example usage and model testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing model architecture...")
    
    # Create a sample input tensor (batch_size=2, channels=3, height=32, width=32)
    test_input = torch.randn(2, 3, 32, 32).to(device)
    
    # Initialize model
    model = initialize_model(device=device)
    
    # Test forward pass
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output: {output[0][:5]}")  # Print first 5 logits of first sample