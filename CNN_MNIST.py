import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. Set random seeds
# =========================
def set_seed(seed_value: int):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)


# =========================
# 2. Get data transforms
# =========================
def get_transforms():
    """Return the image transforms to be applied to the MNIST dataset."""
    return transforms.Compose([transforms.ToTensor()])


# =========================
# 3. Get Data Loaders 
# =========================
def get_data_loaders(batch_size=100):
    """
    Download MNIST dataset and return training and test DataLoaders
    (no validation set).
    """
    transform = get_transforms()

    # Load datasets
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=True
    )

    return train_loader, test_loader


# =========================
# 4. Define CNN Classier Model
# =========================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5, stride=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer
        self.fc1 = nn.Linear(4 * 4 * 10, 100)
        # Output layer
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (N, 10, 24, 24)
        x = self.pool(x)           # (N, 10, 12, 12)
        x = F.relu(self.conv2(x))  # (N, 10, 8, 8)
        x = self.pool(x)           # (N, 10, 4, 4)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ===============================
# 5. Train the model 
# ===============================
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Train the model for a specified number of epochs on the training set.
    This function combines the 'train for one epoch' and 'multi-epoch' logic.
    """
    train_losses, train_accuracies = [], []

    print("Training the classifier")

    for epoch in range(epochs):
        model.train()  # set model to train mode
        total_loss = 0.0
        total_acc = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item() * images.size(0)

            preds = torch.max(outputs, 1)[1]
            total_acc += (preds == labels).sum().item()
            
            total_samples += images.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = total_acc / total_samples

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.4f}"
        )

    print("Training complete!")
    return train_losses, train_accuracies


# ===============================
# 6. Test the model
# ===============================
def test_model(model, test_loader, device):
    """Evaluate the model on the test set and return the accuracy."""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.max(outputs, 1)[1]
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

    test_accuracy = total_correct / total_samples
    return test_accuracy


# ===============================
# 7. Main 
# ===============================
def classifier():
    # 1. Set the seed for reproducibility
    set_seed(1)

    # 2. Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Get data loaders (train + test, no validation)
    batch_size = 100
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # 4. Initialize model
    model = Net().to(device)

    # 5. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 6. Train the model (no separate one-epoch function needed)
    train_losses, train_accuracies = train_model(
        model, train_loader, criterion, optimizer, device, epochs=10
    )

    # Save the model
    # torch.save(model.state_dict(), "trained_classifier.pth")
    # print("Model saved.")

    # 7. Test the model
    test_acc = test_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Plot training curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    # plt.savefig('training_curves.png')
    plt.show()

    return model


if __name__ == "__main__":
    classifier()