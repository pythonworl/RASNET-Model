import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Data Preparation
def prepare_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=0
    )

    return train_loader, test_loader

# 3. Load and Modify ResNet Model
class ResNetModified(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetModified, self).__init__()
        self.resnet = resnet18(pretrained=True)  # Load ResNet-18
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Modify final layer

    def forward(self, x):
        return self.resnet(x)

# 4. Training Function
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_losses = []
    train_accuracies = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_losses.append(running_loss / len(loader))
    train_accuracies.append(100. * correct / total)
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader):.3f}, Accuracy: {100.*correct/total:.2f}%")

    return train_losses, train_accuracies

# 5. Testing Function
def test(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    test_losses = []
    test_accuracies = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_losses.append(test_loss / len(loader))
    test_accuracies.append(100. * correct / total)
    print(f"Test Loss: {test_loss/len(loader):.3f}, Test Accuracy: {100.*correct/total:.2f}%")

    return test_losses, test_accuracies

# Visualization Functions
def plot_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(test_accuracies, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, loader, classes):
    model.eval()  # Set model to evaluation mode
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images, labels = images[:10].to(device), labels[:10].to(device)
    outputs = model(images)
    _, predictions = outputs.max(1)
    
    # Unnormalize images
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1)) * 0.5 + 0.5  # Revert normalization
    
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i])
        plt.title(f"True: {classes[labels[i].item()]}\nPred: {classes[predictions[i].item()]}") 
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, loader, classes):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = outputs.max(1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.show()

# Main Block
if __name__ == "__main__":
    print("Preparing data...")
    train_loader, test_loader = prepare_data()

    print("Initializing model...")
    model = ResNetModified(num_classes=10).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Training and Testing
    num_epochs = 100
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1} ---")
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_accuracy = test(model, test_loader, criterion)

        train_losses.extend(train_loss)
        train_accuracies.extend(train_accuracy)
        test_losses.extend(test_loss)
        test_accuracies.extend(test_accuracy)

    print("Training completed.")

    # Save the trained model
    torch.save(model.state_dict(), 'resnet_model.pth')
    print("Model saved as 'resnet_model.pth'")

    # Visualization
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print("Plotting Accuracy and Loss Curves...")
    plot_curves(train_losses, test_losses, train_accuracies, test_accuracies)

    print("Visualizing Predictions on Test Data...")
    visualize_predictions(model, test_loader, classes)

    print("Plotting Confusion Matrix...")
    plot_confusion_matrix(model, test_loader, classes)
