import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from untitled0 import ResNetModified  # Import the ResNetModified class from the training script

# Define the transformation (same as during training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image path (update with your specific image path)
image_path = r"C:\Users\fnu.sawera\OneDrive - University of Central Asia\Desktop\car.jpg"

# Load and preprocess the image
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Add batch dimension

# Set the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)

# Load the trained model
model = ResNetModified(num_classes=10).to(device)
model.load_state_dict(torch.load('resnet_model.pth'))  # Load the saved weights
model.eval()  # Set the model to evaluation mode

# Perform the prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted = outputs.max(1)
    predicted_class = predicted.item()

# CIFAR-10 class labels (update with your classes if different)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Print the predicted class
print(f"Predicted Class: {classes[predicted_class]}")

# Optionally, display the image with the predicted class
plt.imshow(image.squeeze(0).cpu().numpy().transpose((1, 2, 0)) * 0.5 + 0.5)  # Unnormalize
plt.title(f"Predicted: {classes[predicted_class]}")
plt.axis('off')
plt.show()
