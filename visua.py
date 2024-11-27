# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:55:51 2024

@author: 1000000824
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchsummary import summary

# Define RasNet (Modified ResNet)
class RasNetModified(nn.Module):
    def __init__(self, num_classes=10):
        super(RasNetModified, self).__init__()
        self.resnet = resnet18(pretrained=True)  # Load ResNet-18
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Modify the final layer

    def forward(self, x):
        return self.resnet(x)

# Initialize the model and move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RasNetModified(num_classes=10).to(device)

# Summarize the model
summary(model, (3, 224, 224))
