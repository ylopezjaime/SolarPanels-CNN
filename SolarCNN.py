# Solar CNN using VGG16 Arch.
# --------------------------------------------------------------------------
#  About the code: This code will use the custom dataset of solar panel faults for training a CNN using the 
#  VGG16 Arch
#  
# --------------------------------------------------------------------------
# @author : Yeuris Adolfo Lopez Jaime
# @date : October, 2023
# @Project : Predictive Maintanance using CCN for dectecting faults on Solar panels
# @Workplace : Universidad Central Del Este
# @City : San Pedro De Macoris, Republica Dominicana
# --------------------------------------------------------------------------

# Import all the library for use in the model

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from torchsummary import summary

# --------------------------------------------------------------------------

# Define the custom dataset directory
custom_data_dir = r'/home/reloadzu/SolarPanels/Faulty_solar_panel'                                       # Change this for the directory on your machine

# Define data transformations for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
custom_dataset = datasets.ImageFolder(custom_data_dir, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Load the pre-trained VGG16 model
vgg16 = models.vgg16(pretrained=True)


# Freeze some of the earlier layers
for param in vgg16.features[:-6].parameters():
    param.requires_grad = False                                                                             # Change to "True" for fine tuning the model aand try all the parameters of the CNN

# --------------------------------------------------------------------------

# Replace the classifier with a new one suitable for the dataset
num_classes = len(custom_dataset.classes)                                                                  # Access classes from the original dataset
vgg16.classifier[6] = nn.Linear(4096, num_classes)

# --------------------------------------------------------------------------
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

# Move the model and data to the GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("USING CUDA CORES FOR PROCESSING")

else:
    print("USING CPU FOR PROCESSING")


vgg16.to(device)
summary(vgg16,(3,244,244))                                                                              # Show the model parameters in the terminal.

# --------------------------------------------------------------------------

# Lists to store loss, accuracy, and F1-score values
train_loss_history = []
val_loss_history = []
train_accuracy_history = []
val_accuracy_history = []
train_f1_score_history = []  
val_f1_score_history = []    

# --------------------------------------------------------------------------
# Train the model
num_epochs = 30  # Adjust as needed

for epoch in range(num_epochs):
    vgg16.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_loss_history.append(train_loss)

    train_accuracy = 100 * correct_train / total_train
    train_accuracy_history.append(train_accuracy)

    train_true_labels = []
    train_predicted_labels = []

    # Evaluate training F1-Score
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vgg16(inputs)
            _, predicted = torch.max(outputs.data, 1)

            train_true_labels.extend(labels.cpu().numpy())
            train_predicted_labels.extend(predicted.cpu().numpy())

        train_f1 = f1_score(train_true_labels, train_predicted_labels, average='weighted')
        train_f1_score_history.append(train_f1)

    print(f"Epoch {epoch+1}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy:.2f}%, Training F1-Score: {train_f1:.2f}")

    # Evaluate the model on the validation set
    vgg16.eval()
    correct_val = 0
    total_val = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vgg16(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

        val_accuracy = 100 * correct_val / total_val
        val_accuracy_history.append(val_accuracy)

        val_loss = loss.item()
        val_loss_history.append(val_loss)

        val_f1 = f1_score(true_labels, predicted_labels, average='weighted')
        val_f1_score_history.append(val_f1)


    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy:.2f}%, Validation F1-Score: {val_f1:.2f}")

# --------------------------------------------------------------------------
# Plot the loss, accuracy, F1-score, and training/validation accuracy over epochs
plt.figure(figsize=(16, 6))
plt.subplot(2, 2, 1)
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epoch')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(train_accuracy_history, label='Training Accuracy')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy vs. Epoch')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(train_f1_score_history, label='Training F1-Score')
plt.plot(val_f1_score_history, label='Validation F1-Score')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.legend()
plt.title('F1-Score vs. Epoch')
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------

# Save the trained model
torch.save(vgg16.state_dict(), 'Solar_vgg16.pth')