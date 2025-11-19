# COET 295 Assignment 3
# Logan Davis

# Import necessary libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms.v2 as v2


# Configure the device to use GPU if available, otherwise CPU
# couldn't get the Nvidiq drivers working on my laptop(Linux) so I was just running as CPU the whole time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define a series of transformations to apply to each image
transform = v2.Compose([
    v2.ToImage(),  # Ensures data is treated as images
    v2.Resize((224, 224)),  # Resize for better detail capture (important for detecting grain disease)
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Data augmentation for robustness
    v2.RandomHorizontalFlip(),  # Augmentation: flips images horizontally at random
    v2.RandomRotation(10),  # Augmentation: rotates images slightly
    v2.ToDtype(torch.float32, scale=True),  # Convert to float and scale pixel values
])


# Load dataset from folder with transformation applied
dataset_path = "Files/Wheat"
image_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
classes = image_dataset.classes  # List of class names
num_classes = len(classes)  # Number of output classes

# Split dataset: 80% for training, 20% for testing
train_size = int(0.8 * len(image_dataset))
test_size = len(image_dataset) - train_size
train_dataset, test_dataset = random_split(image_dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Function to perform a single training epoch
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch,(X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # Move data to GPU if available
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Print average loss over training set
    print(f"Train Loss: {total_loss / len(dataloader):.4f}") #gives total loss for current epoch

# Function to evaluate model performance on test set
def test_loop(dataloader, model, loss_fn):
    model.eval()  # Set model to evaluation mode
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # Move data to device
            pred_logits = model(X)

            pred_probability = nn.Softmax(dim=1)(pred_logits)
            test_loss += loss_fn(pred_probability, y).item()
            correct += (pred_probability.argmax(1) == y).type(torch.float).sum().item()
            total += len(y)

    # Print average loss and accuracy
    test_loss /= len(dataloader)
    accuracy = correct / total
    print(f"Test Loss: {test_loss / len(dataloader):.4f}, Accuracy: {accuracy * 100:.2f}%")




#CNN MODEL
class WheatCNN(nn.Module):
    def __init__(self, num_classes, image_size):
        super().__init__()
        #convolutional layers with increasing depth and dropout

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        self.flatten = nn.Flatten() # Flatten for fully connected layer

        self.fc = nn.Sequential(
            nn.Linear(1024 * (image_size // (2 ** 6)) * (image_size // (2 ** 6)), out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


image_classifier_model = WheatCNN(num_classes=num_classes,image_size=224).to(device)
print(image_classifier_model)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(image_classifier_model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(image_classifier_model.parameters(), lr=0.001)



# Train the model for a fixed number of epochs
epochs = 25
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_loop(train_dataloader, image_classifier_model, loss_fn, optimizer)
    test_loop(test_dataloader, image_classifier_model, loss_fn)
print("\nTraining complete!")