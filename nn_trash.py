# Import dependencies
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

import pandas as pd
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using {device} device")

# Define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_path, folder_path, transform=None):
        self.data = pd.read_csv(csv_path, header=None)
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx, 1]
        image_path = os.path.join(self.folder_path, image_name)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.data.iloc[idx, 0]

        if label.isdigit():
            label = int(label)
        else:
            # Handle encoding for letters
            label = label.lower()
            label = ord(label) - ord('a') + 10

        return image, label
    
# Set the data path
label_csv_path = "labels.csv"
image_path = "C:\\Users\\Georg\\Documents\\Computer Science\\SS2023\\signLangML\\data"
train = CustomDataset(label_csv_path, image_path, transform=ToTensor())

# Create the data loader
dataset = DataLoader(train, 32)


# Image Classifier Neural Network
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64 * 122 * 122, 36)  
        )

    def forward(self, x): 
        return self.model(x)

# Instance of the neural network, loss, optimizer 
clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=0.00001)
loss_fn = nn.CrossEntropyLoss() 

# Training flow 

if __name__ == "__main__":
    print("Starting Training...")
    for epoch in range(10): # train for 10 epochs
        for batch in dataset:
            img, label = batch
            img, label = img.to(device), label.to(device)
            yhat = clf(img)
            loss = loss_fn(yhat, label)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
    with open('model_state.pt', 'wb') as f: 
        save(clf.state_dict(), f)
"""

# Testing model
    
    with open('model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    img = Image.open('STTVUKNDDPXUZREU.jpg') 
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    print(torch.argmax(clf(img_tensor)))
"""
