import os
import torch
import pandas as pd
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using {device} device")

# Define a custom dataset class
class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 0]
        img_name = self.data.iloc[idx, 1]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        if label.isdigit():
            label = int(label)
        else:
            label = label.lower()
            label = ord(label) - ord('a') + 10

        return image, label


# Get data
train_dataset = SignLanguageDataset(csv_file='C:\\Users\\Georg\\Documents\\Computer Science\\SS2023\\signLangML\\labels.csv', root_dir='C:\\Users\\Georg\\Documents\\Computer Science\\SS2023\\signLangML\\data',
                                    transform=ToTensor())
dataset = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 122 * 122, 36)  # Update the output size to 36 for 36 classes
        )

    def forward(self, x):
        return self.model(x)


# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    """
    for epoch in range(10):  # train for 10 epochs
        for batch in dataset:
            X, y = batch
            X, y = X.to(device), y.to(device)
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch:{epoch} loss is {loss.item()}")
        with open('model_state.pt', 'wb') as f:
            save(clf.state_dict(), f)
    """

# Testing model
    
    with open('model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    img = Image.open('test\\SKXDYWAZZPCPBQWC.jpg') 
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    print(torch.argmax(clf(img_tensor)))
