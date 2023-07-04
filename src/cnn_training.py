import torch
import os
import csv
import cv2
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torch import nn, save

from string import ascii_lowercase

DATASET_PATH = '../data'
LABELS_FILE = '../data/labels.csv'


def read_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


class SignLangDataset(Dataset):
    """Sign language dataset"""

    def __init__(self, csv_file, root_dir, class_index_map=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.class_index_map = class_index_map
        self.transform = transform
        # List of class names in order
        self.class_names = list(map(str, list(range(10)))) + list(ascii_lowercase)

    def __len__(self):
        """
        Calculates the length of the dataset-
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns one sample (dict consisting of an image and its label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read the image and labels
        image_path = os.path.join(self.root_dir, self.data[idx][1])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Shape of the image should be H,W,C where C=1
        image = np.expand_dims(image, 0)
        # The label is the index of the class name in the list ['0','1',...,'9','a','b',...'z']
        # because we should have integer labels in the range 0-35 (for 36 classes)
        label = self.class_names.index(self.data[idx][0])

        sample = {'image': image, 'label': label}

        return sample


train_dataset = SignLangDataset(csv_file=LABELS_FILE, root_dir=DATASET_PATH, transform=ToTensor())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device} ...")
dataset = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)


# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.6),
            nn.Flatten(),
            nn.Linear(32 * 30 * 30, 36)  # Update the output size to 36 for 36 classes
        )

    def forward(self, x):
        x = x.float().to(device)
        return self.model(x)


# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=0.000096, weight_decay=1e-1)
loss_fn = nn.CrossEntropyLoss()

# Move optimizer to the desired device
for state in opt.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.to(device)

# Training flow
if __name__ == "__main__":
    # Lists to store accuracy scores
    train_acc_scores = []
    val_acc_scores = []
    losses = []
    average_losses = []
    average_validation_losses = []

    print("Training...")
    for epoch in range(10):  # train for 10 epochs
        for batch in dataset:
            X = batch["image"]
            y = batch["label"]
            X, y = X.to(device), y.to(device)
            yhat = clf(X)
            loss = loss_fn(yhat, y)
            losses.append(loss.item())

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        with open('cnn_model.pt', 'wb') as f:
            save(clf.state_dict(), f)

        # Validation loop
        clf.eval()  # Set the model to evaluation mode
        average_loss = np.mean(losses)
        average_losses.append(average_loss)
        losses.clear()

        print(f"Epoch:{epoch} Training loss is {average_loss}. Device: {device}")
