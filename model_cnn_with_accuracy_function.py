import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, io
from torchvision.utils import make_grid
import os
import csv
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score
from torch import nn, save, load

from string import ascii_lowercase

DATASET_PATH = 'C:\\Users\\Remzi\Desktop\\4.Sem\\Maschinelles Lernen\\PS\\Projekt\\sign_lang_train\\'
LABELS_FILE = 'C:\\Users\\Remzi\Desktop\\4.Sem\\Maschinelles Lernen\\PS\\Projekt\\sign_lang_train\\csvFile\\labels.csv'


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

        # if self.transform:
        #    sample = self.transform(sample)

        return sample


train_dataset = SignLangDataset(csv_file=LABELS_FILE, root_dir=DATASET_PATH,
                                transform=ToTensor())
dataset = DataLoader(train_dataset, batch_size=32, shuffle=True)


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
            nn.Flatten(),
            nn.Linear(32 * 30 * 30, 36)  # Update the output size to 36 for 36 classes
        )

    def forward(self, x):
        x = x.float()
        return self.model(x)


# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cpu')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    """
    for epoch in range(10):  # train for 10 epochs
        for batch in dataset:
            X = batch["image"]
            y = batch["label"]
            X, y = X.to('cpu'), y.to('cpu')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch:{epoch} loss is {loss.item()}")
        with open('model_s.pt', 'wb') as f:
            save(clf.state_dict(), f)
            """


# Testing model
def leader_board_predict_fn(input_batch):
    """
    Function for making predictions using your trained model.

    Args:
        input_batch (numpy array): Input images (4D array of shape
                                   [batch_size, 1, 128, 128])

    Returns:
        output (numpy array): Predictions of the your trained model
                             (1D array of int (0-35) of shape [batch_size, ])
    """
    prediction = None

    # Instantiate the network
    model = ImageClassifier().to('cpu')

    # Load the saved weights from the disk
    model.load_state_dict(torch.load("model_s.pt"))

    # Set the network to evaluation mode
    model.eval()

    # Convert the input batch to a torch Tensor and set the data type
    input_tensor = torch.from_numpy(input_batch).float()

    # A forward pass with the input batch produces a batch of logits
    logits = model(input_tensor)

    # Final classification predictions are taken by taking an argmax over the logits
    prediction = torch.argmax(logits, dim=1).numpy()
    assert prediction is not None, "Prediction cannot be None"
    assert isinstance(prediction, np.ndarray), "Prediction must be a numpy array"

    return prediction


def accuracy(dataset_path, max_batches=30):
    """
    Calculates the average prediction accuracy.

    IMPORTANT
    =========
    In this function, we use PyTorch only for loading the data. When your `leader_board_predict_fn`
    function is called, we pass the arguments to it as numpy arrays. The output of `leader_board_predict_fn`
    is also expected to be a numpy array. So, as long as your `leader_board_predict_fn` function takes
    numpy arrays as input and produces numpy arrays as output (with the proper shapes), it does not
    matter what framework you used for training your network or for producing your predictions.

    Args:
        dataset_path (str): Path of the dataset directory

    Returns:
        accuracy (float): Average accuracy score over all images (float in the range 0.0-1.0)
    """

    # Create a Dataset object
    sign_lang_dataset = SignLangDataset(csv_file=LABELS_FILE, root_dir=DATASET_PATH)

    # Create a Dataloader
    sign_lang_dataloader = DataLoader(sign_lang_dataset,
                                      batch_size=64,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=0)

    # Calculate accuracy for each batch
    accuracies = list()
    for batch_idx, sample in enumerate(sign_lang_dataloader):
        x = sample["image"].numpy()
        y = sample["label"].numpy()
        prediction = leader_board_predict_fn(x)
        accuracies.append(accuracy_score(y, prediction, normalize=True))

        # We will consider only the first 30 batches
        if batch_idx == (max_batches - 1):
            break

    assert len(accuracies) == max_batches

    # Return the average accuracy
    mean_accuracy = np.mean(accuracies)
    return mean_accuracy


print(accuracy(dataset_path=DATASET_PATH))