import os
import pickle
import torch
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# Prepare data
root_dir = '../data'
csv_file = '../data/labels.csv'

label_img = pd.read_csv(csv_file, header=None)
img_paths = [os.path.join(root_dir, img_name) for img_name in label_img.iloc[:, 1]]
labels = label_img.iloc[:, 0]

data = np.array([resize(imread(img_path), (64, 64)).flatten() for img_path in img_paths], dtype=np.float32)

# Convert labels
def convert_label(label):
    if label.isdigit():
        return int(label)
    else:
        return ord(label.lower()) - ord('a') + 10

labels = np.array([convert_label(label) for label in labels], dtype=np.int64)

# Train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of the variance
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# Move data to CUDA tensors
x_train = torch.from_numpy(x_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
x_test = torch.from_numpy(x_test).to(device)
y_test = torch.from_numpy(y_test).to(device)

# Train classifier
print('Training classifier...')
parameters = {'gamma': [0.01, 0.001], 'C': [1, 10]}
classifier = GridSearchCV(SVC(), parameters, n_jobs=-1)
classifier.fit(x_train.cpu().numpy(), y_train.cpu().numpy())

# Test performance
print('Testing classifier...')
best_estimator = classifier.best_estimator_
y_prediction = best_estimator.predict(x_test.cpu().numpy())
score = accuracy_score(y_prediction, y_test.cpu().numpy())
print('{:.2f}% of samples were correctly classified'.format(score * 100))

# Save the model
pickle.dump(best_estimator, open('./svm_model.p', 'wb'))
