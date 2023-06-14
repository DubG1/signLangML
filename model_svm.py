import os
import pickle
import torch
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# setting device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

# prepare data
root_dir='C:\\Users\\georg\\OneDrive\\Dokumente\\Studium_Windows\\signLangML\data'
csv_file='C:\\Users\\georg\\OneDrive\\Dokumente\\Studium_Windows\\signLangML\\labels.csv'

data = []
labels = []
label_img = pd.read_csv(csv_file, header=None)

for i in range(len(label_img)):
    label = label_img.iloc[i, 0]
    img_name = label_img.iloc[i, 1]
    img_path = os.path.join(root_dir, img_name)
    img = imread(img_path)
    data.append(img.flatten())
    if label.isdigit():
        label = int(label)
    else:
        label = label.lower()
        label = ord(label) - ord('a') + 10
    labels.append(label)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
x_train = torch.from_numpy(x_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
x_test = torch.from_numpy(x_test).to(device)
y_test = torch.from_numpy(y_test).to(device)
# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))