## Notes

This inlcudes documentation during development

### Method 1

we are using a Neural Network to solve the classification problem, we have a ImageClassifier with multiple layers that uses a ReLU, it then gets trained with our train data and creates a model,
this model can then be loaded and given an image.jpg and it outputs the give class

added CustomDataset class to merge the images and labels from the csv

trainning the model for the first time with learning rate 0.001, 10 epochs and batch size 32, added folders for images to test the model, model is working fine and tested 6 classes with 3 tests each, but the model is too big with ~130mb 

model compressions is needed, pruning layers, quantize, pytroch compression, smaller architecture are available options
changed the architecture and reduced model size from 130mb to 4mb

### Method 2

SVM to solve the classification problem

we are using train_test_split to split the data and labels into training and testing sets, then we started training but it took way too long so we quickly realized dimensionality reduction was needed similiar to what we had to do with our NN, we used PCA to reduce the dimensions and retain 95% of the variance in the data

we moved convert the training and testing data and labels to CUDA tensors, then train the classifier with a few parameters, then do grid search with cross-validation, then we test select the best classifier by letting it predict labels for the testing data and then comparing them to the actual labels

used code:
GridSearchCV() - grid search with cross-validation
classifier.fit() -  train the classifier model, instance of SVC, takes the training data and labels as input and trains the classifier using all possible combinations of hyperparameters specified in the parameter grid
best_estimator_ - attribute of the GridSearchCV object holds the estimator that performed the best during the grid search
