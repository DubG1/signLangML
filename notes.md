## Notes

This inlcudes documentation during development

### Method 1

we are using a Neural Network to solve the classification problem, we have a ImageClassifier with multiple layers that uses a ReLU, it then gets trained with our train data and creates a model,
this model can then be loaded and given an image.jpg and it outputs the give class

added CustomDataset class to merge the images and labels from the csv

trainning the model for the first time with learning rate 0.001, 10 epochs and batch size 32, loss ends up staying around ~3.x so i guess the learning rate needs to be adjusted

lr 0.00001, loss decreasing to ~0.6

added folders for images to test the model
