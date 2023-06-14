## Notes

This inlcudes documentation during development

### Method 1

we are using a Neural Network to solve the classification problem, we have a ImageClassifier with multiple layers that uses a ReLU, it then gets trained with our train data and creates a model,
this model can then be loaded and given an image.jpg and it outputs the give class

added CustomDataset class to merge the images and labels from the csv

trainning the model for the first time with learning rate 0.001, 10 epochs and batch size 32, added folders for images to test the model, model is working fine and tested 6 classes with 3 tests each, but the model is too big with ~130mb 

model compressions is needed, pruning layers, quantize, pytroch compression, smaller architecture are available options
