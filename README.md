# DEFORESTATION IMAGE CLASSIFIER

## Background

Climate change is a major problem in the modern world. New technologies are 
being developed every day to fight against its effects. Developing models that can
quickly estimate the degree of deforestation can be very helpful to help organizations
to decide where to focus the resources

## Results and analysis

After looking for the best hyperparameters using a convolutional neural network, I got a model with a 
score of 0.98 with the validation dataset during training. I considered that it is a good number, 
so I used the model to make predictions on the test dataset. 

I have some concerns, though. I'm afraid that my model is very overfitted, as I didn't have much time to
code measures against it, such as dropout, for example.

## Solution

The network architecture is as follows. Only two convolutional layers, activated with ReLU, followed each one by
a MaxPool layer. Then, the output is flattened and fed to three fully connected layers, the firsts one with
ReLU as activation and the last one with 3 outputs. I used a batch size of 64, and 30 epochs to train. 
As optimizer, I used Adam, with the defaults values that Pytorch provides. 

I built a custom class to manage the data. It's based in Pytorch Dataloaders, and, if the data is ment to used for
training, applies several transformations to the images, trying to do some data augmentation.

I first split the train dataset in train and test, to have some data that would allow me to evaluate
the model and the parameters passed to it. I calculated the f1 score. Seeing that the outcome was satisfying, I then 
retrained the model with the whole dataset. This was done based in the assumption that the more data
the model sees, the better trained will be. After that, I used the best estimator of the grid search
to make the final predictions (saved in the `/predictions` directory).

## Installation

Just run `pip install -r requirements`, and you're ready to go!

To train a new model, go in `/src` in the terminal and run `python main.py`.

## Contact info

Sebastián Dolgonos

[Linkedin](https://www.linkedin.com/in/sebastián-dolgonos-565733226/)

[Portfolio](https://bass2015.github.io)

[Github](https://github.com/Bass2015)

## License 

[MIT](https://opensource.org/licenses/MIT)