"""
In molecule_toxicity.py we used a premade model class, now we will be creating an architecture from scratch
The reason we might want to do this is if we are working on a dataset where no predefined architecture exists.
# todo what is the difference between an architecture and a model?

This works by creating two convolution layerr which is a small square that is a subset of the image.
Then uses two fully connected layers to predict the digit from the local features.

# todo what does that actually mean? [tk include a diagram]
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import deepchem as dc
import deepchem.models.tensorgraph.layers as layers


def create_model():
    """
    Create our own MNIST model from scratch
    :return:
    :rtype:
    """
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

    # the layers from deepchem are the building blocks of what we will use to make our deep learning architecture

    # now we wrap our dataset into a NumpyDataset

    train_dataset = dc.data.NumpyDataset(mnist.train.images, mnist.train.labels)
    test_dataset = dc.data.NumpyDataset(mnist.test.images, mnist.test.labels)

    # we will create a model that will take an input, add multiple layers, where each layer takes input from the
    # previous layers.

    model = dc.models.TensorGraph(model_dir='mnist')

    # 784 corresponds to an image of size 28 X 28
    # 10 corresponds to the fact that there are 10 possible digits (0-9)
    # the None indicates that we can accept any size input (e.g. an empty array or 500 items each with 784 features)
    # our data is also categorical so we must one hot encode, set single array element to 1 and the rest to 0
    feature = layers.Feature(shape=(None, 784))
    labels = layers.Label(shape=(None, 10))

    # in order to apply convolutional layers to our input, we convert flat vector of 785 to 28X28
    # in_layers means it takes our feature layer as an input
    make_image = layers.Reshape(shape=(None, 28, 28), in_layers=feature)

    # now that we have reshaped the input, we pass to convolution layers

    conv2d_1 = layers.Conv2D(num_outputs=32, activation_fn=tf.nn.relu, in_layers=make_image)

    conv2d_2 = layers.Conv2D(num_outputs=64, activation_fn=tf.nn.relu, in_layers=conv2d_1)

    # we want to end by applying fully connected (Dense) layers to the outputs of our convolutional layer
    # but first, we must flatten the layer from a 2d matrix to a 1d vector

    flatten = layers.Flatten(in_layers=conv2d_2)
    dense1 = layers.Dense(out_channels=1024,activation_fn=tf.nn.relu, in_layers=flatten)

    # note that this is final layer so out_channels of 10 represents the 10 outputs and no activation_fn
    dense2 = layers.Dense(out_channels=10,activation_fn=None, in_layers=dense1)

    # next we want to connect this output to a loss function, so we can train the output

    # compute the value of loss function for every sample then average of all samples to get final loss (ReduceMean)
    smce = layers.SoftMaxCrossEntropy(in_layers=[labels, dense2])
    loss = layers.ReduceMean(in_layers=smce)
    model.set_loss(loss)

    # for MNIST we want the probability that a given sample represents one of the 10 digits
    # we can achieve this using a softmax function to get the probabilities, then cross entropy to get the labels

    output = layers.SoftMax(in_layers=dense2)
    model.add_output(output)

    # if our model takes long to train, reduce nb_epoch to 1
    model.fit(train_dataset,nb_epoch=1)

    # our metric is accuracy, the fraction of labels that are accurately predicted
    metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    train_scores = model.evaluate(train_dataset, [metric])
    test_scores = model.evaluate(test_dataset,[metric])

    print('train_scores', train_scores)
    print('test_scores', test_scores)

if __name__ == '__main__':
    create_model()







