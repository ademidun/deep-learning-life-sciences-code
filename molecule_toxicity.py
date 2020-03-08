import deepchem as dc
import numpy as np


def run():
    """
    tox21_tasks is a list of chemical assays and our dataset
    contains training data that will tell us whether a certain molecule binds
    to one of the molecules in
    :return:
    :rtype:
    """
    # first we must load the Toxicity 21 datasets from molnet (MoleculeNet) unto our local machine
    tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()


    # tox21_tasks represent 12 assays or bilogicial targets taht we want to see if our molecule binds to
    print(tox21_tasks)


    # train_dataset is 6264 molecules with a feature vector of length 1024


    # it has a feature vector Y, for each of the 12 assays
    train_dataset, valid_dataset, test_dataset = tox21_datasets

    # the w represents the weights and a weight of zero means that no experiment was run
    # to see if the molecule binds to that assay
    np.count_nonzero(train_dataset.w == 0)

    #  this is a BalancingTransformer because most of the molecules do not bind to most targets
    #  so most of the labels are zero and a model always predicting zero could actually work (but it would be useless!)
    #  BalancingTransformer adjusts dataset's wieghts of individual points so all classes have same total weight
    #  Loss function won't have systematic preference for one class
    print(transformers)

def train_model(train_dataset):
    """
    Train the model using a multitask classifier because there are multiple labels for each sample
    :param train_dataset:
    :type train_dataset:
    :return:
    :rtype:
    """

    # layer_sizes means that we have one hidden layer which has a width of 1,000
    model = dc.models.MultitaskClassifier(n_tasks=12, n_features=1024, layer_sizes=[100])

    # nb_epoch means that we will divide the data into batches, and do one step of gradient descent for each batch
    model.fit(train_dataset, nb_epoch=10)


if __name__ == '__main__':
    run()
