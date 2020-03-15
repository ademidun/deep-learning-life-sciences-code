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

    train_model(train_dataset, test_dataset, transformers)

def train_model(train_dataset, test_dataset, transformers):
    """
    Train the model using a multitask classifier because there are multiple outputs for each sample
    and evaluate model using the mean ROC AUC.
    :param train_dataset:
    :type train_dataset:
    :param transformers:
    :type transformers:
    :return:
    :rtype:
    """

    # this model builds a fully connected network (an MLP)
    #  since we have 12 assays we're testing for, being able to map to multiple outputs is ideal
    # layer_sizes means that we have one hidden layer which has a width of 1,000
    model = dc.models.MultitaskClassifier(n_tasks=12, n_features=1024, layer_sizes=[1000])

    # nb_epoch means that we will divide the data into batches, and do one step of gradient descent for each batch
    model.fit(train_dataset, nb_epoch=10)

    # how do we know how accurate our model is? we will find the mean ROC AUC score across all tasks

    # What is an ROC AUC score? We are trying to predict the toxicity of the molecules,
    # Receiver Operating Characteristic, Area Under Curve
    # If there exists any threshold value where, the true positive rate is 1 and false positive is 0 then score is 1
    # so we pick a threshold of what is considered a toxic molecule
    # if we pick a threshold value that's too low, we will say too many safe molecules are toxic (high false positive)
    # alternatively, if we pick one too igh, we will say that toxic molecules are safe (high false negative)
    # note on understanding false positive terminology.\:
    # Imagine a molecule that is actually toxic. "Is this molecule toxic?" "No." We gave a negative response
    # the answer is relative to what we are testing for, in this case, we are testing if a molecule is toxic
    # so we are making a tradeoff between high false positive vs high false negative so we use something called
    # an ROC AUC curve, which graphs the tradeofff between the false positive rate and the true positive rate

    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

    # evaluate the performance of this model on the train_dataset using the ROC AUC metric

    train_scores = model.evaluate(train_dataset, [metric], transformers)
    test_scores = model.evaluate(test_dataset, [metric], transformers)

    # the train scores are higher than our test scores which shows us that our model has been overfit
    print(f'train_scores: {train_scores}')
    print(f'test_scores: {test_scores}')

if __name__ == '__main__':
    run()
