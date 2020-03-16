"""
Train a model to predict a molecule's solubility (ability to dissolve in water).
"""
import deepchem as dc
from deepchem.models import GraphConvModel
from rdkit import Chem
def create_predict_solubility_model():

    # as explained in the Readme, we will use a GraphConv featurizer
    # which means that the model will learn from itself what features to use to describe the molecule
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')

    train_dataset, valid_dataset, test_dataset = datasets

    # to reduce overfitting we say that dropout=0.2
    # this means that 20% of outputs from each layer will randomly be set to 0
    # n_tasks because there is only one ooutput we are trying to get
    # mode = 'regression' because we want a continuous variable representing the solubility score
    # in contrast to the categorical model we built in chapter 3 that is picking one value from a set of options
    model = GraphConvModel(n_tasks=1, mode='regression', dropout=0.2)
    model.fit(train_dataset, nb_epoch=100)

    # now we will use the pearson coefficient to measure how well our model does
    # pearson coefficient is measuring the linear correlation between two variables
    # todo: why did we pick pearson coefficient?
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

    train_scores = model.evaluate(train_dataset, [metric], transformers)
    test_scores = model.evaluate(test_dataset, [metric], transformers)

    # the train scores are higher than our test scores which shows us that our model has been overfit
    print(f'train_scores: {train_scores}')
    print(f'test_scores: {test_scores}')

    return model

def run():

    model = create_predict_solubility_model()

    smiles = ['COC(C)(C)CCCC(C)CC=CC(C)=CC(=O)OC(C)C',
              'CCOC(=O)CC',
              'CSc1nc(NC(C)C)nc(NC(C)C)n1',
              'CC(C#C)N(C)C(=O)Nc1ccc(Cl)cc1',
              'Cc1cc2ccccc2cc1C']

    # in order to run the SMILE strings on our model we need to convert to a format expected by the graph convolution
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]

    featurizer = dc.feat.ConvMolFeaturizer(mols)
    x = featurizer.featurize(mols)

    predicted_solubility = model.predict_on_batch(x)

    print(predicted_solubility)


if __name__ == '__main__':
    run()
