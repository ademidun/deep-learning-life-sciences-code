import deepchem as dc
import numpy as np


def run():
    # first we must load the Toxicity 21 datasets from molnet (MoleculeNet) unto our local machine
    tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()

    print(tox21_tasks)


if __name__ == '__main__':
    run()
