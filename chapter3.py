import deepchem as dc
import numpy as np


def run():
    # x =  4 random arrays, each array has 5 elements
    # y = 1 random array with 4 elements
    x = np.random.random((4, 5))
    y = np.random.random((1, 4))

    print('x', x)
    print('y', y)


if __name__ == '__main__':
    run()
