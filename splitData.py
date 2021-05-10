import numpy as np


def splitData(dataset):
    columns = len(dataset.columns)

    Y = dataset[0]  # extrai a primeira coluna, que é o label
    X = dataset.loc[:, 1:columns-1]

    # Transforma para Array NumPy
    X = np.array(X)
    Y = np.array(Y)

    return [X, Y]
