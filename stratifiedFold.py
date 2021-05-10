from sklearn.model_selection import StratifiedKFold


def stratifiedFold(X, Y, folds=10):
    kf = StratifiedKFold(n_splits=folds)

    # 10 conjuntos de dados
    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    for train_index, test_index in kf.split(X, Y):
        X_train.append(X[train_index])
        X_test.append(X[test_index])

        Y_train.append(Y[train_index])
        Y_test.append(Y[test_index])

    return [X_train, Y_train, X_test, Y_test]
