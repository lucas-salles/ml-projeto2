from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import numpy as np


def MLPTraining(X_train, Y_train, X_test, Y_test, hidden_layer_sizes, activation='relu', max_iter=2000, folds=10):
    results = []

    for i in range(folds):
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                              activation=activation, max_iter=max_iter, random_state=1)
        model = model.fit(X_train[i], Y_train[i])

        result = model.predict(X_test[i])

        acc = metrics.accuracy_score(result, Y_test[i])

        results.append(acc)

    # print(results)
    accuracy = round(np.mean(results) * 100)
    # print("{}%".format(accuracy))

    return [results, accuracy]
