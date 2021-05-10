from sklearn import tree
from sklearn import metrics
import numpy as np


def decisionTreeTraining(X_train, Y_train, X_test, Y_test, criterion="entropy", folds=10):
    results = []

    for i in range(folds):
        model = tree.DecisionTreeClassifier(criterion=criterion)
        model = model.fit(X_train[i], Y_train[i])

        result = model.predict(X_test[i])

        acc = metrics.accuracy_score(result, Y_test[i])

        results.append(acc)

    accuracy = round(np.mean(results) * 100)

    return [results, accuracy]
