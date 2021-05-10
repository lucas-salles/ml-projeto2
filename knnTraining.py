from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def knnTraining(X_train, Y_train, X_test, Y_test, k, metric="euclidean", folds=10):
    results = []

    for i in range(folds):
        model = KNeighborsClassifier(
            n_neighbors=k, metric=metric, algorithm="brute")
        model = model.fit(X_train[i], Y_train[i])

        result = model.predict(X_test[i])

        acc = metrics.accuracy_score(result, Y_test[i])

        results.append(acc)
    
    accuracy = round(np.mean(results) * 100)

    return [results, accuracy]