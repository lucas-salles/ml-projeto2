import pandas as pd
from splitData import splitData
from stratifiedFold import stratifiedFold
from MLPTraining import MLPTraining
from decisionTreeTraining import decisionTreeTraining
from knnTraining import knnTraining
from KMeansTraining import KMeansTraining


# Carregar base de dados
wineDataset = pd.read_csv("datasets/wine.data", header=None)
glassDataset = pd.read_csv("datasets/glass.data", header=None)
cancerDataset = pd.read_csv("datasets/cancer.data", header=None)

wineX, wineY = splitData(wineDataset)
glassX, glassY = splitData(glassDataset)
cancerX, cancerY = splitData(cancerDataset)

# 10-Fold
wineX_train, wineY_train, wineX_test, wineY_test = stratifiedFold(wineX, wineY)
glassX_train, glassY_train, glassX_test, glassY_test = stratifiedFold(
    glassX, glassY)
cancerX_train, cancerY_train, cancerX_test, cancerY_test = stratifiedFold(
    cancerX, cancerY)

# Treinamento e teste

###############
#
# Árvores de busca
#
###############
# Árvores de busca para wine dataset com o criterion entropy e gini
wineEntropyResults, wineEntropyAccuracy = decisionTreeTraining(
    wineX_train, wineY_train, wineX_test, wineY_test, "entropy")
wineGiniResults, wineGiniAccuracy = decisionTreeTraining(
    wineX_train, wineY_train, wineX_test, wineY_test, "gini")

# Árvores de busca para glass dataset com o criterion entropy e gini
glassEntropyResults, glassEntropyAccuracy = decisionTreeTraining(
    glassX_train, glassY_train, glassX_test, glassY_test, "entropy")
glassGiniResults, glassGiniAccuracy = decisionTreeTraining(
    glassX_train, glassY_train, glassX_test, glassY_test, "gini")

# Árvores de busca para cancer dataset com o criterion entropy e gini
cancerEntropyResults, cancerEntropyAccuracy = decisionTreeTraining(
    cancerX_train, cancerY_train, cancerX_test, cancerY_test, "entropy")
cancerGiniResults, cancerGiniAccuracy = decisionTreeTraining(
    cancerX_train, cancerY_train, cancerX_test, cancerY_test, "gini")

print("=-=-=-=-=-=-=-= Árvores de Busca =-=-=-=-=-=-=-=")

# Wine dataset
print()
print("Wine decision tree with entropy")
print(wineEntropyResults)
print("{}%".format(wineEntropyAccuracy))

print()
print("Wine decision tree with gini")
print(wineGiniResults)
print("{}%".format(wineGiniAccuracy))

# Glass dataset
print()
print("Glass decision tree with entropy")
print(glassEntropyResults)
print("{}%".format(glassEntropyAccuracy))

print()
print("Glass decision tree with gini")
print(glassGiniResults)
print("{}%".format(glassGiniAccuracy))

# Cancer dataset
print()
print("Cancer decision tree with entropy")
print(cancerEntropyResults)
print("{}%".format(cancerEntropyAccuracy))

print()
print("Cancer decision tree with gini")
print(cancerGiniResults)
print("{}%".format(cancerGiniAccuracy))
print()


###############
#
# KNN
#
###############
# KNN com métrica euclidean e k = 5 e 10 para wine dataset
wine5nnResults, wine5nnAccuracy = knnTraining(
    wineX_train, wineY_train, wineX_test, wineY_test, 5)
wine10nnResults, wine10nnAccuracy = knnTraining(
    wineX_train, wineY_train, wineX_test, wineY_test, 10)

# KNN com métrica euclidean e k = 5 e 10 para glass dataset
glass5nnResults, glass5nnAccuracy = knnTraining(
    glassX_train, glassY_train, glassX_test, glassY_test, 5)
glass10nnResults, glass10nnAccuracy = knnTraining(
    glassX_train, glassY_train, glassX_test, glassY_test, 10)

# KNN com métrica euclidean e k = 5 e 10 para cancer dataset
cancer5nnResults, cancer5nnAccuracy = knnTraining(
    cancerX_train, cancerY_train, cancerX_test, cancerY_test, 5)
cancer10nnResults, cancer10nnAccuracy = knnTraining(
    cancerX_train, cancerY_train, cancerX_test, cancerY_test, 10)

print("=-=-=-=-=-=-=-= KNN =-=-=-=-=-=-=-=")

# Wine dataset
print()
print("Wine 5NN")
print(wine5nnResults)
print("{}%".format(wine5nnAccuracy))

print()
print("Wine 10NN")
print(wine10nnResults)
print("{}%".format(wine10nnAccuracy))

# Glass dataset
print()
print("Glass 5NN")
print(glass5nnResults)
print("{}%".format(glass5nnAccuracy))

print()
print("Glass 10NN")
print(glass10nnResults)
print("{}%".format(glass10nnAccuracy))

# Cancer dataset
print()
print("Cancer 5NN")
print(cancer5nnResults)
print("{}%".format(cancer5nnAccuracy))

print()
print("Cancer 10NN")
print(cancer10nnResults)
print("{}%".format(cancer10nnAccuracy))
print()


###############
#
# MLP
#
###############
# MLP com layer_sizes = (5, 4) e (6, 6), activation = tanh e relu e 3000 iterações para wine dataset
wineTanhLayer54Results, wineTanhLayer54Accuracy = MLPTraining(
    wineX_train, wineY_train, wineX_test, wineY_test, (5, 4), 'tanh', 3000)
wineTanhLayer66Results, wineTanhLayer66Accuracy = MLPTraining(
    wineX_train, wineY_train, wineX_test, wineY_test, (6, 6), 'tanh', 3000)
wineReluLayer54Results, wineReluLayer54Accuracy = MLPTraining(
    wineX_train, wineY_train, wineX_test, wineY_test, (5, 4), 'relu', 3000)
wineReluLayer66Results, wineReluLayer66Accuracy = MLPTraining(
    wineX_train, wineY_train, wineX_test, wineY_test, (6, 6), 'relu', 3000)

# MLP com layer_sizes = (5, 4) e (6, 6), activation = tanh e relu e 3000 iterações para glass dataset
glassTanhLayer54Results, glassTanhLayer54Accuracy = MLPTraining(
    glassX_train, glassY_train, glassX_test, glassY_test, (5, 4), 'tanh', 3000)
glassTanhLayer66Results, glassTanhLayer66Accuracy = MLPTraining(
    glassX_train, glassY_train, glassX_test, glassY_test, (6, 6), 'tanh', 3000)
glassReluLayer54Results, glassReluLayer54Accuracy = MLPTraining(
    glassX_train, glassY_train, glassX_test, glassY_test, (5, 4), 'relu', 3000)
glassReluLayer66Results, glassReluLayer66Accuracy = MLPTraining(
    glassX_train, glassY_train, glassX_test, glassY_test, (6, 6), 'relu', 3000)

# MLP com layer_sizes = (5, 4) e (6, 6), activation = tanh e relu e 3000 iterações para cancer dataset
cancerTanhLayer54Results, cancerTanhLayer54Accuracy = MLPTraining(
    cancerX_train, cancerY_train, cancerX_test, cancerY_test, (5, 4), 'tanh', 3000)
cancerTanhLayer66Results, cancerTanhLayer66Accuracy = MLPTraining(
    cancerX_train, cancerY_train, cancerX_test, cancerY_test, (6, 6), 'tanh', 3000)
cancerReluLayer54Results, cancerReluLayer54Accuracy = MLPTraining(
    cancerX_train, cancerY_train, cancerX_test, cancerY_test, (5, 4), 'relu', 3000)
cancerReluLayer66Results, cancerReluLayer66Accuracy = MLPTraining(
    cancerX_train, cancerY_train, cancerX_test, cancerY_test, (6, 6), 'relu', 3000)

print("=-=-=-=-=-=-=-= MLP =-=-=-=-=-=-=-=")

# Wine dataset
print()
print("Wine with layer sizes (5, 4) and tanh activation")
print(wineTanhLayer54Results)
print("{}%".format(wineTanhLayer54Accuracy))

print()
print("Wine with layer sizes (6, 6) and tanh activation")
print(wineTanhLayer66Results)
print("{}%".format(wineTanhLayer66Accuracy))

print()
print("Wine with layer sizes (5, 4) and relu activation")
print(wineReluLayer54Results)
print("{}%".format(wineReluLayer54Accuracy))

print()
print("Wine with layer sizes (6, 6) and relu activation")
print(wineReluLayer66Results)
print("{}%".format(wineReluLayer66Accuracy))

# Glass dataset
print()
print("Glass with layer sizes (5, 4) and tanh activation")
print(glassTanhLayer54Results)
print("{}%".format(glassTanhLayer54Accuracy))

print()
print("Glass with layer sizes (6, 6) and tanh activation")
print(glassTanhLayer66Results)
print("{}%".format(glassTanhLayer66Accuracy))

print()
print("Glass with layer sizes (5, 4) and relu activation")
print(glassReluLayer54Results)
print("{}%".format(glassReluLayer54Accuracy))

print()
print("Glass with layer sizes (6, 6) and relu activation")
print(glassReluLayer66Results)
print("{}%".format(glassReluLayer66Accuracy))

# Cancer dataset
print()
print("Cancer with layer sizes (5, 4) and tanh activation")
print(cancerTanhLayer54Results)
print("{}%".format(cancerTanhLayer54Accuracy))

print()
print("Cancer with layer sizes (6, 6) and tanh activation")
print(cancerTanhLayer66Results)
print("{}%".format(cancerTanhLayer66Accuracy))

print()
print("Cancer with layer sizes (5, 4) and relu activation")
print(cancerReluLayer54Results)
print("{}%".format(cancerReluLayer54Accuracy))

print()
print("Cancer with layer sizes (6, 6) and relu activation")
print(cancerReluLayer66Results)
print("{}%".format(cancerReluLayer66Accuracy))
print()


###############
#
# K-Means
#
###############
# K-Means wine dataset
wineKMeansResults, wineKMeansAccuracy = KMeansTraining(
    wineX_train, wineY_train, wineX_test, wineY_test)

# K-Means glass dataset
glassKMeansResults, glassKMeansAccuracy = KMeansTraining(
    glassX_train, glassY_train, glassX_test, glassY_test)

# K-Means cancer dataset
cancerKMeansResults, cancerKMeansAccuracy = KMeansTraining(
    cancerX_train, cancerY_train, cancerX_test, cancerY_test)

print("=-=-=-=-=-=-=-= K-Means =-=-=-=-=-=-=-=")

# Wine dataset
print()
print("Wine K-Means")
print(wineKMeansResults)
print("{}%".format(wineKMeansAccuracy))

# Glass dataset
print()
print("Glass K-Means")
print(glassKMeansResults)
print("{}%".format(glassKMeansAccuracy))

# Cancer dataset
print()
print("Cancer K-Means")
print(cancerKMeansResults)
print("{}%".format(cancerKMeansAccuracy))
