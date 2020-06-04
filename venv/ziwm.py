import os
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import neighbors
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import math

# wczytanie danych z jednego pliku
def loadDataFromFile(filepath, diagnose_id):
    try:
        file = open(filepath, "r")
        lines = file.readlines()
        file.close()
    except:
        raise Exception

    num_of_records = len(lines[0].split())
    xArray = []
    yArray = []

    for column_num in range(0, num_of_records):
        localXArray = []
        for line in lines:
            localXArray.append(int(line.split()[column_num]))

        xArray.append(localXArray)
        yArray.append(diagnose_id)

    return xArray, yArray

# wczytanie danych z jednego pliku, ale tylko wybranych cech
def loadLinesFromFile(filepath, lines_to_read, diagnose_id):
    try:
        file = open(filepath, "r")
        lines = file.readlines()
        file.close()
    except:
        raise Exception

    num_of_records = len(lines[0].split())
    xArray = []
    yArray = []

    for column_num in range(0, num_of_records):
        localXArray = []
        for line in lines_to_read:
            localXArray.append(int(lines[line].split()[column_num]))

        xArray.append(localXArray)
        yArray.append(diagnose_id)

    return xArray, yArray

# wczytanie wszystkich plików
def createSetForSelection():
    finalX = []
    finalY = []

    X, Y = loadDataFromFile("dane/inne.txt", 1)
    for x in X:
        finalX.append(x)
    for y in Y:
        finalY.append(y)

    X, Y = loadDataFromFile("dane/ang_prect.txt", 2)
    for x in X:
        finalX.append(x)
    for y in Y:
        finalY.append(y)

    X, Y = loadDataFromFile("dane/ang_prct_2.txt", 3)
    for x in X:
        finalX.append(x)
    for y in Y:
        finalY.append(y)

    X, Y = loadDataFromFile("dane/mi.txt", 4)
    for x in X:
        finalX.append(x)
    for y in Y:
        finalY.append(y)

    X, Y = loadDataFromFile("dane/mi_np.txt", 5)
    for x in X:
        finalX.append(x)
    for y in Y:
        finalY.append(y)

    return finalX, finalY

# wczytanie wszystkich plików, ale tylko n najlepszych cech
def createSetWithKBestFeatures(sortedIndexedRank, numOfBestFeatures):
    finalX = []
    finalY = []
    featuresToRead = []

    for i in range(0, numOfBestFeatures):
        featuresToRead.append(sortedIndexedRank[i][2])


    X, Y = loadLinesFromFile("dane/inne.txt", featuresToRead, 1)
    for x in X:
        finalX.append(x)
    for y in Y:
        finalY.append(y)

    X, Y = loadLinesFromFile("dane/ang_prect.txt", featuresToRead, 2)
    for x in X:
        finalX.append(x)
    for y in Y:
        finalY.append(y)

    X, Y = loadLinesFromFile("dane/ang_prct_2.txt", featuresToRead, 3)
    for x in X:
        finalX.append(x)
    for y in Y:
        finalY.append(y)

    X, Y = loadLinesFromFile("dane/mi.txt", featuresToRead, 4)
    for x in X:
        finalX.append(x)
    for y in Y:
        finalY.append(y)

    X, Y = loadLinesFromFile("dane/mi_np.txt", featuresToRead, 5)
    for x in X:
        finalX.append(x)
    for y in Y:
        finalY.append(y)

    return finalX, finalY

# ranking cech
def selection(X, y, numOfFeatures):
    # f_classif - funkcja licząca wartość analizy wariacji
    selector = SelectKBest(f_classif, k=numOfFeatures)
    X_kbest = selector.fit(X, y)
    return X_kbest.scores_

# dodanie oznaczenia cech i posortowanie rankigu malejaco po wyniku
def labelAndSortScores(scores, pathToLabels):
    try:
        file = open(pathToLabels, "r")
        labels = file.read().splitlines()
        file.close()
    except:
        raise Exception

    merged_list = [(labels[i], scores[i], i) for i in range(0, len(scores))]
    return sorted(merged_list, key=lambda x: x[1], reverse = True)

class WeightedEucledianMetric:

    weights = []

    def SetWeights(self, weightsArray):
        self.weights = weightsArray

    def CalculateDistance(self, x, y):
        if len(x) != len(y):
            raise ValueError("x and y need to have the same length")

        return math.sqrt(sum([(((y[i] - x[i]) ** 2) * self.weights[i]) for i in range(len(x))]))


# Program

fullX, fullY = createSetForSelection()
rank = selection(fullX, fullY, 1)
finalRank = labelAndSortScores(rank, "dane/features.txt")

# WYPISANIE RANKINGU CECH DO PLIKU
#outputFile = open("indexed_sortedRank.txt", "w")
#for record in finalRank:
#    outputFile.write(str(record)+"\n")
#outputFile.close()

featuresToInclude = 59
numOfNeighbours = int(math.sqrt(featuresToInclude))

weightedMetric = WeightedEucledianMetric()

sortedWeights = []
for i in range(0, featuresToInclude):
        sortedWeights.append(finalRank[i][2])

weightedMetric.SetWeights(sortedWeights)

amountOfTests = 25

X, Y = createSetWithKBestFeatures(finalRank, featuresToInclude)
nca = neighbors.NeighborhoodComponentsAnalysis()

unweightedDefaultMetricClassifScore = 0
weightedDefaultMetricClassifScore = 0
unweightedCustomMetricClassifScore = 0
weightedCustomMetricClassifScore = 0
NCAUnweightedDefaultMetricClassifScore = 0
NCAWeightedDefaultMetricClassifScore = 0
NCAUnweightedCustomMetricClassifScore = 0
NCAWeightedCustomMetricClassifScore = 0

for i in range(amountOfTests):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    unweightedDefaultMetricClassif = neighbors.KNeighborsClassifier(n_neighbors=numOfNeighbours, algorithm='ball_tree', weights='uniform')
    unweightedDefaultMetricClassif.fit(X_train, Y_train)
    unweightedDefaultMetricClassifScore += unweightedDefaultMetricClassif.score(X_test, Y_test)

    print("Testing progress " + str(round((i+1/8)/float(amountOfTests)*100.0, 2)) + "%", end='\r')

    weightedDefaultMetricClassif = neighbors.KNeighborsClassifier(n_neighbors=numOfNeighbours, algorithm='ball_tree', weights='distance')
    weightedDefaultMetricClassif.fit(X_train, Y_train)
    weightedDefaultMetricClassifScore += weightedDefaultMetricClassif.score(X_test, Y_test)

    print("Testing progress " + str(round((i+2/8)/float(amountOfTests)*100.0, 2)) + "%", end='\r')

    unweightedCustomMetricClassif = neighbors.KNeighborsClassifier(n_neighbors=numOfNeighbours, algorithm='ball_tree', weights='uniform', metric='pyfunc', metric_params={"func":weightedMetric.CalculateDistance})
    unweightedCustomMetricClassif.fit(X_train, Y_train)
    unweightedCustomMetricClassifScore += unweightedCustomMetricClassif.score(X_test, Y_test)

    print("Testing progress " + str(round((i+3/8)/float(amountOfTests)*100.0, 2)) + "%", end='\r')

    weightedCustomMetricClassif = neighbors.KNeighborsClassifier(n_neighbors=numOfNeighbours, algorithm='ball_tree', weights='distance', metric='pyfunc', metric_params={"func":weightedMetric.CalculateDistance})
    weightedCustomMetricClassif.fit(X_train, Y_train)
    weightedCustomMetricClassifScore += weightedCustomMetricClassif.score(X_test, Y_test)

    print("Testing progress " + str(round((i+4/8)/float(amountOfTests)*100.0, 2)) + "%", end='\r')

    NCAUnweightedDefaultMetricClassif = Pipeline([('nca', nca), ('knn-unweighted-default', unweightedDefaultMetricClassif)])
    NCAUnweightedDefaultMetricClassif.fit(X_train, Y_train)
    NCAUnweightedDefaultMetricClassifScore += NCAUnweightedDefaultMetricClassif.score(X_test, Y_test)

    print("Testing progress " + str(round((i+5/8)/float(amountOfTests)*100.0, 2)) + "%", end='\r')

    NCAWeightedDefaultMetricClassif = Pipeline([('nca', nca), ('knn-weighted-default', weightedDefaultMetricClassif)])
    NCAWeightedDefaultMetricClassif.fit(X_train, Y_train)
    NCAWeightedDefaultMetricClassifScore += NCAWeightedDefaultMetricClassif.score(X_test, Y_test)

    print("Testing progress " + str(round((i+6/8)/float(amountOfTests)*100.0, 2)) + "%", end='\r')

    NCAUnweightedCustomMetricClassif = Pipeline([('nca', nca), ('knn-unweighted-custom', unweightedCustomMetricClassif)])
    NCAUnweightedCustomMetricClassif.fit(X_train, Y_train)
    NCAUnweightedCustomMetricClassifScore += NCAUnweightedCustomMetricClassif.score(X_test, Y_test)

    print("Testing progress " + str(round((i+7/8)/float(amountOfTests)*100.0, 2)) + "%", end='\r')

    NCAWeightedCustomMetricClassif = Pipeline([('nca', nca), ('knn-weighted-custom', weightedCustomMetricClassif)])
    NCAWeightedCustomMetricClassif.fit(X_train, Y_train)
    NCAWeightedCustomMetricClassifScore += NCAWeightedCustomMetricClassif.score(X_test, Y_test)

    print("Testing progress " + str(round((i+8/8)/float(amountOfTests)*100.0, 2)) + "%", end='\r')

print("Avg unweighted default metric: " + str(unweightedDefaultMetricClassifScore/float(amountOfTests)))
print("Avg weighted default metric: " + str(weightedDefaultMetricClassifScore/float(amountOfTests)))
print("Avg unweighted custom metric: " + str(unweightedCustomMetricClassifScore/float(amountOfTests)))
print("Avg weighted custom metric: " + str(weightedCustomMetricClassifScore/float(amountOfTests)))
print("Avg unweighted default + NCA: " + str(NCAUnweightedDefaultMetricClassifScore/float(amountOfTests)))
print("Avg weighted default + NCA: " + str(NCAWeightedDefaultMetricClassifScore/float(amountOfTests)))
print("Avg unweighted custom + NCA: " + str(NCAUnweightedCustomMetricClassifScore/float(amountOfTests)))
print("Avg weighted custom + NCA: " + str(NCAWeightedCustomMetricClassifScore/float(amountOfTests)))