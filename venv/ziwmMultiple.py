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

# variables

unweightedDefaultMetricClassifScore = 0
weightedDefaultMetricClassifScore = 0
unweightedCustomMetricClassifScore = 0
weightedCustomMetricClassifScore = 0
NCAUnweightedDefaultMetricClassifScore = 0
NCAWeightedDefaultMetricClassifScore = 0
NCAUnweightedCustomMetricClassifScore = 0
NCAWeightedCustomMetricClassifScore = 0

# format: separators ';' - classifier name; amount of features; amount of neighbours; avg classifier score

def printPartialResultsToFile(numberOfTests, amountOfFeatures, amountOfNeighbours):
    testOutputFile = open("statisticTestResults.txt", "a")
    testOutputFile.write("Unweighted eucledian metric;" + str(amountOfFeatures) + ";" + str(amountOfNeighbours) + ";" + str(unweightedDefaultMetricClassifScore/float(numberOfTests)) + "\n")
    testOutputFile.write("Weighted eucledian metric;" + str(amountOfFeatures) + ";" + str(amountOfNeighbours) + ";" + str(weightedDefaultMetricClassifScore/float(numberOfTests)) + "\n")
    testOutputFile.write("Unweighted custom metric;" + str(amountOfFeatures) + ";" + str(amountOfNeighbours) + ";" + str(unweightedCustomMetricClassifScore/float(numberOfTests)) + "\n")
    testOutputFile.write("Weighted custom metric;" + str(amountOfFeatures) + ";" + str(amountOfNeighbours) + ";" + str(weightedCustomMetricClassifScore/float(numberOfTests)) + "\n")
    testOutputFile.write("Unweighted eucledian metric + NCA;" + str(amountOfFeatures) + ";" + str(amountOfNeighbours) + ";" + str(NCAUnweightedDefaultMetricClassifScore/float(numberOfTests)) + "\n")
    testOutputFile.write("Weighted eucledian metric + NCA;" + str(amountOfFeatures) + ";" + str(amountOfNeighbours) + ";" + str(NCAWeightedDefaultMetricClassifScore/float(numberOfTests)) + "\n")
    testOutputFile.write("Unweighted custom metric + NCA;" + str(amountOfFeatures) + ";" + str(amountOfNeighbours) + ";" + str(NCAUnweightedCustomMetricClassifScore/float(numberOfTests)) + "\n")
    testOutputFile.write("Weighted custom metric + NCA;" + str(amountOfFeatures) + ";" + str(amountOfNeighbours) + ";" + str(NCAWeightedCustomMetricClassifScore/float(numberOfTests)) + "\n")
    testOutputFile.close()

# Program

fullX, fullY = createSetForSelection()
rank = selection(fullX, fullY, 1)
finalRank = labelAndSortScores(rank, "dane/features.txt")

# WYPISANIE RANKINGU CECH DO PLIKU
#outputFile = open("indexed_sortedRank.txt", "w")
#for record in finalRank:
#    outputFile.write(str(record)+"\n")
#outputFile.close()

#initialising custom metric component
weightedMetric = WeightedEucledianMetric()

# initialising analysis component
nca = neighbors.NeighborhoodComponentsAnalysis()

amountOfTests = 25
testOutputFile = open("statisticTestResults.txt", "w")
testOutputFile.write("Testing metrics for averaged " + str(amountOfTests) + " random tests\n")
testOutputFile.close()
# square root from number of cases in the training set (90% of the whole data set)
numOfNeighbours = int(math.sqrt(len(fullY)*0.9))

featuresFromTo = [53, 45, 37, 30, 22, 15, 10]
# for now number of neighbors is based on features
# neighboursFromTo = [5,10,15]

for featuresToInclude in featuresFromTo:

    X, Y = createSetWithKBestFeatures(finalRank, featuresToInclude)
    
    sortedWeights = []
    for i in range(0, featuresToInclude):
            sortedWeights.append(finalRank[i][2])

    weightedMetric.SetWeights(sortedWeights)

    # clearing scores befora a new test iteration    
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

        print("Testing progress " + str(round((featuresFromTo.index(featuresToInclude)*amountOfTests+i+1.0/8.0)/float(amountOfTests*len(featuresFromTo))*100.0, 2)) + "%", end='\r')

        weightedDefaultMetricClassif = neighbors.KNeighborsClassifier(n_neighbors=numOfNeighbours, algorithm='ball_tree', weights='distance')
        weightedDefaultMetricClassif.fit(X_train, Y_train)
        weightedDefaultMetricClassifScore += weightedDefaultMetricClassif.score(X_test, Y_test)

        print("Testing progress " + str(round((featuresFromTo.index(featuresToInclude)*amountOfTests+i+2.0/8.0)/float(amountOfTests*len(featuresFromTo))*100.0, 2)) + "%", end='\r')

        unweightedCustomMetricClassif = neighbors.KNeighborsClassifier(n_neighbors=numOfNeighbours, algorithm='ball_tree', weights='uniform', metric='pyfunc', metric_params={"func":weightedMetric.CalculateDistance})
        unweightedCustomMetricClassif.fit(X_train, Y_train)
        unweightedCustomMetricClassifScore += unweightedCustomMetricClassif.score(X_test, Y_test)

        print("Testing progress " + str(round((featuresFromTo.index(featuresToInclude)*amountOfTests+i+3.0/8.0)/float(amountOfTests*len(featuresFromTo))*100.0, 2)) + "%", end='\r')

        weightedCustomMetricClassif = neighbors.KNeighborsClassifier(n_neighbors=numOfNeighbours, algorithm='ball_tree', weights='distance', metric='pyfunc', metric_params={"func":weightedMetric.CalculateDistance})
        weightedCustomMetricClassif.fit(X_train, Y_train)
        weightedCustomMetricClassifScore += weightedCustomMetricClassif.score(X_test, Y_test)

        print("Testing progress " + str(round((featuresFromTo.index(featuresToInclude)*amountOfTests+i+4.0/8.0)/float(amountOfTests*len(featuresFromTo))*100.0, 2)) + "%", end='\r')

        NCAUnweightedDefaultMetricClassif = Pipeline([('nca', nca), ('knn-unweighted-default', unweightedDefaultMetricClassif)])
        NCAUnweightedDefaultMetricClassif.fit(X_train, Y_train)
        NCAUnweightedDefaultMetricClassifScore += NCAUnweightedDefaultMetricClassif.score(X_test, Y_test)

        print("Testing progress " + str(round((featuresFromTo.index(featuresToInclude)*amountOfTests+i+5.0/8.0)/float(amountOfTests*len(featuresFromTo))*100.0, 2)) + "%", end='\r')

        NCAWeightedDefaultMetricClassif = Pipeline([('nca', nca), ('knn-weighted-default', weightedDefaultMetricClassif)])
        NCAWeightedDefaultMetricClassif.fit(X_train, Y_train)
        NCAWeightedDefaultMetricClassifScore += NCAWeightedDefaultMetricClassif.score(X_test, Y_test)

        print("Testing progress " + str(round((featuresFromTo.index(featuresToInclude)*amountOfTests+i+6.0/8.0)/float(amountOfTests*len(featuresFromTo))*100.0, 2)) + "%", end='\r')

        NCAUnweightedCustomMetricClassif = Pipeline([('nca', nca), ('knn-unweighted-custom', unweightedCustomMetricClassif)])
        NCAUnweightedCustomMetricClassif.fit(X_train, Y_train)
        NCAUnweightedCustomMetricClassifScore += NCAUnweightedCustomMetricClassif.score(X_test, Y_test)

        print("Testing progress " + str(round((featuresFromTo.index(featuresToInclude)*amountOfTests+i+7.0/8.0)/float(amountOfTests*len(featuresFromTo))*100.0, 2)) + "%", end='\r')

        NCAWeightedCustomMetricClassif = Pipeline([('nca', nca), ('knn-weighted-custom', weightedCustomMetricClassif)])
        NCAWeightedCustomMetricClassif.fit(X_train, Y_train)
        NCAWeightedCustomMetricClassifScore += NCAWeightedCustomMetricClassif.score(X_test, Y_test)

        print("Testing progress " + str(round((featuresFromTo.index(featuresToInclude)*amountOfTests+i+8.0/8.0)/float(amountOfTests*len(featuresFromTo))*100.0, 2)) + "%", end='\r')

    printPartialResultsToFile(amountOfTests, featuresToInclude, numOfNeighbours)