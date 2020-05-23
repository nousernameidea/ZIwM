import os
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

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
def createSetForTests(sortedIndexedRank, numOfBestFeatures):
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
    # ranking = X_kbest.scores_
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

fullX, fullY = createSetForSelection()
rank = selection(fullX, fullY, 1)
finalRank = labelAndSortScores(rank, "dane/features.txt")

# TYLKO DO WYPISANIA RANKINGU CECH DO PLIKU
# outputFile = open("indexed_sortedRank.txt", "w")
# for record in finalRank:
#     outputFile.write(str(record)+"\n")
# outputFile.close()

featuresToInclude = 1
XforTest, YforTest = createSetForTests(finalRank, featuresToInclude)



