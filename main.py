import sys
import numpy as np
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.utils import shuffle
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.neural_network import MLPClassifier
from scipy import stats

import pandas as pd

from ACGTClassifier import ACGTClassifier


def repeat_to_length(string_to_expand, length):
    return (string_to_expand * (int(length / len(string_to_expand)) + 1))[:length]

val = {'A': 1, 'C': -.5, 'G': -.25, 'T': 0}


def stringtofeatures(s, nlen):
    arr = [val[c] for c in repeat_to_length(s, nlen)]
    arr = abs(np.fft.rfft(arr, n=nlen))
    return scale(arr, copy=False)


def grid_search(fastas, labels):
    param_grid = {
        'Cval': stats.uniform(-1, 1.5), 'Gval': stats.uniform(-1.5, 2)}

    # param_grid = [
    #     {'Cval': [-1], 'Gval':[-1]}
    # ]

    clf = RandomizedSearchCV(ACGTClassifier(), param_grid,
                             n_jobs=5, n_iter=50, verbose=3, cv=10)
    clf.return_train_score = False
    clf.fit(fastas, labels)
    results = pd.DataFrame(data=clf.cv_results_)
    results = results.sort_values(by='mean_test_score', ascending=False)
    print(results)
    with open("results.csv", 'a') as f:
        results.to_csv(f, header=False)

# calculate the length of a representation according to some heuristic
# i try to keep it with lots of factors of 2 for FFT


def find_seq_length(X):
    minlength = int(min(map(len, X)))
    # minlength = int(np.median(list(map(len, X))))
    cutoffbits = max(0, minlength.bit_length() - 3)
    return (minlength >> cutoffbits) << cutoffbits

# tries one set of val and prints the result


def try_one_val(fastas, labels, classifiers, leaky=True):
    nlen = find_seq_length(fastas)
    ffts = np.matrix([stringtofeatures(s, nlen) for s in fastas])
    if leaky:
        numbers = ffts * ffts.T
    else:
        numtest = 100
        numtrain = len(labels) - numtest
        trains = ffts[:numtrain]
        tests = ffts[-numtest:]
        trainingdata = trains * trains.T
        testingdata = tests * trains.T
        # print(trainingdata.shape, testingdata.shape)
    for cl in classifiers:
        if leaky:
            scores = cross_val_score(
                cl, numbers, labels, cv=10, n_jobs=5, verbose=1)
        else:
            # I am passing in correlation matrices, not dfts
            # CV is too slow in non-leaky case
            kw = {}
            if type(cl) is ACGTClassifier:
                kw['is_ffts'] = False
            # else:
            #     cl = ACGTClassifier(estimator=cl)
            cl.fit(trainingdata, labels[:numtrain], **kw)
            scores = cl.score(testingdata, labels[-numtest:], **kw)
        print(type(cl), np.mean(scores))


if __name__ == '__main__':
    filename = "Fungi"
    with open("cleaned/1" + filename + ".fasta", 'r') as fi:
        N = int(fi.readline())
        sizes = []
        for i in range(N):
            sizes.append(int(fi.readline()))
        fastas, labels = [], []
        for i in range(N):
            for j in range(sizes[i]):
                s = fi.readline().strip()
                fastas.append(s)
                labels.append(i)

    fastas, labels = shuffle(fastas, np.array(labels))

    # classifiers = [NearestCentroid(), SVC(kernel='linear'), SVC(
    #     kernel='poly', degree=2), SVC(kernel='poly', degree=3)]
    # classifiers = [KNeighborsClassifier(n_neighbors=100)]
    # classifiers=[QuadraticDiscriminantAnalysis(),RandomForestClassifier(),DecisionTreeClassifier()]
    # classifiers=[NearestCentroid(),MLPClassifier()]
    # classifiers = [LogisticRegression(),MLPClassifier()]
    classifiers = [SVC(kernel='linear'), LogisticRegression()]
    # classifiers = [ACGTClassifier()]
    try_one_val(fastas, labels, classifiers, leaky=True)
