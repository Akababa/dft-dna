import os
import numpy as np
# from sklearn.preprocessing import scale
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

from ACGTClassifier import ACGTClassifier, getallffts


def meprofile(func):
    import cProfile
    import pstats
    import io

    def foo(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        print(s.getvalue())
        with open("prof.txt", 'w') as f:
            f.write(s.getvalue())

    return foo


# @meprofile
def grid_search(fastas, labels, four=False, iterations=1000):
    fffts = getallffts(fastas, four=True) if four else fastas
    # print(np.array(fffts).shape)
    param_grid = {
        'C': stats.uniform(-2, 4), 'G': stats.uniform(-2, 4)}

    while iterations > 0:
        clf = RandomizedSearchCV(ACGTClassifier(), param_grid,
                                 n_jobs=5, n_iter=100, verbose=1, cv=5)
        clf.return_train_score = False
        clf.fit(fffts, labels)
        results = pd.DataFrame(data=clf.cv_results_)
        results = results[['mean_test_score', 'param_C', 'param_G']]
        results = results.sort_values(by='mean_test_score', ascending=False)
        for i in range(10):
            try:
                filename = "results" + str(i) + ".csv"
                with open(filename, 'a') as f:
                    is_new_file = (os.stat(filename).st_size == 0)
                    results.to_csv(f, header=is_new_file, index=False)
                print("results saved to", filename)
                break
            except:
                print("Oops, error!")
        iterations -= 100


# tries one set of val and prints the result
def try_one_val(fastas, labels, classifiers, val, leaky=True):
    ffts = getallffts(fastas, val, four=False)
    # if leaky:
    #     numbers = ffts * ffts.T
    # else:
    #     numtest = int(len(labels) / 10)
    #     numtrain = len(labels) - numtest
    #     trains = ffts[:numtrain]
    #     tests = ffts[-numtest:]
    #     trainingdata = trains * trains.T
    #     testingdata = tests * trains.T
    #     # print(trainingdata.shape, testingdata.shape)
    for cl in classifiers:
        if leaky:
            samples = ffts * ffts.T
        else:
            cl = ACGTClassifier(**val, estimator=cl)
            samples = ffts
        scores = cross_val_score(
            cl, samples, labels, cv=10, n_jobs=5, verbose=1)
        print(type(cl), np.mean(scores))


if __name__ == '__main__':
    filename = "Protists"
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
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput

    # with PyCallGraph(output=GraphvizOutput()):
    # grid_search(fastas, labels, four=True)

    # myval = {'A': 1, 'C': -1, 'G': -1, 'T': 1}  # P/P

    # myval = {'A': 1, 'C': -.35, 'G': -.6, 'T': 0} #insects
    # myval = {'A': 1, 'C': 0.18, 'G': 0.28, 'T': 0}  # fungi1
    # myval = {'A': 1, 'C': -.305, 'G': 0.464, 'T': 0} #fungi2, 0.87
    # myval = {'A': 1, 'C': -.4, 'G': 0.6, 'T': 0} #fungi3, 0.88
    # myval = {'A': 1, 'C': .1, 'G': 0.8, 'T': 0} #fungi4, 0.87
    # myval = {'A': 1, 'C': -.2, 'G': 0.1, 'T': 0} #fungi4, 0.88 logreg
    myval = {'A': 1, 'C': -.93, 'G': -2, 'T': 0} #protists, 0.9 logreg
    # myval = {'A': 1, 'C': 1.6, 'G': 1.3, 'T': 0} #protists1, 0.88
    # myval = {'A': 1, 'C': -1, 'G': -2, 'T': 0}  # protists2, 0.89 logreg
    # myval = {'A': 1, 'C': -1.5, 'G': 1.8, 'T': 0}  # protists3, 0.88

    # classifiers = [NearestCentroid(), SVC(kernel='linear'), SVC(
    #     kernel='poly', degree=2), SVC(kernel='poly', degree=3)]
    # classifiers = [KNeighborsClassifier(n_neighbors=100)]
    # classifiers=[RandomForestClassifier(n_estimators=200)]#,DecisionTreeClassifier(),QuadraticDiscriminantAnalysis()]
    # classifiers=[NearestCentroid(),MLPClassifier()]
    # classifiers = [LogisticRegression(),MLPClassifier()]

    classifiers = [SVC(kernel='linear'), LogisticRegression()]
    try_one_val(fastas, labels, classifiers, myval, leaky=True)
