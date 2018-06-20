import numpy as np
# from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


def repeat_to_length(string_to_expand, length):
    return (string_to_expand * (int(length / len(string_to_expand)) + 1))[:length]


# calculate the length of a representation according to some heuristic
# i try to keep it with lots of factors of 2 for FFT
def find_seq_length(X):
    mylength = int(np.median(list(map(len, X))))
    # print(mylength)
    cutoffbits = max(0, mylength.bit_length() - 3)
    return (mylength >> cutoffbits) << cutoffbits


def stringtofeatures(s, nlen, val):
    arr = val[np.fromstring(s, dtype=np.uint8)]
    arr = np.abs(np.fft.rfft(arr, n=nlen))
    # print(type(arr), arr.dtype)
    return scale(arr, copy=False)


def getallffts(fastas, val=None, four=False):
    nlen = find_seq_length(fastas)
    if four:
        ffts = np.array([np.matrix([np.abs(np.fft.rfft(np.fromstring(
            s, dtype=np.uint8) == ord(letter), n=nlen)) for letter in "ACGT"]) for s in fastas])
    else:
        assert val is not None
        arr = np.empty(256, dtype=float)
        for k, v in val.items():
            arr[ord(k)] = v
        ffts = np.matrix([stringtofeatures(s, nlen, arr) for s in fastas])
    return ffts


class ACGTClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, A=1, C=-1, G=-1, T=0, estimator=SVC(kernel='linear')):
        self.estimator = estimator
        self.A = A
        self.C = C
        self.G = G
        self.T = T
        self.val = np.empty(256, dtype=float)

    def _update_val(self):
        self.val[ord('A')] = self.A
        self.val[ord('C')] = self.C
        self.val[ord('G')] = self.G
        self.val[ord('T')] = self.T

    def makeX(self, X):
        if type(X[0]) is str:  # only convert it if it's ACGT
            self._update_val()
            return np.matrix([stringtofeatures(s, self.nlen, self.val)
                              for s in X])
        elif X[0].shape[0] == 4:  # it's in separate letter form
            l = np.array([self.A, self.C, self.G, self.T])
            # l = l.reshape((4, 1))
            X = np.transpose(X, (0, 2, 1))
            # print(X.shape)
            # y = np.empty(X.shape[1:])
            # XX = self.A * X[0] + self.C * X[1] + \
            #     self.G * X[2] + self.T * X[3]
            # XX = [self.A * s[0] + l[1] * s[1] + l[2]
            #                 * s[2] + l[3] * s[3] for s in X]
            # XX = np.dot(X, l)
            XX = [np.dot(s, l) for s in X]
            # print(XX.shape,type(XX))
            XX = scale(XX, copy=False, axis=1)
            # print(XX.shape, np.sum(XX[0]))
            # print(type(XX))
            return np.matrix(XX)

    def fit(self, X, y, is_ffts=True):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """

        # if not is_ffts:
        #     return self.estimator.fit(X, y)

        self.nlen = find_seq_length(X)

        self._X = self.makeX(X)
        self._y = y

        # X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        return self.estimator.fit(self._X * self._X.T, y)

    def predict(self, X, is_ffts=True):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        # check_is_fitted(self, ['X_', 'y_'])

        # if not is_ffts:
        #     return self.estimator.predict(X)

        X = self.makeX(X)

        # Input validation
        # X = check_array(X)

        return self.estimator.predict(X * self._X.T)

    def score(self, X, y, is_ffts=True):
        return accuracy_score(y, self.predict(X, is_ffts))
