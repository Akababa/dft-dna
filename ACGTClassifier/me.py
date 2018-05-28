import numpy as np
# from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


class ACGTClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, Aval=1, Cval=-1, Gval=-1, Tval=0, estimator=SVC(kernel='linear')):
        self.est = estimator
        self.vals = {'A': Aval, 'C': Cval, 'G': Gval, 'T': Tval}

    def get_params(self, deep=False):
        params = self.est.get_params(deep=deep)
        vals = {(k + 'val'): v for k, v in self.vals.items()}
        return {**params, **vals}

    def set_params(self, Aval=None, Cval=None, Gval=None, Tval=None, **params):
        if Aval:
            self.vals['A'] = Aval
        if Cval:
            self.vals['C'] = Cval
        if Gval:
            self.vals['G'] = Gval
        if Tval:
            self.vals['T'] = Tval
        self.est.set_params(**params)
        return self

    def _stringtofeatures(self, s):
        arr = [self.vals[c] for c in s[:self.nlen]]
        arr = abs(np.fft.rfft(arr, n=self.nlen))
        return scale(arr, copy=False)

    @staticmethod
    def _find_seq_length(X):
        minlength = min(map(len, X))
        cutoffbits = max(0, minlength.bit_length() - 3)
        return (minlength >> cutoffbits) << cutoffbits

    def _help(self, X, y):
        self.nlen = self._find_seq_length(X)

        if type(X[0]) is str:
            X = np.matrix([self._stringtofeatures(s) for s in X])

        # X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self._X = X
        self._y = y

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

        if not is_ffts:
            return self.est.fit(X, y)

        self._help(X, y)

        return self.est.fit(self._X * self._X.T, y)

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

        if not is_ffts:
            return self.est.predict(X)

        if type(X[0]) is str:
            X = np.matrix([self._stringtofeatures(s) for s in X])

        # Input validation
        # X = check_array(X)

        return self.est.predict(X * self._X.T)

    def score(self, X, y, is_ffts=True):
        return accuracy_score(y, self.predict(X, is_ffts))
