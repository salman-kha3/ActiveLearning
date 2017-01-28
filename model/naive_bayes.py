"""SVM

An interface for scikit-learn's C-Support Vector Classifier model.
"""
import logging

import sklearn.linear_model

LOGGER = logging.getLogger(__name__)

import numpy as np
from sklearn.naive_bayes import MultinomialNB

from libact.base.interfaces import ContinuousModel


class NaiveBayes(ContinuousModel):

    """C-Support Vector Machine Classifier

    References
    ----------
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def     __init__(self, *args, **kwargs):
        self.model = sklearn.naive_bayes.MultinomialNB(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn()+args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        if hasattr(self.model, "decision_function"):
            dvalue = self.model.decision_function(feature, *args, **kwargs)
        else:
            dvalue = self.model.predict_proba(feature, *args, **kwargs)
            # [:, 1]

        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue