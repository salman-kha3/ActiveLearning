"""SVM

An interface for scikit-learn's C-Support Vector Classifier model.
"""
import logging

import nltk
import sklearn.linear_model

LOGGER = logging.getLogger(__name__)

import numpy as np
from sklearn.naive_bayes import MultinomialNB

from libact.base.interfaces import ContinuousModel


class NLTKNaiveBayes(ContinuousModel):

    """C-Support Vector Machine Classifier

    References
    ----------
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    # classifier = nltk.NaiveBayesClassifier.train(train_set)

    def __init__(self, *args, **kwargs):
        self.model = nltk.NaiveBayesClassifier(*args, **kwargs)

    def train(self,*args, **kwargs):
        return self.model.train(*args, **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.classify(feature, *args, **kwargs)

    def score(self, classifier, testing_dataset, *args, **kwargs):
        return nltk.classify.accuracy(classifier, testing_dataset)

    def predict_real(self, feature, *args, **kwargs):
        if hasattr(self.model, "decision_function"):
            dvalue = self.model.decision_function(feature, *args, **kwargs)
        else:
            dvalue = self.model.prob_classify_many(feature, *args, **kwargs)           # [:, 1]

        prob_array = []

        for item in dvalue:
            prob_list = []
            for exact in item._prob_dict.values():
                prob_list.append(exact)

            prob_list_to_array = np.array(prob_list)
            prob_array.append(prob_list_to_array)

        prob_feature = np.array(prob_array)

        if len(np.shape(prob_feature)) == 1:  # n_classes == 2
            return np.vstack((-prob_feature, prob_feature)).T
        else:
            return prob_feature