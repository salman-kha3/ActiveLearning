import nltk
import numpy as np

from libact.base.interfaces import ContinuousModel


class NaiveBayesLibact(ContinuousModel):

    def __init__(self, *args, **kwargs):
        self.model = nltk.NaiveBayesClassifier(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.model.train(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.classify(*args, **kwargs)

    def score(self, *args, **kwargs):
        return nltk.classify.accuracy(*args, **kwargs)

    def predict_real(self, *args, **kwargs):
        dvalue = self.model.predict_proba(*args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue


