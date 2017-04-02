import keras
from libact.base.interfaces import ContinuousModel
import numpy as np


class KerasClassifier(ContinuousModel):

    def __init__(self, *args, **kwargs):
        self.model = keras.models.Sequential(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn()+args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

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

    def add(self, *args, **kwargs):
        self.model.add(*args, **kwargs)

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

