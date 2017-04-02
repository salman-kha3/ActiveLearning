import sklearn
import sklearn.linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from model.interfaces import Classifier


class ScikitNaiveBayes(Classifier):

    def __init__(self, *args, **kwargs):
        self.model = MultinomialNB(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, feature_set, *args, **kwargs):
        return self.model.predict(feature_set, *args, **kwargs)

    def score(self, *args, **kwargs):
        return self.model.score(*args, **kwargs)

    def pipeline(self):
        return Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', self.model,)])