from libact.base.interfaces import ContinuousModel
import nltk
import numpy as np


class NaiveBayes(ContinuousModel):

    def __init__(self, *args):
        # arg0: label_probdist
        # arg1: feature_probdist
        self.model = nltk.NaiveBayesClassifier(*args)

    def train(self, *args):
        return self.model.train(*args)

    def predict(self, feature_set, *args, **kwargs):
        return self.model.classify(feature_set, *args, **kwargs)

    def score(self, classifier, testing_dataset, *args, **kwargs):
        return nltk.classify.accuracy(classifier, testing_dataset)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.prob_classify_many(feature, *args, **kwargs)  # [:, 1]
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
