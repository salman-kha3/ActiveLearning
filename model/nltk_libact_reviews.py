import random

import nltk
import numpy as np
from libact.query_strategies import UncertaintySampling

from libact.labelers import IdealLabeler

from libact.base.dataset import Dataset
from nltk.corpus import movie_reviews, stopwords
import matplotlib.pyplot as plt

from model.nltk_libact_classifier import NLTKNaiveBayes

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
#
filtered_words = [word for word in all_words if word not in stopwords.words('english')]

word_features = list(filtered_words)[:3000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        # features['contains({})'.format(word)] = (word in document_words)
        features[word] = (word in document_words)
    return features

def word_feats(document):
    document_words = set(document)
    return dict([(word, True) for word in document_words])

featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[:1600], featuresets[1600:]
review_train, sentiment_train = zip(*train_set)
review_test, sentiment_test = zip(*test_set)

# dataset_train = Dataset(review_train, sentiment_train[:100])
dataset_train = Dataset(review_train, np.concatenate(
        [sentiment_train[:10], [None] * (len(sentiment_train) - 10)]))
dataset_test = Dataset(review_test, sentiment_test)
fully_labelled = Dataset(review_train, sentiment_train)
lbr = IdealLabeler(fully_labelled)

def get_dist(labeled_featuresets, estimator=nltk.ELEProbDist):
    """
    :param labeled_featuresets: A list of classified featuresets,
        i.e., a list of tuples ``(featureset, label)``.
    """
    label_freqdist = nltk.FreqDist()
    feature_freqdist = nltk.defaultdict(nltk.FreqDist)
    feature_values = nltk.defaultdict(set)
    fnames = set()

    # Count up how many times each feature value occurred, given
    # the label and featurename.
    for featureset, label in labeled_featuresets:
        label_freqdist[label] += 1
        for fname, fval in featureset.items():
            # Increment freq(fval|label, fname)
            feature_freqdist[label, fname][fval] += 1
            # Record that fname can take the value fval.
            feature_values[fname].add(fval)
            # Keep a list of all feature names.
            fnames.add(fname)

    # If a feature didn't have a value given for an instance, then
    # we assume that it gets the implicit value 'None.'  This loop
    # counts up the number of 'missing' feature values for each
    # (label,fname) pair, and increments the count of the fval
    # 'None' by that amount.
    for label in label_freqdist:
        num_samples = label_freqdist[label]
        for fname in fnames:
            count = feature_freqdist[label, fname].N()
            # Only add a None key when necessary, i.e. if there are
            # any samples with feature 'fname' missing.
            if num_samples - count > 0:
                feature_freqdist[label, fname][None] += num_samples - count
                feature_values[fname].add(None)

    # Create the P(label) distribution
    label_probdist = estimator(label_freqdist)

    # Create the P(fval|label, fname) distribution
    feature_probdist = {}
    for ((label, fname), freqdist) in feature_freqdist.items():
        probdist = estimator(freqdist, bins=len(feature_values[fname]))
        feature_probdist[label, fname] = probdist

    return label_probdist, feature_probdist


label_probdist, feature_probdist = get_dist(train_set)
model = NLTKNaiveBayes(label_probdist, feature_probdist)
qs = UncertaintySampling(dataset_train, method='sm', model= model)
quota = len(sentiment_train) - 10

def run(trn_ds, tst_ds, lbr, model, qs, quota):
    error_trained_data, error_test_data = [], []

    for _ in range(quota):
        # Standard usage of libact objects
        print('Quota:', _)
        ask_id = qs.make_query()
        print('askid:', ask_id)
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)

        trained_model=model.train(trn_ds.data)
        error_trained_data = np.append(error_trained_data, model.score(trained_model, trn_ds.data))
        error_test_data = np.append(error_test_data, model.score(trained_model, tst_ds.data))

    return error_trained_data, error_test_data

error_uncertainty_train, error_uncertainty_test = run(dataset_train, dataset_test, lbr, model, qs, quota)

query_num = np.arange(1, quota + 1)

plt.plot(query_num, error_uncertainty_train, 'b', label='Uncertainty Training')
plt.plot(query_num, error_uncertainty_test, 'g', label='Uncertainty Test')
plt.xlabel('Number of Queries')
plt.ylabel('Accuracy')
plt.title('Experiment Result')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
plt.show()









