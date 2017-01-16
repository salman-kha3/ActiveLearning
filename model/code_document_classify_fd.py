import nltk
from nltk.corpus import movie_reviews
from libact.base.dataset import Dataset

import random

from model.naive_bayes import NaiveBayes
from query_strategies.QueryStrategyImpl import QueryStrategyImpl

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[1000:], featuresets[:100]
# classifier = nltk.NaiveBayesClassifier.train(train_set)

test_train= Dataset(train_set)
print(test_set)

qs = QueryStrategyImpl(test_train, method='lc', model= NaiveBayes())

# dataset = Dataset(train_set, ['pos'])
# qs = UncertaintySampling(dataset, model=LogisticRegression())
# # query_strategy = QueryStrategy(dataset) # declare a QueryStrategy instance
# labeler = Labeler() # declare Labeler instance
# model = Model()
#
# query_id = qs.make_query() # let the specified QueryStrategy suggest a data to query
# lbl = labeler.label(dataset.data[1][0]) # query the label of the example at query_id
# dataset.update(1, lbl) # update the dataset with newly-labeled example
# model.train(dataset)