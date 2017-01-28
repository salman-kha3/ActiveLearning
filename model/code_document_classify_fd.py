import nltk
import sys
from nltk.corpus import movie_reviews, stopwords
from sklearn.model_selection import train_test_split
from libact.base.dataset import Dataset
from sklearn.feature_extraction.text import CountVectorizer

import random

from sklearn.datasets import load_files

from model.naive_bayes import NaiveBayes
from query_strategies.QueryStrategyImpl import QueryStrategyImpl

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
# random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
#
# filtered_words = [word for word in all_words if word not in stopwords.words('english')]

# filtered_word_list = all_words[:] #make a copy of the word_list
# for word in all_words: # iterate over word_list
#   if word in stopwords.words('english'):
#     filtered_word_list.remove(word)


word_features = list(all_words)[:2000]
#
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        # features['contains({})'.format(word)] = (word in document_words)
        features[word] = (word in document_words)
    return features
#
featuresets = [(document_features(d), c) for (d, c) in documents]

print(featuresets[102])
train_set, test_set = featuresets[100:], featuresets[:100]
#
# print(train_set)
# print(test_set)
#
classifier = nltk.NaiveBayesClassifier.train(train_set)

# for test_words in test_set:
#     print(test_words, "\n")
# sys.argv[0]= '/home/salman/thesis/corpuses/movie_reviews'
# movie_reviews_data_folder = sys.argv[0]
# dataset = load_files(movie_reviews_data_folder, shuffle=False)
# print("n_samples: %d" % len(dataset.data))
#
# reviews, sentiments= Dataset(dataset.data, dataset.target).format_sklearn()

# print(reviews[2], sentiments[2])
# reviews_train, reviews_test, sentiments_train, sentiments_test = train_test_split(
#     reviews, sentiments, test_size=0.25)
#
# count_vect = CountVectorizer()
# reviews_train_counts = count_vect.fit_transform(reviews_train)
# print(reviews_train_counts.shape)




# trn_ds = Dataset(train_set, np.concatenate(
#         [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
#
# classifier = nltk.NaiveBayesClassifier.train(train_set)
# print(nltk.classify.accuracy(classifier, test_set))
# classifier.show_most_informative_features(5)

# print(featuresets.shape)

# test_train= Dataset(train_set)
# print(test_set)
#
# qs = QueryStrategyImpl(test_train, method='lc', model= NaiveBayes())

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