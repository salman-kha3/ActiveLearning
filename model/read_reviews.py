import sys

import copy
import numpy as np
from libact.labelers import IdealLabeler
from libact.query_strategies import RandomSampling
from libact.query_strategies import UncertaintySampling
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from libact.base.dataset import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from model.naive_bayes import NaiveBayes
import matplotlib.pyplot as plt


def run(trn_ds, tst_ds, lbr, model, qs, quota):
    error_trained_data, error_test_data = [], []

    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        error_trained_data = np.append(error_trained_data, 1 - model.score(trn_ds))
        error_test_data = np.append(error_test_data, 1 - model.score(tst_ds))

    return error_trained_data, error_test_data

sys.argv[0]= '/home/salman/thesis/corpuses/movie_reviews'

movie_reviews_data_folder = sys.argv[0]
movie_reviews_dataset = load_files(movie_reviews_data_folder, shuffle=False)
print("n_samples: %d" % len(movie_reviews_dataset.data))

reviews, sentiments= Dataset(movie_reviews_dataset.data, movie_reviews_dataset.target).format_sklearn()
reviews_train, reviews_test, sentiments_train, sentiments_test = train_test_split(
    reviews, sentiments, test_size=0.25)

# S = set(X) # collect unique label names
# D = dict( zip(S, range(len(S))) ) # assign each string an integer, and put it in a dict
# Y = [D[y2_] for y2_ in X]

count_vectorizer = CountVectorizer()
reviews_train_counts = count_vectorizer.fit_transform(reviews_train)
print(reviews_train_counts.shape)

tfidf_transformer = TfidfTransformer()
reviews_train_tfidf = tfidf_transformer.fit_transform(reviews_train_counts)
print(reviews_train_tfidf.shape)

train_ds = Dataset(reviews_train_tfidf.toarray(), np.concatenate(
        [sentiments_train[:10], [None] * (len(sentiments_train) - 10)]))
classifier = MultinomialNB().fit(reviews_train_tfidf, sentiments_train)

reviews_test_counts= count_vectorizer.transform(reviews_test)
reviews_test_tfidf= tfidf_transformer.transform(reviews_test_counts)
print(reviews_test_tfidf.shape)
test_ds = Dataset(reviews_test_tfidf.toarray(), sentiments_test)

train_ds_copy = copy.deepcopy(train_ds)
fully_labeled_trn_ds = Dataset(reviews_train_tfidf.toarray(), sentiments_train)
lbr = IdealLabeler(fully_labeled_trn_ds)
quota = len(sentiments_train) - 10
qs = UncertaintySampling(train_ds, method='sm', model= NaiveBayes())
model = NaiveBayes()
error_uncertainty_train, error_uncertainty_test = run(train_ds, test_ds, lbr, model, qs, quota)

qs2 = RandomSampling(train_ds_copy)
model = NaiveBayes()

error_sample_train, error_sample_test = run(train_ds_copy, test_ds, lbr, model, qs2, quota)

#     # Plot the learning curve of UncertaintySampling to RandomSampling
#     # The x-axis is the number of queries, and the y-axis is the corresponding
#     # error rate.
query_num = np.arange(1, quota + 1)
plt.plot(query_num, error_uncertainty_train, 'b', label='Uncertainty Training')
plt.plot(query_num, error_sample_train, 'r', label='Random Train')
plt.plot(query_num, error_uncertainty_test, 'g', label='Uncertainty Test')
plt.plot(query_num, error_sample_test, 'k', label='Random Test')
plt.xlabel('Number of Queries')
plt.ylabel('Error')
plt.title('Experiment Result')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
plt.show()
