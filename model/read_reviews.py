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
    E_in, E_out = [], []

    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out

sys.argv[0]= '/home/salman/thesis/corpuses/movie_reviews'

movie_reviews_data_folder = sys.argv[0]
dataset = load_files(movie_reviews_data_folder, shuffle=False)
print("n_samples: %d" % len(dataset.data))

X,y= Dataset(dataset.data, dataset.target).format_sklearn()
docs_train, docs_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)

# S = set(X) # collect unique label names
# D = dict( zip(S, range(len(S))) ) # assign each string an integer, and put it in a dict
# Y = [D[y2_] for y2_ in X]

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(docs_train)
print(X_train_counts.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

category=['pos', 'neg']

trn_ds = Dataset(X_train_tfidf.toarray(), np.concatenate(
        [y_train[:10], [None] * (len(y_train) - 10)]))
clf = MultinomialNB().fit(X_train_tfidf, y_train)

docs_new = ['it is ok movie', 'bad dialogues']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

X_test_counts= count_vect.transform(docs_test)
X_test_tfidf= tfidf_transformer.transform(X_test_counts)
print(X_test_tfidf.shape)
tst_ds = Dataset(X_test_tfidf.toarray(), y_test)

for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, dataset.target_names[category]))

trn_ds2 = copy.deepcopy(trn_ds)
fully_labeled_trn_ds = Dataset(X_train_tfidf.toarray(), y_train)
lbr = IdealLabeler(fully_labeled_trn_ds)
quota = len(y_train) - 10
qs = UncertaintySampling(trn_ds, method='lc', model= NaiveBayes())
model = NaiveBayes()
E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota)

qs2 = RandomSampling(trn_ds2)
model = NaiveBayes()

E_in_2, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota)

#     # Plot the learning curve of UncertaintySampling to RandomSampling
#     # The x-axis is the number of queries, and the y-axis is the corresponding
#     # error rate.
query_num = np.arange(1, quota + 1)
plt.plot(query_num, E_in_1, 'b', label='qs Ein')
plt.plot(query_num, E_in_2, 'r', label='random Ein')
plt.plot(query_num, E_out_1, 'g', label='qs Eout')
plt.plot(query_num, E_out_2, 'k', label='random Eout')
plt.xlabel('Number of Queries')
plt.ylabel('Error')
plt.title('Experiment Result')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
plt.show()
