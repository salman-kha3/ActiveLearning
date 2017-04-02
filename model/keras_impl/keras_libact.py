from __future__ import print_function

import sys
import copy

import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

np.random.seed(1337)  # for reproducibility

from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from model.keras_impl.keras_classifier import KerasClassifier
from libact.labelers import IdealLabeler
from libact.query_strategies import RandomSampling
from libact.query_strategies import UncertaintySampling
from libact.base.dataset import Dataset
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

max_words = 1000
batch_size = 32
nb_epoch = 5

print('Loading data...')
sys.argv[0]= '/home/salman/thesis/corpuses/movie_reviews'

movie_reviews_data_folder = sys.argv[0]
movie_reviews_dataset = load_files(movie_reviews_data_folder, shuffle=False)

reviews, sentiments= Dataset(movie_reviews_dataset.data, movie_reviews_dataset.target).format_sklearn()

X_train, X_test, y_train, y_test = \
    train_test_split(reviews, sentiments, test_size=0.25)

# (X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

nb_classes = np.max(y_train) + 1
print(nb_classes, 'classes')

# print('Vectorizing sequence data...')
# tokenizer = Tokenizer(nb_words=max_words)
# X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
# X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)
#
# print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
# print('Y_train shape:', Y_train.shape)
# print('Y_test shape:', Y_test.shape)

count_vectorizer = CountVectorizer()
reviews_train_counts = count_vectorizer.fit_transform(X_train)
print(reviews_train_counts.shape)

tfidf_transformer = TfidfTransformer()
reviews_train_tfidf = tfidf_transformer.fit_transform(reviews_train_counts)
print(reviews_train_tfidf.shape)

train_ds = Dataset(reviews_train_tfidf.toarray(), np.concatenate(
        [y_train[:10], [None] * (len(y_train) - 10)]))

# train_ds = Dataset(X_train, Y_train[:10])

train_ds_copy = copy.deepcopy(train_ds)
test_ds = Dataset(X_test, y_test)
fully_labeled_trn_ds = Dataset(X_train, y_train)
lbr = IdealLabeler(fully_labeled_trn_ds)
quota = len(y_train) - 10


print('Building model...')
# model = Sequential()
model = KerasClassifier()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
qs = UncertaintySampling(train_ds, method='sm', model= model)

error_uncertainty_train, error_uncertainty_test = run(train_ds, test_ds, lbr, model, qs, quota)

query_num = np.arange(1, quota + 1)
plt.plot(query_num, error_uncertainty_train, 'b', label='Uncertainty Training')
# plt.plot(query_num, error_sample_train, 'r', label='Random Train')
plt.plot(query_num, error_uncertainty_test, 'g', label='Uncertainty Test')
# plt.plot(query_num, error_sample_test, 'k', label='Random Test')
plt.xlabel('Number of Queries')
plt.ylabel('Error')
plt.title('Experiment Result')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
plt.show()


