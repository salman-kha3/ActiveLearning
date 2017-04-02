import sys
import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from model.scikit_naive_bayes_impl.scikit_naive_bayes import ScikitNaiveBayes

sys.argv[0]= '/home/salman/thesis/corpuses/movie_reviews'

movie_reviews_data_folder = sys.argv[0]
movie_reviews_dataset = load_files(movie_reviews_data_folder, shuffle=False)

reviews_train, reviews_test, sentiments_train, sentiments_test = \
    train_test_split(movie_reviews_dataset.data, movie_reviews_dataset.target, test_size=0.40)

model = ScikitNaiveBayes()

reviews_train
text_clf = model.pipeline()
text_clf = text_clf.fit(reviews_train, sentiments_train)

test= ['good movie']

predicted = text_clf.predict(reviews_test)
# print(np.mean(predicted == sentiments_test))






# count_vectorizer = CountVectorizer()
# reviews_train_counts = count_vectorizer.fit_transform(reviews_train)
#
# tfidf_transformer = TfidfTransformer()
# reviews_train_tfidf = tfidf_transformer.fit_transform(reviews_train_counts)
#
# reviews_test_counts= count_vectorizer.transform(reviews_test)
# reviews_test_tfidf= tfidf_transformer.transform(reviews_test_counts)
#
# model = ScikitNaiveBayes()
# classifier = model.train(reviews_train_tfidf, sentiments_train)
# pred= model.predict(reviews_test_tfidf)
# print(model.score(reviews_test_tfidf, sentiments_test))
#
# docs_new = ['good movie']
# X_new_counts = count_vectorizer.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#
# predicted = model.predict(X_new_tfidf)
#
for doc, category in zip(reviews_test, predicted):
    print('%r => %s' % (doc, movie_reviews_dataset.target_names[category]))
    break

def update(clf, review, label):
    clf.fit(review, [1])
    return clf


print('Active learning on...')
clf = update(text_clf, test, 'pos')
predictActive= clf.predict(test)
print('after active learning:',predictActive)

for doc, category in zip(test, predictActive):
    print('%r => %s' % (doc, movie_reviews_dataset.target_names[category]))

print('checking modelled classifier..')

predictAgain = text_clf.predict(reviews_test)
for doc, category in zip(reviews_test, predictAgain):
    print('%r => %s' % (doc, movie_reviews_dataset.target_names[category]))
    break

self.data[entry_id] = (self.data[entry_id][0], new_label)











