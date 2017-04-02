import random

import nltk
from nltk.corpus import movie_reviews, stopwords

class Classification:

    def __init__(self):
        all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
        filtered_words = [word for word in all_words if word not in stopwords.words('english')]
        self.word_list = list(filtered_words)[:3000]

    def get_docs(self):
        documents = [(list(movie_reviews.words(fileid)), category)
                     for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]
        random.shuffle(documents)
        return documents

    def word_feats(document):
        document_words = set(document)
        return dict([(word, True) for word in document_words])

    def document_features(self, document, word_list):
        document_words = set(document)
        features = {}
        for word in word_list:
            # features['contains({})'.format(word)] = (word in document_words)
            features[word] = (word in document_words)
        return features

    def get_train_test(self):
        # all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
        # filtered_words = [word for word in all_words if word not in stopwords.words('english')]
        # word_list = list(filtered_words)[:2000]
        featuresets = [(self.document_features(d, self.word_list), c) for (d, c) in self.get_docs()]
        train_set, test_set = featuresets[:1500], featuresets[1500:]

        return train_set, test_set


    def get_dist(self, labeled_featuresets, estimator=nltk.ELEProbDist):
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

    def get_words(self, sent):
        words_list = nltk.word_tokenize(sent)
        # return dict([(word, True) for word in words_list])
        return words_list

    def update(self, model, dataset_train, review, label):
        fet_words = {}
        fet_words = self.get_words(review)
        fet = (self.document_features(review, self.word_list), label)
        dataset_train.data.append(fet)
        updated_classifier = model.train(dataset_train.data)
        return updated_classifier













