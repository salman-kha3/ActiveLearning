from data_set.dataset import Dataset
from model.nltk_naive_bayes_impl.code_document_classify_fd import Classification
from model.nltk_naive_bayes_impl.nltk_naive_bayes import Bayes


def main():
    cls = Classification()
    train_set, test_set = cls.get_train_test()
    review_train, sentiment_train = zip(*train_set)
    dataset_train = Dataset(review_train, sentiment_train)
    label_probdist, feature_probdist = cls.get_dist(train_set)
    model = Bayes(label_probdist, feature_probdist)
    classifier= model.train(dataset_train.data)
    re = 'awesome'
    predict = model.predict(cls.get_words(re))
    print(predict)
    print('accuracy before Active learning', model.score(classifier, test_set))





# dataset_train = Dataset(list1, list2)
#
#
# classifier = model.train(dataset_train.data)
#     # reviews ='', 'is a bad one']
#
#
#
# print('accuracy before Active learning', model.score(classifier, test_set))
# print('Active learning on...')
# model, updated_classifier = update(model, re, 'pos')
# predictActive = model.predict(document_features(re))
# print('after active learning:', predictActive)
# print('accuracy after active learning', model.score(updated_classifier, test_set))
#
# predictActive1 = model.predict(document_features('worst movie ever'))
# print('checking again', predictActive1)

if __name__ == "__main__":
    main()