from flask import Flask,render_template
from flask import json
from flask import redirect
from flask import request
from libact.base.dataset import Dataset

from model.nltk_naive_bayes_impl.code_document_classify_fd import Classification
from model.nltk_naive_bayes_impl.nltk_naive_bayes import Bayes

application= Flask(__name__)

@application.route("/sendQuery", methods=['GET', 'POST'])
def sendQuery():
    sentiment = request.data
    striin = sentiment.decode("utf-8")
    re = cls.get_words(striin)
    predict = model.predict(cls.document_features(re, cls.word_list))
    accuracy = model.score(classifier, test_set)
    return predict

@application.route("/update", methods=['GET', 'POST'])
def updateClassfier():

    # data = request.data
    json_data = request.json['review']
    review = json_data['rev']
    sent = json_data['sent']
    classifier = cls.update(model, dataset_train, review, sent)
    return "ok"



@application.after_request
def apply_caching(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept';
    return response

@application.route('/')
def showMainView():
    return redirect('http://localhost:63342/ActiveLearningMovieReviews/View/MainView.html')

if __name__ == "__main__":
    cls = Classification()
    train_set, test_set = cls.get_train_test()
    review_train, sentiment_train = zip(*train_set)
    label_probdist, feature_probdist = cls.get_dist(train_set)
    dataset_train = Dataset(review_train, sentiment_train)
    model = Bayes(label_probdist, feature_probdist)
    classifier = model.train(dataset_train.data)
    application.run(host= '127.0.0.1')
