from six import with_metaclass

from abc import ABCMeta, abstractmethod

class Classifier(with_metaclass(ABCMeta, object)):

    """Classification Model

    A Model returns a class-predicting function for future samples after
    trained on a training dataset.
    """
    @abstractmethod
    def train(self, *args, **kwargs):
        """Train a model according to the given training dataset.

        Parameters
        ----------
        train_set :  object
             The training dataset the model is to be trained on.

        Returns
        -------
        self : object
            Returns self.
        """
        pass

    @abstractmethod
    def predict(self, feature, *args, **kwargs):
        """Predict the class labels for the input samples

        Parameters
        ----------
        feature : array-like, shape (n_samples, n_features)
            The unlabeled samples whose labels are to be predicted.

        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            The class labels for samples in the feature array.
        """
        pass

    @abstractmethod
    def score(self, *args, **kwargs):
        """Return the mean accuracy on the test dataset

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        pass

    @abstractmethod
    def predict_real(self, feature, *args, **kwargs):
        """Predict confidence scores for samples.

        Parameters
        ----------
        feature : array-like, shape (n_samples, n_features)
            The samples whose confidence scores are to be predicted.

        Returns
        -------
        X : array-like, shape (n_samples, n_classes)
            Each entry is the confidence scores per (sample, class)
            combination.
        """
        pass
