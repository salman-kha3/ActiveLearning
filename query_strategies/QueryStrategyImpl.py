""" Uncertainty Sampling

This module contains a class that implements two of the most well-known
uncertainty sampling query strategies: the least confidence method and the
smallest margin method (margin sampling).

"""
import numpy as np

from libact.base.interfaces import QueryStrategy, ContinuousModel
from libact.base.dataset import Dataset
from libact.utils import inherit_docstring_from, zip


class UncertaintySampling(QueryStrategy):

    """Uncertainty Sampling

    This class implements Uncertainty Sampling active learning algorithm [1]_.

    Parameters
    ----------
    model: :py:class:`libact.base.interfaces.ContinuousModel` object instance
        The base model used for training.

    method: {'lc', 'sm'}, optional (default='lc')
        least confidence (lc), it queries the instance whose posterior
        probability of being positive is nearest 0.5 (for binary
        classification);
        smallest margin (sm), it queries the instance whose posterior
        probability gap between the most and the second probable labels is
        minimal;


    Attributes
    ----------
    model: :py:class:`libact.base.interfaces.ContinuousModel` object instance
        The model trained in last query.

    References
    ----------

    .. [1] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11.
    """

    def __init__(self, *args, **kwargs):
        super(UncertaintySampling, self).__init__(*args, **kwargs)

        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        if not isinstance(self.model, ContinuousModel):
            raise TypeError(
                "model has to be a ContinuousModel"
            )
        if isinstance(self.dataset, Dataset):
            self.model.train(self.dataset.data)
        else:
            self.model.train(self.dataset)

        self.method = kwargs.pop('method', 'lc')
        if self.method not in ['lc', 'sm']:
            raise TypeError(
                "supported methods are ['lc', 'sm'], the given one is: " +
                self.method
            )

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        self.model.train(dataset.data)

        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        X_pool_list = list(X_pool)
        if self.method == 'lc':  # least confident
            ask_id = np.argmin(
                np.max(self.model.predict_real(X_pool_list), axis=1)
            )

        elif self.method == 'sm':  # smallest margin
            dvalue = self.model.predict_real(X_pool_list)

            if np.shape(dvalue)[1] > 2:
                # Find 2 largest decision values
                dvalue = -(np.partition(-dvalue, 2, axis=1)[:, :2])

            margin = np.abs(dvalue[:, 0] - dvalue[:, 1])
            ask_id = np.argmin(margin)

        return unlabeled_entry_ids[ask_id]
