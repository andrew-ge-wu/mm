import numpy as np


class TrainTestIterator:
    """
    A decorator of an iterator that allows to split a dataset in training and test set without the need to
    store all results and support for batching.

    :param
        :param iterator_factory: a function that returns an iterator
        :param training_condition: a function of two positional arguments (value, index, random_state) that returns true or false to
        use the element on the training set or not (default to 0.8 test set).
        :param batch_size: the size of the batch. Batching is done on iterator. Last batch is not guaranteed to have the requested size.
        :param seed: the seed for reproducibility. Same seed will always return the same state

    Relevant methods:
    * `__next__` to iterate by one batch.
    * `reset` to reset to the original state
    * `test_samples` list of test samples (not-batched). This list is populated on during iterator. E.g. if you iterate over 100 samples so far
    with a test ratio of 0.2, it will contain on average 20 samples (and always the same).
    """
    def __init__(self, iterator_factory, training_condition=lambda value, index, state: state.rand() < 0.8, batch_size=1, seed=None):
        self._iterator_factory = iterator_factory
        self.seed = seed
        self._batch_size = batch_size
        self._training_condition = training_condition
        self.reset()

    def reset(self):
        self._iterator = self._iterator_factory()
        self._state = np.random.RandomState(self.seed)  # so others do not interfere with this state
        self._samples_so_far = 0
        self._test_indexes = set()

    def _next_single(self):
        result = self._iterator.__next__()  # advances the base iterator
        if self._training_condition(result, self._samples_so_far, self._state):  # advances the random state
            # is a training sample
            self._samples_so_far += 1
            yield result
        else:
            self._samples_so_far += 1
            # it is a test sample
            self._test_indexes.add(self._samples_so_far)

    def __iter__(self):
        return self

    def __next__(self):
        _current_batch = []
        while len(_current_batch) < self._batch_size:
            _current_batch.append(self._next_single().__next__())
        return _current_batch

    def test_samples(self):
        return [sample for i, sample in enumerate(self._iterator_factory()) if i in self._test_indexes]
