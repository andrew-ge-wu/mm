import numpy as np


class DummyIterator:
    def __init__(self, sample_size=1000, batch_size=256):
        self.remaining_sample_size = sample_size
        self.batch_size = batch_size

        # Set the random seed for reproducibility.
        np.random.seed(42)

    def __iter__(self):
        return self

    def __next__(self):

        # Calculate the remaining sample size.
        self.remaining_sample_size = self.remaining_sample_size - self.batch_size

        # If there is no more data, so stop the iterator.
        chunk_size = min(self.batch_size, self.remaining_sample_size)
        if chunk_size <= 0:
            raise StopIteration

        # Return a pandas dataframe with random entries.
        x = np.random.random_sample(size=(chunk_size, 2))
        y = np.random.randint(2, size=(chunk_size, 1))

        return np.concatenate((x, y), axis=1)
