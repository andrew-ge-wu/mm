import pandas as pd


class CSVIterator:
    def __init__(self, file_path, batch_size, is_infinite=False):
        self._file_path = file_path
        self._batch_size = batch_size
        self._is_infinite = is_infinite
        self.reset()

    def reset(self):
        self._df = pd.read_csv(self._file_path, chunksize=self._batch_size)

    def get_single(self):
        return pd.read_csv(self._file_path)

    def __iter__(self):
        return self

    def __next__(self):
        if self._is_infinite:
            try:
                return self._df.__next__()
            except StopIteration:
                self.reset()
        return self._df.__next__()
