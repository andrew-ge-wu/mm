import pickle


class PickleIterator:
    """
    An iterator over pickle instances on a file.
    """
    def __init__(self, file_path, batch_size, is_infinite=False):
        self._file_path = file_path
        self._batch_size = batch_size
        self._is_infinite = is_infinite
        self._descriptor = None
        self.reset()

    def reset(self):
        if self._descriptor is not None:
            self._descriptor.close()
        self._descriptor = open(self._file_path, 'rb')
        self._unpickler = pickle.Unpickler(self._descriptor)

    def __iter__(self):
        return self

    def get_single(self):
        if self._is_infinite:
            try:
                return self._unpickler.load()
            except EOFError:
                self.reset()
                return self._unpickler.load()
        else:
            try:
                return self._unpickler.load()
            except EOFError as e:
                self._descriptor.close()
                raise e

    def __next__(self):
        values = []
        for i in range(self._batch_size):
            try:
                value = self.get_single()
            except EOFError:
                raise StopIteration
            values.append(value)
        if self._batch_size == 1:
            return values[0]
        return values

    def __del__(self):
        if not self._descriptor.closed:
            self._descriptor.close()
