class Pipe:

    def __init__(self, states=None):
        if states is None:
            states = {}
        assert isinstance(states, dict)
        self.states = states

    def transform(self, dataset, supervised=False):
        # cannot modify states
        # can modify dataset (ideally inplace)
        return dataset

    def fit(self, dataset):
        # can modify states
        # cannot modify dataset
        pass

    def fit_partial(self, dataset):
        pass

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)


class F(Pipe):
    """
    A function. Use the optional argument for applying a different function when in supervised mode.
    """
    def __init__(self, func, func_supervised=None):
        self.func = func
        if func_supervised is None:
            func_supervised = func
        self.func_supervised = func_supervised
        super().__init__({})

    def transform(self, dataset, supervised=False):
        if supervised:
            return self.func_supervised(dataset)
        else:
            return self.func(dataset)


class Copy(Pipe):
    """
    Copies the dataset so that every subsequent operation does not affect original data
    """
    def transform(self, dataset, supervised=False):
        return dataset.copy()


class Callback(Pipe):
    """
    Executes a function whenever it fits or transforms. Used to inspect the data along the pipeline.
    """
    def __init__(self, fit=lambda dataset: None, transform=lambda dataset: None):
        self._fit = fit
        self._transform = transform
        super().__init__({})

    def fit(self, dataset):
        self._fit(dataset)
        return super().fit(dataset)

    def transform(self, X, supervised=False):
        self._transform(X)
        return super().transform(X)


class SK(Pipe):

    def __init__(self, sk_element):
        super().__init__({'element': sk_element})

    def fit(self, dataset):
        self.states['element'].fit(dataset[0], dataset[1])

    def fit_partial(self, dataset):
        self.states['element'].partial_fit(dataset[0], dataset[1])

    def predict(self, x):
        return self.states['element'].predict(x)

    def predict_proba(self, x):
        return self.states['element'].predict_proba(x)
