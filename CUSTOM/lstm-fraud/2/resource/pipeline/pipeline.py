from .pipe import Pipe


class Pipeline(Pipe):
    """
    A sequence of pipes that are trained in sequence.
    The last pipe is called "classifier", the remaining "transformers".
    """
    def __init__(self, pipes):
        assert isinstance(pipes, list)
        for pipe in pipes:
            assert isinstance(pipe, Pipe)
        self.pipes = pipes
        super().__init__({})  # stateless

    def transform(self, dataset, supervised=True):
        for pipe in self.pipes:
            dataset = pipe.transform(dataset, supervised)
        return dataset

    def transform_transformers(self, dataset, supervised=True):
        for pipe in self.pipes[:-1]:
            dataset = pipe.transform(dataset, supervised=supervised)
        return dataset

    def fit(self, dataset):
        """
        Fits all pipes in sequence.
        """
        for pipe in self.pipes[:-1]:
            pipe.fit(dataset)
            dataset = pipe.transform(dataset, supervised=True)
        self.pipes[-1].fit(dataset)

    def fit_transformers(self, dataset):
        """
        Fits all but the last pipe in sequence.
        """
        for pipe in self.pipes[:-1]:
            pipe.fit(dataset)
            dataset = pipe.transform(dataset, supervised=True)

    def fit_classifier(self, dataset):
        """
        Transforms the data with the transformers and fit the classifier
        """
        dataset = self.transform_transformers(dataset, supervised=True)
        self.pipes[-1].fit(dataset)

    def fit_partial(self, dataset):
        for pipe in self.pipes:
            pipe.fit_partial(dataset)
            dataset = pipe.transform(dataset, supervised=True)

    def fit_partial_transformers(self, dataset):
        for pipe in self.pipes[:-1]:
            pipe.fit_partial(dataset)
            dataset = pipe.transform(dataset, supervised=True)

    def fit_partial_classifier(self, dataset):
        dataset = self.transform_transformers(dataset, supervised=True)
        self.pipes[-1].fit_partial(dataset)

    def predict(self, dataset):
        assert hasattr(self.pipes[-1], 'predict')
        for pipe in self.pipes[:-1]:
            dataset = pipe.transform(dataset, supervised=False)
        return self.pipes[-1].predict(dataset)

    def predict_proba(self, dataset):
        assert hasattr(self.pipes[-1], 'predict_proba')
        dataset = self.transform_transformers(dataset, supervised=False)
        return self.pipes[-1].predict_proba(dataset)

    def transformers_iterator(self, supervised=True):
        """
        Returns a "decorator" of an iterator. Pass an iterator to the result of this function and
        use it as a generator where each item of the generator passes through the transformers.
        """
        def transformed_iterator(iterator):
            for dataset in iterator:
                for pipe in self.pipes[:-1]:
                    dataset = pipe.transform(dataset, supervised=supervised)
                yield dataset
        return transformed_iterator

    def iterator(self, supervised=True):
        """
        Returns a "decorator" of an iterator. Pass an iterator to the result of this function and
        use it as a generator where each item of the generator passes through the pipeline.
        """
        def transformed_iterator(iterator):
            for dataset in iterator:
                for pipe in self.pipes:
                    dataset = pipe.transform(dataset, supervised=supervised)
                yield dataset
        return transformed_iterator
