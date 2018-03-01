import numpy as np
from .pipe import Pipe
from .pipeline import Pipeline

# required for the Local Explainer
try:
    import numdifftools as nd
except ImportError:
    nd = None

# required for the Lime Explainer
try:
    import lime.lime_tabular
except ImportError:
    lime = None


class ExplanatoryPipeline(Pipeline):
    def __init__(self, pipes):
        super().__init__(pipes)

        self._explainer = None
        for pipe in pipes:
            if hasattr(pipe, 'explain'):
                if self._explainer is not None:
                    raise AttributeError('There can only exist one explainer in the pipeline.')
                self._explainer = pipe
                break
        if self._explainer is None:
            raise AttributeError('There must exist one explainer in the pipeline.')

    def explain(self, x_i, **kwargs):

        # transform the element using the transformers up to the explainer
        remaining_pipes = []
        for i, pipe in enumerate(self.pipes):
            if pipe == self._explainer:
                remaining_pipes = self.pipes[i+1:]
                break
            x_i = pipe.transform(x_i, supervised=False)

        # build a predict_prob function as the application of the remaining transformers + the classifier
        def predict_proba(x_i):
            for pipe in remaining_pipes[:-1]:
                x_i = pipe.transform(x_i, supervised=False)
            return remaining_pipes[-1].predict_proba(x_i)

        return self._explainer.explain(x_i[0], predict_proba, **kwargs)


class LimeExplainer(Pipe):
    def __init__(self, to_x_y, feature_names, mode='classification', num_features=10):
        self._mode = mode
        self._num_features = num_features
        self._to_x_y = to_x_y
        self._feature_names = feature_names
        super().__init__({'explainer': None})

    def fit(self, dataset):
        if lime is None:
            raise ImportError('lime required (`pip install lime`) to use the LimeExplainer')

        x, y = self._to_x_y(dataset)
        self.states['explainer'] = lime.lime_tabular.LimeTabularExplainer(
            x, mode=self._mode,
            feature_names=self._feature_names,
            class_names=[str(y) for y in sorted(np.unique(y))],
            discretize_continuous=True)

    def explain(self, x_i, predict_proba, **kwargs):
        return self.states['explainer'].explain_instance(x_i, predict_proba, **kwargs)


class LocalExplainer(Pipe):
    def __init__(self, to_x_y, columns, names):
        if nd is None:
            raise ImportError('lime required (`pip install numdifftools`) to use the LocalExplainer')
        self._to_x_y = to_x_y
        assert len(names) == len(columns)
        self._names = names
        self._columns = columns
        super().__init__({})

    def explain(self, x_i, predict_proba):
        assert (x_i.shape[0] >= len(self._columns))

        f = lambda x: predict_proba(x)[0][1]

        jacobian = nd.Jacobian(f, method='central', order=4)
        jacobian = jacobian(x_i)[0]

        result = {}
        for i, name in enumerate(self._names):
            result[name] = jacobian[i]
        return result
