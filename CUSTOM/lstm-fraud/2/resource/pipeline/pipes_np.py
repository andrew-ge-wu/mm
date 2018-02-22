import numpy as np

from .pipe import Pipe, F


class Drop(Pipe):
    def __init__(self, columns):
        self._columns = columns
        super().__init__({})

    def transform(self, dataset, supervised=False):
        return np.delete(dataset, self._columns, axis=1)


class Zscore(Pipe):
    """
    Scales columns to `z = (x - mu)/std` with support to `fit_partial`.
    """
    def __init__(self, columns):
        self._columns = columns
        super().__init__({'count': None, 'sum': None, 'sum2': None})

    def _init(self):
        self.states['count'] = 0.0
        self.states['sum'] = 0.0
        self.states['sum2'] = 0.0

    def fit(self, dataset):
        self._init()
        self.fit_partial(dataset)

    def fit_partial(self, dataset):
        if self.states['count'] is None:
            self._init()
        count = len(dataset)
        self.states['count'] += count
        self.states['sum'] += np.sum(dataset[:, self._columns], axis=0)
        self.states['sum2'] += np.sum(np.square(dataset[:, self._columns]), axis=0)

    @property
    def mean(self):
        return self.states['sum']/self.states['count']

    @property
    def variance(self):
        return self.states['sum2'] / self.states['count'] - np.square(self.mean)

    @property
    def std(self):
        return np.sqrt(self.variance)

    def transform(self, dataset, supervised=False):
        dataset[:, self._columns] -= self.mean
        dataset[:, self._columns] /= self.std
        return dataset


class SurroundWithCorrelatedFeatures(Pipe):
    """
     Constructs matrix around each element. The matrix contains the element at the center, and the values of the
     variables most correlated with the element around it.
    """

    def __init__(self, augmented_matrix_dim=3, n_timesteps=9):
        if (augmented_matrix_dim % 2 == 0) or (augmented_matrix_dim < 3):
            raise ValueError("The value for augmented_matrix_dim must be an odd number that is greater than or equal "
                             "to 3.") # This is to ensure that the matrix construction will be square.
        self._augmented_matrix_dim = augmented_matrix_dim
        self._n_timesteps = n_timesteps
        super().__init__({'count': None, 'cross_prod_sum': None, 'means': None})

    def _init(self, dataset):
        shape = np.shape(dataset)
        if np.alen(shape) > 2:
            raise ValueError("The Pipe SurroundWithCorrelatedFeatures only works on datasets of the shape "
                             "observations by features.")

        self.states['count'] = 0
        self.states['cross_prod_sum'] = np.full(shape=(shape[1], shape[1]), fill_value=0, dtype=float)
        self.states['means'] = np.full(shape=shape[1], fill_value=0, dtype=float)

    def fit(self, dataset):
        self._init(dataset)
        self.fit_partial(dataset)

    def fit_partial(self, dataset):
        """
        Calculates the covariance batch-wise.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """
        if self.states['count'] is None:
            self._init(dataset)

        count = len(dataset)
        self.states['count'] += count
        dx = dataset - self.states['means']
        self.states['means'] += np.sum(dx, axis=0) / self.states['count']
        self.states['cross_prod_sum'] += np.dot(np.transpose(dx), dataset - self.states['means'])
        
    @property
    def mean(self):
        return self.states['mean']

    @property
    def cov(self):
        return self.states['cross_prod_sum'] / (self.states['count']-1)

    @property
    def cor(self):
        n_vars = len(self.states['means'])
        variances = self.cov[np.identity(n_vars)==1]
        return self.cov/np.sqrt(np.outer(variances, variances))

    def pairwise_correlation_ranking(self):
        n_most_related = np.square(self._augmented_matrix_dim)-1
        n_variables = len(self.states['means'])
        corr_coef = self.cor

        ranking = np.full(shape=[n_variables, n_most_related], fill_value=-42, dtype=np.int)

        for ivariable in range(0, n_variables):
            ranking[ivariable, :] = np.argsort(corr_coef[ivariable, :])[1:(n_most_related+1)]

        return ranking

    def correlated_feature_augment(self, x, ranking):

        n_rows = np.shape(x)[0]
        n_cols = np.shape(x)[1]

        augmented_x = np.full(shape=[self._augmented_matrix_dim * n_rows,
                                     self._augmented_matrix_dim * n_cols],
                              fill_value=-42.0, dtype=float)

        for irow in range(0, n_rows):
            augmented_x[irow * self._augmented_matrix_dim:((irow + 1) * self._augmented_matrix_dim), :] = \
                self.correlated_feature_augment_one_row(x[irow,], ranking)

        return augmented_x

    def correlated_feature_augment_one_row(self, x, ranking):

        n_vars = np.shape(x)[0]

        augmented_x_row = np.full(shape=[self._augmented_matrix_dim,
                                         self._augmented_matrix_dim * n_vars],
                                  fill_value=-42.0, dtype=float)

        for ivar in range(0, n_vars):
            temp_row_partial = np.empty(shape=np.square(self._augmented_matrix_dim))
            n_cor = np.int(
                np.floor((np.square(self._augmented_matrix_dim) - 1) / 2))  # number of correlated variables to augment with
            temp_row_partial[0:n_cor] = x[ranking[ivar, 0:n_cor]]
            temp_row_partial[n_cor] = x[ivar]
            temp_row_partial[(n_cor + 1):np.square(self._augmented_matrix_dim)] = x[
                ranking[ivar, n_cor:(np.square(self._augmented_matrix_dim) - 1)]]

            reshaped_to_matrix = np.reshape(temp_row_partial,
                                            newshape=[self._augmented_matrix_dim, self._augmented_matrix_dim])

            augmented_x_row[:,
            np.array(range(ivar * self._augmented_matrix_dim,
                           ((ivar + 1) * self._augmented_matrix_dim)))] = reshaped_to_matrix

        return augmented_x_row

    def transform(self, dataset):
        """
        Turn an n_observation by n_features dataset into a dataset of dimensions
        (n_observations-self.n_timesteps) by self.n_timesteps*self.augmented_matrix_dim by n_features*self.augmetned_matrix_dim
        :param dataset:
        :return:
        """

        ranking = self.pairwise_correlation_ranking()

        n_obs = np.shape(dataset)[0]
        n_vars = np.shape(dataset)[1]

        newx = np.full(
            shape=[n_obs - self._n_timesteps,
                   (self._n_timesteps + 1) * self._augmented_matrix_dim, n_vars * self._augmented_matrix_dim],
            fill_value=-42.0, dtype=float)

        for iobs in range(0, n_obs - self._n_timesteps):
            newx[iobs, :] = self.correlated_feature_augment(dataset[iobs:(iobs + self._n_timesteps + 1), :], ranking)

        return newx


class ToXy(F):
    """
    Splits the numpy in a 2-element tuple where the first is the first columns, and the last it the last column
    """
    def __init__(self):
        super().__init__(lambda chunk: chunk, lambda chunk: (chunk[:, :-1], chunk[:, -1]))
