"""
Contains an API to use pandas data frames in sklearn pipelines.
"""
from .pipe import Pipe, F


class Copy(Pipe):
    """Copies the dataset, so that every subsequent operation does not affect original data"""
    def transform(self, df, supervised=False):
        return df.copy()


class ManyToOne(Pipe):
    """
    A N->1 transformation (many columns to one column)
        * `columns_in`: a single column (string)
        * `column_out`: another column (default: same as column_in).
        * `name`: optional transformation name
    """
    def __init__(self, columns_in, transform, column_out):
        self._columns_in = columns_in
        self._column_out = column_out
        super().__init__({'transform': transform})

    def fit(self, df):
        self.states['transform'].fit(df[self._columns_in])
        return self

    def transform(self, df, supervised=False):
        df[self._column_out] = self.states['transform'].transform(df.loc[:, self._columns_in].values)
        return df


class OneToOne(ManyToOne):
    """A 1->1 transformation (one column to one column)
    * `column_in`: a single column (string)
    * `column_out`: another column (default: same as column_in).
    * `name`: optional transformation name
    """
    def __init__(self, column_in, transform, column_out=None):
        if column_out is None:
            column_out = column_in
        super().__init__([column_in], transform, column_out)


class ApplyToAll(Pipe):
    """
    Similar to a OneToOne, but fits the columns and applies the operation over all
    """
    def __init__(self, transform, columns=None):
        self._fit_columns = columns is None
        super().__init__({'transform': transform, 'columns': columns})

    def fit(self, df):
        if self._fit_columns:
            self.states['columns'] = df.columns
        self.states['transform'].fit(df[self.states['columns']])
        return self

    def transform(self, df, supervised=False):
        df[self.states['columns']] = self.states['transform'].transform(df.loc[:, self.states['columns']].values)
        return df


class Select(Pipe):
    """Selects a set of columns (drops all others)"""
    def __init__(self, columns_to_keep=None):
        self._fit_columns = columns_to_keep is None
        super().__init__({'columns_to_keep': columns_to_keep})

    def fit(self, df):
        if self._fit_columns:
            self.states['columns_to_keep'] = df.columns

    def transform(self, df, supervised=False):
        return df[self.states['columns_to_keep']]


class Drop(Pipe):
    def __init__(self, columns_to_drop):
        super().__init__()
        self._columns_to_drop = columns_to_drop

    def transform(self, df, supervised=False):
        df.drop(self._columns_to_drop, axis=1, inplace=True)
        return df


class OneToMany(Pipe):
    """A 1->N transformation (one column to many columns)
    * `column_in`: a single column (string)
    * `column_out`: another column (default: same as column_in).
    * `name`: optional transformation name
    """
    def __init__(self, column_in, transform):
        self._column_in = column_in
        super().__init__({'transform': transform})

    def fit(self, df):
        self.states['transform'].fit(df[[self._column_in]])
        return self

    def transform(self, df, supervised=False):
        result = self.states['transform'].transform(df.loc[:, [self._column_in]])
        column_names = ['%s_%d' % (self._column_in, i) for i in range(result.shape[1])]
        for i, column in enumerate(column_names):
            df[column] = result[i]
        return df


class Callback(Pipe):
    """
    Executes a function whenever it fits or transforms. Used to inspect the data along the pipeline.
    """
    def __init__(self, fit=lambda df: None, transform=lambda X: None):
        self._fit = fit
        self._transform = transform
        super().__init__()

    def fit(self, df):
        self._fit(df)
        return super().fit(df)

    def transform(self, df, supervised=False):
        self._transform(df)
        return super().transform(df)


class SelectToNumpy(Select):
    """Selects a set of columns and convert them to a numpy matrix"""

    def transform(self, df, supervised=False):
        return df[self.states['columns_to_keep']].values


class ToXy(F):
    """
    Splits the df in a 2-element tuple where the first is all but the y_column, and the last it the y_column
    """
    def __init__(self, y_column):
        super().__init__(lambda chunk: chunk, lambda chunk: (chunk.drop(y_column, axis=1), chunk[y_column]))
