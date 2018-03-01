# Pipeline

This is a framework to write in-memory transparent data transformations in Python.

This framework contains 2 main elements:

1. the pipeline API for transparent data transformations.
2. a set of iterators for batch iterating over a relational database

## Installation

1. Run `pip install -r requirements.txt`.
2. Run `pip install .`

## Run tests

1. Run `python -m unittest`

To run tests with coverage:

1. `pip install coverage`
2. `coverage run --source=pipeline,iterators -m unittest discover -s tests`
3. `coverage html`
4. open `htmlcov/index.html` in the browser

## Examples

See `examples/`. See also `tests`, that also contain several examples of usage.

## Pipeline API

The core element of this framework is the `Pipe`, a base Python class
that contains 3 methods:

* `fit(dataset)`
* `fit_partial(dataset)`
* `transform(dataset)`

and one attribute:

* `states`

`fit/fit_partial` changes `states`, `transform` uses `states` to change `dataset`.

`Pipe` can have constant attributes initialized at its `__init__`.

The set of all constant attributes and `states` fully determine the data transformation done by `transform`.

A `Pipeline` is a stateless `Pipe` that contains a sequence of (stateful) pipes
on which each pipe's methods is applied to the transformed data from the previous pipe's transform.
These two elements define a powerful pattern to design stateful data transformations.

### Interface with Keras models

In `pipeline/keras_classifier.py` you can find a `Pipe` element that
allows you to use a Keras model. It supports both `fit` and `fit_partial`.

### Interface with LIME

In `pipeline/explainers.py` you can find a `Pipeline` and `Pipe`s
that allows you to use explain predictions. See `tests/test_pipeline/test_explainers.py`
for a minimal and complete example of usage.

### Serialization

In `pipeline/serialization.py` you can find elements to load and save pipelines.

There are 4 functions:

1. `save_untrained(pipeline, directory)` saves the trained pipeline as pickle and (in case it exists) the
untrained Keras model as `.json`.

2. `pipeline = load_untrained(directory)` reverse of `save_untrained`.

3. `save_trained(pipeline, directory)` saves the pipeline as pickle and the trained Keras model
as `.json` and tensorflow binary format

4. `pipeline = load_trained(directory)` reverse of `save_trained`.

### Existing pipe elements

See `pipeline/pipes_df.py` for elements operating on pandas dataframes

See `pipeline/pipes_np.py` for elements operating on numpy arrays

See `pipeline/pipes.py` for general pipes. This includes e.g. an element to
interface with any Sklearn element.

### Why not sklearn pipelines

Sklearn pipelines operate on numpy arrays. We want a generic format as the data
because e.g. we can have a combination of tensors with scalars.

Another issue with sklearn elements is that they do not distinguish between
values initialized during `__init__` and fitted values (`states`). We
do this in Pipe.

This pipeline API does not commit to a particular data format. It is
the programmer's responsibility to guarantee interoperability between
the the pipes `n` and `n+1` on a pipeline.

### Why not sklearn-pandas

Because pandas is intrinsically 2D (matrixes). We want to be open to tensors
for deep learning.

## Iterators

Another core element of this package is the `iterators` package.
There, you can find Python classes that can iterate over a database in batches (i.e. lower RAM foot-print).
There are 4 iterators:

1. `MysqlIterator`: to iterate over a mysql database
2. `SQLIterator`: to iterate over a SQL database that supports the `over` operator (TDC, Oracle, Postgres)
3. `CSVIterator`: to iterate over a CSV file, line by line
4. `PickleIterator`: to iterate over a pickle file, object by object

Each iterator has a different initialization, depending on their respective connectors.
For example, to iterate over a file `test.csv` in batches of 10, use

    from iterators import CSVIterator
    for chunk in iterators.CSVIterator('test.csv', 10):
        print(chunk)  # chunk is a pandas df with 10 elements
