# Fetch in batches is often hard. See e.g. https://stackoverflow.com/a/7390660/931303
# This is a common solution:
# from https://bitbucket.org/zzzeek/sqlalchemy/wiki/UsageRecipes/WindowedRangeQuery

import sqlalchemy
from sqlalchemy import and_, func, create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session


def column_windows(session, column, windowsize):
    """Return a series of WHERE clauses against
    a given column that break it into windows.

    Result is an iterable of tuples, consisting of
    ((start, end), whereclause), where (start, end) are the ids.

    Requires a database that supports window functions,
    i.e. Postgresql, SQL Server, Oracle.

    Enhance this yourself !  Add a "where" argument
    so that windows of just a subset of rows can
    be computed.
    """
    def int_for_range(start_id, end_id):
        if end_id is not None:
            return and_(column >= start_id, column < end_id)
        else:
            return column >= start_id

    q = session.query(
        column,
        func.row_number().\
        over(order_by=column).\
        label('rownum')
    ).from_self(column)
    if windowsize > 1:
        q = q.filter(sqlalchemy.text("rownum %% %d=1" % windowsize))

    intervals = [id for id, in q]

    while intervals:
        start = intervals.pop(0)
        if intervals:
            end = intervals[0]
        else:
            end = None
        yield int_for_range(start, end)


def windowed_query(q, column, windowsize):
    """"Break a Query into windows on a given column."""

    for whereclause in column_windows(q.session, column, windowsize):
        yield q.filter(whereclause).order_by(column)

## Copy ends here. Below is the iterator


class SQLIterator:
    def __init__(self, connection, database, table_name, pk_column, batch_size):
        engine = create_engine('{}/{}'.format(connection, database))

        # build the schema from the existing database
        automap = automap_base()
        automap.prepare(engine, reflect=True)

        # get the sqlalchemy class and primary key from table_name
        self._Base = getattr(automap.classes, table_name)
        self._pk = getattr(self._Base, pk_column)

        self.batch_size = batch_size

        self._engine = engine
        self.session = Session(engine)
        self._iterator = windowed_query(self.session.query(self._Base), self._pk, self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._iterator.__next__()
        except StopIteration:
            self.session.close()
            raise StopIteration
