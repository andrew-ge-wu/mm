import numpy as np

class MysqlIterator:
    """
    Mocks what the generic iterators.sqlalchemy would achieve in mysql, if mysql would support window operations.
    """
    def __init__(self, user, password, host, port, database, table, columns, query, batch_size):
        import pymysql

        self.batch_size = batch_size

        self.cnx = pymysql.connect(user=user, password=password, host=host, database=database, port=port)

        self.cursor = self.cnx.cursor()
        self.columns = columns
        self.cursor.execute(query.format(columns=','.join(['`%s`' % column for column in columns]), table=table))

    def __iter__(self):
        return self

    def __next__(self):
        rows = self.cursor.fetchmany(self.batch_size)

        # If the array is empty after we tried filling it,
        # assume there's no more data, so stop the iterator
        if len(rows) == 0:
            raise StopIteration

        return np.array(rows)
