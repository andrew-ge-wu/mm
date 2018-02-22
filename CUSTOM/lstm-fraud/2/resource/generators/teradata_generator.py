import pandas as pd
import teradata

class TeradataGenerator:
    """
    Creates a generator for Teradata
    """
    def __init__(self, user, passwd, host, query, batch_size):
        # Set some defaults
        # TODO: Improve by detecting available driver or terminating gracefully
        self.app_name = "TDGenerator"
        self.method = "odbc"
        self.driver = "Teradata Database ODBC Driver 16.10"
        self.batch_size = batch_size
        self.query = query
        # Setup connection and load data
        udaexec = teradata.UdaExec(appName = self.app_name, version ="1.0", logConsole=False)
        self.connection = udaexec.connect(method = self.method, system = host, username = user, password = passwd, driver = self.driver)
        self.reset()

    # Reloads data to DF
    def reset(self):
        self._df = pd.read_sql(self.query,self.connection, chunksize = self.batch_size)

    # Generates data in loop; when empty, reloads data
    def generate(self):
        while 1:
            try:
                yield self._df.__next__().as_matrix()
            except StopIteration:
                self.reset()
                yield self._df.__next__().as_matrix()
