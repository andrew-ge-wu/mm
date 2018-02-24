from unittest import TestCase
from pathlib import Path
import configparser
import numpy as np
import pandas as pd
import teradata
import unittest
import sys

class TestCase(TestCase):
    """
        Tests the TeradataGenerator can continually generate off of one query
        NOTE: Requires testconfig.ini configuration file
    """
    def setUp(self):
        config = configparser.ConfigParser()
        config.read('../test_resources/testconfig.ini')

        self._username = config['generators']['user']
        self._passwd = config['generators']['passwd']
        self._host = config['generators']['host']
        # Add module
        sys.path.insert(0, "../../generators")
        # Set fields
        self._query = "SELECT td.account_id,td.amount,td.score,td.time_stamp,td.fraudulent,td.bustout FROM (SELECT account_id,amount,score,time_stamp,fraudulent,bustout,RANDOM(1,1000000) AS marker FROM aifraud.transaction_analysis_test) AS td ORDER BY td.marker"
        self._batch_size = 1
        self._count = 0

    @unittest.skipIf(not Path('../test_resources/testconfig.ini').exists(),"Requires configuration file")
    def test_basic(self):
        import teradata_generator
        generator = teradata_generator.TeradataGenerator(self._username,self._passwd,self._host,self._query,self._batch_size)
        generated = generator.generate()
        for resultset in generated:
            for row in resultset:
                self.assertNotEqual(row[1], 0)  # Ensure account ID isn't empty
                self._count += 1
            if self._count >15:
                break
        # Query limits to 10; if we have greater than 15, then we've looped
        # generation
        self.assertGreater(self._count,15)