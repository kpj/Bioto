from unittest import TestCase

import sys
from io import StringIO

import stats


class TestBMStatsMan(TestCase):
    def setUp(self):
        self.tmp_out = StringIO()
        sys.stdout = self.tmp_out

        self.stat = stats.BMStatsMan()

    def test_initialization(self):
        self.assertEqual(self.stat.early_stops, 0)
        self.assertEqual(self.stat.discrete_runs, 0)

    def test_info(self):
        self.stat.early_stops += 5
        self.stat.discrete_runs += 9
        self.stat.info()

        out = self.tmp_out.getvalue()
        self.assertEqual(out, 'early stops | discrete runs | quotient : 5 | 9 | 0.56\n')

    def test_zero_runs(self):
        self.stat.info()

        out = self.tmp_out.getvalue()
        self.assertEqual(out, 'early stops | discrete runs | quotient : 0 | 0 | 0\n')
