import csv
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from release_smoke import check_factorial, same
ROOT=Path(__file__).resolve().parents[1]
def rows(name):
    with (ROOT/'results/md_factorial/analysis'/name).open(newline='') as f:return list(csv.DictReader(f))
class ReleaseSmokeTests(unittest.TestCase):
    def setUp(self):
        self.values=rows('per_trajectory_windows.csv')
        self.contrasts=rows('seed_contrasts.csv')
        self.summaries=rows('descriptive_summary.csv')
    def test_rejects_missing_branch(self):
        with self.assertRaises(ValueError):check_factorial(self.values[:-1],self.contrasts,self.summaries)
    def test_rejects_wrong_interaction(self):
        next(r for r in self.contrasts if r['contrast']=='(D-C)-(B-A)')['value']='100'
        with self.assertRaisesRegex(ValueError,'arithmetic'):check_factorial(self.values,self.contrasts,self.summaries)
    def test_rejects_summary_that_hides_range(self):
        self.summaries[0]['maximum']='100'
        with self.assertRaisesRegex(ValueError,'summary'):check_factorial(self.values,self.contrasts,self.summaries)
    def test_rejects_nonfinite(self):
        self.assertFalse(same('nan','nan'))
        self.assertFalse(same('inf','inf'))
if __name__=='__main__':unittest.main()
