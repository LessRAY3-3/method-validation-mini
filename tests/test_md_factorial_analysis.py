import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_md_factorial import contrasts, SEEDS, validate_completed

class FactorialAnalysisTests(unittest.TestCase):
    def rows(self):
        return [dict(cell=c,seed=s,window_ns='1-3',metric='distance',value=v)
                for c,v in zip('ABCD',[2.,5.,7.,11.]) for s in SEEDS]
    def test_contrast_direction_and_interaction(self):
        rows=contrasts(self.rows())
        self.assertEqual(len(rows),15)
        expected={'B-A':3.,'D-C':4.,'C-A':5.,'D-B':6.,'(D-C)-(B-A)':1.}
        for r in rows:self.assertEqual(r['value'],expected[r['contrast']])
    def test_missing_cell_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing'):contrasts(self.rows()[:-1])
    def test_duplicate_endpoint_rejected(self):
        rows=self.rows()
        with self.assertRaisesRegex(ValueError,'Duplicate'):contrasts(rows+[rows[0]])
    def test_incomplete_run_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name in ('md.tpr','md.xtc','md.log'):(p/name).write_text('')
            (p/'md.done.json').write_text(json.dumps({'nsteps':5000000}))
            (p/'provenance.json').write_text(json.dumps({'seed':SEEDS[0],'target_production_ns':11}))
            with self.assertRaisesRegex(ValueError,'Incomplete'):validate_completed(p,SEEDS[0])
    def test_numerical_warning_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name in ('md.tpr','md.xtc'):(p/name).write_text('')
            (p/'md.log').write_text('LINCS WARNING')
            (p/'md.done.json').write_text(json.dumps({'nsteps':5500000}))
            (p/'provenance.json').write_text(json.dumps({'seed':SEEDS[0],'target_production_ns':11}))
            with self.assertRaisesRegex(ValueError,'failure'):validate_completed(p,SEEDS[0])
if __name__=='__main__':unittest.main()
