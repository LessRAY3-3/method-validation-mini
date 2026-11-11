import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_md_confirmation_batches import load_verified_batch,verify_files,summarize,validate_contacts,WINDOWS,OLD

class ComparisonTests(unittest.TestCase):
    def test_incomplete_batch_rejected_before_csv(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'summary.json').write_text(json.dumps({'status':'running','seeds':[20261113,20261114,20261115]}))
            with self.assertRaises(ValueError):load_verified_batch(p,True)
    def test_bad_hash_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'x').write_text('changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_files(p,{'x':'0'*64})
            with self.assertRaisesRegex(ValueError,'invalid evidence'):verify_files(p,{'../outside':'0'*64})
    def contact_fixture(self):
        from itertools import product
        return [dict(seed=str(seed),window_ns=window,native_cutoff_A=str(cutoff),retention_factor=str(factor),residue_i=str(pair[0]),residue_j=str(pair[1]),reference_distance_A='8',selection_status='excluded' if cutoff<8 else 'included',occupancy='' if cutoff<8 else '0.5') for seed,window,cutoff,factor,pair in product(OLD,WINDOWS,(7,8,9),(1.1,1.2,1.3),((3,19),(12,18)))]
    def test_contact_missing_duplicate_rejected(self):
        rows=self.contact_fixture();validate_contacts(rows,OLD)
        with self.assertRaisesRegex(ValueError,'Missing'):validate_contacts(rows[:-1],OLD)
        with self.assertRaisesRegex(ValueError,'Duplicate'):validate_contacts(rows+[rows[0]],OLD)
    def test_excluded_contact_cannot_be_zero(self):
        rows=self.contact_fixture();rows[0]['occupancy']='0'
        with self.assertRaisesRegex(ValueError,'null occupancy'):validate_contacts(rows,OLD)
    def test_stats_count_trajectories_not_frames(self):
        result=summarize([1,2,3])
        self.assertEqual(result['n_trajectories'],3)
        self.assertEqual(result['mean'],2)
        self.assertEqual(result['between_trajectory_sd'],1)

if __name__=='__main__':unittest.main()
