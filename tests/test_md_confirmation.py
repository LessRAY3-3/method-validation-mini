import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_md_confirmation_batch import aggregate_command, selected_occupancy, episodes
from analyze_md_replicates import window_selection

class ConfirmationTests(unittest.TestCase):
    def test_new_seeds_forwarded_to_aggregate(self):
        cmd=aggregate_command(Path('/input'),Path('/output'),(20261113,20261114,20261115),'/gmx')
        ix=cmd.index('--seeds')
        self.assertEqual(cmd[ix+1:ix+4],['20261113','20261114','20261115'])
        self.assertIn('--keep-coordinates',cmd)
    def test_excluded_contact_is_not_zero_occupancy(self):
        self.assertEqual(selected_occupancy(8.1,[8,8.2],8,1.2),('excluded',None))
        self.assertEqual(selected_occupancy(8,[9.6,9.7],8,1.2),('included',.5))
    def test_final_endpoint_and_sample_span(self):
        t=np.arange(0,11001,10)
        self.assertEqual(window_selection(t,1,3,11).sum(),200)
        self.assertEqual(window_selection(t,8,11,11).sum(),301)
        self.assertEqual(episodes([0,10,20,30],[True,True,False,True]),[(0,10,2),(30,30,1)])
        with self.assertRaises(ValueError):episodes([0,20],[True,True])

if __name__=='__main__':unittest.main()
