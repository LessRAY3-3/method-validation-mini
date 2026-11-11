import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_md_temporal_robustness import episodes,block_means,acf
class TemporalChecks(unittest.TestCase):
 def test_sampled_episode_span_and_separation(self):
  self.assertEqual(episodes([0,10,20,30,40],[1,1,0,1,0]),[(0,10,2),(30,30,1)])
  self.assertEqual(episodes([0,10],[0,0]),[])
 def test_blocks_exclude_endpoint_and_reject_missing(self):
  times=np.arange(0,101,10);vals=np.ones(11);vals[-1]=999
  self.assertEqual(block_means(times,vals,0,100,50),[(0,1.),(50,1.)])
  with self.assertRaises(ValueError):block_means(times[1:],vals[1:],0,100,50)
 def test_autocorrelation_constant_and_alternating(self):
  self.assertTrue(np.isnan(acf(np.ones(5))).all())
  np.testing.assert_allclose(acf([1,-1,1,-1],2),[1,-.75,.5])
if __name__=='__main__':unittest.main()
