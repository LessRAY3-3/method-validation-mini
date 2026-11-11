import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_md_torsions import delta,circular_summary,torsion
class TorsionTests(unittest.TestCase):
 def test_wrap_and_aromatic_symmetry(self):
  self.assertEqual(delta(-179,179),2)
  self.assertEqual(delta(-89,89,180),2)
  self.assertEqual(delta(10,190,180),0)
 def test_circular_mean_not_zero_at_wrap(self):
  mean,r=circular_summary([179,-179])
  self.assertAlmostEqual(abs(mean),180)
  self.assertGreater(r,.99)
 def test_torsion_rigid_invariant_and_undefined(self):
  x=np.array([[1.,0,0],[0,0,0],[0,1,0],[0,1,1]])
  rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
  self.assertAlmostEqual(torsion(x),torsion(x@rotation+4))
  with self.assertRaises(ValueError):torsion(np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]]))
