from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_md_local_geometry import pair_geometry, rmsd_series

class LocalGeometryTests(unittest.TestCase):
    def test_pair_and_fit_invariant_under_rigid_motion(self):
        ref=np.random.default_rng(42).normal(size=(20,3))
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        xyz=np.stack([ref,ref@rot+[9,-4,17]])
        *_,signed=pair_geometry(xyz,ref)
        np.testing.assert_allclose(signed,0,atol=1e-12)
        np.testing.assert_allclose(rmsd_series(xyz,ref,np.arange(20)),0,atol=1e-12)

    def test_terminal_distortion_does_not_corrupt_internal_refit(self):
        ref=np.random.default_rng(8).normal(size=(20,3))
        xyz=ref.copy();xyz[0]+=[10,20,30]
        self.assertGreater(rmsd_series(xyz[None],ref,np.arange(20))[0],1)
        np.testing.assert_allclose(rmsd_series(xyz[None],ref,np.arange(2,18)),0,atol=1e-12)
        i,j,_,_,signed=pair_geometry(xyz[None],ref)
        np.testing.assert_allclose(signed[:,(i>=2)&(j<=17)],0)
        self.assertGreater(np.abs(signed).max(),10)
