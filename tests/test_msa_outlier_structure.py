import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_msa_outlier_structure import distance_map, dihedral_deg, angular_delta


class StructuralGeometryTests(unittest.TestCase):
    def test_distance_map_is_rigid_motion_invariant(self):
        xyz=np.array([[0.,0.,0.],[1.,2.,3.],[-2.,1.,.5],[2.,3.,-2.]])
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        np.testing.assert_allclose(distance_map(xyz),distance_map(xyz @ rotation + [12.,-2.,7.]),atol=1e-12)
        self.assertAlmostEqual(distance_map(xyz)[0,1],np.sqrt(14))
        np.testing.assert_allclose(np.diag(distance_map(xyz)),0)

    def test_known_signed_right_angle_torsions(self):
        a=np.array([0.,1.,0.]);b=np.array([0.,0.,0.]);c=np.array([1.,0.,0.])
        self.assertAlmostEqual(dihedral_deg(a,b,c,np.array([1.,0.,1.])),90.)
        self.assertAlmostEqual(dihedral_deg(a,b,c,np.array([1.,0.,-1.])),-90.)
        self.assertAlmostEqual(abs(dihedral_deg(a,b,c,np.array([1.,-1.,0.]))),180.)
        self.assertAlmostEqual(float(angular_delta(179.,-179.)),-2.)

    def test_collinear_torsion_is_rejected(self):
        with self.assertRaises(ValueError):
            dihedral_deg(np.array([-1.,0.,0.]),np.array([0.,0.,0.]),np.array([1.,0.,0.]),np.array([1.,1.,0.]))


if __name__=='__main__':
    unittest.main()
