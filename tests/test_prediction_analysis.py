import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prediction_utils import fitted_ca_rmsd, ca_models
from analyze_predictions import validate_scores


class PredictionAnalysisTests(unittest.TestCase):
    def test_rigid_transform_is_removed_but_reflection_is_not(self):
        reference = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.assertLess(fitted_ca_rmsd(reference @ rotation + [5, -8, 10], reference), 1e-12)
        reflected = reference * [-1, 1, 1]
        self.assertGreater(fitted_ca_rmsd(reflected, reference), .1)

    def test_wrong_sequence_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, 'sequence mismatch'):
            ca_models(root / 'data/raw/1L2Y.pdb', 'A' * 20)

    def test_missing_ca_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        lines = (root / 'results/prediction/esmfold_1L2Y.pdb').read_text().splitlines()
        lines = [line for line in lines if not (line.startswith('ATOM') and line[12:16].strip() == 'CA' and int(line[22:26]) == 2)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'missing_ca.pdb'
            path.write_text('\n'.join(lines))
            with self.assertRaisesRegex(ValueError, 'sequence mismatch'):
                ca_models(path, 'NLYIQWLKDGGPSSGRPPPS')

    def test_wrong_pae_dimensions_and_invalid_confidence_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_scores({'plddt': [90]*20, 'pae': [[1]*19]*19}, 20)
        with self.assertRaises(ValueError):
            validate_scores({'plddt': [float('nan')]*20, 'pae': [[1]*20]*20}, 20)


class PredictionEvidenceTests(unittest.TestCase):
    def test_changed_pdb_cannot_pass_with_stale_provenance(self):
        import shutil
        from verify_prediction_repeat import verify_evidence
        root = Path(__file__).resolve().parents[1]
        source = root / 'results/prediction/reproducibility_check'
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            for path in source.iterdir():
                if path.is_file():
                    shutil.copy2(path, target / path.name)
            verify_evidence(target)
            pdb = next(target.glob('*.pdb'))
            pdb.write_text(pdb.read_text() + '\nREMARK CHANGED AFTER PROVENANCE\n')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                verify_evidence(target)


if __name__ == '__main__':
    unittest.main()
