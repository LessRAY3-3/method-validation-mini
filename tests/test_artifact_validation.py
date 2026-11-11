import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from validate_project import validate


class ArtifactValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ('data/raw', 'data/prepared', 'results/docking'):
            shutil.copytree(ROOT / name, self.root / name)

    def tearDown(self):
        self.temp.cleanup()

    def test_corrupted_source_is_rejected(self):
        source = self.root / 'data/raw/1L2Y.pdb'
        source.write_bytes(source.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
            validate(self.root)

    def test_wrong_docking_score_is_rejected(self):
        path = self.root / 'results/docking/all_poses.csv'
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        rows[0]['affinity_kcal_mol'] = '999'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'Score mismatch'):
            validate(self.root)


if __name__ == '__main__':
    unittest.main()
