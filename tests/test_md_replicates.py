import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_md_replicates import validate_times, window_selection, geometry_metrics, fitted_coordinates, discover_runs
from prediction_utils import ca_models, fasta_sequence
ROOT = Path(__file__).resolve().parents[1]


class MDReplicateTests(unittest.TestCase):
    def test_rigid_motion_does_not_change_q_contacts_or_fitted_rmsd(self):
        sequence = fasta_sequence(ROOT/'data/prepared/1L2Y.fasta')
        coords = ca_models(ROOT/'data/raw/1L2Y.pdb', sequence)[0][1]
        identities = [(i+1, 'ALA', 'CA') for i in range(20)] + [(i+1, 'ALA', 'CB') for i in range(20)]
        reference = coords.copy()
        coords = np.concatenate([coords, coords + [.5, 0, 0]])
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        trajectory = np.stack([coords, coords @ rotation + [11, -20, 8]])
        metrics, _, _ = geometry_metrics(trajectory, identities, reference)
        for key, values in metrics.items():
            np.testing.assert_allclose(values[0], values[1], atol=1e-12)
        np.testing.assert_allclose(metrics['q_ca_8A'], 1)
        np.testing.assert_allclose(metrics['ca_rmsd_nmr1_angstrom'], 0, atol=1e-12)

    def test_sidechain_distance_excludes_backbone(self):
        sequence = fasta_sequence(ROOT/'data/prepared/1L2Y.fasta')
        reference = ca_models(ROOT/'data/raw/1L2Y.pdb', sequence)[0][1]
        identities = [(i+1, 'ALA', 'CA') for i in range(20)] + [(i+1, 'ALA', 'CB') for i in range(20)]
        xyz = np.concatenate([reference, reference + [.5, 0, 0]])
        xyz[25] += [100, 0, 0]
        metrics, _, _ = geometry_metrics(xyz[None], identities, reference)
        self.assertGreater(metrics['sidechain_min_6_12_angstrom'][0], 80)
        self.assertLess(metrics['heavy_min_6_12_angstrom'][0], 20)

    def test_no_duplicate_times_or_window_endpoints(self):
        times = np.arange(0, 11001, 10)
        validate_times(times, 11)
        bad = times.copy(); bad[800] = bad[799]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_times(bad, 11)
        masks = [window_selection(times, i, i+1, 11) for i in (8,9,10)]
        self.assertEqual([int(m.sum()) for m in masks], [100,100,101])
        self.assertFalse(np.any(sum(m.astype(int) for m in masks)>1))
        self.assertEqual(int(window_selection(times,8,11,11).sum()),301)

    def test_incomplete_study_cannot_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Missing completed replicate'):
                discover_runs(root,11)
            for seed in (20261110,20261111):
                directory=root/f'seed_{seed}'; directory.mkdir()
                (directory/'md.done.json').write_text(json.dumps({'nsteps':5500000}))
                (directory/'provenance.json').write_text(json.dumps({'seed':seed,'target_production_ns':11,'gromacs':'test','input_sha256':{'x':'same'}}))
                (directory/'md.xtc').touch();(directory/'md.tpr').touch()
            with self.assertRaisesRegex(ValueError, 'Missing completed replicate'):
                discover_runs(root,11)


if __name__=='__main__':
    unittest.main()
