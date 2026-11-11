import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_prediction_ablation import conditions_from_a3m, parse_a3m, main, SEEDS
from analyze_prediction_ablation import paired_deltas


class AblationTests(unittest.TestCase):
    def test_query_and_deduplicated_conditions_preserve_query(self):
        query = 'NLYIQWLKDGGPSSGRPPPS'
        text = '#20\t1\n>q\n' + query + '\n>q2\n' + query + '\n>h\n-LFIEWLKNGGPSSGAPPPS\n>h2\n-LFIEWLKNGGPSSGAPPPS\n'
        inputs = conditions_from_a3m(text, query)
        self.assertEqual([len(parse_a3m(inputs[c], query)) for c in ('query_only', 'deduplicated_saved_msa', 'raw_saved_msa')], [1, 2, 4])
        self.assertEqual(inputs['raw_saved_msa'], text)
        with self.assertRaisesRegex(ValueError, 'query'):
            parse_a3m(text, 'A' * 20)

    def test_pairs_match_seed_not_row_order_and_reject_missing(self):
        rows = [{'condition': c, 'seed': s, 'median_ca_rmsd_angstrom': i + (1 if c == 't' else 0), 'mean_plddt': 80} for c in ('t', 'r') for i, s in enumerate(SEEDS)]
        result = paired_deltas(list(reversed(rows)), 't', 'r')
        self.assertEqual([r['delta_median_ca_rmsd_angstrom'] for r in result], [1, 1, 1])
        with self.assertRaisesRegex(ValueError, 'requires all'):
            paired_deltas(rows[:-1], 't', 'r')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            paired_deltas(rows + rows[:1], 't', 'r')

    def test_default_prepares_without_starting_a_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            weights = base / 'cache/params'
            weights.mkdir(parents=True)
            (weights / 'params_model_1_ptm.npz').write_bytes(b'unit-test-placeholder')
            (weights / 'download_finished.txt').touch()
            out = base / 'plan'
            with patch.object(sys, 'argv', ['run_prediction_ablation.py', '--output-dir', str(out), '--weights-dir', str(weights.parent)]), patch('run_prediction_ablation.subprocess.run', side_effect=AssertionError('Default must not compute')):
                main()
            status = json.loads((out / 'status.json').read_text())
            self.assertEqual([r['state'] for r in status['runs']], ['pending'] * 9)
            self.assertFalse((out / 'runs').exists())


if __name__ == '__main__':
    unittest.main()
