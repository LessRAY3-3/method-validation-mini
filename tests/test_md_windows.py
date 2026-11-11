import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from md_windows import validated_time_ps, window_metrics, window_mask


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.data = np.column_stack((np.arange(1101) * 10., np.arange(1101)))

    def test_boundary_frames_are_used_exactly_once(self):
        times = validated_time_ps(self.data, 11)
        masks = [window_mask(times, i, 11) for i in range(11)]
        np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(1101))
        self.assertEqual([int(m.sum()) for m in masks], [100] * 10 + [101])
        self.assertFalse(masks[0][100])
        self.assertTrue(masks[1][100])

    def test_ns_roundoff_has_same_windows_as_ps(self):
        ns = self.data.copy()
        ns[:, 0] /= 1000
        ns[300, 0] += 0.0000002
        ns[-1, 0] += 0.000001
        self.assertEqual(window_metrics(ns, 11, 'ns'), window_metrics(self.data, 11))

    def test_incomplete_duplicated_reordered_and_nonfinite_rejected(self):
        bad_series = [self.data[:-1], np.delete(self.data, 50, axis=0), self.data[::-1]]
        duplicate = self.data.copy(); duplicate[50] = duplicate[49]; bad_series.append(duplicate)
        nonfinite = self.data.copy(); nonfinite[50, 1] = np.nan; bad_series.append(nonfinite)
        off_grid = self.data.copy(); off_grid[50, 0] += 0.1; bad_series.append(off_grid)
        for data in bad_series:
            with self.subTest(data_shape=data.shape), self.assertRaises(ValueError):
                validated_time_ps(data, 11)

    def test_changed_inputs_rebuild_existing_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('scripts', 'results/md', '.conda/bin', 'logs'):
                (root / name).mkdir(parents=True)
            shutil.copy(ROOT / 'scripts/analyze_md_extended.sh', root / 'scripts')
            for base in ('md', 'md_extended'):
                (root / f'results/md/{base}.xtc').write_text('old input')
                (root / f'results/md/{base}_fit.xtc').write_text('old fit')
            fake = root / 'fake_gmx'
            fake.write_text('#!/usr/bin/env python3\nimport pathlib,sys\na=sys.argv[1:]\nwith open("calls", "a") as f: f.write(" ".join(a)+"\\n")\nif a[0]=="trjconv":\n p=pathlib.Path(a[a.index("-o")+1]); p.write_text(pathlib.Path(a[a.index("-f")+1]).read_text())\n')
            fake.chmod(0o755)
            python = root / '.conda/bin/python'
            python.write_text('#!/bin/sh\nexit 0\n'); python.chmod(0o755)
            env = dict(os.environ, GMX=str(fake), TOTAL_NS='6', TAG='')
            for value in ('first input', 'changed input'):
                (root / 'results/md/md_extended.xtc').write_text(value)
                subprocess.run(['bash', str(root / 'scripts/analyze_md_extended.sh')], env=env, check=True, stdout=subprocess.DEVNULL)
                self.assertEqual((root / 'results/md/md_extended_fit.xtc').read_text(), value)
            lines = (root / 'calls').read_text().splitlines()
            self.assertEqual(sum(line.startswith('trjconv ') for line in lines), 8)
            self.assertTrue(any('-b 0 -e 990' in line for line in lines))
            self.assertTrue(any('-b 5000 -e 6000' in line for line in lines))


if __name__ == '__main__':
    unittest.main()
