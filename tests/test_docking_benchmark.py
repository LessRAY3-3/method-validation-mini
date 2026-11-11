import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import analyze_docking_benchmark as analysis
import run_docking_benchmark as runner


class DockingBenchmarkTests(unittest.TestCase):
    def test_small_sample_intervals_do_not_imply_certainty(self):
        lo, hi = analysis.wilson(5, 5)
        self.assertAlmostEqual(hi, 1)
        self.assertLess(lo, 0.6)
        lo0, hi0 = analysis.wilson(0, 5)
        self.assertAlmostEqual(lo0, 0)
        self.assertGreater(hi0, 0.4)

    def test_contact_definition_is_residue_level_inclusive(self):
        receptor = {'A:ALA1': np.array([[4., 0., 0.], [3., 0., 0.]]),
                    'A:GLY2': np.array([[4.001, 0., 0.]])}
        self.assertEqual(analysis.contacts(np.array([[0., 0., 0.]]), receptor), {'A:ALA1'})
        self.assertEqual(analysis.contacts(np.array([[50., 0., 0.]]), receptor), set())

    def test_default_only_reports_pending_without_launching_tools(self):
        with tempfile.TemporaryDirectory() as td, patch.object(runner, 'OUT', Path(td)), patch.object(sys, 'argv', ['run_docking_benchmark.py']), patch.object(runner.subprocess, 'run') as run, patch.object(runner.subprocess, 'check_output') as version:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                runner.main()
            self.assertEqual(json.loads(output.getvalue())['counts'], {'pending': 15})
            run.assert_not_called()
            version.assert_not_called()

    def test_custom_directory_status_does_not_launch_or_create(self):
        with tempfile.TemporaryDirectory() as td, patch.object(runner, 'ROOT', Path(td)), patch.object(runner, 'OUT', Path(td) / 'original'), patch.object(sys, 'argv', ['run_docking_benchmark.py', '--output-dir', 'results/new_m3']), patch.object(runner.subprocess, 'run') as run, patch.object(runner.subprocess, 'check_output') as version:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                runner.main()
            self.assertEqual(runner.OUT, (Path(td) / 'results/new_m3').resolve())
            self.assertEqual(json.loads(output.getvalue())['counts'], {'pending': 15})
            self.assertFalse(runner.OUT.exists())
            run.assert_not_called()
            version.assert_not_called()


if __name__ == '__main__':
    unittest.main()
