import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import investigate_msa_outlier as runner
from analyze_msa_followup import inspect_run


class FollowupTests(unittest.TestCase):
    def test_default_prepares_without_inference(self):
        with tempfile.TemporaryDirectory() as td, patch.object(sys,'argv',['study','--output-dir',str(Path(td)/'prepared')]), patch.object(runner.subprocess,'run') as run, contextlib.redirect_stdout(io.StringIO()):
            runner.main()
            run.assert_not_called()
            plan=json.loads((Path(td)/'prepared/plan.json').read_text())
            self.assertEqual(len(plan['runs']),14)
            self.assertFalse((Path(td)/'prepared/runs').exists())

    def test_changed_input_cannot_reuse_completed_prediction(self):
        source=ROOT/'results/prediction_msa_diagnosis/followup'
        plan=json.loads((source/'plan.json').read_text());spec=plan['runs'][0]
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            shutil.copy2(source/'plan.json',base/'plan.json')
            shutil.copytree(source/'inputs',base/'inputs')
            shutil.copytree(source/'runs'/spec['id'],base/'runs'/spec['id'])
            path=base/plan['inputs'][spec['condition']]['path']
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'MSA changed'):
                inspect_run(base,plan,spec)


if __name__=='__main__':unittest.main()
