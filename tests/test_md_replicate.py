"""Exercise runner control flow with a fake executable; no MD is performed."""
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ReplicateRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ('scripts', 'results/md', 'config'):
            (self.root / name).mkdir(parents=True)
        shutil.copy(ROOT / 'scripts/run_md_replicate.py', self.root / 'scripts')
        for name in ('minim.gro', 'topol.top', 'posre.itp', 'md.tpr'):
            (self.root / 'results/md' / name).write_text('fixture')
        for name in ('nvt', 'npt', 'md'):
            shutil.copy(ROOT / 'config' / f'{name}.mdp', self.root / 'config')
        self.gmx = self.root / 'fake_gmx'
        self.gmx.write_text('''#!/usr/bin/env python3
import os,sys,pathlib
a=sys.argv[1:]
if a[0]=='--version': print('GROMACS version: fake\\nGPU support: disabled')
elif a[0]=='grompp': pathlib.Path(a[a.index('-o')+1]).write_text('tpr')
elif a[0]=='mdrun': pathlib.Path(a[a.index('-deffnm')+1]+'.cpt').write_text('checkpoint')
elif a[0]=='dump': print('step = '+os.environ.get('FAKE_STEP','1000000'))
''')
        self.gmx.chmod(0o755)
        self.argv = [sys.executable, '-B', str(self.root/'scripts/run_md_replicate.py'), '--seed','20261110','--total-ns','2','--gmx',str(self.gmx),'--output-root','results/md_replicates_gpu']
        self.out=self.root/'results/md_replicates_gpu/seed_20261110'

    def tearDown(self):
        self.tmp.cleanup()

    def test_cuda_precheck_rejects_cpu_build(self):
        result=subprocess.run(self.argv+['--execution','cuda'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('does not report CUDA',result.stderr)
        self.assertFalse((self.out/'commands.jsonl').exists())

    def test_partial_checkpoint_is_not_marked_finished(self):
        result=subprocess.run(self.argv,env=dict(os.environ,FAKE_STEP='35000'),capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertTrue((self.out/'nvt.cpt').exists())
        self.assertFalse((self.out/'nvt.done.json').exists())
        self.assertFalse((self.out/'npt.tpr').exists())

    def test_completed_stages_are_not_repeated(self):
        subprocess.run(self.argv,capture_output=True,text=True,check=True)
        subprocess.run(self.argv,capture_output=True,text=True,check=True)
        calls=[json.loads(x)['argv'] for x in (self.out/'commands.jsonl').read_text().splitlines()]
        runs=[c for c in calls if c[1]=='mdrun']
        self.assertEqual(len(runs),3)
        self.assertTrue(all('-ntmpi' in c and '-ntomp' in c for c in runs))
        self.assertEqual(json.loads((self.out/'md.done.json').read_text())['nsteps'],1000000)

if __name__=='__main__': unittest.main()
