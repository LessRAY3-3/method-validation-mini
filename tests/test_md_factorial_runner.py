import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_md_factorial import SEEDS, NPT_PS, check_stage_text, check_tpr, source_files, stage_mdp, validate_protocol

class FactorialRunnerTests(unittest.TestCase):
    def test_protocol_mutation_and_design_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); (r/'input').write_text('immutable')
            p=dict(seeds=SEEDS,output_root='results/md_factorial',production_ns_each=11,nvt_ps=100,npt_ps_by_cell=NPT_PS,
                   gromacs_binary_sha256='frozen',input_sha256={'input':hashlib.sha256(b'immutable').hexdigest()},source_sha256={'input':hashlib.sha256(b'immutable').hexdigest()})
            f=r/'protocol.json';f.write_text(json.dumps(p));validate_protocol(r,f)
            p['npt_ps_by_cell']={**NPT_PS,'D':100};f.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'design'):validate_protocol(r,f)
            p['npt_ps_by_cell']=NPT_PS;f.write_text(json.dumps(p));(r/'input').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen'):validate_protocol(r,f)

    def test_tpr_duration_restraint_and_clock_are_checked(self):
        md=' nsteps = 5500000\n dt = 0.002\n tinit = 0\n init-step = 0\n'
        check_tpr(md,'md',5500000)
        with self.assertRaisesRegex(ValueError,'restraints'):check_tpr(md+'functype[354]=POSRES,','md',5500000)
        with self.assertRaisesRegex(ValueError,'clock'):check_tpr(md.replace('tinit = 0','tinit = 1000'),'md',5500000)
        with self.assertRaisesRegex(ValueError,'duration'):check_tpr(md,'md',50000)
        with self.assertRaisesRegex(ValueError,'restraints'):check_tpr(md,'npt',5500000)
        check_tpr(md+'functype[354]=POSRES,','npt',5500000)

    def test_checkpoint_must_finish_and_gpu_warnings_rejected(self):
        log='short-ranged interactions on the GPU\nPME tasks will do all aspects on the GPU'
        check_stage_text(log,'step = 500000\n',500000)
        with self.assertRaisesRegex(ValueError,'exact'):check_stage_text(log,'step = 50000\n',500000)
        with self.assertRaisesRegex(ValueError,'Numerical'):check_stage_text(log+'LINCS WARNING','step = 500000',500000)
        with self.assertRaisesRegex(ValueError,'GPU'):check_stage_text('cpu','step = 500000',500000)

    def test_branch_requires_complete_short_npt_and_all_append_files(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)
            for name in ['minim.gro','topol.top','posre.itp','nvt.gro','nvt.cpt','nvt.tpr','nvt.log','npt.gro','npt.cpt','npt.tpr','npt.log','npt.edr','npt.xtc']:(r/name).write_text('data')
            with self.assertRaisesRegex(ValueError,'completed'):source_files(r,True)
            (r/'npt.done.json').write_text('{"nsteps":50000}')
            self.assertIn('npt.edr',source_files(r,True));self.assertNotIn('npt.done.json',source_files(r,True))
            (r/'npt.edr').unlink()
            with self.assertRaisesRegex(ValueError,'Missing'):source_files(r,True)

    def test_seed_and_duration_substitution_preserves_physics(self):
        text='nsteps = 50000\ngen_seed = 1\nld-seed = 1\ngen_vel = no\nref_t = 300 300\n'
        got=stage_mdp(text,20261113,5500000)
        self.assertIn('nsteps = 5500000',got);self.assertIn('ld-seed = 20261113',got)
        self.assertIn('gen_vel = no',got);self.assertIn('ref_t = 300 300',got)

if __name__=='__main__':unittest.main()
