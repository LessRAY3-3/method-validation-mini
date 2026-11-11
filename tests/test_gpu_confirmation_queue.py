import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_gpu_formal_queue import validate_protocol,require_gpu_logs

class ConfirmationQueueTests(unittest.TestCase):
 def test_protocol_rejects_changed_input_or_seed(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);(r/'input').write_text('frozen');h=hashlib.sha256(b'frozen').hexdigest()
   p={'seeds':[13,14,15],'output_root':'new','production_ns_each':11,'nvt_ps':100,'npt_ps':100,'input_sha256':{'input':h},'source_sha256':{'input':h}}
   f=r/'protocol.json';f.write_text(json.dumps(p))
   validate_protocol(r,f,(13,14,15),'new')
   with self.assertRaisesRegex(ValueError,'batch'):validate_protocol(r,f,(10,11,12),'new')
   (r/'input').write_text('changed')
   with self.assertRaisesRegex(ValueError,'Frozen file changed'):validate_protocol(r,f,(13,14,15),'new')
 def test_gpu_fallback_or_lincs_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);good='short-ranged interactions on the GPU\nPME tasks will do all aspects on the GPU\n'
   for stage in ['nvt','npt','md']:(r/f'{stage}.log').write_text(good)
   require_gpu_logs(r)
   (r/'md.log').write_text(good+'LINCS WARNING')
   with self.assertRaisesRegex(ValueError,'Numerical'):require_gpu_logs(r)
   (r/'md.log').write_text('CPU only')
   with self.assertRaisesRegex(ValueError,'GPU offload'):require_gpu_logs(r)
if __name__=='__main__':unittest.main()
