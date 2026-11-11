"""Package completed factorial evidence locally, verifying every archived member."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    root=ROOT;study=root/'results/md_factorial';analysis=study/'analysis'
    protocol=root/'config/experiments/md_factorial_protocol.json'
    p=json.loads(protocol.read_text());q=json.loads((study/'queue_status.json').read_text())
    assert q['status']=='complete' and len(q['completed'])==9
    assert sha(protocol)==q['protocol_sha256']
    files={protocol,study/'queue_status.json',study/'local_transfer_verification.json',analysis/'summary.json',root/'results/validation/md_factorial_m3_independent.json'}
    for field in ('input_sha256','source_sha256'):
        for rel,want in p[field].items():
            path=root/rel;assert sha(path)==want,rel;files.add(path)
    for marker in study.glob('*/seed_*/complete.json'):
        files.add(marker)
        for rel,want in json.loads(marker.read_text())['output_sha256'].items():
            path=marker.parent/rel;assert sha(path)==want,rel;files.add(path)
    for rel,want in json.loads((analysis/'summary.json').read_text())['outputs_sha256'].items():
        path=analysis/rel;assert sha(path)==want,rel;files.add(path)
    # Analysis inputs include reused A baseline and immutable per-run provenance.
    prefix='/home/lessray/codex-runs/method-validation-20261110-gpu/'
    for rel,want in json.loads((analysis/'manifest.json').read_text())['input_sha256'].items():
        path=root/rel.removeprefix(prefix);assert sha(path)==want,rel;files.add(path)
    manifest=root/'results/validation/md_factorial_evidence_manifest.json'
    manifest.write_text(json.dumps({'status':'local archive, not uploaded','scope':'Frozen inputs/sources, twelve analyzed trajectories, nine new branch outputs, complete analysis and M3 audit; excludes historical first three trajectories','files':[{'path':str(f.relative_to(root)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(files)]},indent=2)+'\n');files.add(manifest)
    expected={str(f.relative_to(root)):sha(f) for f in files}
    out=root/'artifacts/md-factorial-evidence.tar.gz';out.parent.mkdir(exist_ok=True)
    if out.exists():raise ValueError('Archive exists; refusing overwrite')
    tmp=out.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz',compresslevel=3) as t:
        for f in sorted(files):
            assert not f.is_symlink();t.add(f,arcname=str(f.relative_to(root)),recursive=False)
    with tarfile.open(tmp,'r:gz') as t:
        assert {m.name for m in t.getmembers()}==set(expected)
        for m in t:
            assert m.isfile()
            with t.extractfile(m) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected[m.name],m.name
    tmp.replace(out);checksum=sha(out)
    out.with_suffix(out.suffix+'.sha256').write_text(checksum+'  '+out.name+'\n')
    result={'status':'verified local archive, not uploaded','path':str(out.relative_to(root)),'bytes':out.stat().st_size,'sha256':checksum,'members':len(files)}
    (root/'results/validation/md_factorial_evidence_archive.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
