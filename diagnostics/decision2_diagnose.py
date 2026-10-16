"""Read-only checkpoint verification and full exception capture."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
REV = 'cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764'
REPO = 'vllm-sr/Decision-2.0-Kai-0.6B'
OUT = ROOT / 'results/decision2_diagnostics'
OUT.mkdir(exist_ok=True)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoModel
    root = ROOT / f'.cache/huggingface/hub/models--vllm-sr--Decision-2.0-Kai-0.6B/snapshots/{REV}'
    manifest = json.loads((root / 'MODEL_MANIFEST.json').read_text())
    info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
    latest = HfApi().model_info(REPO)
    remote_manifest_path = hf_hub_download(REPO, 'MODEL_MANIFEST.json', revision=REV, force_download=True, cache_dir=str(ROOT / '.cache/decision2-verification'))
    remote_manifest = json.loads(Path(remote_manifest_path).read_text())
    verification = []
    for sibling in info.siblings:
        path = root / sibling.rfilename
        if not path.is_file():
            verification.append({'file': sibling.rfilename, 'present': False})
            continue
        actual = sha(path)
        expected = manifest['files_sha256'].get(sibling.rfilename)
        lfs_sha = sibling.lfs.sha256 if sibling.lfs else None
        git_sha = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0' + path.read_bytes()).hexdigest() if not sibling.lfs else None
        verification.append({'file': sibling.rfilename, 'sha256': actual, 'manifest_sha256': expected,
            'manifest_match': actual == expected if expected else None, 'official_lfs_sha256': lfs_sha,
            'official_blob_id': sibling.blob_id, 'official_match': actual == lfs_sha if lfs_sha else git_sha == sibling.blob_id})
    report = {'pinned_revision': REV, 'official_resolved_revision': info.sha, 'official_main_revision': latest.sha,
        'manifest_matches_fresh_official_download': manifest == remote_manifest,
        'versions': {name: importlib.metadata.version(name) for name in ['transformers', 'torch', 'safetensors', 'huggingface-hub']},
        'files': verification}
    try:
        AutoModel.from_pretrained(REPO, revision=REV, trust_remote_code=True, cache_dir=str(ROOT / '.cache/huggingface/hub'), local_files_only=True)
    except Exception as exc:
        (OUT / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        tb = exc.__traceback__
        while tb:
            frame = tb.tb_frame
            if 'identity' in frame.f_locals and 'manifest' in frame.f_locals:
                identity = frame.f_locals['identity']
                report['computed_identity'] = identity
                report['expected_identity'] = frame.f_locals['manifest']['identity']
                report['raising_source'] = {'path': frame.f_code.co_filename, 'line': tb.tb_lineno}
                normalized = {key.replace('\\', '/'): value for key, value in identity['files_sha256'].items()}
                canonical = frame.f_locals['checkpoint_fingerprint'].__globals__['canonical']
                report['posix_normalized_identity'] = hashlib.sha256(canonical(normalized).encode()).hexdigest()
                report['normalized_files_match_manifest'] = normalized == manifest['identity']['fingerprint_files']
            tb = tb.tb_next
        report['error'] = str(exc)
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
