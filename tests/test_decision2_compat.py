import json
from src.policy_engine import ROOT
from src.decision2_compat import portable_fingerprint, SUPPORTED_REVISION
import pytest


def manifest_identity():
    path = ROOT / f'.cache/huggingface/hub/models--vllm-sr--Decision-2.0-Kai-0.6B/snapshots/{SUPPORTED_REVISION}/MODEL_MANIFEST.json'
    if not path.exists():
        pytest.skip('Downloaded model manifest unavailable')
    return json.loads(path.read_text())['identity']


def test_windows_fingerprint_matches_verified_checkpoint():
    identity = manifest_identity()
    files = {key.replace('/', '\\'): value for key, value in identity['fingerprint_files'].items()}
    fixed = portable_fingerprint({'files_sha256': files})
    assert fixed['files_sha256'] == identity['fingerprint_files']
    assert fixed['model_sha256'] == identity['model_sha256']


def test_changed_weight_digest_still_fails_identity():
    identity = manifest_identity()
    files = dict(identity['fingerprint_files'])
    files['backbone/model.safetensors'] = '0' * 64
    assert portable_fingerprint({'files_sha256': files})['model_sha256'] != identity['model_sha256']


def test_ambiguous_paths_are_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        portable_fingerprint({'files_sha256': {'a/b': 'x', 'a\\b': 'y'}})
