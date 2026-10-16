"""Pinned Decision 2.0 Windows fingerprint portability correction.

Only the checkpoint hash's relative path spelling is normalized. The upstream
bundle SHA-256 checks, checkpoint identity comparison and parameter checks remain
active. Downloaded source, manifests and weights are never modified.
"""
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

SUPPORTED_REVISION = 'cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764'


def portable_fingerprint(identity: dict) -> dict:
    files = {key.replace('\\', '/'): value for key, value in identity['files_sha256'].items()}
    if len(files) != len(identity['files_sha256']):
        raise ValueError('Fingerprint path normalization produced duplicate paths')
    # Same canonical JSON representation used by the vendored data.canonical.
    encoded = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return {'model_sha256': hashlib.sha256(encoded.encode('utf-8')).hexdigest(), 'files_sha256': files}


def prepare_windows_fingerprint(repo: str, revision: str, cache_dir: str) -> None:
    if os.name != 'nt':
        return
    if repo != 'vllm-sr/Decision-2.0-Kai-0.6B' or revision != SUPPORTED_REVISION:
        raise ValueError('Windows compatibility fix only supports the verified revision')
    from huggingface_hub import snapshot_download
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    root = Path(snapshot_download(repo, revision=revision, cache_dir=cache_dir))
    manifest = json.loads((root / 'MODEL_MANIFEST.json').read_text(encoding='utf-8'))
    model_class = get_class_from_dynamic_module('modeling_decision2.Decision2Model', repo,
        revision=revision, cache_dir=cache_dir)
    model_module = sys.modules[model_class.__module__]
    runtime = model_module._import_runtime(root, manifest)
    # Verify the original bundle before applying an in-memory compatibility fix.
    runtime.verify_bundle(root)
    infer = importlib.import_module(runtime.__name__ + '._vendor.dev2model.infer')
    original = infer.checkpoint_fingerprint
    if getattr(original, '_agentgate_portable', False):
        return
    if manifest['profile'] != 'qwen-full':
        raise ValueError('Compatibility fix requires the verified full checkpoint')

    def fingerprint(path, source_path=None):
        return portable_fingerprint(original(path, source_path))

    fingerprint._agentgate_portable = True
    infer.checkpoint_fingerprint = fingerprint
