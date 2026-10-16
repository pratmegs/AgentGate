"""Run the official model-card inputs after the verified Windows loader fix."""
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
from src.decision2_compat import prepare_windows_fingerprint, SUPPORTED_REVISION
from transformers import AutoModel

REPO = 'vllm-sr/Decision-2.0-Kai-0.6B'
OUT = ROOT / 'results/decision2_diagnostics'
OUT.mkdir(exist_ok=True)

try:
    prepare_windows_fingerprint(REPO, SUPPORTED_REVISION, str(ROOT / '.cache/huggingface/hub'))
    model = AutoModel.from_pretrained(REPO, revision=SUPPORTED_REVISION, trust_remote_code=True,
        cache_dir=str(ROOT / '.cache/huggingface/hub'))
    result = model.system_one(
        state='The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.',
        questions={
            'route': {'type': 'choice', 'instructions': 'Which team should handle this request?',
                'criteria': {'returns': 'Refunds, replacements and damaged deliveries', 'billing': 'Payments, invoices and charges', 'technical': 'Product setup and faults'}},
            'receipt': {'type': 'noul', 'instructions': 'Does the customer have a receipt?'},
            'urgency': {'type': 'score', 'instructions': 'How urgent is this request?', 'criteria': ['Routine', 'Soon', 'Today']},
        })
    (OUT / 'official_model_card_output.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['answers'], indent=2))
except Exception:
    (OUT / 'official_model_card_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    raise
