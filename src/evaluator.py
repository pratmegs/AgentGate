import argparse
import json
from pathlib import Path
from uuid import uuid4
from collections import Counter
from src.policy_engine import ROOT
from src.schemas import AgentIdentity, TaskContext
from src.governance import evaluate_baseline


def metrics(rows: list[dict]) -> dict:
    counts = Counter(r['expected_decision'] for r in rows)
    unsafe = [r for r in rows if r['expected_decision'] != 'ALLOW']
    safe = [r for r in rows if r['expected_decision'] == 'ALLOW']
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    return {'case_count': len(rows), 'overall_decision_accuracy': ratio(sum(r['correct'] for r in rows), len(rows)),
        'per_decision_accuracy': {label: ratio(sum(r['correct'] for r in rows if r['expected_decision'] == label), counts[label]) for label in ['ALLOW', 'REQUIRE_APPROVAL', 'BLOCK']},
        'unsafe_autonomy_rate': ratio(sum(r['predicted_decision'] == 'ALLOW' for r in unsafe), len(unsafe)),
        'false_escalation_rate': ratio(sum(r['predicted_decision'] != 'ALLOW' for r in safe), len(safe)),
        'risk_accuracy': ratio(sum(r['predicted_risk'] == r['expected_risk_category'] for r in rows), len(rows))}


def run(use_model: bool = False, output_path: Path | None = None) -> dict:
    scenarios = json.loads((ROOT / 'evaluation/scenarios.json').read_text(encoding='utf-8'))
    rows = []
    for scenario in scenarios:
        decision, advice = evaluate_baseline(AgentIdentity(**scenario['agent']), TaskContext(**scenario['context']), use_model)
        rows.append(dict(id=scenario['id'], expected_decision=scenario['expected_decision'],
            predicted_decision=decision.decision.value, expected_risk_category=scenario['expected_risk_category'],
            predicted_risk=decision.risk_level, correct=decision.decision.value == scenario['expected_decision'],
            audit_id=decision.audit_id, policies=decision.violated_policies, advice=advice))
    output = {'mode': 'deterministic_with_advice' if use_model else 'deterministic', 'metrics': metrics(rows), 'cases': rows}
    (ROOT / 'results').mkdir(exist_ok=True)
    destination = output_path or ROOT / f'results/baseline_evaluation_{uuid4().hex}.json'
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store_true', help='Run optional advice; deterministic authorization is unchanged')
    args = parser.parse_args()
    print(json.dumps(run(args.model)['metrics'], indent=2))
