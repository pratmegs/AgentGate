"""Offline comparison. Model-only predictions are never authorization."""
import json
from src.evaluator import metrics
from src.policy_engine import ROOT, PolicyEngine
from src.schemas import AgentIdentity, TaskContext
from src.decision_model import get_advice


def model_prediction(advice: dict) -> tuple[str, str] | None:
    if advice['status'] != 'available':
        return None
    answers = advice['result']['answers']
    risk = answers['consequence']['choice']
    if answers['within_authority']['choice'] == 'NO':
        decision = 'BLOCK'
    elif answers['human_approval']['choice'] == 'YES' or answers['autonomous']['choice'] == 'NO':
        decision = 'REQUIRE_APPROVAL'
    else:
        decision = 'ALLOW'
    return decision, risk


def run() -> dict:
    scenarios = json.loads((ROOT / 'evaluation/scenarios.json').read_text())
    rows = {'deterministic': [], 'model_only': [], 'combined': []}
    failures = []
    for scenario in scenarios:
        agent, context = AgentIdentity(**scenario['agent']), TaskContext(**scenario['context'])
        deterministic = PolicyEngine().evaluate(agent, context)
        advice = get_advice(agent, context)
        prediction = model_prediction(advice)
        for mode in rows:
            if mode == 'model_only' and prediction is None:
                continue
            decision, risk = prediction if mode == 'model_only' else (deterministic.decision.value, deterministic.risk_level)
            rows[mode].append({'id': scenario['id'], 'expected_decision': scenario['expected_decision'],
                'predicted_decision': decision, 'correct': decision == scenario['expected_decision'],
                'expected_risk_category': scenario['expected_risk_category'], 'predicted_risk': risk,
                'advice': advice if mode != 'deterministic' else {'status': 'disabled'}})
        if prediction is None:
            failures.append({'id': scenario['id'], 'advice': advice})
            # Loading failed: don't repeatedly attempt a large unavailable download.
            break
    complete = not failures and len(rows['model_only']) == len(scenarios)
    output = {'status': 'complete' if complete else 'incomplete', 'expected_case_count': len(scenarios),
              'metrics': {mode: metrics(values) for mode, values in rows.items()} if complete else None,
              'failures': failures, 'cases': rows}
    (ROOT / 'results/comparison.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    result = run()
    print(json.dumps({'status': result['status'], 'metrics': result['metrics'], 'failures': result['failures']}, indent=2))
