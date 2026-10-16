from unittest.mock import Mock
from src import decision_model
from src.governance import evaluate
from src.schemas import AgentIdentity, TaskContext
from src.policy_engine import ROOT
from src.demo_authority import provision_demo_authority
import json


def sample():
    scenario = json.loads((ROOT / 'evaluation/scenarios.json').read_text())[6]
    return AgentIdentity(**scenario['agent']), TaskContext(**scenario['context'])


def test_model_failure_preserves_block(monkeypatch):
    monkeypatch.setattr(decision_model, 'load_model', Mock(side_effect=RuntimeError('offline')))
    agent, context = sample()
    decision, advice = evaluate(agent, context, use_model=True, authority_store=provision_demo_authority(agent, context), task_id='demo-task')
    assert decision.decision == 'BLOCK'
    assert advice['status'] == 'unavailable'


def test_model_cannot_override_authority(monkeypatch):
    answers = {key: {'type': 'choice', 'choice': next(iter(q['criteria'])), 'probabilities': {label: .9 if label == next(iter(q['criteria'])) else .1 / (len(q['criteria']) - 1) for label in q['criteria']}} for key, q in decision_model.QUESTIONS.items()}
    fake = Mock()
    fake.system_one.return_value = {'answers': answers}
    monkeypatch.setattr(decision_model, 'load_model', lambda: fake)
    agent, context = sample()
    decision, advice = evaluate(agent, context, use_model=True, authority_store=provision_demo_authority(agent, context), task_id='demo-task')
    assert decision.decision == 'BLOCK'
    assert advice['result']['answers'] == answers


def test_malformed_model_falls_back(monkeypatch):
    fake = Mock()
    fake.system_one.return_value = {'answers': {}}
    monkeypatch.setattr(decision_model, 'load_model', lambda: fake)
    assert decision_model.get_advice(*sample())['status'] == 'unavailable'
