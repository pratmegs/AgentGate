from pathlib import Path

import pytest
from src.approval_manager import ApproverIdentity
from src.audit_trail import AuditTrail
from src.dashboard_session import provision_dashboard
from src.execution_gateway import ExecutionGateway
from src.policy_engine import ROOT
from diagnostics import pilot_evaluation
from diagnostics.pilot_evaluation import (SCENARIO_PATH, build_authority, build_parser,
    load_frozen_cases, model_only_decision, system_metrics)


@pytest.mark.parametrize(('path', 'expected_count', 'expected_per_label'), [
    (SCENARIO_PATH, 12, 4),
    (ROOT / 'evaluation/pilot_scenarios_v2.json', 57, 19),
])
def test_frozen_pilot_has_balanced_unique_labels(path, expected_count, expected_per_label):
    cases = load_frozen_cases(path)['cases']
    assert len(cases) == expected_count
    assert len({case['id'] for case in cases}) == expected_count
    assert {label: sum(case['expected_decision'] == label for case in cases)
            for label in ('ALLOW', 'REQUIRE_APPROVAL', 'BLOCK')} == {
                'ALLOW': expected_per_label, 'REQUIRE_APPROVAL': expected_per_label,
                'BLOCK': expected_per_label}


@pytest.mark.parametrize('path', [
    SCENARIO_PATH,
    ROOT / 'evaluation/pilot_scenarios_v2.json',
])
def test_authority_provisioning_uses_stable_delegation_per_case(path):
    cases = load_frozen_cases(path)['cases']
    store, credentials, task_ids = build_authority(cases)
    assert len(task_ids) == len(cases)
    assert len(set(task_ids.values())) == len(cases)
    assert set(credentials.values()) == {case['trusted_authorization']['agent_id'] for case in cases}
    for case in cases:
        agent_id = case['trusted_authorization']['agent_id']
        agent, delegation = store.resolve(agent_id, task_ids[case['id']])
        assert agent.agent_id == agent_id
        assert delegation.original_goal == case['delegated_task']
        assert delegation.task_id == f"{agent_id}:{case['id']}"


def test_cli_selects_v2_and_refund_investigation_is_opt_in(monkeypatch):
    defaults = build_parser().parse_args([])
    assert defaults.scenarios == SCENARIO_PATH
    assert defaults.investigate_refund is False
    calls = []
    monkeypatch.setattr(pilot_evaluation, 'run', lambda **kwargs: calls.append(kwargs))
    selected = ROOT / 'evaluation/pilot_scenarios_v2.json'
    pilot_evaluation.main(['--scenarios', str(selected), '--investigate-refund'])
    assert calls == [{'offline': True, 'scenario_path': Path(selected),
                      'include_refund_investigation': True}]


def test_model_only_mapping_is_fixed_and_fail_closed(advice_factory):
    assert model_only_decision(advice_factory())[0] == 'ALLOW'
    assert model_only_decision(advice_factory(authority_drift='YES'))[0] == 'BLOCK'
    assert model_only_decision(advice_factory(human_approval='YES'))[0] == 'REQUIRE_APPROVAL'
    assert model_only_decision({'status': 'unavailable'}) == ('REQUIRE_APPROVAL', ['MODEL_UNAVAILABLE'])


def test_customer_support_250_refund_is_blocked_and_cannot_be_approved(tmp_path):
    store, scenarios = provision_dashboard()
    scenario = next(item for item in scenarios if item.name == 'Customer Support: excessive refund')
    audit = AuditTrail(tmp_path / 'refund-250-audit.jsonl')
    gateway = ExecutionGateway(authority_store=store,
        agent_credentials={'agent-credential': scenario.agent_id},
        approver_credentials={'reviewer-credential': ApproverIdentity(
            approver_id='supervisor', roles=('supervisor',))}, audit=audit)
    outcome = gateway.submit(agent_credential='agent-credential', task_id=scenario.task_id,
        tool_call=scenario.proposal, use_model=False)
    assert outcome.status == 'BLOCKED'
    assert outcome.supervision.decision.decision.value == 'BLOCK'
    assert outcome.supervision.authority.status == 'PROHIBITED'
    assert outcome.supervision.authority.reasons == [
        'OUTSIDE_TRUSTED_FINANCIAL_BOUNDARY', 'NO_APPLICABLE_SCOPED_GRANT']
    assert outcome.approval is None
    attempt = gateway.approve(approval_id=outcome.request_id,
        approver_credential='reviewer-credential',
        interception_id=outcome.supervision.intercepted.interception_id,
        tool_call=scenario.proposal)
    assert attempt.status == 'REJECTED' and attempt.reason == 'UNKNOWN_APPROVAL'
    assert not gateway.tool_state()['invocations']
    request_events = [record['event'] for record in audit.records()
                      if record['request_id'] == outcome.request_id]
    assert request_events == ['action_proposed', 'action_intercepted', 'authority_verified',
                              'model_assessment_received', 'governance_decision', 'tool_rejected']


def test_metric_definitions_count_unsafe_allow_and_false_escalation():
    rows = [
        {'expected_decision': 'ALLOW', 'prediction': 'REQUIRE_APPROVAL'},
        {'expected_decision': 'REQUIRE_APPROVAL', 'prediction': 'ALLOW'},
        {'expected_decision': 'BLOCK', 'prediction': 'BLOCK'},
    ]
    metrics = system_metrics(rows, 'prediction')
    assert metrics['decision_accuracy_count'] == '1/3'
    assert metrics['unsafe_autonomy_count'] == '1/2'
    assert metrics['false_escalation_count'] == '1/1'
    assert metrics['block_recall_count'] == '1/1'
