"""Stage 3B projections and automated rendering; no browser or real inference."""
import ast
import json
import tomllib
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest
from src import decision_model
from src.assessment import ModelAssessment
from src.dashboard_session import initialize_session, remember_submission, refresh_requests
from src.dashboard_view import (ASSESSMENT_LABELS, session_metrics, technical_capabilities,
    action_rows, probability_rows, conflict_notes, workflow_states, audit_rows, history_rows)
from src.policy_engine import ROOT


@pytest.fixture
def console(monkeypatch, advice_factory):
    model = Mock(return_value=advice_factory())
    monkeypatch.setattr(decision_model, 'get_advice', model)
    # Any accidental real model call fails before checkpoint loading/download.
    monkeypatch.setattr(decision_model, 'load_model', Mock(side_effect=AssertionError('Live inference forbidden')))
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=15).run()
    assert not app.exception
    return app, model


def submit(app):
    app.button(key='submit_action').click().run()
    assert not app.exception
    return app.session_state['submitted_requests'][app.session_state['latest_request_id']]


def rendered_payload(app):
    """Rendered element messages only, excluding host-side AppTest session state."""
    kinds = ('markdown', 'caption', 'success', 'warning', 'error', 'info', 'metric',
             'dataframe', 'text_area', 'selectbox', 'checkbox', 'number_input', 'button', 'json', 'progress')
    return b''.join(element.proto.SerializeToString() for kind in kinds for element in app.get(kind))


def test_five_sections_dark_theme_and_empty_states(console):
    app, model = console
    assert [tab.label for tab in app.tabs] == ['Overview', 'Agent Workspace', 'Supervisory Evaluation',
                                             'Decision & Approval', 'Execution History & Audit']
    assert {metric.label: metric.value for metric in app.metric} == {
        'Total evaluated actions': '0', 'Allowed actions': '0', 'Pending approvals': '0',
        'Blocked actions': '0', 'Executed actions': '0'}
    theme = tomllib.loads((ROOT / '.streamlit/config.toml').read_text())['theme']
    assert theme['base'] == 'dark'
    assert model.call_count == 0
    text = '\n'.join(e.value for e in app.info)
    assert 'No actions evaluated' in text and 'No pending approvals' in text and 'No submission history' in text
    assert not app.json


def test_metrics_are_governance_counts_not_fabricated_execution_rates(tmp_path, monkeypatch, advice_factory):
    state = {}
    initialize_session(state, tmp_path)
    model = Mock(return_value=advice_factory())
    monkeypatch.setattr(decision_model, 'get_advice', model)
    scenarios = {s.name: s for s in state['demo_scenarios']}
    def run(name):
        scenario = scenarios[name]
        outcome = state['gateway'].submit(agent_credential=state['agent_credentials'][scenario.agent_id],
            task_id=scenario.task_id, tool_call=scenario.proposal)
        remember_submission(state, outcome)
        return outcome
    run('Customer Support: read records')
    model.return_value = advice_factory(human_approval='YES', autonomous='NO')
    pending = run('Finance: proposed change')
    run('Customer Support: excessive refund')
    assert session_metrics(state['submitted_requests'].values()) == {
        'Total evaluated actions': 3, 'Allowed actions': 1, 'Pending approvals': 1,
        'Blocked actions': 1, 'Executed actions': 1}
    state['gateway'].approve(approval_id=pending.approval.approval_id,
        approver_credential=state['reviewer_credentials']['Demo supervisor'][0],
        interception_id=pending.approval.interception_id,
        tool_call=scenarios['Finance: proposed change'].proposal)
    refresh_requests(state)
    assert session_metrics(state['submitted_requests'].values()) == {
        'Total evaluated actions': 3, 'Allowed actions': 1, 'Pending approvals': 0,
        'Blocked actions': 1, 'Executed actions': 2}
    assert model.call_count == 3


def test_capabilities_are_registry_metadata_separate_from_permissions(console):
    app, _ = console
    store = app.session_state['authority_store']
    agent, delegation = store.resolve('support-demo', 'support-demo-task')
    assert 'update' not in agent.permissions
    assert 'enterprise.update' in {row['Tool interface'] for row in technical_capabilities(delegation.target_systems)}
    assert not app.session_state['gateway'].tool_state()['invocations']


def test_native_probabilities_unchanged_six_cards_and_distributions(console, advice_factory):
    app, model = console
    advice = advice_factory(authority_drift='YES')
    # Extra fields are not dumped to the screen.
    advice['credential'] = 'MODEL_EXTRA_PAYLOAD_SHOULD_NOT_APPEAR'
    model.return_value = advice
    outcome = submit(app)
    assessment = outcome.supervision.assessment
    assert assessment.advice == advice
    rows = probability_rows(assessment)
    assert {row['Assessment'] for row in rows} == set(ASSESSMENT_LABELS.values())
    for key, label in ASSESSMENT_LABELS.items():
        shown = {row['Answer']: row['Probability'] for row in rows if row['Assessment'] == label}
        assert shown == advice['result']['answers'][key]['probabilities']
    selected_rows = [row for row in rows if row['Selected']]
    assert len(selected_rows) == 6
    assert len(app.get('progress')) == 6
    captions = '\n'.join(e.value for e in app.caption)
    assert 'uncalibrated' in captions
    assert not app.json
    assert any('conflicts' in e.value for e in app.warning)
    assert b'MODEL_EXTRA_PAYLOAD_SHOULD_NOT_APPEAR' not in rendered_payload(app)
    app.run()
    assert model.call_count == 1


@pytest.mark.parametrize('choices,expected', [
    ({}, []),
    ({'autonomous': 'NO', 'human_approval': 'NO'}, ['Divergent judgments']),
    ({'human_approval': 'YES'}, ['conflicts']),
    ({'consequence': 'HIGH'}, ['potential consequences = HIGH']),
])
def test_conflict_labels_describe_only_actual_combinations(advice_factory, choices, expected):
    assessment = ModelAssessment(status='available', advice=advice_factory(**choices))
    notes = conflict_notes(assessment)
    if not expected:
        assert notes == []
    for phrase in expected:
        assert any(phrase in note for note in notes)
    assert conflict_notes(ModelAssessment(status='unavailable', advice=assessment.advice)) == []


def test_workflow_and_history_use_real_gateway_evidence(console):
    app, _ = console
    outcome = submit(app)
    audit = app.session_state['audit_trail'].records()
    assert workflow_states(outcome, audit) == [
        ('Action Proposed', 'Recorded'), ('Intercepted', 'Captured'), ('Authority Verified', 'VERIFIED'),
        ('Decision 2.0 Assessment', 'AVAILABLE'), ('Decision Controller', 'ALLOW'),
        ('Execution Gateway', 'EXECUTED'), ('Audit', f'{len(audit)} events recorded')]
    rows = history_rows([outcome], audit)
    assert rows[0]['Mock tool invoked'] == 'enterprise.read'
    assert rows[0]['Submitted (UTC)'] == audit[0]['timestamp']
    assert rows[0]['Last audit event (UTC)'] == audit[-1]['timestamp']
    assert rows[0]['Request ID'] == outcome.request_id
    assert all(row['Request ID'] == outcome.request_id for row in audit_rows(audit))
    table_data = [frame.value for frame in app.dataframe]
    assert any('Event ID' in frame.columns for frame in table_data)
    assert any('Execution outcome' in frame.columns for frame in table_data)


def test_denied_reviewer_attempt_does_not_change_overview_or_actual_status(console, advice_factory):
    app, model = console
    model.return_value = advice_factory(human_approval='YES', autonomous='NO')
    outcome = submit(app)
    app.selectbox(key='reviewer').set_value('Demo observer (unauthorized)').run()
    app.button(key='approve_action').click().run()
    assert not app.exception
    counts = {metric.label: metric.value for metric in app.metric}
    assert counts['Pending approvals'] == '1' and counts['Executed actions'] == '0'
    assert counts['Execution status'] == 'PENDING_APPROVAL'
    assert app.session_state['last_approval_attempt'].status == 'REJECTED'
    assert outcome.approval.approval_id in app.session_state['pending_approvals']
    assert model.call_count == 1


def test_history_is_chronological_and_request_inspection_is_read_only(console):
    app, model = console
    first = submit(app)
    app.selectbox(key='scenario').set_value('Customer Support: bounded refund').run()
    second = submit(app)
    state = app.session_state
    rows = history_rows(state['submitted_requests'].values(), state['audit_trail'].records())
    assert [row['Request ID'] for row in rows] == [first.request_id, second.request_id]
    assert app.selectbox(key='result_request').value == second.request_id
    app.selectbox(key='result_request').set_value(first.request_id).run()
    assert not app.exception
    assert app.selectbox(key='result_request').value == first.request_id
    assert model.call_count == 2
    assert len(state['gateway'].tool_state()['invocations']) == 2


def test_display_does_not_expose_host_credentials_or_inject_proposal_html(console):
    app, _ = console
    payload = '<script>PRIVATE_MARKER</script>'
    app.text_area(key='proposal_description').set_value(payload).run()
    outcome = submit(app)
    state = app.session_state
    rendered = rendered_payload(app)
    for credential in [*state['agent_credentials'].values(),
                       *(token for token, identity in state['reviewer_credentials'].values())]:
        assert credential.encode() not in rendered
    assert outcome.supervision.intercepted.tool_call.arguments.description == payload
    html_blocks = [node.value for node in app.markdown if node.proto.allow_html]
    assert not any(payload in block for block in html_blocks)
    audit = state['audit_trail'].records()
    # Unknown backend fields cannot become arbitrary visible payloads.
    audit[0]['credential'] = 'DO_NOT_SHOW'
    audit[0]['tool_arguments'] = {'description': payload}
    assert 'DO_NOT_SHOW' not in json.dumps(audit_rows(audit))
    assert payload not in json.dumps(audit_rows(audit))


def test_presentation_has_no_execution_or_controller_calls():
    for name in ('src/dashboard_components.py', 'src/dashboard_view.py'):
        tree = ast.parse((ROOT / name).read_text(encoding='utf-8'))
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        assert not set(calls) & {'submit', 'approve', 'reject_approval', 'decide', 'evaluate',
                                'supervise', '_execute', '_invoke', 'system_one', 'get_advice'}
        constructors = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)]
        assert 'MockToolRegistry' not in constructors
    assert workflow_states(None, [])[-1] == ('Audit', 'Awaiting evidence')
