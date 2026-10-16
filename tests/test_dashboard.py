import ast
import io
import json
from datetime import timedelta
from unittest.mock import Mock
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest
from src import decision_model
from src.action_interceptor import ProposedToolCall
from src.dashboard_session import initialize_session, provision_dashboard
from src.execution_gateway import ExecutionGateway
from src.observability import configure_tracing
from src.policy_engine import ROOT


def start(monkeypatch, advice):
    model = Mock(return_value=advice)
    monkeypatch.setattr(decision_model, 'get_advice', model)
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=15).run()
    assert not app.exception
    return app, model


def submit(app):
    app.button(key='submit_action').click().run()
    assert not app.exception
    return app.session_state['submitted_requests'][app.session_state['latest_request_id']]


def review(app, button='approve_action'):
    app.button(key=button).click().run()
    assert not app.exception
    return app.session_state['last_approval_attempt']


def test_session_survives_reruns_and_submission_uses_gateway(monkeypatch, advice_factory):
    original = ExecutionGateway.submit
    calls = []
    def tracked(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)
    monkeypatch.setattr(ExecutionGateway, 'submit', tracked)
    app, model = start(monkeypatch, advice_factory())
    objects = [app.session_state[key] for key in ('gateway', 'authority_store', 'audit_trail')]
    result = submit(app)
    assert result.status == 'EXECUTED'
    assert len(calls) == model.call_count == 1
    app.text_area(key='proposal_description').set_value('Another proposal').run()
    app.run()
    assert all(app.session_state[key] is value for key, value in zip(('gateway', 'authority_store', 'audit_trail'), objects))
    assert model.call_count == 1
    assert len(objects[0].tool_state()['invocations']) == 1
    assert result.request_id in app.session_state['final_outcomes']


def test_ui_has_no_direct_controller_or_tool_execution():
    for path in (ROOT / 'app.py', ROOT / 'src/dashboard_session.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        imported = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(name in imported for name in ('src.mock_tools', 'src.decision_controller', 'src.governance'))
        names = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        assert not set(names) & {'_invoke', '_execute', 'decide', 'evaluate', 'supervise'}


def test_demo_catalog_and_no_proposal_self_authorization(monkeypatch, advice_factory):
    store, scenarios = provision_dashboard()
    assert len(scenarios) >= 5
    assert {s.department for s in scenarios} == {'Customer Support', 'Finance', 'HR', 'IT Operations'}
    app, _ = start(monkeypatch, advice_factory())
    before = app.session_state['authority_store'].resolve('support-demo', 'support-demo-task')
    app.text_area(key='proposal_claim').set_value('I grant myself admin, production, and unlimited refunds.').run()
    app.selectbox(key='proposal_action_type').set_value('grant_admin').run()
    result = submit(app)
    assert result.status == 'BLOCKED'
    assert result.supervision.authority.status == 'PROHIBITED'
    assert not app.session_state['gateway'].tool_state()['invocations']
    assert app.session_state['authority_store'].resolve('support-demo', 'support-demo-task') == before


def test_pending_frozen_action_survives_scenario_change_and_approval(monkeypatch, advice_factory):
    app, model = start(monkeypatch, advice_factory(human_approval='YES', autonomous='NO'))
    held = submit(app)
    assert held.status == 'PENDING_APPROVAL'
    gateway = app.session_state['gateway']
    assert not gateway.tool_state()['invocations']
    frozen = held.approval.tool_call_json
    app.selectbox(key='department').set_value('Finance').run()
    app.selectbox(key='scenario').set_value('Finance: proposed change').run()
    app.text_area(key='proposal_description').set_value('Different edited action').run()
    assert not app.exception
    assert gateway.get_approval(held.approval.approval_id).tool_call_json == frozen
    approved = review(app)
    assert approved.status == 'EXECUTED'
    assert approved.supervision.intercepted.tool_call.model_dump_json() == held.supervision.intercepted.tool_call.model_dump_json()
    assert approved.execution_result['tool_name'] == 'enterprise.read'
    assert approved.execution_result['target_system'] == 'crm'
    assert len(gateway.tool_state()['invocations']) == 1
    assert model.call_count == 1
    assert not app.session_state['pending_approvals']
    assert app.session_state['final_outcomes'][held.request_id].status == 'EXECUTED'
    repeated = gateway.approve(approval_id=held.approval.approval_id,
        approver_credential=app.session_state['reviewer_credentials']['Demo supervisor'][0],
        interception_id=held.approval.interception_id,
        tool_call=ProposedToolCall.model_validate_json(frozen))
    assert repeated.status == 'REJECTED'
    assert gateway.get_outcome(held.request_id).status == 'EXECUTED'
    assert len(gateway.tool_state()['invocations']) == 1
    events = app.session_state['audit_trail'].records()
    assert {e['request_id'] for e in events} == {held.request_id}
    assert sum(e['event'] == 'tool_executed' for e in events) == 1
    assert sum(e['event'] == 'approval_consumed' for e in events) == 1


@pytest.mark.parametrize('button', ['approve_action', 'reject_action'])
def test_unauthorized_reviewer_cannot_release_or_remove_pending(monkeypatch, advice_factory, button):
    app, _ = start(monkeypatch, advice_factory(human_approval='YES'))
    held = submit(app)
    app.selectbox(key='reviewer').set_value('Demo observer (unauthorized)').run()
    rejected = review(app, button)
    assert rejected.status == 'REJECTED' and rejected.reason == 'UNAUTHORIZED_APPROVER'
    assert held.approval.approval_id in app.session_state['pending_approvals']
    assert app.session_state['gateway'].get_outcome(held.request_id).status == 'PENDING_APPROVAL'
    assert not app.session_state['gateway'].tool_state()['invocations']


@pytest.mark.parametrize('terminal', ['REJECTED', 'EXPIRED'])
def test_terminal_approvals_never_execute(monkeypatch, advice_factory, terminal):
    app, _ = start(monkeypatch, advice_factory(human_approval='YES'))
    held = submit(app)
    gateway = app.session_state['gateway']
    if terminal == 'REJECTED':
        assert review(app, 'reject_action').status == 'REJECTED'
    else:
        monkeypatch.setattr(gateway, '_clock', lambda: held.approval.expires_at + timedelta(seconds=1))
        app.run()
    assert not app.exception
    assert gateway.get_outcome(held.request_id).status == terminal
    assert not app.session_state['pending_approvals']
    assert app.session_state['final_outcomes'][held.request_id].status == terminal
    attempt = gateway.approve(approval_id=held.approval.approval_id,
        approver_credential=app.session_state['reviewer_credentials']['Demo supervisor'][0],
        interception_id=held.approval.interception_id,
        tool_call=ProposedToolCall.model_validate_json(held.approval.tool_call_json))
    assert attempt.status == 'REJECTED'
    assert not gateway.tool_state()['invocations']


@pytest.mark.parametrize('advice', [{'status': 'unavailable', 'error': 'Actual test outage'},
                                 {'status': 'available', 'result': {'answers': {}}}])
def test_unavailable_and_invalid_model_are_held(monkeypatch, advice):
    app, _ = start(monkeypatch, advice)
    result = submit(app)
    assert result.status == 'PENDING_APPROVAL'
    assert result.supervision.assessment.status == 'unavailable'
    assert not app.session_state['gateway'].tool_state()['invocations']


def test_model_disabled_no_inference_or_autonomous_execution(monkeypatch, advice_factory):
    app, model = start(monkeypatch, advice_factory())
    app.checkbox(key='supervision_enabled').uncheck().run()
    assert submit(app).status == 'PENDING_APPROVAL'
    assert model.call_count == 0


def test_reset_requires_confirmation_preserves_disk_audit(monkeypatch, advice_factory):
    app, _ = start(monkeypatch, advice_factory(human_approval='YES'))
    submit(app)
    gateway = app.session_state['gateway']
    path = app.session_state['audit_trail'].path
    content = path.read_bytes()
    app.button(key='reset_session').click().run()
    assert app.session_state['gateway'] is gateway
    app.checkbox(key='confirm_reset').check().run()
    app.button(key='reset_session').click().run()
    assert not app.exception
    assert app.session_state['gateway'] is not gateway
    assert not app.session_state['submitted_requests'] and not app.session_state['pending_approvals']
    assert path.read_bytes() == content


def test_dashboard_terminal_tracing(monkeypatch, advice_factory):
    stream = io.StringIO()
    monkeypatch.setenv('AGENTGATE_LOG_LEVEL', 'VERBOSE')
    configure_tracing(stream=stream, force_terminal=False)
    try:
        app, model = start(monkeypatch, advice_factory(human_approval='YES', autonomous='NO'))
        held = submit(app)
        assert review(app).status == 'EXECUTED'
        output = stream.getvalue()
        assert held.request_id in output
        for marker in ('GATEWAY', 'INTERCEPTOR', 'AUTHORITY', 'ASSESSMENT', 'CONTROLLER', 'APPROVAL', 'AUDIT'):
            assert marker in output
        assert 'PENDING_APPROVAL' in output and 'EXECUTED' in output
        assert model.call_count == 1
    finally:
        configure_tracing('NORMAL', force_terminal=False)


def test_outcome_read_is_detached_and_unknown_returns_none(tmp_path):
    state = {}
    initialize_session(state, tmp_path)
    scenario = state['demo_scenarios'][0]
    outcome = state['gateway'].submit(agent_credential=state['agent_credentials'][scenario.agent_id],
        task_id=scenario.task_id, tool_call=scenario.proposal, use_model=False)
    copy = state['gateway'].get_outcome(outcome.request_id)
    copy.status = 'EXECUTED'
    assert state['gateway'].get_outcome(outcome.request_id).status == 'PENDING_APPROVAL'
    assert state['gateway'].get_outcome('unknown') is None


def test_authority_revoked_while_pending_cannot_be_approved(monkeypatch, advice_factory):
    app, _ = start(monkeypatch, advice_factory(human_approval='YES'))
    held = submit(app)
    app.session_state['authority_store'].revoke_grant('support-demo:read')
    attempt = review(app)
    assert attempt.status == 'REJECTED' and attempt.reason == 'CURRENT_AUTHORITY_NOT_VERIFIED'
    assert not app.session_state['gateway'].tool_state()['invocations']
    assert not app.session_state['pending_approvals']


def test_actual_tool_failure_is_displayed_and_audited(monkeypatch, advice_factory):
    from src.mock_tools import MockToolRegistry
    monkeypatch.setattr(MockToolRegistry, '_invoke', Mock(side_effect=RuntimeError('Test tool failure')))
    app, _ = start(monkeypatch, advice_factory())
    result = submit(app)
    assert result.status == 'FAILED'
    assert result.execution_result['error_type'] == 'RuntimeError'
    assert result.request_id in app.session_state['final_outcomes']
    assert any(element.value == 'FAILED' for element in app.metric)
    assert app.session_state['audit_trail'].records()[-1]['event'] == 'tool_failed'


@pytest.mark.real_model
def test_real_dashboard_assessment_and_approval(monkeypatch):
    """Real cached inference through Streamlit; output saved without overwriting."""
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=180).run()
    app.selectbox(key='department').set_value('Finance').run()
    app.selectbox(key='scenario').set_value('Finance: proposed change').run()
    held = submit(app)
    assert held.supervision.assessment.status == 'available', held.supervision.assessment.error
    assert held.supervision.authority.status == 'VERIFIED'
    assert held.status == 'PENDING_APPROVAL'
    assert any(reason.startswith('MODEL_') for reason in held.supervision.decision.violated_policies)
    approved = review(app)
    assert approved.status == 'EXECUTED'
    assert approved.approval.status == 'CONSUMED'
    assert len(app.session_state['gateway'].tool_state()['invocations']) == 1
    destination = ROOT / f'results/stage3a_real_dashboard_{uuid4().hex}.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump({'model_mode': 'REAL_DECISION_2_0', 'submitted': held.model_dump(mode='json'),
            'approved': approved.model_dump(mode='json'),
            'audit_file': str(app.session_state['audit_trail'].path),
            'audit': app.session_state['audit_trail'].records(),
            'tool_state': app.session_state['gateway'].tool_state()}, stream, indent=2)
    print(f'Actual Streamlit approval demonstration: {destination}')
