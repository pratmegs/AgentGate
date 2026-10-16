from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock
import pytest
from src import decision_model
from src.action_interceptor import ProposedToolCall
from src.approval_manager import ApproverIdentity
from src.audit_trail import AuditTrail
from src.authority_verifier import AuthorityStore, Delegation, PermissionGrant
from src.decision_controller import ControllerConfig
from src.execution_gateway import ExecutionGateway, GatewayConfig
from src.schemas import AgentIdentity, RequestedAction


@pytest.fixture
def gateway_setup(monkeypatch, advice_factory, tmp_path):
    agent = AgentIdentity(agent_id='finance', role='Finance operator', department='Finance',
        permissions=['read', 'refund', 'configure'], autonomous_financial_limit=100)
    delegation = Delegation(task_id='refund-task', agent_id='finance', original_goal='Process refunds and inspect payment records.',
        allowed_actions=('read', 'refund', 'configure'), target_systems=('payments',),
        environments=('staging',), data_classifications=('internal',), max_amount=100)
    grants = [PermissionGrant(grant_id=action, agent_id='finance', tool_name=f'enterprise.{action}',
        action_type=action, target_system='payments', environment='staging',
        data_classification='internal', max_amount=100) for action in ('read', 'refund', 'configure')]
    store = AuthorityStore([agent], [delegation], grants)
    audit = AuditTrail(tmp_path / 'audit.jsonl')
    now = [datetime.now(timezone.utc)]
    gateway = ExecutionGateway(authority_store=store, agent_credentials={'agent-token': 'finance'},
        approver_credentials={'approver-token': ApproverIdentity(approver_id='manager', roles=('supervisor',)),
                              'viewer-token': ApproverIdentity(approver_id='viewer', roles=('viewer',))},
        audit=audit, clock=lambda: now[0], config=GatewayConfig(approval_ttl_seconds=60))
    action = RequestedAction(action_type='refund', description='Issue mock refund.', target_system='payments',
        amount=25, environment='staging', data_classification='internal', reversible=True, external_effect=False)
    call = ProposedToolCall(tool_name='enterprise.refund', arguments=action)
    monkeypatch.setattr(decision_model, 'get_advice', lambda *args: advice_factory())
    return gateway, store, audit, call, now


def submit(setup, **kwargs):
    gateway, _, _, call, _ = setup
    return gateway.submit(agent_credential='agent-token', task_id='refund-task', tool_call=call, **kwargs)


def approve(setup, outcome, **kwargs):
    gateway, _, _, call, _ = setup
    arguments = dict(approval_id=outcome.approval.approval_id, approver_credential='approver-token',
                     interception_id=outcome.approval.interception_id, tool_call=call)
    arguments.update(kwargs)
    return gateway.approve(**arguments)


def hold(setup, monkeypatch, advice_factory):
    monkeypatch.setattr(decision_model, 'get_advice', lambda *args: advice_factory(human_approval='YES', autonomous='NO'))
    return submit(setup)


def test_allow_executes_correct_tool(gateway_setup):
    outcome = submit(gateway_setup)
    assert outcome.status == 'EXECUTED'
    assert outcome.execution_result['tool_name'] == 'enterprise.refund'
    assert gateway_setup[0].tool_state()['state']['payments']['refunds'] == 1


def test_block_never_executes_and_preserves_low_model_consequence(gateway_setup):
    gateway_setup[3].arguments.amount = 101
    outcome = submit(gateway_setup)
    assert outcome.status == 'BLOCKED'
    assert outcome.supervision.decision.authorization_outcome == 'PROHIBITED'
    assert outcome.supervision.decision.model_consequence_severity == 'LOW'
    assert outcome.supervision.decision.risk_level == 'LOW'
    assert not gateway_setup[0].tool_state()['invocations']


def test_approval_holds_and_releases_exact_action(gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    assert pending.status == 'PENDING_APPROVAL'
    assert not gateway_setup[0].tool_state()['invocations']
    assert pending.approval.delegation_version == 1
    outcome = approve(gateway_setup, pending)
    assert outcome.status == 'EXECUTED'
    assert outcome.approval.status == 'CONSUMED'
    assert outcome.approval.approved_by == 'manager'
    assert json.loads(outcome.approval.execution_outcome_json)['status'] == 'SUCCEEDED'


@pytest.mark.parametrize('credential', ['self-asserted-supervisor', 'viewer-token'])
def test_unauthorized_approver_cannot_release(credential, gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    assert approve(gateway_setup, pending, approver_credential=credential).reason == 'UNAUTHORIZED_APPROVER'
    assert gateway_setup[0].get_approval(pending.approval.approval_id).status == 'PENDING'
    assert not gateway_setup[0].tool_state()['invocations']


def test_changed_arguments_invalidate_approval(gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    changed = gateway_setup[3].model_copy(deep=True)
    changed.arguments.amount = 26
    outcome = approve(gateway_setup, pending, tool_call=changed)
    assert outcome.reason == 'APPROVED_ACTION_CHANGED'
    assert outcome.approval.status == 'REJECTED'
    assert not gateway_setup[0].tool_state()['invocations']


def test_interception_mismatch_invalidates_approval(gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    assert approve(gateway_setup, pending, interception_id='different-request').reason == 'APPROVAL_REQUEST_MISMATCH'
    assert not gateway_setup[0].tool_state()['invocations']


def test_expired_approval_cannot_execute(gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    gateway_setup[4][0] += timedelta(seconds=60)
    outcome = approve(gateway_setup, pending)
    assert outcome.status == 'EXPIRED'
    assert outcome.approval.status == 'EXPIRED'
    assert not gateway_setup[0].tool_state()['invocations']


def test_replay_cannot_execute_twice(gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    assert approve(gateway_setup, pending).status == 'EXECUTED'
    assert approve(gateway_setup, pending).reason == 'APPROVAL_NOT_PENDING'
    assert len(gateway_setup[0].tool_state()['invocations']) == 1


def test_concurrent_approval_executes_once(gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: approve(gateway_setup, pending), range(8)))
    assert sum(o.status == 'EXECUTED' for o in outcomes) == 1
    assert len(gateway_setup[0].tool_state()['invocations']) == 1


@pytest.mark.parametrize('revoke', ['grant', 'permission', 'delegation'])
def test_revoked_authority_prevents_approved_execution(revoke, gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    store = gateway_setup[1]
    agent, delegation = store.resolve('finance', 'refund-task')
    if revoke == 'grant':
        store.revoke_grant('refund')
    elif revoke == 'permission':
        store.replace_agent(agent.model_copy(update={'permissions': ['read']}))
    else:
        store.replace_delegation(delegation.model_copy(update={'allowed_actions': ('read',)}))
    outcome = approve(gateway_setup, pending)
    assert outcome.status == 'REJECTED'
    assert outcome.approval.status == 'REJECTED'
    assert not gateway_setup[0].tool_state()['invocations']


def test_delegation_version_change_invalidates_snapshot(gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    _, delegation = gateway_setup[1].resolve('finance', 'refund-task')
    gateway_setup[1].replace_delegation(delegation)
    assert approve(gateway_setup, pending).reason == 'TRUSTED_AUTHORIZATION_CONTEXT_CHANGED'
    assert not gateway_setup[0].tool_state()['invocations']


def test_revocation_during_model_inference_prevents_allow_execution(gateway_setup, monkeypatch, advice_factory):
    def infer(*args):
        gateway_setup[1].revoke_grant('refund')
        return advice_factory()
    monkeypatch.setattr(decision_model, 'get_advice', infer)
    assert submit(gateway_setup).status == 'REJECTED'
    assert not gateway_setup[0].tool_state()['invocations']


@pytest.mark.parametrize('bad_output', [{'status': 'unavailable'}, {'status': 'unavailable', 'result': None},
    {'status': 'available', 'result': {'answers': {}}}])
def test_missing_model_never_executes_autonomously(gateway_setup, monkeypatch, bad_output):
    monkeypatch.setattr(decision_model, 'get_advice', lambda *args: bad_output)
    pending = submit(gateway_setup)
    assert pending.status == 'PENDING_APPROVAL'
    assert not gateway_setup[0].tool_state()['invocations']


def test_unresolved_original_authority_cannot_be_released(gateway_setup, monkeypatch):
    for grant in gateway_setup[1].grants_for('finance'):
        gateway_setup[1].revoke_grant(grant.grant_id)
    pending = submit(gateway_setup, use_model=False)
    assert pending.supervision.authority.status == 'UNRESOLVED'
    assert approve(gateway_setup, pending).reason == 'ORIGINAL_AUTHORITY_UNRESOLVED'
    assert not gateway_setup[0].tool_state()['invocations']


def test_forged_decision_and_direct_tool_calls_are_rejected(gateway_setup):
    gateway, _, _, call, _ = gateway_setup
    with pytest.raises(TypeError):
        gateway.submit(agent_credential='agent-token', task_id='refund-task', tool_call=call, decision='ALLOW')
    with pytest.raises(PermissionError):
        gateway._tools._invoke(object(), call)
    with pytest.raises(PermissionError):
        gateway._execute(object(), 'forged-request', None)
    assert gateway.submit(agent_credential='finance', task_id='refund-task', tool_call=call).reason == 'UNKNOWN_AGENT_CREDENTIAL'
    assert not gateway.tool_state()['invocations']


def test_mandatory_prohibition_cannot_be_approved(gateway_setup):
    gateway_setup[3].arguments.data_classification = 'restricted'
    blocked = submit(gateway_setup)
    assert blocked.status == 'BLOCKED'
    assert blocked.approval is None
    assert gateway_setup[0].approve(approval_id=blocked.request_id, approver_credential='approver-token',
        interception_id=blocked.supervision.intercepted.interception_id, tool_call=gateway_setup[3]).reason == 'UNKNOWN_APPROVAL'
    assert not gateway_setup[0].tool_state()['invocations']


def test_idempotent_submissions_and_changed_retries(gateway_setup):
    first = submit(gateway_setup, idempotency_key='operation-1')
    second = submit(gateway_setup, idempotency_key='operation-1')
    assert first == second
    gateway_setup[3].arguments.amount = 30
    assert submit(gateway_setup, idempotency_key='operation-1').reason == 'IDEMPOTENCY_REQUEST_CHANGED'
    assert len(gateway_setup[0].tool_state()['invocations']) == 1


def test_audit_captures_lifecycle_without_payloads_or_credentials(gateway_setup, monkeypatch, advice_factory):
    gateway_setup[3].arguments.description = 'secret-password=DO_NOT_LOG'
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    approve(gateway_setup, pending)
    records = gateway_setup[2].records()
    events = [r['event'] for r in records]
    assert events == ['action_proposed', 'action_intercepted', 'authority_verified', 'model_assessment_received',
        'governance_decision', 'approval_requested', 'approval_authority_rechecked', 'approval_granted',
        'authority_rechecked', 'approval_consumed', 'tool_execution_started', 'tool_executed']
    assert all(r['timestamp'] and r['request_id'] for r in records)
    assert 'probabilities' in json.dumps(records)
    text = gateway_setup[2].path.read_text()
    assert 'DO_NOT_LOG' not in text and 'agent-token' not in text and 'approver-token' not in text
    assert len(text.splitlines()) == len(records)


def test_human_rejection_and_lazy_expiration(gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    rejected = gateway_setup[0].reject_approval(approval_id=pending.approval.approval_id, approver_credential='approver-token')
    assert rejected.approval.status == 'REJECTED'
    second = hold(gateway_setup, monkeypatch, advice_factory)
    gateway_setup[4][0] += timedelta(seconds=61)
    assert gateway_setup[0].get_approval(second.approval.approval_id).status == 'EXPIRED'
    assert not gateway_setup[0].tool_state()['invocations']


def test_tool_failure_consumes_approval_without_retry(gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    failing = Mock(side_effect=RuntimeError('tool failure'))
    monkeypatch.setattr(gateway_setup[0]._tools, '_invoke', failing)
    first = approve(gateway_setup, pending)
    assert first.status == 'FAILED' and first.approval.status == 'CONSUMED'
    approve(gateway_setup, pending)
    assert failing.call_count == 1


def test_audit_failure_before_execution_is_fail_closed(gateway_setup, monkeypatch):
    monkeypatch.setattr(gateway_setup[2], 'record', Mock(side_effect=OSError('disk full')))
    with pytest.raises(OSError):
        submit(gateway_setup, idempotency_key='audit-failure')
    assert submit(gateway_setup, idempotency_key='audit-failure').status == 'FAILED'
    assert not gateway_setup[0].tool_state()['invocations']


def test_probability_threshold_is_configurable_and_contradictions_hold(gateway_setup, monkeypatch, advice_factory):
    from src.decision_controller import DecisionController
    from src.assessment import ModelAssessment
    pending = submit(gateway_setup)
    authority = pending.supervision.authority
    advice = advice_factory()
    advice['result']['answers']['task_alignment']['probabilities'] = {'YES': .55, 'NO': .45}
    assessment = ModelAssessment(status='available', advice=advice)
    assert DecisionController().decide(authority, assessment).decision == 'REQUIRE_APPROVAL'
    assert DecisionController(ControllerConfig(probability_threshold=.5)).decide(authority, assessment).decision == 'ALLOW'
    conflict = ModelAssessment(status='available', advice=advice_factory(human_approval='YES', autonomous='YES'))
    assert 'MODEL_CONTRADICTION' in DecisionController().decide(authority, conflict).violated_policies


def test_scopes_cannot_be_joined_across_grants(gateway_setup):
    store = gateway_setup[1]
    agent, delegation = store.resolve('finance', 'refund-task')
    other = [PermissionGrant(grant_id='wrong-environment', agent_id='finance', tool_name='enterprise.refund',
        action_type='refund', target_system='payments', environment='production', data_classification='internal', max_amount=100),
        PermissionGrant(grant_id='wrong-tool', agent_id='finance', tool_name='enterprise.read', action_type='read',
        target_system='payments', environment='staging', data_classification='internal', max_amount=100)]
    gateway_setup[0]._store = AuthorityStore([agent], [delegation], other)
    blocked = submit(gateway_setup)
    assert blocked.status == 'BLOCKED'
    assert 'NO_APPLICABLE_SCOPED_GRANT' in blocked.supervision.authority.reasons
    assert not gateway_setup[0].tool_state()['invocations']


def test_returned_outcomes_and_approval_snapshots_cannot_change_authority(gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    pending.supervision.decision.decision = 'ALLOW'
    pending.supervision.intercepted.tool_call.arguments.amount = 1
    with pytest.raises(ValueError):
        pending.approval.tool_call_json = '{}'
    outcome = approve(gateway_setup, pending)
    assert outcome.status == 'EXECUTED'
    assert outcome.supervision.intercepted.tool_call.arguments.amount == 25


@pytest.mark.parametrize('domain,system,action', [('Customer Support', 'crm', 'read'),
    ('Finance', 'payments', 'refund'), ('HR', 'payroll', 'update'), ('IT Operations', 'infrastructure', 'configure')])
def test_registered_domain_tools(domain, system, action, monkeypatch, advice_factory, tmp_path):
    agent = AgentIdentity(agent_id=domain, role='Operator', department=domain, permissions=[action], autonomous_financial_limit=100)
    delegation = Delegation(task_id='demo', agent_id=domain, original_goal=f'Perform mock {action}.',
        allowed_actions=(action,), target_systems=(system,), environments=('staging',), data_classifications=('internal',), max_amount=100)
    grant = PermissionGrant(grant_id='single-scope', agent_id=domain, tool_name=f'enterprise.{action}', action_type=action,
        target_system=system, environment='staging', data_classification='internal', max_amount=100)
    call = ProposedToolCall(tool_name=f'enterprise.{action}', arguments=RequestedAction(action_type=action,
        description=f'Simulated {action}', target_system=system, amount=25 if action == 'refund' else None,
        environment='staging', data_classification='internal', reversible=True, external_effect=False))
    monkeypatch.setattr(decision_model, 'get_advice', lambda *args: advice_factory())
    gateway = ExecutionGateway(authority_store=AuthorityStore([agent], [delegation], [grant]),
        agent_credentials={'agent': domain}, approver_credentials={}, audit=AuditTrail(tmp_path / 'domain.jsonl'))
    outcome = gateway.submit(agent_credential='agent', task_id='demo', tool_call=call)
    assert outcome.status == 'EXECUTED'
    assert outcome.execution_result['tool_name'] == f'enterprise.{action}'
    assert outcome.execution_result['target_system'] == system


def test_expiry_during_audit_does_not_invoke_tool(gateway_setup, monkeypatch, advice_factory):
    pending = hold(gateway_setup, monkeypatch, advice_factory)
    original = gateway_setup[2].record
    def record(event, *args, **kwargs):
        original(event, *args, **kwargs)
        if event == 'tool_execution_started':
            gateway_setup[4][0] += timedelta(seconds=61)
    monkeypatch.setattr(gateway_setup[2], 'record', record)
    outcome = approve(gateway_setup, pending)
    assert outcome.status == 'EXPIRED'
    assert not gateway_setup[0].tool_state()['invocations']


def test_post_execution_audit_failure_does_not_make_retry_execute_again(gateway_setup, monkeypatch):
    original = gateway_setup[2].record
    def record(event, *args, **kwargs):
        if event == 'tool_executed':
            raise OSError('disk full after simulated execution')
        original(event, *args, **kwargs)
    monkeypatch.setattr(gateway_setup[2], 'record', record)
    with pytest.raises(OSError):
        submit(gateway_setup, idempotency_key='post-execution-audit')
    assert len(gateway_setup[0].tool_state()['invocations']) == 1
    assert submit(gateway_setup, idempotency_key='post-execution-audit').status == 'FAILED'
    assert len(gateway_setup[0].tool_state()['invocations']) == 1


def test_terminal_trace_preserves_replay_and_never_logs_credentials(gateway_setup, monkeypatch, advice_factory):
    import io
    from src.observability import configure_tracing
    stream = io.StringIO()
    configure_tracing('VERBOSE', stream=stream, force_terminal=False)
    try:
        gateway_setup[3].arguments.description = 'SENSITIVE_TOOL_ARGUMENT'
        pending = hold(gateway_setup, monkeypatch, advice_factory)
        first = approve(gateway_setup, pending)
        second = approve(gateway_setup, pending)
        assert first.status == 'EXECUTED' and second.reason == 'APPROVAL_NOT_PENDING'
        assert len(gateway_setup[0].tool_state()['invocations']) == 1
        output = stream.getvalue()
        assert pending.request_id in output and pending.approval.approval_id in output
        assert 'SENSITIVE_TOOL_ARGUMENT' not in output
        assert 'agent-token' not in output and 'approver-token' not in output
        assert 'Approval consumed' in output and 'Execution rejected' in output
        assert [r['event'] for r in gateway_setup[2].records()].count('tool_executed') == 1
    finally:
        configure_tracing('NORMAL', force_terminal=False)
