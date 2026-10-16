"""Workflow demonstrations with explicitly mocked model assessments.

These demonstrate gateway enforcement, not model prediction quality. Actual model
inference is covered separately by the opt-in real gateway integration test.
"""
import json
from pathlib import Path
from secrets import token_urlsafe
import sys
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.action_interceptor import ProposedToolCall
from src.approval_manager import ApproverIdentity
from src.audit_trail import AuditTrail
from src.authority_verifier import AuthorityStore, Delegation, PermissionGrant
from src.decision_model import QUESTIONS
from src.execution_gateway import ExecutionGateway
from src.schemas import AgentIdentity, RequestedAction


def mocked_assessment(needs_review: bool = False) -> dict:
    choices = dict(task_alignment='YES', authority_drift='NO', within_authority='YES',
        consequence='LOW', human_approval='YES' if needs_review else 'NO', autonomous='NO' if needs_review else 'YES')
    answers = {key: {'type': 'choice', 'choice': choices[key], 'probabilities':
        {label: .9 if label == choices[key] else .1 / (len(question['criteria']) - 1) for label in question['criteria']}}
        for key, question in QUESTIONS.items()}
    return {'status': 'available', 'model_id': 'MOCK_FOR_WORKFLOW_DEMO', 'result': {'answers': answers}}


def run() -> Path:
    agent = AgentIdentity(agent_id='finance-demo', role='Finance operator', department='Finance',
        permissions=['refund'], autonomous_financial_limit=100)
    delegation = Delegation(task_id='refund-demo', agent_id=agent.agent_id, original_goal='Issue simulated refunds within the delegated limit.',
        allowed_actions=('refund',), target_systems=('payments',), environments=('staging',),
        data_classifications=('internal',), max_amount=100)
    grant = PermissionGrant(grant_id='refund-staging-internal', agent_id=agent.agent_id,
        tool_name='enterprise.refund', action_type='refund', target_system='payments', environment='staging',
        data_classification='internal', max_amount=100)
    run_id = uuid4().hex
    audit = AuditTrail(ROOT / f'results/stage2_demo_audit_{run_id}.jsonl')
    agent_token, approver_token = token_urlsafe(32), token_urlsafe(32)
    gateway = ExecutionGateway(authority_store=AuthorityStore([agent], [delegation], [grant]),
        agent_credentials={agent_token: agent.agent_id},
        approver_credentials={approver_token: ApproverIdentity(approver_id='trusted-demo-manager', roles=('supervisor',))}, audit=audit)
    call = ProposedToolCall(tool_name='enterprise.refund', arguments=RequestedAction(action_type='refund',
        description='Issue simulated refund.', target_system='payments', amount=25, environment='staging',
        data_classification='internal', reversible=True, external_effect=False))
    with patch('src.decision_model.get_advice', return_value=mocked_assessment()):
        allowed = gateway.submit(agent_credential=agent_token, task_id=delegation.task_id, tool_call=call)
    assert allowed.status == 'EXECUTED'
    call.arguments.amount = 50
    with patch('src.decision_model.get_advice', return_value=mocked_assessment(True)):
        pending = gateway.submit(agent_credential=agent_token, task_id=delegation.task_id, tool_call=call)
    assert pending.status == 'PENDING_APPROVAL' and len(gateway.tool_state()['invocations']) == 1
    approved = gateway.approve(approval_id=pending.approval.approval_id, approver_credential=approver_token,
        interception_id=pending.approval.interception_id, tool_call=call)
    assert approved.status == 'EXECUTED'
    call.arguments.amount = 101
    with patch('src.decision_model.get_advice', return_value=mocked_assessment()):
        blocked = gateway.submit(agent_credential=agent_token, task_id=delegation.task_id, tool_call=call)
    assert blocked.status == 'BLOCKED' and len(gateway.tool_state()['invocations']) == 2
    records = audit.records()
    by_request = {outcome.request_id: [r['event'] for r in records if r['request_id'] == outcome.request_id]
                  for outcome in (allowed, approved, blocked)}
    assert 'tool_executed' in by_request[allowed.request_id]
    assert {'approval_requested', 'approval_granted', 'approval_consumed', 'tool_executed'} <= set(by_request[approved.request_id])
    assert 'tool_rejected' in by_request[blocked.request_id] and 'tool_executed' not in by_request[blocked.request_id]
    output = {'model_mode': 'MOCKED_FOR_WORKFLOW_DEMO', 'allow': allowed.model_dump(mode='json'),
        'pending': pending.model_dump(mode='json'), 'approved': approved.model_dump(mode='json'),
        'block': blocked.model_dump(mode='json'), 'audit': records, 'tool_state': gateway.tool_state()}
    destination = ROOT / f'results/stage2_demo_{run_id}.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps({'trace': str(destination), 'ALLOW': allowed.status, 'REQUIRE_APPROVAL': pending.status,
        'authorized_approval': approved.status, 'BLOCK': blocked.status, 'mock_invocations': 2}, indent=2))
    return destination


if __name__ == '__main__':
    run()
