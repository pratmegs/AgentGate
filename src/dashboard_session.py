"""Fixed host provisioning and persistent dashboard bookkeeping."""
from dataclasses import dataclass
from pathlib import Path
from secrets import token_urlsafe
from uuid import uuid4
from src.action_interceptor import ProposedToolCall
from src.approval_manager import ApproverIdentity
from src.audit_trail import AuditTrail
from src.authority_verifier import AuthorityStore, Delegation, PermissionGrant
from src.execution_gateway import ExecutionGateway, GatewayOutcome
from src.schemas import AgentIdentity, RequestedAction


@dataclass(frozen=True)
class DemoScenario:
    name: str
    department: str
    agent_id: str
    task_id: str
    proposal: ProposedToolCall


def provision_dashboard():
    """Widget proposals never participate in trusted provisioning."""
    specs = [
        ('Customer Support', 'support-demo', 'Support specialist', 'crm', ('read', 'refund'),
         100, 'Resolve the customer ticket and refund at most 100 if warranted.'),
        ('Finance', 'finance-demo', 'Finance analyst', 'payments', ('read', 'configure'),
         1000, 'Investigate payment discrepancies and report their cause.'),
        ('HR', 'hr-demo', 'HR coordinator', 'payroll', ('read', 'update'),
         0, 'Review and correct the employee record in staging.'),
        ('IT Operations', 'it-demo', 'IT operator', 'infrastructure', ('read', 'configure'),
         0, 'Investigate service latency and report the cause.'),
    ]
    agents, delegations, grants, scenarios = [], [], [], []
    for department, agent_id, role, system, actions, limit, goal in specs:
        task_id = agent_id + '-task'
        agents.append(AgentIdentity(agent_id=agent_id, role=role, department=department,
            permissions=list(actions), autonomous_financial_limit=limit))
        delegations.append(Delegation(task_id=task_id, agent_id=agent_id, original_goal=goal,
            allowed_actions=actions, target_systems=(system,), environments=('staging',),
            data_classifications=('internal',), max_amount=limit))
        for action in actions:
            grants.append(PermissionGrant(grant_id=f'{agent_id}:{action}', agent_id=agent_id,
                tool_name=f'enterprise.{action}', action_type=action, target_system=system,
                environment='staging', data_classification='internal', max_amount=limit))
        def scenario(name, action, description, amount=None):
            return DemoScenario(name, department, agent_id, task_id,
                ProposedToolCall(tool_name=f'enterprise.{action}', arguments=RequestedAction(
                    action_type=action, description=description, target_system=system,
                    amount=amount, environment='staging', data_classification='internal',
                    reversible=True, external_effect=False)))
        scenarios.append(scenario(f'{department}: read records', 'read', f'Read relevant {system} records.'))
        if department == 'Customer Support':
            scenarios.extend([
                scenario('Customer Support: bounded refund', 'refund', 'Refund 50 for the customer ticket.', 50),
                scenario('Customer Support: excessive refund', 'refund', 'Refund 250 for the customer ticket.', 250),
            ])
        else:
            action = actions[1]
            scenarios.append(scenario(f'{department}: proposed change', action,
                f'Change the {system} configuration.' if action == 'configure' else 'Correct the employee record.'))
    return AuthorityStore(agents, delegations, grants), tuple(scenarios)


SESSION_KEYS = ('gateway', 'authority_store', 'audit_trail', 'demo_scenarios',
                'agent_credentials', 'reviewer_credentials', 'current_scenario',
                'submitted_requests', 'pending_approvals', 'final_outcomes',
                'latest_request_id', 'last_approval_attempt')


def initialize_session(state, audit_directory: Path):
    if 'gateway' in state:
        return
    store, scenarios = provision_dashboard()
    agent_credentials = {s.agent_id: token_urlsafe(32) for s in scenarios}
    reviewers = {'Demo supervisor': (token_urlsafe(32), ApproverIdentity(
        approver_id='demo-supervisor', roles=('supervisor',))),
        'Demo observer (unauthorized)': (token_urlsafe(32), ApproverIdentity(
        approver_id='demo-observer', roles=('observer',)))}
    audit = AuditTrail(audit_directory / f'dashboard-{uuid4()}.jsonl')
    values = dict(gateway=ExecutionGateway(authority_store=store,
        agent_credentials={token: agent for agent, token in agent_credentials.items()},
        approver_credentials={token: identity for token, identity in reviewers.values()}, audit=audit),
        authority_store=store, audit_trail=audit, demo_scenarios=scenarios,
        agent_credentials=agent_credentials, reviewer_credentials=reviewers,
        current_scenario=None, submitted_requests={}, pending_approvals={}, final_outcomes={},
        latest_request_id=None, last_approval_attempt=None)
    for key, value in values.items():
        state[key] = value


def remember_submission(state, outcome: GatewayOutcome):
    state['submitted_requests'][outcome.request_id] = outcome
    state['latest_request_id'] = outcome.request_id
    refresh_requests(state)


def refresh_requests(state):
    pending, final = {}, {}
    for request_id, original in list(state['submitted_requests'].items()):
        outcome = state['gateway'].get_outcome(request_id) or original
        state['submitted_requests'][request_id] = outcome
        if outcome.status == 'PENDING_APPROVAL' and outcome.approval:
            pending[outcome.approval.approval_id] = state['gateway'].get_approval(outcome.approval.approval_id)
        else:
            final[request_id] = outcome
    state['pending_approvals'] = pending
    state['final_outcomes'] = final


def reset_session(state):
    """Called after explicit confirmation; append-only audit files are retained."""
    for key in SESSION_KEYS:
        state.pop(key, None)
