import pytest
from src.schemas import AgentIdentity, RequestedAction
from src.authority_verifier import AuthorityStore, Delegation, PermissionGrant
from src.action_interceptor import ProposedToolCall
from src.decision_model import QUESTIONS


def pytest_addoption(parser):
    parser.addoption('--real-model', action='store_true', default=False, help='Run cached real Decision 2.0 integration')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--real-model'):
        for item in items:
            if 'real_model' in item.keywords:
                item.add_marker(pytest.mark.skip(reason='Use --real-model for actual cached inference'))


@pytest.fixture
def supervisory_input():
    agent = AgentIdentity(agent_id='it-agent', role='Operator', department='IT Operations',
        permissions=['read', 'update', 'configure', 'refund'], autonomous_financial_limit=100)
    delegation = Delegation(task_id='task-1', agent_id=agent.agent_id,
        original_goal='Investigate service latency and report the cause.',
        allowed_actions=('read', 'update', 'configure', 'refund'), target_systems=('infrastructure',),
        environments=('staging', 'production'), data_classifications=('public', 'internal'), max_amount=100)
    action = RequestedAction(action_type='read', description='Read service latency report.',
        target_system='infrastructure', environment='staging', data_classification='internal',
        reversible=True, external_effect=False)
    grants = [PermissionGrant(grant_id=f'it:{a}:{e}:{c}', agent_id=agent.agent_id,
        tool_name=f'enterprise.{a}', action_type=a, target_system='infrastructure',
        environment=e, data_classification=c, max_amount=100)
        for a in ('read', 'update', 'configure', 'refund')
        for e in ('staging', 'production') for c in ('public', 'internal')]
    return dict(agent_id=agent.agent_id, task_id=delegation.task_id,
        tool_call=ProposedToolCall(tool_name='enterprise.read', arguments=action),
        authority_store=AuthorityStore([agent], [delegation], grants))


@pytest.fixture
def advice_factory():
    def create(**changes):
        choices = dict(task_alignment='YES', authority_drift='NO', within_authority='YES',
                       human_approval='NO', consequence='LOW', autonomous='YES')
        choices.update(changes)
        answers = {key: {'type': 'choice', 'choice': choices[key],
            'probabilities': {label: .9 if label == choices[key] else .1 / (len(q['criteria']) - 1) for label in q['criteria']}}
            for key, q in QUESTIONS.items()}
        return {'status': 'available', 'result': {'answers': answers}}
    return create
