import pytest
from src.schemas import AgentIdentity, RequestedAction, TaskContext, Decision
from src.policy_engine import PolicyEngine


def inputs(**changes):
    agent = AgentIdentity(agent_id='demo', role='Analyst', department='Finance',
                          permissions=['read', 'refund'], autonomous_financial_limit=100)
    action = dict(action_type='refund', description='Refund customer', target_system='payments',
                  amount=50, environment='staging', data_classification='internal',
                  reversible=True, external_effect=False)
    action.update(changes)
    return agent, TaskContext(original_goal='Process customer refund', requested_action=RequestedAction(**action))


@pytest.mark.parametrize('changes,expected', [({}, Decision.ALLOW),
    ({'amount': 101}, Decision.REQUIRE_APPROVAL), ({'amount': 100}, Decision.ALLOW),
    ({'amount': None}, Decision.BLOCK), ({'data_classification': 'restricted'}, Decision.BLOCK),
    ({'environment': 'production', 'reversible': False}, Decision.REQUIRE_APPROVAL),
    ({'target_system': 'payroll'}, Decision.REQUIRE_APPROVAL),
    ({'action_type': 'grant_admin'}, Decision.BLOCK), ({'target_system': 'unknown'}, Decision.BLOCK),
    ({'amount': 1000, 'data_classification': 'restricted'}, Decision.BLOCK)])
def test_rules(changes, expected):
    assert PolicyEngine().evaluate(*inputs(**changes)).decision == expected


def test_negative_amount():
    with pytest.raises(ValueError):
        inputs(amount=-1)
