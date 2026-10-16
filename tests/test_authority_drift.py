from src.authority_drift import check_authority_drift
from src.schemas import TaskContext, RequestedAction
from src.policy_engine import PolicyEngine
from src.schemas import AgentIdentity, Decision


def test_investigation_cannot_configure_production():
    context = TaskContext(original_goal='Investigate why customer refunds are delayed.',
        requested_action=RequestedAction(action_type='configure', description='Modify the production payment configuration.', target_system='payments', environment='production', data_classification='internal', reversible=True, external_effect=False))
    assert check_authority_drift(context).exceeds_authority
    agent = AgentIdentity(agent_id='a', role='Operator', department='Finance', permissions=['configure'], autonomous_financial_limit=100)
    assert PolicyEngine().evaluate(agent, context).decision == Decision.REQUIRE_APPROVAL


def test_explicit_delegation():
    context = TaskContext(original_goal='Update the payment configuration.', requested_action=RequestedAction(action_type='configure', description='Update config', target_system='payments', environment='staging', data_classification='internal', reversible=True, external_effect=False))
    assert not check_authority_drift(context).exceeds_authority
