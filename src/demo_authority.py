"""Trusted demo operator provisioning, never inferred from action descriptions."""
from src.schemas import AgentIdentity, TaskContext
from src.authority_verifier import AuthorityStore, Delegation, PermissionGrant, ACTION_PERMISSIONS

DOMAIN_SYSTEMS = {'Customer Support': 'crm', 'Finance': 'payments', 'HR': 'payroll', 'IT Operations': 'infrastructure'}


def provision_demo_authority(agent: AgentIdentity, context: TaskContext) -> AuthorityStore:
    """UI operator simulates the trusted delegating host, not an enterprise agent.

    Fixed structured boundaries are independent of the proposed action and goal
    text. Production callers must supply authenticated records instead.
    """
    system = DOMAIN_SYSTEMS.get(agent.department)
    scope = Delegation(task_id='demo-task', agent_id=agent.agent_id, original_goal=context.original_goal,
        allowed_actions=('read', 'draft', 'refund', 'transfer', 'update', 'delete', 'export', 'configure'),
        target_systems=(system,) if system else (), environments=('development', 'staging', 'production'),
        data_classifications=('public', 'internal', 'confidential', 'restricted'),
        max_amount=agent.autonomous_financial_limit, allow_external_effect=False, allow_irreversible=False)
    # Explicitly materialize only the intersection authorized by the old demo.
    # This is trusted operator provisioning, not inference from proposed text.
    grants = [PermissionGrant(grant_id=f'demo:{action}:{env}:{classification}', agent_id=agent.agent_id,
        tool_name=f'enterprise.{action}', action_type=action, target_system=system,
        environment=env, data_classification=classification, max_amount=scope.max_amount)
        for action in scope.allowed_actions if ACTION_PERMISSIONS[action] in agent.permissions
        for env in scope.environments for classification in scope.data_classifications
        if system and (classification != 'restricted' or 'restricted_data' in agent.permissions)]
    return AuthorityStore([agent], [scope], grants)
