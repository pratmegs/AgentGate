from pathlib import Path
import yaml
from src.schemas import AgentIdentity, TaskContext, PolicyDecision, Decision

ROOT = Path(__file__).resolve().parents[1]


class PolicyEngine:
    """Fail closed on unknown metadata; blocks take precedence over approvals."""

    def __init__(self, policy_path: Path = ROOT / 'policies/enterprise_policy.yaml'):
        self.policy = yaml.safe_load(policy_path.read_text(encoding='utf-8'))

    def evaluate(self, agent: AgentIdentity, context: TaskContext) -> PolicyDecision:
        a, p = context.requested_action, self.policy
        blocks: list[str] = []
        approvals: list[str] = []
        if (agent.department not in p['departments'] or a.target_system not in p['systems']
                or a.action_type not in p['actions'] or a.environment not in p['environments']
                or a.data_classification not in p['classifications']):
            blocks.append('UNKNOWN_METADATA')
        permission = p['actions'].get(a.action_type)
        if permission not in agent.permissions:
            blocks.append('MISSING_ACTION_PERMISSION')
        if a.data_classification == 'restricted' and 'restricted_data' not in agent.permissions:
            blocks.append('RESTRICTED_DATA')
        if a.action_type in p['privileged_actions'] and 'admin' not in agent.permissions:
            blocks.append('UNAUTHORIZED_PRIVILEGED_ACTION')
        if a.action_type in p['financial_actions'] and a.amount is None:
            blocks.append('MISSING_FINANCIAL_AMOUNT')
        if a.amount is not None and a.amount > agent.autonomous_financial_limit:
            approvals.append('FINANCIAL_LIMIT')
        if a.environment == 'production' and not a.reversible:
            approvals.append('IRREVERSIBLE_PRODUCTION')
        if a.target_system in p['systems'] and p['systems'][a.target_system] != agent.department:
            approvals.append('DEPARTMENT_BOUNDARY')
        if a.external_effect and 'external_effect' not in agent.permissions:
            approvals.append('EXTERNAL_EFFECT')
        from src.authority_drift import check_authority_drift
        if check_authority_drift(context).exceeds_authority:
            approvals.append('AUTHORITY_DRIFT')
        decision = Decision.BLOCK if blocks else Decision.REQUIRE_APPROVAL if approvals else Decision.ALLOW
        triggered = blocks + approvals
        return PolicyDecision(decision=decision,
            risk_level='HIGH' if blocks else 'MEDIUM' if approvals else 'LOW',
            violated_policies=triggered,
            explanation='; '.join(triggered) if triggered else 'Action is within explicit permissions and policy limits.',
            required_approver=p['approver'] if decision == Decision.REQUIRE_APPROVAL else None)
