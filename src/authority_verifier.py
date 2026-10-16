"""Focused verification of trusted identity and explicit delegation records."""
from typing import Literal
from threading import RLock
from contextlib import contextmanager
from pydantic import ConfigDict, Field
from src.schemas import StrictModel, AgentIdentity, TaskContext
from src.action_interceptor import InterceptedAction
from src.observability import trace, VERBOSE

# Registry of proposed tool contracts only; these are not executable functions.
TOOL_ACTIONS = {f'enterprise.{action}': action for action in
                ['read', 'draft', 'refund', 'transfer', 'update', 'delete', 'export', 'configure', 'grant_admin']}
ACTION_PERMISSIONS = {action: ('admin' if action == 'grant_admin' else action) for action in TOOL_ACTIONS.values()}


class Delegation(StrictModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    task_id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    agent_id: str = Field(min_length=1)
    original_goal: str = Field(min_length=1)
    allowed_actions: tuple[str, ...]
    target_systems: tuple[str, ...]
    environments: tuple[str, ...]
    data_classifications: tuple[str, ...]
    max_amount: float = Field(ge=0, allow_inf_nan=False)
    allow_external_effect: bool = False
    allow_irreversible: bool = False


class PermissionGrant(StrictModel):
    """One atomic authorization scope; different grants cannot be combined."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    grant_id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    agent_id: str
    tool_name: str
    action_type: str
    target_system: str
    environment: str
    data_classification: str
    max_amount: float = Field(ge=0, allow_inf_nan=False)
    allow_external_effect: bool = False
    allow_irreversible: bool = False


class AuthorityVerification(StrictModel):
    status: Literal['VERIFIED', 'PROHIBITED', 'UNRESOLVED']
    reasons: list[str]
    agent: AgentIdentity | None = None
    context: TaskContext | None = None
    delegation: Delegation | None = None
    applicable_grants: tuple[PermissionGrant, ...] = ()


class AuthorityStore:
    """Provisioned by trusted host code, not passed in a proposed tool call.

    In-memory records demonstrate the trust boundary; authentication is outside
    Stage 1. Copies prevent edits to caller-owned objects from changing grants.
    """
    def __init__(self, agents: list[AgentIdentity], delegations: list[Delegation],
                 grants: list[PermissionGrant] | None = None):
        if len({a.agent_id for a in agents}) != len(agents):
            raise ValueError('Duplicate trusted agent identity')
        if len({d.task_id for d in delegations}) != len(delegations):
            raise ValueError('Duplicate trusted delegation')
        self._agents = {a.agent_id: a.model_copy(deep=True) for a in agents}
        self._delegations = {d.task_id: d.model_copy(deep=True) for d in delegations}
        grants = grants or []
        if len({g.grant_id for g in grants}) != len(grants):
            raise ValueError('Duplicate trusted grant')
        self._grants = {g.grant_id: g.model_copy(deep=True) for g in grants}
        self._lock = RLock()

    @contextmanager
    def locked(self):
        """Serialize trusted changes with the gateway's last check and execution."""
        with self._lock:
            yield

    def revoke_grant(self, grant_id: str) -> None:
        with self._lock:
            self._grants.pop(grant_id, None)

    def replace_agent(self, agent: AgentIdentity) -> None:
        with self._lock:
            self._agents[agent.agent_id] = agent.model_copy(deep=True)

    def replace_delegation(self, delegation: Delegation) -> None:
        with self._lock:
            previous = self._delegations.get(delegation.task_id)
            version = previous.version + 1 if previous else delegation.version
            self._delegations[delegation.task_id] = delegation.model_copy(update={'version': version}, deep=True)

    def grants_for(self, agent_id: str) -> tuple[PermissionGrant, ...]:
        with self._lock:
            return tuple(g.model_copy(deep=True) for _, g in sorted(self._grants.items()) if g.agent_id == agent_id)

    def resolve(self, agent_id: str, task_id: str) -> tuple[AgentIdentity | None, Delegation | None]:
        with self._lock:
            agent, delegation = self._agents.get(agent_id), self._delegations.get(task_id)
            return (agent.model_copy(deep=True) if agent else None,
                    delegation.model_copy(deep=True) if delegation else None)


class AuthorityVerifier:
    def verify(self, intercepted: InterceptedAction, store: AuthorityStore) -> AuthorityVerification:
        trace('AUTHORITY', 'Resolving trusted agent identity and delegation.', level=VERBOSE,
              agent_id=intercepted.agent_id, task_id=intercepted.task_id)
        agent, delegation = store.resolve(intercepted.agent_id, intercepted.task_id)
        if agent is None or delegation is None:
            trace('AUTHORITY', 'Trusted authority could not be resolved.', status='UNRESOLVED')
            return AuthorityVerification(status='UNRESOLVED', reasons=['TRUSTED_AUTHORITY_MISSING'])
        action = intercepted.tool_call.arguments
        context = TaskContext(original_goal=delegation.original_goal, requested_action=action.model_copy(deep=True))
        reasons = []
        trace('AUTHORITY', 'Checking trusted permissions, action and resource scope.', level=VERBOSE,
              tool_name=intercepted.tool_call.tool_name, target_system=action.target_system,
              environment=action.environment, data_classification=action.data_classification)
        if delegation.agent_id != agent.agent_id:
            reasons.append('DELEGATION_AGENT_MISMATCH')
        registered_action = TOOL_ACTIONS.get(intercepted.tool_call.tool_name)
        if registered_action is None or registered_action != action.action_type:
            reasons.append('UNREGISTERED_OR_MISMATCHED_TOOL')
        permission = ACTION_PERMISSIONS.get(action.action_type)
        if permission is None or permission not in agent.permissions:
            reasons.append('MISSING_TRUSTED_PERMISSION')
        if action.action_type not in delegation.allowed_actions:
            reasons.append('OUTSIDE_DELEGATED_ACTIONS')
        if action.target_system not in delegation.target_systems:
            reasons.append('OUTSIDE_DELEGATED_SYSTEMS')
        if action.environment not in delegation.environments:
            reasons.append('OUTSIDE_DELEGATED_ENVIRONMENTS')
        if action.data_classification not in delegation.data_classifications:
            reasons.append('OUTSIDE_DELEGATED_DATA_SCOPE')
        if action.data_classification == 'restricted' and 'restricted_data' not in agent.permissions:
            reasons.append('RESTRICTED_DATA_PERMISSION_MISSING')
        if action.external_effect and (not delegation.allow_external_effect or 'external_effect' not in agent.permissions):
            reasons.append('EXTERNAL_EFFECT_NOT_AUTHORIZED')
        if not action.reversible and not delegation.allow_irreversible:
            reasons.append('IRREVERSIBLE_ACTION_NOT_DELEGATED')
        trace('AUTHORITY', 'Verifying financial boundaries.', level=VERBOSE)
        if action.action_type in {'refund', 'transfer'} and action.amount is None:
            reasons.append('FINANCIAL_AMOUNT_UNRESOLVED')
            status = 'UNRESOLVED'
        else:
            status = 'VERIFIED'
        if action.amount is not None and action.amount > min(agent.autonomous_financial_limit, delegation.max_amount):
            reasons.append('OUTSIDE_TRUSTED_FINANCIAL_BOUNDARY')
        grants = store.grants_for(agent.agent_id)
        trace('AUTHORITY', 'Trusted scoped grants loaded.', level=VERBOSE, grant_count=len(grants))
        applicable = tuple(g for g in grants if
            g.tool_name == intercepted.tool_call.tool_name and g.action_type == action.action_type
            and g.target_system == action.target_system and g.environment == action.environment
            and g.data_classification == action.data_classification
            and (action.amount is None or action.amount <= g.max_amount)
            and (not action.external_effect or g.allow_external_effect)
            and (action.reversible or g.allow_irreversible))
        if not grants:
            reasons.append('SCOPED_GRANTS_MISSING')
            status = 'UNRESOLVED'
        elif not applicable:
            reasons.append('NO_APPLICABLE_SCOPED_GRANT')
        prohibitions = [r for r in reasons if r not in {'FINANCIAL_AMOUNT_UNRESOLVED', 'SCOPED_GRANTS_MISSING'}]
        if prohibitions:
            status = 'PROHIBITED'
        trace('AUTHORITY', 'Authority verification complete.', status=status, reasons=reasons)
        return AuthorityVerification(status=status, reasons=reasons, agent=agent, context=context,
                                     delegation=delegation, applicable_grants=applicable)
