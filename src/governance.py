"""Trusted supervisory orchestration, separate from gateway execution."""
from src.schemas import AgentIdentity, TaskContext, PolicyDecision, StrictModel
from src.action_interceptor import ActionInterceptor, ProposedToolCall, InterceptedAction
from src.authority_verifier import AuthorityStore, AuthorityVerifier, AuthorityVerification
from src.assessment import DecisionAssessmentLayer, ModelAssessment
from src.decision_controller import DecisionController, ControllerConfig
from src.policy_engine import PolicyEngine
from src.observability import trace, traced_request, VERBOSE


class SupervisoryResult(StrictModel):
    intercepted: InterceptedAction
    authority: AuthorityVerification
    assessment: ModelAssessment
    decision: PolicyDecision


@traced_request
def supervise(*, agent_id: str, task_id: str, tool_call: ProposedToolCall,
              authority_store: AuthorityStore, use_model: bool = True,
              controller_config: ControllerConfig | None = None) -> SupervisoryResult:
    trace('WORKFLOW', 'Supervisory workflow started.', level=VERBOSE, agent_id=agent_id, task_id=task_id)
    with authority_store.locked():
        _, delegation = authority_store.resolve(agent_id, task_id)
        trusted_context = TaskContext(original_goal=delegation.original_goal,
            requested_action=tool_call.arguments) if delegation else None
        captured = ActionInterceptor().capture(agent_id=agent_id, task_id=task_id,
            tool_call=tool_call, delegated_context=trusted_context)
        authority = AuthorityVerifier().verify(captured, authority_store)
    if authority.agent is not None and authority.context is not None:
        assessment = DecisionAssessmentLayer().assess(authority.agent, authority.context, enabled=use_model)
    else:
        assessment = ModelAssessment(status='unavailable', error='Trusted task context unavailable')
    decision = DecisionController(controller_config).decide(authority, assessment)
    trace('WORKFLOW', 'Supervisory workflow complete; forwarding decision.', level=VERBOSE, decision=decision.decision.value)
    return SupervisoryResult(intercepted=captured, authority=authority, assessment=assessment, decision=decision)


def evaluate(agent: AgentIdentity, context: TaskContext, use_model: bool = True,
             authority_store: AuthorityStore | None = None, task_id: str = 'unresolved') -> tuple[PolicyDecision, dict]:
    """Compatibility facade. Supplied text and identity never provision authority."""
    result = supervise(agent_id=agent.agent_id, task_id=task_id,
        tool_call=ProposedToolCall(tool_name=f'enterprise.{context.requested_action.action_type}',
            arguments=context.requested_action, claimed_task_description=context.original_goal),
        authority_store=authority_store if authority_store is not None else AuthorityStore([], []), use_model=use_model)
    details = result.assessment.advice or {'status': result.assessment.status, 'error': result.assessment.error}
    return result.decision, {**details, 'supervision': result.model_dump(mode='json')}


def evaluate_baseline(agent: AgentIdentity, context: TaskContext, use_model: bool = False) -> tuple[PolicyDecision, dict]:
    """Explicit legacy benchmark path; never used by the supervisory controller."""
    advice = {'status': 'disabled'}
    if use_model:
        from src.decision_model import get_advice
        advice = get_advice(agent, context)
    return PolicyEngine().evaluate(agent, context), advice
