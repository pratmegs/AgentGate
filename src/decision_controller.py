"""Model-driven supervisory decisions, bounded by verified authority."""
from src.schemas import PolicyDecision, Decision
from src.authority_verifier import AuthorityVerification
from src.assessment import ModelAssessment
from src.decision_model import validate_answers
from src.schemas import StrictModel
from pydantic import ConfigDict, Field
from src.observability import trace, VERBOSE


class ControllerConfig(StrictModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    # Experimental and uncalibrated: not a validated confidence guarantee.
    probability_threshold: float = Field(default=0.6, ge=0, le=1, allow_inf_nan=False)
    approver_role: str = Field(default='supervisor', min_length=1)


class DecisionController:
    def __init__(self, config: ControllerConfig | None = None):
        self.config = config or ControllerConfig()

    def decide(self, authority: AuthorityVerification, assessment: ModelAssessment) -> PolicyDecision:
        trace('CONTROLLER', 'Receiving verified authority and model assessment.', level=VERBOSE,
              status=authority.status)
        trace('CONTROLLER', 'Checking conflicts and applying decision precedence.', level=VERBOSE,
              threshold=self.config.probability_threshold)
        reasons = list(authority.reasons)
        risk = 'UNKNOWN'
        severity = None
        answers = None
        if assessment.status == 'available':
            try:
                answers = validate_answers(assessment.advice.get('result'))
                risk = answers['consequence']['choice']
                severity = risk
            except Exception:
                reasons.append('INVALID_MODEL_OUTPUT')
        if authority.status == 'PROHIBITED':
            decision = Decision.BLOCK
            explanation = 'Verified authority prohibits this tool call; model judgments cannot override the prohibition.'
        elif authority.status != 'VERIFIED' or authority.agent is None or authority.context is None or authority.delegation is None or not authority.applicable_grants:
            decision = Decision.REQUIRE_APPROVAL
            reasons.append('AUTHORIZATION_UNRESOLVED')
            explanation = 'Trusted authorization is unresolved. Human review and a valid delegation are required.'
        elif answers is None:
            decision = Decision.REQUIRE_APPROVAL
            reasons.append('MODEL_ASSESSMENT_UNAVAILABLE')
            explanation = 'A valid model assessment is required for autonomous authorization.'
        else:
            checks = {
                'MODEL_TASK_MISALIGNMENT': answers['task_alignment']['choice'] == 'NO',
                'MODEL_AUTHORITY_DRIFT': answers['authority_drift']['choice'] == 'YES',
                'MODEL_OUTSIDE_TASK_AUTHORITY': answers['within_authority']['choice'] == 'NO',
                'MODEL_HUMAN_APPROVAL': answers['human_approval']['choice'] == 'YES',
                'MODEL_UNSUITABLE_AUTONOMY': answers['autonomous']['choice'] == 'NO',
                'MODEL_HIGH_CONSEQUENCE': risk in {'HIGH', 'CRITICAL'},
                'MODEL_UNCERTAINTY': any(a['probabilities'][a['choice']] < self.config.probability_threshold for a in answers.values()),
                'MODEL_CONTRADICTION': answers['autonomous']['choice'] == 'YES' and (
                    answers['human_approval']['choice'] == 'YES' or answers['task_alignment']['choice'] == 'NO'
                    or answers['authority_drift']['choice'] == 'YES' or answers['within_authority']['choice'] == 'NO'
                    or risk in {'HIGH', 'CRITICAL'}),
            }
            reasons.extend(name for name, triggered in checks.items() if triggered)
            decision = Decision.REQUIRE_APPROVAL if reasons else Decision.ALLOW
            explanation = ('Decision 2.0 flags risk, scope expansion, approval need, or uncertainty; human review is required.'
                if reasons else 'Trusted authority is verified and Decision 2.0 supports aligned, bounded autonomous execution.')
        trace('CONTROLLER', 'Final supervisory decision.', decision=decision.value, reasons=list(dict.fromkeys(reasons)))
        return PolicyDecision(decision=decision, risk_level=risk, violated_policies=list(dict.fromkeys(reasons)),
            explanation=explanation, required_approver=self.config.approver_role if decision == Decision.REQUIRE_APPROVAL else None,
            authorization_outcome=authority.status, model_consequence_severity=severity)
