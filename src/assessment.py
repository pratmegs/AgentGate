"""Real model assessment independent of authorization and controller rules."""
from typing import Literal
from pydantic import Field
from src.schemas import StrictModel, AgentIdentity, TaskContext
from src import decision_model
from src.observability import trace, VERBOSE


class ModelAssessment(StrictModel):
    status: Literal['available', 'unavailable', 'disabled']
    advice: dict = Field(default_factory=dict)
    error: str | None = None


class DecisionAssessmentLayer:
    def assess(self, agent: AgentIdentity, context: TaskContext, enabled: bool = True) -> ModelAssessment:
        trace('ASSESSMENT', 'Contextual assessment requested.', level=VERBOSE, model_enabled=enabled)
        if not enabled:
            trace('ASSESSMENT', 'Model disabled. Autonomous authorization unavailable.', status='DISABLED')
            return ModelAssessment(status='disabled', error='Model assessment disabled; autonomous execution is unavailable.')
        try:
            advice = decision_model.get_advice(agent, context)
            if advice.get('status') != 'available':
                trace('ASSESSMENT', 'Model unavailable. Human review remains required.', status='UNAVAILABLE')
                return ModelAssessment(status='unavailable', advice=advice, error=advice.get('error', 'Model unavailable'))
            decision_model.validate_answers(advice.get('result'))
            trace('ASSESSMENT', 'Validated model assessment received.', level=VERBOSE, status='VERIFIED')
            return ModelAssessment(status='available', advice=advice)
        except Exception as exc:
            trace('ASSESSMENT', 'Model assessment failed validation.', status='UNAVAILABLE', error_type=type(exc).__name__)
            return ModelAssessment(status='unavailable', error=f'{type(exc).__name__}: {exc}')
