import re
from pydantic import BaseModel
from src.schemas import TaskContext


class DriftAssessment(BaseModel):
    exceeds_authority: bool
    reason: str


def check_authority_drift(context: TaskContext) -> DriftAssessment:
    """Conservative heuristic; free-text goals are not a source of permissions."""
    goal = context.original_goal.lower()
    action = context.requested_action
    investigative = bool(re.search(r'\b(investigate|review|analyze|analyse|inspect|diagnose|report|summarize|summarise|audit|monitor)\b', goal))
    mutation = action.action_type not in {'read', 'draft'}
    explicit_execution = bool(re.search(r'\b(execute|process|issue|modify|update|delete|configure|transfer|refund|grant|export|deploy)\b', goal))
    drift = investigative and mutation and not explicit_execution
    return DriftAssessment(exceeds_authority=drift,
        reason='Investigative delegation does not authorize a state-changing action.' if drift else 'No scope expansion detected by the V1 heuristic.')
