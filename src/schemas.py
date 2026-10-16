from enum import StrEnum
from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field


class Decision(StrEnum):
    ALLOW = 'ALLOW'
    REQUIRE_APPROVAL = 'REQUIRE_APPROVAL'
    BLOCK = 'BLOCK'


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class AgentIdentity(StrictModel):
    agent_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    department: str
    permissions: list[str]
    autonomous_financial_limit: float = Field(ge=0, allow_inf_nan=False)


class RequestedAction(StrictModel):
    action_type: str
    description: str = Field(min_length=1)
    target_system: str
    amount: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    environment: str
    data_classification: str
    reversible: bool
    external_effect: bool


class TaskContext(StrictModel):
    original_goal: str = Field(min_length=1)
    requested_action: RequestedAction


class PolicyDecision(StrictModel):
    decision: Decision
    risk_level: str
    violated_policies: list[str]
    explanation: str
    required_approver: str | None = None
    audit_id: str = Field(default_factory=lambda: str(uuid4()))
    authorization_outcome: Literal['VERIFIED', 'PROHIBITED', 'UNRESOLVED'] | None = None
    model_consequence_severity: Literal['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] | None = None
