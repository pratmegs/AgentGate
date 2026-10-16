"""Immutable approval snapshots; mutation belongs to the owning gateway."""
from datetime import datetime
from typing import Literal
from pydantic import ConfigDict
from src.schemas import StrictModel
from src.observability import trace

ApprovalStatus = Literal['PENDING', 'APPROVED', 'REJECTED', 'EXPIRED', 'CONSUMED']


class ApproverIdentity(StrictModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    approver_id: str
    roles: tuple[str, ...]


class ApprovalRequest(StrictModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    approval_id: str
    request_id: str
    interception_id: str
    agent_id: str
    task_id: str
    tool_call_json: str
    action_digest: str
    authorization_context_json: str
    authorization_digest: str
    delegation_version: int | None
    supervisor_decision_json: str
    required_approver_role: str
    status: ApprovalStatus = 'PENDING'
    created_at: datetime
    expires_at: datetime
    updated_at: datetime
    approved_by: str | None = None
    approved_at: datetime | None = None
    execution_outcome_json: str | None = None


class ApprovalManager:
    """No public approval mutation or executor; gateway owns the capability."""
    def __init__(self, capability: object):
        self.__capability = capability
        self._records: dict[str, ApprovalRequest] = {}

    def _check(self, capability: object) -> None:
        if capability is not self.__capability:
            raise PermissionError('Approval mutation requires the owning gateway')

    def _put(self, capability: object, record: ApprovalRequest) -> None:
        self._check(capability)
        previous = self._records.get(record.approval_id)
        self._records[record.approval_id] = record.model_copy(deep=True)
        if previous is None or previous.status != record.status:
            messages = {'PENDING': 'Approval request created. Waiting for human approval.',
                        'APPROVED': 'Authorized human approval received.', 'REJECTED': 'Approval rejected.',
                        'EXPIRED': 'Approval expired.', 'CONSUMED': 'Approval consumed for one execution attempt.'}
            trace('APPROVAL', messages[record.status], request_id=record.request_id,
                  approval_id=record.approval_id, status=record.status)

    def get(self, approval_id: str) -> ApprovalRequest | None:
        record = self._records.get(approval_id)
        return record.model_copy(deep=True) if record else None
