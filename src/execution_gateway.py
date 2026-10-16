"""Single trusted entry point for simulated execution and human approval.

Host-provisioned opaque credentials identify agents and approvers. This prototype
does not sandbox arbitrary Python code running inside the trusted host process.
"""
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable, Literal
from uuid import uuid4
from pydantic import Field
from src.schemas import StrictModel
from src.action_interceptor import ProposedToolCall
from src.authority_verifier import AuthorityStore, AuthorityVerifier, AuthorityVerification
from src.decision_controller import ControllerConfig
from src.governance import supervise, SupervisoryResult
from src.approval_manager import ApprovalManager, ApprovalRequest, ApproverIdentity
from src.audit_trail import AuditTrail
from src.mock_tools import MockToolRegistry
from src.observability import trace, traced_request, bind_request, VERBOSE


def canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value: dict) -> str:
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


class GatewayConfig(StrictModel):
    approval_ttl_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)


class GatewayOutcome(StrictModel):
    request_id: str
    status: Literal['EXECUTED', 'PENDING_APPROVAL', 'BLOCKED', 'REJECTED', 'FAILED', 'EXPIRED']
    supervision: SupervisoryResult | None = None
    approval: ApprovalRequest | None = None
    execution_result: dict | None = None
    reason: str | None = None


class ExecutionGateway:
    def __init__(self, *, authority_store: AuthorityStore, agent_credentials: dict[str, str],
                 approver_credentials: dict[str, ApproverIdentity], audit: AuditTrail,
                 config: GatewayConfig | None = None, clock: Callable[[], datetime] | None = None):
        self._store = authority_store
        self._agents = dict(agent_credentials)
        self._approvers = {key: value.model_copy(deep=True) for key, value in approver_credentials.items()}
        if not all(self._agents) or not all(self._approvers):
            raise ValueError('Empty host credential is not allowed')
        self._audit = audit
        self._config = (config or GatewayConfig()).model_copy(deep=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = RLock()
        self.__capability = object()
        self._tools = MockToolRegistry(self.__capability)
        self._approvals = ApprovalManager(self.__capability)
        self._requests: dict[str, str] = {}
        self._outcomes: dict[str, GatewayOutcome] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, str]] = {}
        self._attempted: set[str] = set()

    def _record(self, event: str, request_id: str, supervision: SupervisoryResult | None = None, **evidence) -> None:
        common = {}
        if supervision:
            captured = supervision.intercepted
            common = {'interception_id': captured.interception_id, 'agent_id': captured.agent_id,
                      'task_id': captured.task_id, 'action_digest': digest(captured.tool_call.model_dump(mode='json'))}
        self._audit.record(event, request_id, **common, **evidence)

    def _save(self, outcome: GatewayOutcome) -> GatewayOutcome:
        self._outcomes[outcome.request_id] = outcome.model_copy(deep=True)
        trace('GATEWAY', 'Final execution status.', request_id=outcome.request_id, status=outcome.status,
              level=logging.ERROR if outcome.status == 'FAILED' else logging.INFO)
        return outcome.model_copy(deep=True)

    def _reject(self, request_id: str, supervision: SupervisoryResult | None, reason: str,
                approval: ApprovalRequest | None = None, status: str = 'REJECTED') -> GatewayOutcome:
        trace('GATEWAY', 'Execution rejected; tool will not be invoked.', request_id=request_id, status=status, reason=reason)
        self._record('tool_rejected', request_id, supervision, reason=reason, execution_status=status)
        return GatewayOutcome(request_id=request_id, status=status, supervision=supervision,
                              approval=approval, reason=reason)

    @traced_request
    def submit(self, *, agent_credential: str, task_id: str, tool_call: ProposedToolCall,
               idempotency_key: str | None = None, use_model: bool = True) -> GatewayOutcome:
        """Caller cannot submit decisions, grants or supervisory result objects."""
        with self._lock:
            request_id = str(uuid4())
            bind_request(request_id)
            trace('GATEWAY', 'Proposed action received for execution.', task_id=task_id)
            agent_id = self._agents.get(agent_credential)
            if agent_id is None:
                return self._reject(request_id, None, 'UNKNOWN_AGENT_CREDENTIAL')
            # Validate even objects built with unchecked Pydantic construction/copy.
            call = ProposedToolCall.model_validate(tool_call.model_dump(mode='json'))
            proposal_digest = digest({'task_id': task_id, 'call': call.model_dump(mode='json')})
            if idempotency_key is not None:
                key = (agent_id, idempotency_key)
                previous = self._idempotency.get(key)
                if previous:
                    previous_id, previous_digest = previous
                    if previous_digest != proposal_digest:
                        return self._reject(request_id, None, 'IDEMPOTENCY_REQUEST_CHANGED')
                    trace('GATEWAY', 'Reusing previous request outcome; no new tool execution.', request_id=previous_id,
                          status=self._outcomes[previous_id].status)
                    return self._outcomes[previous_id].model_copy(deep=True)
                # Reserve before any audit/execution side effect. A failure cannot
                # make a retry with this key invoke the tool a second time.
                self._idempotency[key] = (request_id, proposal_digest)
                self._outcomes[request_id] = GatewayOutcome(request_id=request_id, status='FAILED',
                    reason='REQUEST_PROCESSING_NOT_COMPLETED')
            self._audit.record('action_proposed', request_id, agent_id=agent_id, task_id=task_id,
                action_digest=digest(call.model_dump(mode='json')), tool_name=call.tool_name,
                target_system=call.arguments.target_system, environment=call.arguments.environment,
                data_classification=call.arguments.data_classification, execution_status='NOT_EXECUTED')
            result = supervise(agent_id=agent_id, task_id=task_id, tool_call=call,
                authority_store=self._store, use_model=use_model, controller_config=self._config.controller)
            self._requests[request_id] = result.model_dump_json()
            self._record('action_intercepted', request_id, result, execution_status='NOT_EXECUTED')
            self._record('authority_verified', request_id, result,
                authorization_outcome=result.authority.status, reasons=result.authority.reasons,
                grant_ids=[g.grant_id for g in result.authority.applicable_grants],
                delegation_version=result.authority.delegation.version if result.authority.delegation else None)
            advice = result.assessment.advice
            native = advice.get('result', {}) if result.assessment.status == 'available' else {}
            if not isinstance(native, dict):
                native = {}
            self._record('model_assessment_received', request_id, result,
                model_status=result.assessment.status, model_id=advice.get('model_id'), revision=advice.get('revision'),
                model_output={key: native[key] for key in ('model', 'answers', 'usage') if key in native})
            self._record('governance_decision', request_id, result,
                decision=result.decision.model_dump(mode='json'), execution_status='NOT_EXECUTED')
            with self._store.locked():
                if result.decision.decision == 'BLOCK':
                    outcome = self._reject(request_id, result, 'SUPERVISOR_BLOCK', status='BLOCKED')
                elif result.decision.decision == 'REQUIRE_APPROVAL':
                    trace('GATEWAY', 'REQUIRE_APPROVAL: execution placed on hold.', status='PENDING_APPROVAL')
                    now = self._clock()
                    authority_json = canonical(result.authority.model_dump(mode='json'))
                    approval = ApprovalRequest(approval_id=str(uuid4()), request_id=request_id,
                        interception_id=result.intercepted.interception_id, agent_id=agent_id, task_id=task_id,
                        tool_call_json=canonical(call.model_dump(mode='json')), action_digest=digest(call.model_dump(mode='json')),
                        authorization_context_json=authority_json, authorization_digest=digest(result.authority.model_dump(mode='json')),
                        delegation_version=result.authority.delegation.version if result.authority.delegation else None,
                        supervisor_decision_json=canonical(result.decision.model_dump(mode='json')),
                        required_approver_role=self._config.controller.approver_role,
                        created_at=now, updated_at=now, expires_at=now + timedelta(seconds=self._config.approval_ttl_seconds))
                    self._approvals._put(self.__capability, approval)
                    self._record('approval_requested', request_id, result, approval_id=approval.approval_id,
                        expires_at=approval.expires_at.isoformat(), required_approver_role=approval.required_approver_role,
                        execution_status='HELD')
                    outcome = GatewayOutcome(request_id=request_id, status='PENDING_APPROVAL', supervision=result, approval=approval)
                else:
                    outcome = self._execute(self.__capability, request_id, result)
            saved = self._save(outcome)
            if idempotency_key is not None:
                self._idempotency[(agent_id, idempotency_key)] = (request_id, proposal_digest)
            return saved

    def _current_authority(self, result: SupervisoryResult) -> tuple[AuthorityVerification, str | None]:
        trace('GATEWAY', 'Rechecking current trusted tool authorization.', level=VERBOSE)
        current = AuthorityVerifier().verify(result.intercepted, self._store)
        if result.authority.status != 'VERIFIED':
            return current, 'ORIGINAL_AUTHORITY_UNRESOLVED'
        if current.status != 'VERIFIED':
            return current, 'CURRENT_AUTHORITY_NOT_VERIFIED'
        if digest(current.model_dump(mode='json')) != digest(result.authority.model_dump(mode='json')):
            return current, 'TRUSTED_AUTHORIZATION_CONTEXT_CHANGED'
        return current, None

    def _execute(self, capability: object, request_id: str, result: SupervisoryResult,
                 approval: ApprovalRequest | None = None) -> GatewayOutcome:
        """Private, gateway-owned path; called with gateway and store locks held."""
        if capability is not self.__capability:
            raise PermissionError('Execution requires the owning gateway capability')
        if request_id not in self._requests or request_id in self._attempted:
            return self._reject(request_id, result, 'UNKNOWN_OR_ALREADY_ATTEMPTED_REQUEST', approval)
        trusted = SupervisoryResult.model_validate_json(self._requests[request_id])
        if canonical(trusted.model_dump(mode='json')) != canonical(result.model_dump(mode='json')):
            return self._reject(request_id, trusted, 'CALLER_SUPERVISION_CHANGED', approval)
        if approval and self._approvals.get(approval.approval_id) != approval:
            return self._reject(request_id, trusted, 'UNTRUSTED_APPROVAL_RECORD')
        if result.decision.decision != 'ALLOW' and (approval is None or approval.status != 'APPROVED'):
            return self._reject(request_id, result, 'EXECUTION_NOT_AUTHORIZED', approval)
        current, reason = self._current_authority(result)
        self._record('authority_rechecked', request_id, result, authorization_outcome=current.status,
                     reasons=current.reasons, execution_status='NOT_EXECUTED')
        if reason:
            return self._reject(request_id, result, reason, approval)
        if not self._tools.supports(result.intercepted.tool_call):
            return self._reject(request_id, result, 'UNREGISTERED_MOCK_TOOL', approval)
        if approval and self._clock() >= approval.expires_at:
            return self._finish_approval(approval, result, 'EXPIRED', 'APPROVAL_EXPIRED')
        self._attempted.add(request_id)
        if approval:
            approval = approval.model_copy(update={'status': 'CONSUMED', 'updated_at': self._clock()})
            self._approvals._put(self.__capability, approval)
            self._record('approval_consumed', request_id, result, approval_id=approval.approval_id)
        self._record('tool_execution_started', request_id, result, execution_status='STARTED')
        # Audit I/O itself may cross the expiry boundary. Do not invoke an
        # expired action even if admission was reserved moments earlier.
        if approval and self._clock() >= approval.expires_at:
            return self._finish_approval(approval, result, 'EXPIRED', 'APPROVAL_EXPIRED')
        try:
            trace('GATEWAY', 'Releasing registered mock tool after approval.' if approval else 'ALLOW: executing registered mock tool.',
                  tool_name=result.intercepted.tool_call.tool_name, status='APPROVED' if approval else 'ALLOW')
            execution = self._tools._invoke(self.__capability, result.intercepted.tool_call)
        except Exception as exc:
            execution = {'status': 'FAILED', 'error_type': type(exc).__name__, 'simulated': True}
            status = 'FAILED'
        else:
            status = 'EXECUTED'
        if approval:
            approval = approval.model_copy(update={'execution_outcome_json': canonical(execution), 'updated_at': self._clock()})
            self._approvals._put(self.__capability, approval)
        self._record('tool_executed' if status == 'EXECUTED' else 'tool_failed', request_id, result,
                     execution_result=execution, execution_status=status)
        return GatewayOutcome(request_id=request_id, status=status, supervision=result, approval=approval, execution_result=execution)

    def _finish_approval(self, record: ApprovalRequest, result: SupervisoryResult, status: str, reason: str) -> GatewayOutcome:
        record = record.model_copy(update={'status': status, 'updated_at': self._clock(),
            'execution_outcome_json': canonical({'status': 'NOT_EXECUTED', 'reason': reason})})
        self._approvals._put(self.__capability, record)
        self._record('approval_expired' if status == 'EXPIRED' else 'approval_denied', record.request_id, result,
                     approval_id=record.approval_id, reason=reason, approval_status=status)
        return self._save(self._reject(record.request_id, result, reason, record,
                                      status='EXPIRED' if status == 'EXPIRED' else 'REJECTED'))

    @traced_request
    def approve(self, *, approval_id: str, approver_credential: str, interception_id: str,
                tool_call: ProposedToolCall) -> GatewayOutcome:
        with self._lock, self._store.locked():
            record = self._approvals.get(approval_id)
            if record is None:
                return self._reject(str(uuid4()), None, 'UNKNOWN_APPROVAL')
            bind_request(record.request_id)
            trace('APPROVAL', 'Approval submission received; verifying trusted approver and action binding.', level=VERBOSE,
                  approval_id=approval_id)
            result = SupervisoryResult.model_validate_json(self._requests[record.request_id])
            if record.status != 'PENDING':
                return self._reject(record.request_id, result, 'APPROVAL_NOT_PENDING', record)
            if self._clock() >= record.expires_at:
                return self._finish_approval(record, result, 'EXPIRED', 'APPROVAL_EXPIRED')
            approver = self._approvers.get(approver_credential)
            if approver is None or record.required_approver_role not in approver.roles:
                self._record('approval_denied', record.request_id, result, approval_id=approval_id,
                             reason='UNAUTHORIZED_APPROVER', approval_status='PENDING')
                return self._reject(record.request_id, result, 'UNAUTHORIZED_APPROVER', record)
            if record.interception_id != interception_id:
                return self._finish_approval(record, result, 'REJECTED', 'APPROVAL_REQUEST_MISMATCH')
            if canonical(tool_call.model_dump(mode='json')) != record.tool_call_json:
                return self._finish_approval(record, result, 'REJECTED', 'APPROVED_ACTION_CHANGED')
            current, reason = self._current_authority(result)
            self._record('approval_authority_rechecked', record.request_id, result,
                authorization_outcome=current.status, reasons=current.reasons, approval_id=approval_id)
            if reason:
                return self._finish_approval(record, result, 'REJECTED', reason)
            if result.decision.decision != 'REQUIRE_APPROVAL':
                return self._finish_approval(record, result, 'REJECTED', 'ORIGINAL_DECISION_NOT_APPROVABLE')
            if not self._tools.supports(result.intercepted.tool_call):
                return self._finish_approval(record, result, 'REJECTED', 'UNREGISTERED_MOCK_TOOL')
            record = record.model_copy(update={'status': 'APPROVED', 'approved_by': approver.approver_id,
                'approved_at': self._clock(), 'updated_at': self._clock()})
            self._approvals._put(self.__capability, record)
            self._record('approval_granted', record.request_id, result, approval_id=approval_id,
                         approver_id=approver.approver_id, approval_status='APPROVED')
            return self._save(self._execute(self.__capability, record.request_id, result, record))

    @traced_request
    def reject_approval(self, *, approval_id: str, approver_credential: str) -> GatewayOutcome:
        with self._lock:
            record = self._approvals.get(approval_id)
            if record is None:
                return self._reject(str(uuid4()), None, 'UNKNOWN_APPROVAL')
            bind_request(record.request_id)
            result = SupervisoryResult.model_validate_json(self._requests[record.request_id])
            if record.status != 'PENDING':
                return self._reject(record.request_id, result, 'APPROVAL_NOT_PENDING', record)
            if self._clock() >= record.expires_at:
                return self._finish_approval(record, result, 'EXPIRED', 'APPROVAL_EXPIRED')
            identity = self._approvers.get(approver_credential)
            if identity is None or record.required_approver_role not in identity.roles:
                self._record('approval_denied', record.request_id, result, approval_id=approval_id,
                             reason='UNAUTHORIZED_APPROVER', approval_status='PENDING')
                return self._reject(record.request_id, result, 'UNAUTHORIZED_APPROVER', record)
            return self._finish_approval(record, result, 'REJECTED', 'HUMAN_REJECTED')

    @traced_request
    def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        with self._lock:
            record = self._approvals.get(approval_id)
            if record:
                bind_request(record.request_id)
            if record and record.status == 'PENDING' and self._clock() >= record.expires_at:
                result = SupervisoryResult.model_validate_json(self._requests[record.request_id])
                self._finish_approval(record, result, 'EXPIRED', 'APPROVAL_EXPIRED')
                record = self._approvals.get(approval_id)
            return record

    def get_outcome(self, request_id: str) -> GatewayOutcome | None:
        """Return a detached authoritative outcome, including lazy approval expiry."""
        with self._lock:
            outcome = self._outcomes.get(request_id)
            if outcome and outcome.approval:
                self.get_approval(outcome.approval.approval_id)
                outcome = self._outcomes[request_id]
            return outcome.model_copy(deep=True) if outcome else None

    def tool_state(self) -> dict:
        with self._lock:
            return self._tools.snapshot()
