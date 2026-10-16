"""Capture proposals without invoking a tool or accepting agent credentials."""
from datetime import datetime, timezone
from uuid import uuid4
from pydantic import ConfigDict, Field
from src.schemas import StrictModel, RequestedAction, TaskContext
from src.observability import trace, VERBOSE, bind_interception


class ProposedToolCall(StrictModel):
    tool_name: str = Field(min_length=1)
    arguments: RequestedAction
    claimed_task_description: str | None = None


class InterceptedAction(StrictModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    interception_id: str = Field(default_factory=lambda: str(uuid4()))
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: str
    task_id: str
    tool_call: ProposedToolCall
    delegated_context: TaskContext | None = None


class ActionInterceptor:
    def capture(self, *, agent_id: str, task_id: str, tool_call: ProposedToolCall,
                delegated_context: TaskContext | None = None) -> InterceptedAction:
        """IDs come from the authenticated caller, never from tool arguments."""
        trace('INTERCEPTOR', 'Incoming proposed tool call.', level=VERBOSE, agent_id=agent_id,
              task_id=task_id, tool_name=tool_call.tool_name, target_system=tool_call.arguments.target_system)
        captured = InterceptedAction(agent_id=agent_id, task_id=task_id,
                                 tool_call=tool_call.model_copy(deep=True),
                                 delegated_context=delegated_context.model_copy(deep=True) if delegated_context else None)
        bind_interception(captured.interception_id)
        trace('INTERCEPTOR', 'Agent action captured. Forwarding for verification.',
              interception_id=captured.interception_id, agent_id=agent_id, task_id=task_id)
        return captured
