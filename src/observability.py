"""Best-effort live terminal observability, separate from durable audit behavior."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import wraps
import logging
import os
import re
from threading import RLock
from time import perf_counter
from typing import TextIO
from uuid import uuid4
from rich.console import Console
from rich.text import Text

VERBOSE = 15
logging.addLevelName(VERBOSE, 'VERBOSE')
LEVELS = {'QUIET': logging.ERROR, 'NORMAL': logging.INFO, 'VERBOSE': VERBOSE, 'DEBUG': logging.DEBUG}
COLORS = {'INTERCEPTOR': 'blue', 'AUTHORITY': 'blue', 'DECISION 2.0': 'cyan',
          'ASSESSMENT': 'cyan', 'CONTROLLER': 'blue', 'GATEWAY': 'blue',
          'APPROVAL': 'yellow', 'AUDIT': 'dim', 'WORKFLOW': 'blue', 'DEMO': 'blue'}
FIELDS = {'request_id', 'interception_id', 'approval_id', 'audit_id', 'agent_id', 'task_id',
          'tool_name', 'target_system', 'environment', 'data_classification', 'status',
          'decision', 'reasons', 'reason', 'question', 'questions', 'choice', 'probability',
          'duration_s', 'model_id', 'revision', 'grant_count', 'threshold', 'error_type',
          'event', 'execution_status', 'expected_decision', 'actual_decision', 'scenario',
          'model_enabled', 'model_available', 'invocation_count', 'trace_file'}
_LOGGER = logging.getLogger('agentgate.trace')
_LOGGER.propagate = False
_LOCK = RLock()


@dataclass(frozen=True)
class TraceContext:
    request_id: str
    started: float
    interception_id: str | None = None


_CONTEXT: ContextVar[TraceContext | None] = ContextVar('agentgate_trace_context', default=None)


def _clean(value) -> str:
    if isinstance(value, (list, tuple)):
        return ','.join(_clean(item) for item in value)[:400]
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '?', str(value))
    text = re.sub(r'(?i)\b(password|secret|credential|token|api[_-]?key)\s*[=:]\s*[^\s,;]+', r'\1=<redacted>', text)
    return text[:400]


class TerminalTraceHandler(logging.Handler):
    def __init__(self, stream: TextIO | None = None, force_terminal: bool | None = None):
        super().__init__()
        self._agentgate_trace_handler = True
        self.console = Console(file=stream, force_terminal=force_terminal, width=180,
                               highlight=False, markup=False, soft_wrap=True)

    def emit(self, record: logging.LogRecord) -> None:
        # Do not render arbitrary log messages, exception payloads or locals.
        try:
            data = getattr(record, 'agentgate_trace', None)
            if not isinstance(data, dict):
                return
            fields = data['fields']
            status = fields.get('status', fields.get('decision', fields.get('execution_status')))
            color = 'dim' if record.levelno == logging.DEBUG else COLORS.get(data['component'], 'blue')
            if status in {'PROHIBITED', 'BLOCK', 'BLOCKED', 'REJECTED', 'FAILED', 'ERROR'}:
                color = 'red'
            elif status in {'VERIFIED', 'ALLOW', 'EXECUTED', 'APPROVED', 'CONSUMED'}:
                color = 'green'
            elif status in {'UNRESOLVED', 'REQUIRE_APPROVAL', 'PENDING', 'PENDING_APPROVAL', 'EXPIRED', 'UNAVAILABLE', 'DISABLED'}:
                color = 'yellow'
            if record.levelno >= logging.ERROR:
                color = 'red'
            timestamp = datetime.fromtimestamp(record.created, timezone.utc).strftime('%H:%M:%SZ')
            line = Text(f'{timestamp} +{data["elapsed"]:.3f}s ', style='dim')
            line.append(f'[{data["component"]}] ', style=color)
            request_id = fields.get('request_id', '-')
            line.append(f'request={request_id} ', style='dim')
            line.append(data['message'], style=color)
            details = [f'{key}={value}' for key, value in fields.items() if key != 'request_id']
            if details:
                line.append(' | ' + ' '.join(details), style='dim' if record.levelno <= VERBOSE else color)
            self.console.print(line)
            self.console.file.flush()
        except Exception:
            # Terminal output is optional. Audit I/O remains independently enforced.
            return


def configure_tracing(level: str | None = None, *, stream: TextIO | None = None,
                      force_terminal: bool | None = None) -> logging.Logger:
    with _LOCK:
        selected = (level or os.environ.get('AGENTGATE_LOG_LEVEL', 'NORMAL')).upper()
        _LOGGER.setLevel(LEVELS.get(selected, logging.INFO))
        owned = [h for h in _LOGGER.handlers if getattr(h, '_agentgate_trace_handler', False)]
        if stream is not None or force_terminal is not None or not owned:
            for handler in owned:
                _LOGGER.removeHandler(handler)
                handler.close()
            _LOGGER.addHandler(TerminalTraceHandler(stream, force_terminal))
        else:
            for handler in owned[1:]:
                _LOGGER.removeHandler(handler)
                handler.close()
        return _LOGGER


@contextmanager
def request_context(request_id: str | None = None):
    parent = _CONTEXT.get()
    token = _CONTEXT.set(parent if parent and request_id is None else TraceContext(request_id or str(uuid4()), perf_counter()))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def traced_request(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with request_context():
            return function(*args, **kwargs)
    return wrapped


def bind_request(request_id: str) -> None:
    context = _CONTEXT.get()
    if context:
        _CONTEXT.set(replace(context, request_id=request_id))


def bind_interception(interception_id: str) -> None:
    context = _CONTEXT.get()
    if context:
        _CONTEXT.set(replace(context, interception_id=interception_id))


def trace(component: str, message: str, *, level: int = logging.INFO, **metadata) -> None:
    """Only intentionally selected metadata is emitted; logging cannot raise."""
    try:
        if not any(getattr(h, '_agentgate_trace_handler', False) for h in _LOGGER.handlers):
            configure_tracing()
        if not _LOGGER.isEnabledFor(level):
            return
        context = _CONTEXT.get()
        fields = {}
        if context:
            fields['request_id'] = _clean(context.request_id)
            if context.interception_id:
                fields['interception_id'] = _clean(context.interception_id)
        fields.update({key: _clean(value) for key, value in metadata.items() if key in FIELDS and value is not None})
        _LOGGER.log(level, message, extra={'agentgate_trace': {
            'component': _clean(component), 'message': _clean(message), 'fields': fields,
            'elapsed': max(0, perf_counter() - context.started) if context else 0,
        }})
    except Exception:
        return
