"""Append-only local lifecycle records. Local storage is not tamper-proof."""
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4
from src.observability import trace, VERBOSE


class AuditTrail:
    def __init__(self, path: Path | None = None):
        self.path = path
        self._lock = RLock()
        self._events: list[dict] = []
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, request_id: str, **evidence) -> None:
        record = {'event_id': str(uuid4()), 'timestamp': datetime.now(timezone.utc).isoformat(),
                  'event': event, 'request_id': request_id, **evidence}
        serialized = json.dumps(record, allow_nan=False)
        with self._lock:
            if self.path:
                with self.path.open('a', encoding='utf-8') as stream:
                    stream.write(serialized + '\n')
                    stream.flush()
            self._events.append(json.loads(serialized))
            trace('AUDIT', 'Audit record written.' if self.path else 'Audit record recorded in memory.',
                  level=VERBOSE, request_id=request_id, audit_id=record['event_id'], event=event)

    def records(self) -> list[dict]:
        with self._lock:
            return json.loads(json.dumps(self._events))
