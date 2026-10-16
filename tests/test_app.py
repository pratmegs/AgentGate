from streamlit.testing.v1 import AppTest
from src.policy_engine import ROOT
from src import decision_model
import json


def test_app_evaluation(monkeypatch):
    monkeypatch.setattr(decision_model, 'get_advice', lambda *args: {'status': 'unavailable', 'error': 'Unit test offline'})
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=10).run()
    assert not app.exception
    app.button(key='submit_action').click().run()
    assert not app.exception
    result = next(iter(app.session_state['submitted_requests'].values()))
    assert result.supervision.decision.decision == 'REQUIRE_APPROVAL'
    assert result.status == 'PENDING_APPROVAL'


def test_app_audit_redacts_task_and_tool_descriptions(monkeypatch):
    monkeypatch.setattr(decision_model, 'get_advice', lambda *args: {'status': 'unavailable'})
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=10).run()
    app.text_area(key='proposal_claim').set_value('SECRET_TASK_PAYLOAD').run()
    app.text_area(key='proposal_description').set_value('SECRET_ACTION_PAYLOAD').run()
    app.button(key='submit_action').click().run()
    assert not app.exception
    audit = app.session_state['audit_trail']
    content = audit.path.read_text(encoding='utf-8')
    assert 'SECRET_TASK_PAYLOAD' not in content and 'SECRET_ACTION_PAYLOAD' not in content
    assert json.loads(content.splitlines()[-1])['execution_status'] == 'HELD'
