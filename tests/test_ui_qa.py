"""Regression checks for defects confirmed during Stage 3C visual inspection."""
from streamlit.testing.v1 import AppTest
import pytest
from src.dashboard_view import format_timestamp, audit_summary_rows
from src.policy_engine import ROOT


def qa_app():
    app = AppTest.from_file(ROOT / 'diagnostics/ui_qa_app.py', default_timeout=15).run()
    assert not app.exception
    return app


def submit(app):
    app.button(key='submit_action').click().run()
    assert not app.exception
    return app.session_state['submitted_requests'][app.session_state['latest_request_id']]


def test_tab_labels_stay_stable_as_queue_count_changes():
    app = qa_app()
    labels = [tab.label for tab in app.tabs]
    held = submit(app)
    assert held.status == 'PENDING_APPROVAL'
    assert [tab.label for tab in app.tabs] == labels
    assert 'Decision & Approval' in labels
    app.button(key='approve_action').click().run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == labels
    assert app.session_state['submitted_requests'][held.request_id].status == 'EXECUTED'


def test_approval_labels_readable_parameters_inspectable_and_colors_consistent():
    app = qa_app()
    held = submit(app)
    labels = app.selectbox(key='selected_approval').options
    assert any('read → crm' in label and 'support-demo' in label for label in labels)
    assert app.selectbox(key='selected_approval').value == held.approval.approval_id
    collapsed = {expander.label: expander.proto.expanded for expander in app.expander}
    assert collapsed['Original frozen action parameters'] is False
    assert collapsed['Triggered decision reasons'] is False
    assert any('Approval lifecycle · PENDING' in element.value for element in app.warning)
    app.button(key='approve_action').click().run()
    assert not app.exception
    assert any('Approval lifecycle · CONSUMED' in element.value for element in app.success)


def test_history_summary_exposes_outcome_and_expanded_evidence_preserves_ids():
    app = qa_app()
    held = submit(app)
    summary = next(frame.value for frame in app.dataframe if 'Execution outcome' in frame.value.columns)
    assert list(summary.columns) == ['Submitted (UTC)', 'Requested action', 'Target system',
                                    'Governance decision', 'Execution outcome']
    assert summary.iloc[0]['Execution outcome'] == 'PENDING_APPROVAL'
    assert summary.iloc[0]['Submitted (UTC)'] == format_timestamp(held.supervision.intercepted.captured_at.isoformat())
    evidence = next(frame.value for frame in app.dataframe if 'Event ID' in frame.value.columns)
    assert all(evidence['Request ID'] == held.request_id)
    assert any(held.request_id in element.value for element in app.markdown)
    timeline = next(frame.value for frame in app.dataframe if 'Event' in frame.value.columns and 'Status' in frame.value.columns)
    assert list(timeline.columns) == ['Timestamp (UTC)', 'Event', 'Status', 'Reason']
    collapsed = {expander.label: expander.proto.expanded for expander in app.expander}
    assert collapsed['Recorded model probabilities'] is False
    assert collapsed['Full audit correlation identifiers'] is False


def test_audit_summary_keeps_actual_status_and_hides_unnecessary_payloads():
    events = [{'timestamp': '2026-10-13T14:10:14+00:00', 'event': 'authority_verified',
               'authorization_outcome': 'VERIFIED', 'credential': 'DO_NOT_RENDER'}]
    assert audit_summary_rows(events) == [{'Timestamp (UTC)': '2026-10-13 14:10:14',
        'Event': 'authority_verified', 'Status': 'VERIFIED', 'Reason': ''}]


def test_visual_qa_harness_can_show_unavailable_without_loading_model():
    app = qa_app()
    app.selectbox(key='qa_fixture').set_value('Unavailable').run()
    outcome = submit(app)
    assert outcome.status == 'PENDING_APPROVAL'
    assert outcome.supervision.assessment.status == 'unavailable'
    assert not app.session_state['gateway'].tool_state()['invocations']
    assert not app.get('progress')
    assert any('Visual QA fixture: model unavailable' in element.value for element in app.warning)


def test_previous_approval_is_distinguished_from_a_new_blocked_request():
    app = qa_app()
    held = submit(app)
    app.button(key='approve_action').click().run()
    app.selectbox(key='scenario').set_value('Customer Support: excessive refund').run()
    blocked = submit(app)
    assert blocked.status == 'BLOCKED'
    assert any(f'Last reviewer attempt for read → crm ({held.request_id[:8]})' in element.value for element in app.success)
    assert any('different submitted request' in element.value for element in app.caption)


@pytest.mark.parametrize('source,expected', [
    ('2026-10-13T14:10:14.349186+00:00', '2026-10-13 14:10:14'),
    ('2026-10-13T07:10:14-07:00', '2026-10-13 14:10:14'),
    ('', 'Not recorded')])
def test_compact_timestamps_preserve_utc_meaning(source, expected):
    assert format_timestamp(source) == expected
