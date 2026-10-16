"""Streamlit presentation only; mutation remains in the existing gateway callbacks."""
from html import escape
import json
import streamlit as st
from src.action_interceptor import ProposedToolCall
from src.dashboard_view import (ASSESSMENT_LABELS, session_metrics, technical_capabilities,
    grant_rows, action_rows, probability_rows, conflict_notes, workflow_states, audit_rows,
    audit_summary_rows, history_rows, format_timestamp)

STYLE = """
<style>
.block-container {max-width: 1480px; padding: 4.5rem 2rem 2rem;}
.block-container h1 {font-size: 2.1rem; line-height: 1.2; padding: .25rem 0 .5rem;}
.block-container h3 {font-size: 1.25rem; padding: .25rem 0 .5rem;}
h1 {letter-spacing: -.045em; font-weight: 700;}
h2, h3 {letter-spacing: -.02em;}
[data-testid="stMetric"] {border: 1px solid #29364c; border-radius: 12px; padding: 12px; background: #111d30;}
[data-testid="stMetricLabel"] {color: #a8b7cc; min-height: 2rem; overflow: visible;}
[data-testid="stMetricLabel"] p {white-space: normal; overflow: visible; text-overflow: clip; font-size: .8rem; line-height: 1.3;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
button[data-baseweb="tab"] {padding: 12px 18px;}
.console-eyebrow {color: #80adff; font-size: .75rem; font-weight: 700; letter-spacing: .15em; margin-bottom: .2rem;}
.workflow {display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 8px; margin: 16px 0 24px;}
.workflow-stage {padding: 12px; background: #111d30; border: 1px solid #29364c; border-radius: 10px; min-height: 96px;}
.workflow-stage small {display: block; color: #91a3bb; margin-bottom: 7px; font-size: .7rem;}
.workflow-stage strong {display: block; font-size: .85rem; line-height: 1.35; margin-bottom: 12px;}
.state {font-size: .72rem; font-weight: 650; overflow-wrap: anywhere; color: #bcc9dc;}
.state.good {color: #78d5ad;} .state.hold {color: #efc478;} .state.bad {color: #f4979e;}
@media (max-width: 1550px) {.workflow {grid-template-columns: repeat(4, minmax(0, 1fr));}}
@media (max-width: 650px) {.workflow {grid-template-columns: repeat(2, minmax(0, 1fr));}}
</style>
"""


def apply_style():
    st.markdown(STYLE, unsafe_allow_html=True)


def table(rows):
    if rows:
        st.dataframe(rows, hide_index=True, width='stretch')
    else:
        st.caption('No records available.')


def status_message(status, text=None):
    message = text or status
    if status in ('ALLOW', 'EXECUTED', 'VERIFIED', 'CONSUMED', 'APPROVED', 'AVAILABLE'):
        st.success(message)
    elif status in ('BLOCK', 'BLOCKED', 'REJECTED', 'FAILED', 'PROHIBITED'):
        st.error(message)
    elif status in ('REQUIRE_APPROVAL', 'PENDING_APPROVAL', 'PENDING', 'EXPIRED', 'UNRESOLVED', 'UNAVAILABLE', 'DISABLED'):
        st.warning(message)
    else:
        st.info(message)


def render_overview(state):
    st.subheader('Session overview')
    st.write('Supervising AI-agent actions by evaluating delegated authority, contextual risk, and autonomous execution suitability before tool execution.')
    metrics = session_metrics(state.submitted_requests.values())
    for column, (label, value) in zip(st.columns(5), metrics.items()):
        column.metric(label, value)
    st.caption('Session data only. Allowed counts autonomous ALLOW verdicts; executed includes approved mock actions. No benchmark or model performance metrics are shown.')
    with st.container(border=True):
        st.markdown('**A simulated enterprise environment**')
        st.write('Propose a tool action in Agent Workspace, inspect the supervisory evidence, then review any held action in Decision & Approval. All tool effects are confined to in-memory enterprise mocks.')
        st.caption('Technical capability describes available tool interfaces. Trusted authority bounds their permitted use. Model judgments cannot override verified prohibitions.')
    if not state.submitted_requests:
        st.info('No actions evaluated yet. Select an agent and scenario, then submit a proposed action.')


def render_trusted_agent(agent, delegation, grants):
    with st.container(border=True):
        st.subheader(agent.role)
        st.caption(f'{agent.department} · {agent.agent_id}')
        st.markdown('**Original delegated objective**')
        st.write(delegation.original_goal)
        st.markdown('**Technical capabilities**')
        st.caption('Registered mock tool interfaces for this system. Availability does not grant permission.')
        table(technical_capabilities(delegation.target_systems))
    with st.container(border=True):
        st.markdown('**Trusted authority · read-only**')
        st.caption('Host-provisioned permissions and delegation. Proposed text cannot expand these boundaries.')
        st.write('Permissions:', ', '.join(agent.permissions) or 'None')
        st.write('Delegated actions:', ', '.join(delegation.allowed_actions) or 'None')
        table([{'Boundary': label, 'Trusted value': str(value)} for label, value in (
            ('Target systems', ', '.join(delegation.target_systems)),
            ('Environments', ', '.join(delegation.environments)),
            ('Data scope', ', '.join(delegation.data_classifications)),
            ('Agent financial limit', agent.autonomous_financial_limit),
            ('Task amount limit', delegation.max_amount),
            ('External effects permitted', delegation.allow_external_effect),
            ('Irreversible actions permitted', delegation.allow_irreversible))])
        with st.expander('Atomic scoped grants'):
            table(grant_rows(grants))
            st.caption(f'Task {delegation.task_id} · delegation version {delegation.version}. A single applicable grant must cover the complete action scope.')


def render_workflow(outcome, events):
    stages = workflow_states(outcome, events)
    cards = []
    for index, (name, status) in enumerate(stages, 1):
        tone = ('good' if status in ('Recorded', 'Captured', 'VERIFIED', 'AVAILABLE', 'ALLOW', 'EXECUTED')
                else 'bad' if status in ('BLOCK', 'BLOCKED', 'PROHIBITED', 'REJECTED', 'FAILED')
                else 'hold' if status in ('REQUIRE_APPROVAL', 'PENDING_APPROVAL', 'EXPIRED', 'UNAVAILABLE', 'DISABLED', 'UNRESOLVED') else '')
        cards.append(f'<div class="workflow-stage"><small>{index:02d} {"→" if index < 7 else ""}</small><strong>{escape(name)}</strong><span class="state {tone}">{escape(status)}</span></div>')
    st.markdown('<div class="workflow">' + ''.join(cards) + '</div>', unsafe_allow_html=True)
    st.caption('Stages reflect the selected request’s recorded backend results. Authority verification can prohibit an action; it does not always pass.')


def render_assessment(assessment):
    status_message(assessment.status.upper(), f'Decision 2.0 · {assessment.status}')
    st.caption('Probabilities are uncalibrated model scores, not validated confidence or guarantees of correctness.')
    if assessment.status != 'available':
        st.warning(assessment.error or 'No model assessment is available. Autonomous authorization is unavailable.')
        return
    answers = assessment.advice['result']['answers']
    labels = list(ASSESSMENT_LABELS.items())
    for offset in (0, 3):
        for column, (key, label) in zip(st.columns(3), labels[offset:offset + 3]):
            answer = answers[key]
            selected_probability = answer['probabilities'][answer['choice']]
            with column, st.container(border=True):
                st.caption(label)
                st.markdown(f'**{answer["choice"]}**')
                st.progress(float(selected_probability), text=f'{selected_probability:.1%} selected probability')
    for note in conflict_notes(assessment):
        st.warning(note + ' The final controller verdict is shown separately.')
    with st.expander('Full probability distributions and model metadata'):
        table(probability_rows(assessment))
        st.write('Model:', assessment.advice.get('model_id', assessment.advice['result'].get('model', 'Not supplied')))
        st.write('Pinned revision:', assessment.advice.get('revision', 'Not supplied'))
        usage = assessment.advice['result'].get('usage', {})
        if usage:
            table([{'Usage field': key, 'Value': value} for key, value in usage.items()
                   if key in ('input_tokens', 'output_tokens')])
        native = [{'Assessment': ASSESSMENT_LABELS[key], 'Native confidence (uncalibrated)': value['confidence']}
                  for key, value in answers.items() if key in ASSESSMENT_LABELS and 'confidence' in value]
        if native:
            table(native)


def render_authority(authority):
    with st.container(border=True):
        st.markdown('**Trusted authority findings**')
        status_message(authority.status, f'Authority verification · {authority.status}')
        if authority.reasons:
            table([{'Finding': code.replace('_', ' ').capitalize(), 'Reason code': code} for code in authority.reasons])
        else:
            st.caption('No authority violations reported by the verifier.')
        if authority.agent:
            st.write('Trusted permissions:', ', '.join(authority.agent.permissions))
        if authority.delegation:
            delegation = authority.delegation
            st.write('Original objective:', delegation.original_goal)
            st.write('Task:', delegation.task_id, '· version', delegation.version)
            st.write('Limits:', ', '.join(delegation.target_systems), ' / ', ', '.join(delegation.environments),
                ' / ', ', '.join(delegation.data_classifications), ' · amount ≤ ', delegation.max_amount)
            st.write('External effects:', delegation.allow_external_effect, ' · irreversible:', delegation.allow_irreversible)
        with st.expander('Applicable verified grants'):
            table(grant_rows(authority.applicable_grants))


def render_verdict(outcome):
    if not outcome:
        st.info('No governance verdict yet. Submit an action in Agent Workspace.')
        return
    result = outcome.supervision
    with st.container(border=True):
        if result:
            decision = result.decision
            status_message(decision.decision.value, f'Final governance verdict · {decision.decision.value}')
            st.write(decision.explanation)
            cols = st.columns(3)
            cols[0].metric('Model consequence severity', decision.model_consequence_severity or 'Unknown')
            cols[1].metric('Execution status', outcome.status)
            cols[2].metric('Required approver', decision.required_approver or 'None')
            if decision.violated_policies:
                with st.expander('Triggered decision reasons'):
                    table([{'Triggered reason': code.replace('_', ' ').capitalize(), 'Reason code': code}
                           for code in decision.violated_policies])
            else:
                st.caption('No adverse decision reasons were triggered.')
            with st.expander('Request and audit identifiers'):
                table([{'Identifier': key, 'Value': value} for key, value in (
                    ('Request ID', outcome.request_id), ('Governance audit ID', decision.audit_id),
                    ('Interception ID', result.intercepted.interception_id),
                    ('Approval ID', outcome.approval.approval_id if outcome.approval else 'None'))])
        else:
            status_message(outcome.status, f'Execution status · {outcome.status}')
            st.write('Request ID:', outcome.request_id)
        if outcome.reason:
            st.write('Gateway reason:', outcome.reason)
        if outcome.approval:
            status_message(outcome.approval.status, f'Approval lifecycle · {outcome.approval.status}')
            if outcome.approval.approved_by:
                st.caption(f'Reviewer: {outcome.approval.approved_by}')
        if outcome.execution_result:
            table([{'Execution field': key, 'Actual result': str(value)} for key, value in outcome.execution_result.items()
                   if key in ('simulated', 'tool_name', 'target_system', 'invocation_number', 'status', 'error_type')])


def render_frozen_action(record):
    call = ProposedToolCall.model_validate_json(record.tool_call_json)
    st.markdown(f'**Frozen action: {call.arguments.action_type} → {call.arguments.target_system}**')
    with st.expander('Original frozen action parameters'):
        table(action_rows(call))
    st.markdown('**Original proposed action**')
    st.write(call.arguments.description)
    snapshot = json.loads(record.authorization_context_json)
    delegation = snapshot.get('delegation') or {}
    if delegation.get('original_goal'):
        st.write('Original delegated objective:', delegation['original_goal'])
    st.caption(f'Agent {record.agent_id} · task {record.task_id} · delegation version {record.delegation_version}')
    st.write('Expires at (UTC):', record.expires_at.isoformat())
    st.write('Required reviewer role:', record.required_approver_role)
    with st.expander('Frozen action binding identifiers'):
        table([{'Identifier': key, 'Value': value} for key, value in (
            ('Request ID', record.request_id), ('Approval ID', record.approval_id),
            ('Interception ID', record.interception_id), ('Action digest', record.action_digest),
            ('Authorization digest', record.authorization_digest))])


def render_history(state):
    st.subheader('Execution history & audit explorer')
    if not state.submitted_requests:
        st.info('No submission history yet. Gateway outcomes and audit events will appear here after submission.')
        return
    events = state.audit_trail.records()
    rows = history_rows(state.submitted_requests.values(), events)
    table([{key: format_timestamp(row[key]) if key == 'Submitted (UTC)' else row[key]
            for key in ('Submitted (UTC)', 'Requested action', 'Target system', 'Governance decision', 'Execution outcome')}
           for row in rows])
    st.caption('Chronological submission order. A governance ALLOW verdict and a completed execution are separate facts.')
    for row in rows:
        outcome = state.submitted_requests[row['Request ID']]
        label = f'{format_timestamp(row["Submitted (UTC)"])} UTC · {row["Requested action"]} → {row["Target system"]} · {outcome.status} · {outcome.request_id[:8]}'
        with st.expander(label):
            st.write('Agent:', row['Agent'])
            st.write('Request ID:', outcome.request_id)
            st.write('Submitted (UTC):', row['Submitted (UTC)'])
            st.write('Last audit event (UTC):', row['Last audit event (UTC)'])
            st.write('Mock tool invoked:', row['Mock tool invoked'])
            if outcome.supervision:
                with st.expander('Frozen action parameters'):
                    table(action_rows(outcome.supervision.intercepted.tool_call))
                st.write('Original proposed action:', outcome.supervision.intercepted.tool_call.arguments.description)
                st.write('Governance audit ID:', outcome.supervision.decision.audit_id)
                st.write('Decision reasons:', ', '.join(outcome.supervision.decision.violated_policies) or 'None')
                st.write('Authority verification:', outcome.supervision.authority.status)
                st.write('Authority findings:', ', '.join(outcome.supervision.authority.reasons) or 'None')
                st.write('Model availability:', outcome.supervision.assessment.status)
                with st.expander('Recorded model probabilities'):
                    table(probability_rows(outcome.supervision.assessment))
            if outcome.approval:
                st.write('Approval lifecycle:', outcome.approval.status)
                st.write('Approval created (UTC):', outcome.approval.created_at.isoformat())
                st.write('Expires at (UTC):', outcome.approval.expires_at.isoformat())
                if outcome.approval.approved_at:
                    st.write('Approved at (UTC):', outcome.approval.approved_at.isoformat())
            request_events = [e for e in events if e['request_id'] == outcome.request_id]
            st.markdown('**Audit event timeline**')
            table(audit_summary_rows(request_events))
            with st.expander('Full audit correlation identifiers'):
                table(audit_rows(request_events))
    st.caption('Audit evidence is append-only local storage, not tamper-proof. History is scoped to this running session.')
