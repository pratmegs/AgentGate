"""Enterprise governance console; all execution authority stays in the gateway."""
from pathlib import Path
from uuid import uuid4
import streamlit as st
from pydantic import ValidationError
from src.action_interceptor import ProposedToolCall
from src.authority_verifier import TOOL_ACTIONS
from src.dashboard_session import initialize_session, remember_submission, refresh_requests, reset_session
from src.dashboard_components import (apply_style, render_overview, render_trusted_agent,
    render_workflow, render_assessment, render_authority, render_verdict,
    render_frozen_action, render_history, status_message)
from src.schemas import RequestedAction

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='AgentGate | Governance Console', page_icon='🛡️', layout='wide')
apply_style()
initialize_session(st.session_state, ROOT / 'results/dashboard_audit')
state = st.session_state
refresh_requests(state)

st.sidebar.title('AgentGate')
st.sidebar.caption('ENTERPRISE GOVERNANCE · SIMULATION')
st.sidebar.header('Agent workspace')
department = st.sidebar.selectbox('Department',
    ['Customer Support', 'Finance', 'HR', 'IT Operations'], key='department')
available = [s for s in state.demo_scenarios if s.department == department]
agent_id = st.sidebar.selectbox('Predefined demo agent',
    list(dict.fromkeys(s.agent_id for s in available)), key='demo_agent')
available = [s for s in available if s.agent_id == agent_id]
name = st.sidebar.selectbox('Business scenario', [s.name for s in available], key='scenario')
scenario = next(s for s in available if s.name == name)
if state.current_scenario != scenario.name:
    values = scenario.proposal.arguments.model_dump()
    for key, value in values.items():
        if key != 'amount':
            state['proposal_' + key] = value
    state.proposal_has_amount = values['amount'] is not None
    state.proposal_amount = float(values['amount'] or 0)
    state.proposal_claim = ''
    state.current_scenario = scenario.name

st.sidebar.divider()
use_model = st.sidebar.checkbox('Enable Decision 2.0 assessment', value=True, key='supervision_enabled')
st.sidebar.caption('Disabled or unavailable inference cannot authorize autonomous execution.')
with st.sidebar.expander('Reset this session'):
    st.caption('Starts a fresh gateway and discards in-memory pending requests. Previous audit files are retained.')
    confirm_reset = st.checkbox('Confirm reset: discard this session and its pending approvals', key='confirm_reset')
    if st.button('Reset session', key='reset_session'):
        if confirm_reset:
            reset_session(state)
            st.rerun()
        else:
            st.warning('Select the reset confirmation first. Audit files are retained.')

st.markdown('<div class="console-eyebrow">AGENTGATE / GOVERNANCE CONSOLE</div>', unsafe_allow_html=True)
st.title('Supervise before execution.')
st.caption('Simulated enterprise environment · Decision 2.0 + trusted authority verification · Session-scoped evidence')
outcome = None
if state.submitted_requests:
    ids = list(reversed(state.submitted_requests))
    inspect = state.pop('dashboard_inspect_request', None)
    if inspect in ids:
        state.result_request = inspect
    request_labels = {}
    for request_id, record in state.submitted_requests.items():
        call = record.supervision.intercepted.tool_call if record.supervision else None
        action = f'{call.arguments.action_type} → {call.arguments.target_system}' if call else 'Unassessed request'
        request_labels[request_id] = f'{action} · {record.status} · {request_id[:8]}'
    selected = st.selectbox('Inspect submitted action', ids,
        format_func=lambda request_id, labels=request_labels: labels[request_id],
        index=ids.index(state.latest_request_id) if state.latest_request_id in ids else 0, key='result_request')
    outcome = state.gateway.get_outcome(selected) or state.submitted_requests[selected]
    verdict = outcome.supervision.decision.decision.value if outcome.supervision else 'Not assessed'
    status_message(outcome.status, f'{verdict} · Gateway outcome: {outcome.status}')

overview_tab, workspace_tab, evaluation_tab, approval_tab, history_tab = st.tabs([
    'Overview', 'Agent Workspace', 'Supervisory Evaluation',
    'Decision & Approval', 'Execution History & Audit'])

with overview_tab:
    render_overview(state)

with workspace_tab:
    st.subheader('Agent workspace')
    st.caption(f'{department} · {scenario.name}. Scenario selection changes the next proposal, not existing requests.')
    trusted, proposed = st.columns([1, 1], gap='medium')
    agent, delegation = state.authority_store.resolve(agent_id, scenario.task_id)
    with trusted:
        render_trusted_agent(agent, delegation, state.authority_store.grants_for(agent_id))
    with proposed, st.container(border=True):
        st.subheader('Proposed tool action')
        st.caption('Editable agent proposal · these inputs do not provision authority.')
        action = st.selectbox('Action', list(TOOL_ACTIONS.values()), key='proposal_action_type')
        description = st.text_area('Proposed action description', key='proposal_description', height=100)
        with st.expander('Optional agent task claim · untrusted'):
            claim = st.text_area('Agent task description (untrusted)', key='proposal_claim')
            st.caption('This text is captured as a claim. It cannot replace the trusted objective or grant rights.')
        targets, scopes = st.columns(2)
        with targets:
            target = st.selectbox('Target system', ['crm', 'payments', 'payroll', 'infrastructure'], key='proposal_target_system')
            environment = st.selectbox('Environment', ['development', 'staging', 'production'], key='proposal_environment')
        with scopes:
            classification = st.selectbox('Data classification', ['public', 'internal', 'confidential', 'restricted'], key='proposal_data_classification')
            has_amount = st.checkbox('Include amount', key='proposal_has_amount')
            amount = st.number_input('Amount', min_value=0.0, step=10.0, key='proposal_amount', disabled=not has_amount)
        reversible = st.checkbox('Reversible', key='proposal_reversible')
        external = st.checkbox('External effect', key='proposal_external_effect')
        if st.button('Submit Action', type='primary', key='submit_action', width='stretch'):
            try:
                call = ProposedToolCall(tool_name=f'enterprise.{action}', arguments=RequestedAction(
                    action_type=action, description=description, target_system=target,
                    amount=amount if has_amount else None, environment=environment,
                    data_classification=classification, reversible=reversible, external_effect=external),
                    claimed_task_description=claim or None)
            except ValidationError:
                st.error('Invalid proposed action. Enter a description and valid finite, non-negative amount.')
            else:
                with st.spinner('Processing gateway assessment and authorization. Model inference may take time…', show_time=True):
                    submitted = state.gateway.submit(agent_credential=state.agent_credentials[agent_id],
                        task_id=scenario.task_id, tool_call=call, idempotency_key=str(uuid4()), use_model=use_model)
                remember_submission(state, submitted)
                state.dashboard_inspect_request = submitted.request_id
                st.rerun()

with evaluation_tab:
    st.subheader('Supervisory workflow')
    events = [e for e in state.audit_trail.records() if outcome and e['request_id'] == outcome.request_id]
    render_workflow(outcome, events)
    if outcome and outcome.supervision:
        render_assessment(outcome.supervision.assessment)
        render_authority(outcome.supervision.authority)
    else:
        st.info('Submit an action to inspect its actual authority findings and six model assessments. No stages or predictions are simulated.')

with approval_tab:
    st.subheader('Governance decision')
    render_verdict(outcome)
    st.subheader('Human approval queue')
    st.caption(f'{len(state.pending_approvals)} pending approval request(s)')
    st.caption('Simulated reviewer selection is not production authentication. Approval binds to the original frozen action and rechecks trusted authority.')
    if state.pending_approvals:
        approval_labels = {}
        for pending_id, pending_record in state.pending_approvals.items():
            frozen = ProposedToolCall.model_validate_json(pending_record.tool_call_json)
            approval_labels[pending_id] = (f'{frozen.arguments.action_type} → {frozen.arguments.target_system}'
                f' · {pending_record.agent_id} · {pending_id[:8]}')
        approval_id = st.selectbox('Pending approval request', list(state.pending_approvals),
            format_func=lambda value, labels=approval_labels: labels[value], key='selected_approval')
        record = state.gateway.get_approval(approval_id)
        if record:
            with st.container(border=True):
                render_frozen_action(record)
                reviewer = st.selectbox('Simulated trusted reviewer', list(state.reviewer_credentials), key='reviewer')
                identity = state.reviewer_credentials[reviewer][1]
                st.caption(f'Reviewer identity: {identity.approver_id} · trusted roles: {", ".join(identity.roles)}')
                approve_col, reject_col = st.columns(2)
                approve = approve_col.button('Approve original action', type='primary', key='approve_action', disabled=record.status != 'PENDING', width='stretch')
                reject = reject_col.button('Reject original action', key='reject_action', disabled=record.status != 'PENDING', width='stretch')
                if approve or reject:
                    credential = state.reviewer_credentials[reviewer][0]
                    with st.spinner('Verifying reviewer, frozen action binding and current authority…'):
                        if approve:
                            attempt = state.gateway.approve(approval_id=record.approval_id,
                                approver_credential=credential, interception_id=record.interception_id,
                                tool_call=ProposedToolCall.model_validate_json(record.tool_call_json))
                        else:
                            attempt = state.gateway.reject_approval(approval_id=record.approval_id, approver_credential=credential)
                    state.last_approval_attempt = attempt
                    state.latest_request_id = record.request_id
                    state.dashboard_inspect_request = record.request_id
                    refresh_requests(state)
                    st.rerun()
    else:
        st.info('No pending approvals. Held requests appear here until approved, rejected or expired.')
    if state.last_approval_attempt:
        attempt = state.last_approval_attempt
        reviewed_call = attempt.supervision.intercepted.tool_call if attempt.supervision else None
        reviewed_action = (f'{reviewed_call.arguments.action_type} → {reviewed_call.arguments.target_system}'
            if reviewed_call else 'Unassessed request')
        status_message(attempt.status, f'Last reviewer attempt for {reviewed_action} ({attempt.request_id[:8]}): '
            f'{attempt.status} · {attempt.reason or "Original action executed"}')
        if outcome and outcome.request_id != attempt.request_id:
            st.caption('This reviewer attempt belongs to a different submitted request than the selected verdict.')
        st.caption(f'Request {attempt.request_id}. A denied reviewer attempt does not necessarily close the pending request; consult its actual gateway outcome.')

with history_tab:
    render_history(state)
