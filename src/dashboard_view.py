"""Read-only presentation projections. These never authorize or execute actions."""
from datetime import datetime, timezone
from src.mock_tools import MockToolRegistry

ASSESSMENT_LABELS = {
    'task_alignment': 'Task alignment',
    'authority_drift': 'Authority drift',
    'within_authority': 'Within delegated authority',
    'human_approval': 'Human approval needed',
    'consequence': 'Potential consequences',
    'autonomous': 'Autonomous execution suitable',
}


def format_timestamp(value):
    """Compact UTC display; full-precision source remains in expanded evidence."""
    if not value:
        return 'Not recorded'
    instant = datetime.fromisoformat(value)
    return instant.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def session_metrics(outcomes):
    outcomes = list(outcomes)
    evaluated = [o for o in outcomes if o.supervision is not None]
    return {
        'Total evaluated actions': len(evaluated),
        'Allowed actions': sum(o.supervision.decision.decision == 'ALLOW' for o in evaluated),
        'Pending approvals': sum(o.status == 'PENDING_APPROVAL' and o.approval is not None
                                 and o.approval.status == 'PENDING' for o in outcomes),
        'Blocked actions': sum(o.supervision.decision.decision == 'BLOCK' for o in evaluated),
        'Executed actions': sum(o.status == 'EXECUTED' for o in outcomes),
    }


def technical_capabilities(systems):
    """Read registry metadata only: tool availability is not an authorization grant."""
    return [{'Tool interface': tool, 'Target systems': ', '.join(sorted(set(systems) & targets))}
            for tool, targets in MockToolRegistry.SYSTEMS.items() if set(systems) & targets]


def grant_rows(grants):
    return [{'Grant ID': g.grant_id, 'Tool': g.tool_name, 'System': g.target_system,
             'Environment': g.environment, 'Data': g.data_classification,
             'Amount limit': g.max_amount, 'External effect': g.allow_external_effect,
             'Irreversible': g.allow_irreversible, 'Version': g.version} for g in grants]


def action_rows(call):
    action = call.arguments
    return [{'Parameter': label, 'Frozen value': str(value)} for label, value in (
        ('Tool', call.tool_name), ('Action', action.action_type), ('Target system', action.target_system),
        ('Amount', action.amount if action.amount is not None else 'Not specified'),
        ('Environment', action.environment), ('Data classification', action.data_classification),
        ('Reversible', action.reversible), ('External effect', action.external_effect))]


def probability_rows(assessment):
    if assessment.status != 'available':
        return []
    answers = assessment.advice.get('result', {}).get('answers', {})
    return [{'Assessment': label, 'Answer': option, 'Probability': probability,
             'Selected': option == answers[key]['choice']}
            for key, label in ASSESSMENT_LABELS.items() if key in answers
            for option, probability in answers[key]['probabilities'].items()]


def conflict_notes(assessment):
    """Describe observed answer combinations; never invent model reasoning."""
    if assessment.status != 'available':
        return []
    answers = assessment.advice.get('result', {}).get('answers', {})
    choices = {key: value['choice'] for key, value in answers.items()}
    notes = []
    if choices.get('autonomous') == 'YES':
        for key, adverse in (('task_alignment', 'NO'), ('authority_drift', 'YES'),
                             ('within_authority', 'NO'), ('human_approval', 'YES')):
            if choices.get(key) == adverse:
                notes.append(f'Autonomous execution = YES conflicts with {ASSESSMENT_LABELS[key]} = {adverse}.')
        if choices.get('consequence') in ('HIGH', 'CRITICAL'):
            notes.append(f'Autonomous execution = YES accompanies potential consequences = {choices["consequence"]}.')
    elif choices.get('autonomous') == 'NO' and choices.get('human_approval') == 'NO':
        notes.append('Divergent judgments: human approval needed = NO, but autonomous execution suitable = NO.')
    return notes


def workflow_states(outcome, events):
    """Completed stages require backend evidence; no speculative progress."""
    seen = {e['event'] for e in events}
    result = outcome.supervision if outcome else None
    return [
        ('Action Proposed', 'Recorded' if 'action_proposed' in seen else 'Awaiting evidence'),
        ('Intercepted', 'Captured' if 'action_intercepted' in seen else 'Awaiting evidence'),
        ('Authority Verified', result.authority.status if result else 'Awaiting evidence'),
        ('Decision 2.0 Assessment', result.assessment.status.upper() if result else 'Awaiting evidence'),
        ('Decision Controller', result.decision.decision.value if result else 'Awaiting evidence'),
        ('Execution Gateway', outcome.status if outcome else 'Awaiting evidence'),
        ('Audit', f'{len(events)} events recorded' if events else 'Awaiting evidence'),
    ]


def audit_rows(events):
    """Allowlisted audit summary: no arbitrary payloads or host credentials."""
    return [{'Timestamp (UTC)': e['timestamp'], 'Event': e['event'],
             'Execution status': e.get('execution_status', ''),
             'Approval status': e.get('approval_status', ''),
             'Reason': e.get('reason', ''), 'Event ID': e['event_id'],
             'Request ID': e['request_id'], 'Interception ID': e.get('interception_id', ''),
             'Approval ID': e.get('approval_id', ''), 'Agent ID': e.get('agent_id', ''),
             'Task ID': e.get('task_id', '')} for e in events]


def audit_summary_rows(events):
    """Compact event timeline; full correlation fields remain separately available."""
    return [{'Timestamp (UTC)': format_timestamp(e['timestamp']), 'Event': e['event'],
             'Status': e.get('execution_status') or e.get('approval_status')
                       or e.get('authorization_outcome') or e.get('model_status') or '',
             'Reason': e.get('reason', '')} for e in events]


def history_rows(outcomes, events):
    grouped = {}
    for event in events:
        grouped.setdefault(event['request_id'], []).append(event)
    rows = []
    for outcome in outcomes:
        result = outcome.supervision
        records = grouped.get(outcome.request_id, [])
        proposal = next((e for e in records if e['event'] == 'action_proposed'), None)
        submitted = proposal['timestamp'] if proposal else (
            result.intercepted.captured_at.isoformat() if result else '')
        action = result.intercepted.tool_call.arguments if result else None
        rows.append({'Submitted (UTC)': submitted, 'Last audit event (UTC)': records[-1]['timestamp'] if records else '',
            'Agent': result.intercepted.agent_id if result else 'Unknown',
            'Requested action': action.action_type if action else 'Unknown',
            'Target system': action.target_system if action else 'Unknown',
            'Governance decision': result.decision.decision.value if result else 'Not assessed',
            'Execution outcome': outcome.status,
            'Mock tool invoked': outcome.execution_result.get('tool_name', 'Invocation failed') if outcome.execution_result else 'None',
            'Approval lifecycle': outcome.approval.status if outcome.approval else 'Not required',
            'Request ID': outcome.request_id})
    return sorted(rows, key=lambda row: row['Submitted (UTC)'])
