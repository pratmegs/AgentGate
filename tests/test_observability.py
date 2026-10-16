import io
import logging
from unittest.mock import Mock
import pytest
from src import decision_model
from src.observability import configure_tracing, trace, request_context, VERBOSE, TerminalTraceHandler
from src.governance import supervise
from src.demo_trace import run


@pytest.fixture
def terminal():
    stream = io.StringIO()
    logger = configure_tracing('VERBOSE', stream=stream, force_terminal=False)
    yield stream, logger
    configure_tracing('NORMAL', force_terminal=False)


@pytest.mark.parametrize('setting,expected', [
    ('QUIET', ['error']), ('NORMAL', ['normal', 'error']),
    ('VERBOSE', ['verbose', 'normal', 'error']), ('DEBUG', ['debug', 'verbose', 'normal', 'error'])])
def test_levels(setting, expected, terminal):
    stream, _ = terminal
    configure_tracing(setting)
    for name, level in [('debug', logging.DEBUG), ('verbose', VERBOSE), ('normal', logging.INFO), ('error', logging.ERROR)]:
        trace('WORKFLOW', f'event-{name}', level=level)
    text = stream.getvalue()
    assert [name for name in ('debug', 'verbose', 'normal', 'error') if f'event-{name}' in text] == expected
    assert '\x1b[' not in text


def test_environment_and_no_duplicate_handlers(monkeypatch, terminal):
    _, logger = terminal
    monkeypatch.setenv('AGENTGATE_LOG_LEVEL', 'DEBUG')
    for _ in range(5):
        configure_tracing()
    assert logger.level == logging.DEBUG
    assert len([h for h in logger.handlers if getattr(h, '_agentgate_trace_handler', False)]) == 1
    trace('WORKFLOW', 'single-line')
    assert terminal[0].getvalue().count('single-line') == 1


def test_logging_does_not_change_decisions_and_handlers_cannot_break_them(monkeypatch, terminal, supervisory_input, advice_factory):
    monkeypatch.setattr(decision_model, 'get_advice', lambda *args: advice_factory())
    configure_tracing('QUIET')
    quiet = supervise(**supervisory_input)
    configure_tracing('DEBUG')
    loud = supervise(**supervisory_input)
    assert quiet.decision.decision == loud.decision.decision == 'ALLOW'
    assert quiet.authority == loud.authority and quiet.assessment == loud.assessment
    handler = next(h for h in terminal[1].handlers if isinstance(h, TerminalTraceHandler))
    monkeypatch.setattr(handler, 'emit', Mock(side_effect=OSError('broken terminal')))
    broken = supervise(**supervisory_input)
    assert broken.decision.decision == 'ALLOW'


def test_safe_metadata_and_model_error_do_not_leak_payloads(terminal, monkeypatch, supervisory_input):
    sentinel = 'VERY_PRIVATE_PAYLOAD'
    supervisory_input['tool_call'].arguments.description = sentinel
    supervisory_input['tool_call'].claimed_task_description = sentinel
    monkeypatch.setattr(decision_model, 'load_model', Mock(side_effect=RuntimeError(sentinel)))
    trace('WORKFLOW', 'Safe metadata.', agent_credential=sentinel, password=sentinel,
          tool_arguments={'description': sentinel}, original_goal=sentinel)
    result = supervise(**supervisory_input)
    assert result.decision.decision == 'REQUIRE_APPROVAL'
    assert sentinel not in terminal[0].getvalue()
    assert 'RuntimeError' in terminal[0].getvalue()


def test_correlation_ids_distinguish_requests(terminal, monkeypatch, supervisory_input, advice_factory):
    monkeypatch.setattr(decision_model, 'get_advice', lambda *args: advice_factory())
    with request_context('correlation-one'):
        supervise(**supervisory_input)
    with request_context('correlation-two'):
        supervise(**supervisory_input)
    output = terminal[0].getvalue()
    assert 'request=correlation-one' in output and 'request=correlation-two' in output
    assert 'interception_id=' in output


def test_inference_start_is_emitted_before_forward_pass_and_no_repeat(terminal, monkeypatch, supervisory_input, advice_factory):
    agent, delegation = supervisory_input['authority_store'].resolve('it-agent', 'task-1')
    from src.schemas import TaskContext
    context = TaskContext(original_goal=delegation.original_goal, requested_action=supervisory_input['tool_call'].arguments)
    expected = advice_factory()['result']
    def forward(**kwargs):
        output = terminal[0].getvalue()
        assert 'Inference started' in output
        assert 'Inference completed' not in output
        return expected
    model = Mock()
    model.system_one.side_effect = forward
    monkeypatch.setattr(decision_model, 'load_model', lambda: model)
    result = decision_model.get_advice(agent, context)
    assert result['result'] == expected
    assert model.system_one.call_count == 1
    output = terminal[0].getvalue()
    assert 'Inference completed' in output and 'duration_s=' in output
    assert 'question=task_alignment' in output and 'probability=0.9' in output


def test_no_model_demo_is_not_autonomous_allow(terminal):
    output = run(no_model=True)
    assert all(s['actual']['supervision']['decision']['decision'] != 'ALLOW' for s in output['scenarios'])
    assert output['scenarios'][0]['actual']['status'] == 'PENDING_APPROVAL'
    assert output['scenarios'][1]['after_review']['status'] == 'EXECUTED'
    assert output['scenarios'][2]['actual']['status'] == 'BLOCKED'
    assert len(output['mock_tools']['invocations']) == 1
    assert 'Desired branch was not reached' in terminal[0].getvalue()
