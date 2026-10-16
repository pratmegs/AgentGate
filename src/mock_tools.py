"""In-memory enterprise simulation. Only the gateway holds invocation capability."""
from copy import deepcopy
from src.action_interceptor import ProposedToolCall


class MockToolRegistry:
    SYSTEMS = {
        'enterprise.read': {'crm', 'payments', 'payroll', 'infrastructure'},
        'enterprise.refund': {'crm', 'payments'},
        'enterprise.update': {'payroll', 'crm', 'infrastructure'},
        'enterprise.configure': {'infrastructure', 'payments'},
    }

    def __init__(self, capability: object):
        self.__capability = capability
        self._invocations: list[dict] = []
        self._state = {'crm': {'tickets': 3}, 'payments': {'refunds': 0},
                       'payroll': {'employee_version': 0}, 'infrastructure': {'configuration_version': 0}}

    def supports(self, call: ProposedToolCall) -> bool:
        return call.arguments.target_system in self.SYSTEMS.get(call.tool_name, set())

    def _invoke(self, capability: object, call: ProposedToolCall) -> dict:
        if capability is not self.__capability:
            raise PermissionError('Mock tools require gateway authorization')
        if not self.supports(call):
            raise ValueError('Unsupported mock tool/system combination')
        action = call.arguments
        state = self._state[action.target_system]
        if action.action_type == 'refund':
            state['refunds'] = state.get('refunds', 0) + 1
        elif action.action_type == 'update':
            state['employee_version'] = state.get('employee_version', 0) + 1
        elif action.action_type == 'configure':
            state['configuration_version'] = state.get('configuration_version', 0) + 1
        result = {'simulated': True, 'tool_name': call.tool_name, 'target_system': action.target_system,
                  'invocation_number': len(self._invocations) + 1, 'status': 'SUCCEEDED',
                  'simulated_state': deepcopy(state)}
        self._invocations.append(deepcopy(result))
        return result

    def snapshot(self) -> dict:
        return {'state': deepcopy(self._state), 'invocations': deepcopy(self._invocations)}
