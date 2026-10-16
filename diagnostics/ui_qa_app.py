"""Browser QA harness: runs the actual app with existing unit-test model fixtures.

Never used by the normal launch command. No real model can load in this harness.
"""
from pathlib import Path
import runpy
import sys
import importlib
from unittest.mock import patch
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import decision_model

# The harness lives below the normal app directory. Refresh presentation modules
# explicitly so browser QA reruns reflect edits without restarting the gateway.
for module_name in ('src.dashboard_view', 'src.dashboard_components'):
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

factory = runpy.run_path(str(ROOT / 'tests/conftest.py'))['advice_factory'].__wrapped__()
fixture = st.sidebar.selectbox('Visual QA model fixture', ['Approval required', 'Allow', 'Unavailable'], key='qa_fixture')
st.sidebar.caption('MOCKED MODEL · visual QA only. Real inference disabled.')
advice = ({'status': 'unavailable', 'error': 'Visual QA fixture: model unavailable'} if fixture == 'Unavailable'
          else factory(human_approval='YES', autonomous='NO') if fixture == 'Approval required'
          else factory())
with patch.object(decision_model, 'get_advice', return_value=advice), patch.object(
        decision_model, 'load_model', side_effect=AssertionError('Real inference disabled in UI QA')):
    runpy.run_path(str(ROOT / 'app.py'), run_name='__main__')
