"""Start the local Streamlit server, verify health, and stop the child process."""
import json
import argparse
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['stage1', 'stage2'], default='stage1')
stage = parser.parse_args().stage
with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
run_id = uuid4().hex
log_path = ROOT / f'results/{stage}_streamlit_{run_id}.log'
with log_path.open('x', encoding='utf-8') as log:
    child = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app.py'),
        '--server.headless=true', '--server.address=127.0.0.1', f'--server.port={port}',
        '--browser.gatherUsageStats=false'], cwd=ROOT, stdout=log, stderr=log)
    try:
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError(f'Streamlit exited with {child.returncode}; inspect {log_path}')
            try:
                with urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=1) as response:
                    body = response.read().decode()
                    if response.status == 200 and body == 'ok':
                        output = {'status': 'started', 'health': body, 'http_status': 200, 'port': port,
                                  'log_path': str(log_path), 'server_stopped_after_check': True}
                        break
            except OSError:
                pass
            time.sleep(.2)
        else:
            raise TimeoutError('Streamlit health check timed out')
    finally:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
with (ROOT / f'results/{stage}_streamlit_startup_{run_id}.json').open('x', encoding='utf-8') as result:
    json.dump(output, result, indent=2)
print(json.dumps(output, indent=2))
