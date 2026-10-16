# AgentGate

Capability does not imply authority. AgentGate intercepts proposed tool calls and
combines real Decision 2.0 assessments with trusted authority verification.

> **Experimental research prototype.** AgentGate uses in-memory mock tools and
> simulated identities. It is not a production authorization system and must not be
> connected to real enterprise tools, credentials, or financial workflows without
> independent security engineering and validation.

```text
Enterprise Agent → Action Interceptor → Decision 2.0 + Authority Verification
                 → Decision Controller → Execution Gateway → Audit Trail
```

The supervisory core feeds a simulated execution gateway and approval workflow.
Registered mock tools operate only on in-memory state. There are no real enterprise
connections or financial transactions. The Streamlit console submits actions through
the gateway and exposes its actual outcomes, pending approvals and audit evidence.
The interface organizes session evidence into a dark Streamlit governance console.

## Run (PowerShell)

```powershell
cd D:\Megha\AgentGate
uv sync --extra decision
uv run --extra decision streamlit run app.py
```

## Governance console

Five sections keep the workflow and evidence easy to locate:

- **Overview:** actual session counts for evaluated, allowed, pending, blocked and
  executed actions. ALLOW verdicts are distinct from approved executions.
- **Agent Workspace:** registered mock capabilities, read-only trusted authority,
  original objectives and editable proposed tool parameters.
- **Supervisory Evaluation:** backend workflow milestones, all six Decision 2.0
  answers, uncalibrated probability bars, full distributions and separate authority
  findings. Bars show model scores, not inference progress.
- **Decision & Approval:** the selected request's verdict and execution status, plus
  pending requests with their original frozen actions and simulated reviewer controls.
- **Execution History & Audit:** chronological submissions, approval lifecycle states,
  gateway events, timestamps and correlation identifiers in expandable tables.

The console reuses Stage 3A sessions and gateway callbacks. It does not infer again
to populate views. `STAGE3B_REPORT.md` documents automated validation and the browser
and live-model checks left for the user's final review. No live inference or manual
browser demonstration was performed during Stage 3B.

## Live terminal tracing

AgentGate emits live component milestones through standard Python logging with a
Rich terminal handler. Configure it before starting the app or demo:

```powershell
cd D:\Megha\AgentGate
$env:AGENTGATE_LOG_LEVEL = 'VERBOSE'
uv run --extra decision python -m src.demo_trace
```

The demonstration requests real Decision 2.0 assessments for three scenarios:
a harmless public ticket-count read, a payment configuration change under an
investigative objective, and a refund outside its explicit financial grant. Desired
branches are ALLOW, REQUIRE_APPROVAL and BLOCK, but actual model outcomes are always
reported. The program never changes a threshold or fabricates predictions to force
a branch. The review scenario simulates a host-provisioned human reviewer; no
external approval or enterprise system is contacted.

For model-disabled diagnostics:

```powershell
uv run python -m src.demo_trace --no-model
```

Disabled inference cannot produce autonomous ALLOW. A simulated authorized human
review may still release a permitted held action through the existing gateway.
Each demonstration saves a new `results/terminal_demo_<id>.json` and a separate
append-only audit file. Terminal traces do not replace or weaken persistent audit.

| Setting | AgentGate terminal events |
|---|---|
| QUIET | ERROR and CRITICAL only |
| NORMAL (default) | High-level workflow milestones |
| VERBOSE | Component steps, model choices/probabilities and audit writes |
| DEBUG | VERBOSE plus diagnostic transitions |

VERBOSE uses logging level 15, between DEBUG and INFO. Repeated configuration does
not add duplicate handlers. The dedicated logger does not change root, Streamlit
or third-party loggers. Their existing warnings/loading progress may still appear.
Rich detects terminal color support; redirected output and `NO_COLOR` remain
readable. Timestamps are UTC (`Z`). `+...s` is elapsed time for the current entry
point; approval callbacks reuse the original request ID but start a new call timer.
Model loading and inference durations are separately measured actual milestones.
No internal reasoning or invented progress percentages are displayed.

Tracing only forwards selected IDs, scope metadata, status/reason codes and validated
bounded model choices. Credentials, task text, tool descriptions, argument objects
and exception payloads are not forwarded. Terminal write failures are best-effort
and cannot authorize an action or suppress enforced audit errors. Governance
policies, model questions, thresholds and execution controls remain unchanged.

Nine predefined business scenarios span four demo agents. Supervision is enabled by
default. Without optional dependencies or valid model inference, decisions require
human review or are blocked; there is no autonomous fallback to baseline policies.
The sidebar switch can disable inference, but doing so cannot produce ALLOW.

## Core components

- `action_interceptor.py`: copies the proposed registered tool contract, arguments,
  task reference and trusted delegated context before any execution.
- `authority_verifier.py`: resolves identities and structured delegation records
  from a host-provisioned `AuthorityStore`. Checks permissions, task ownership,
  actions, target systems, environments, classifications, financial boundaries,
  external effects and irreversible actions. A single trusted `PermissionGrant`
  must cover the exact agent/tool/action/system/environment/classification and
  amount. Separate grants cannot be joined to create authorization. Text cannot
  create authority. No grants means UNRESOLVED; incompatible grants mean PROHIBITED.
- `assessment.py` and `decision_model.py`: invoke the verified Decision 2.0
  `system_one` API to assess alignment, authority drift, task authority, consequences,
  approval need and autonomous suitability. All six native probability maps are
  preserved. Missing, ambiguous, inconsistent or nonfinite predictions are rejected.
- `decision_controller.py`: verified prohibitions always BLOCK. Unresolved authority
  or failed/disabled inference requires review. Otherwise model misalignment, drift,
  approval need, unsuitable autonomy, HIGH/CRITICAL consequences or a selected
  probability below the configured threshold requires approval. Contradictions
  never allow autonomy. `ControllerConfig(probability_threshold=0.6)` retains the
  experimental, uncalibrated default. Authorization outcome, model consequence
  severity and explanation are separate; BLOCK does not change LOW model severity.
- `governance.py`: orchestration and the existing UI compatibility facade.

`supervise()` remains a decision-only API. Simulated execution is available only
through `ExecutionGateway.submit(agent_credential=..., task_id=..., tool_call=...)`.
The gateway invokes supervision internally; it cannot accept agent-supplied ALLOW
objects as execution authority. Agent and approver credentials are opaque tokens
provisioned by the trusted host, with server-side identity and role mappings.

The UI shows read-only trusted identities, scoped grants and delegation limits
from `dashboard_session.py`. Editable fields are agent proposals; their text cannot
provision permissions. Gateway, authority, audit, submissions and approvals survive
normal Streamlit reruns. Scenario changes affect only the next proposal. The approval
panel uses the original frozen action and simulated host-provisioned reviewer
credentials. Reviewer selection is not production authentication. A confirmed reset
starts a fresh gateway session while retaining prior audit files.

## Model verification and Windows compatibility

Decision 2.0 remains pinned to `cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764` using
Transformers with `trust_remote_code=True`; no other LLM was added. The Windows-only
fingerprint path normalization in `decision2_compat.py` is preserved. Manifest file
hashes, checkpoint identity and loaded parameter count are still checked. Cached
code, manifests and weights are never edited. See the original investigation in
`results/decision2_diagnostics/REPORT.md` and the
[official model card](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B).

## Tests

Ordinary tests use mocks for model edge cases and never download the checkpoint:

```powershell
uv run pytest -q --basetemp .tmp/unit-tests
```

Run actual model inference against the existing verified cache:

```powershell
uv run --extra decision pytest -q --real-model --basetemp .tmp/real-tests
```

Opt-in real tests run offline against the existing verified checkpoint. They verify
supervision and the complete approval-to-mock-execution path, preserving actual
inputs, probabilities, approval status and audit events in unique result files.
No historical result is overwritten. The real dashboard test also saves its actual
model-informed submission and approval output. See `STAGE3A_REPORT.md` for results.

## Gateway and approval workflow

`execution_gateway.py` is the only simulated execution entry point. ALLOW rechecks
current scoped grants and delegation before invoking a registered mock tool.
BLOCK never invokes a tool. REQUIRE_APPROVAL stores a PENDING immutable snapshot
of the proposed action, delegation version, trusted authorization context and
controller decision in `approval_manager.py`.

`approve()` requires the host-provisioned approver credential, original interception
ID and exact tool call. It checks the required role, expiry, action binding and
current authority before execution. Unresolved original authority cannot be cured
by an approval; a newly authorized proposal must be submitted. Revoked permissions,
changed delegation versions, changed actions, expired approvals and replays are
rejected. `reject_approval()` records human rejection. Approval TTL defaults to 300
seconds and is configurable through `GatewayConfig(approval_ttl_seconds=...)`.

Approvals move through PENDING, APPROVED and CONSUMED, or become REJECTED/EXPIRED.
Consumption reserves one execution attempt, including tool failure. Gateway and
authority-store locks serialize approval, revocation checks and simulated execution.
Concurrent approval attempts cannot execute the same request twice within one
gateway. An optional idempotency key protects repeated agent submissions.

`mock_tools.py` supports CRM reads, mock refunds, employee-record version updates
and infrastructure/payment configuration version updates. Invocation requires the
gateway-owned capability. `audit_trail.py` appends JSONL lifecycle events and
preserves model probabilities separately from verified authority facts. It logs
action hashes and scope metadata rather than raw task text, credentials or tool
descriptions. Local files are **not tamper-proof**.

Run the three enforcement demonstrations:

```powershell
uv run python diagnostics/stage2_demo.py
```

This script explicitly uses mocked model judgments to demonstrate ALLOW execution,
approval release and BLOCK enforcement. Its output is labeled accordingly. It is
not a real-model quality test; actual inference is exercised by `--real-model`.

## Separate deterministic baseline

`policy_engine.py` and `authority_drift.py` preserve the original independently
testable baseline. They do not drive supervisory decisions. `evaluate_baseline()`
is an explicitly named legacy evaluation path. The existing evaluator still
measures that baseline and writes unique `baseline_evaluation_<id>.json` files;
its regression tests write only to temporary workspace files.

Historical `results/evaluation.json`, `results/comparison.json`, model probes and
diagnostic outputs are preserved. Their prior metrics are not metrics for this
supervisory architecture. The old comparison utility describes the earlier baseline
plus advice design; no new comparison or benchmark has been run for Stage 2.

## Limits

Model judgment is probabilistic and has not been calibrated or benchmarked for this
stage. Structured tool arguments and identities need an authenticated trusted host
in production. Credential provisioning, grants, mock state, approvals and replay
protection are in memory and scoped to one gateway process. Python-private
capabilities protect the gateway API boundary, not against malicious Python code
with access to host memory. Restart recovery, multi-process coordination and
transactional execution/audit commits are not implemented. Audit I/O failure after
a simulated effect may leave incomplete evidence; consumed approvals and reserved
idempotency keys still prevent retrying that attempt within the running gateway.
Transformers still emits the upstream model-type and
tokenizer-regex warnings observed in the original investigation. The compatibility
hook uses private APIs from a pinned runtime and should be replaced by an upstream
portability correction when available. No cloud infrastructure or new dependencies
were introduced.

## License

AgentGate source code is released under the [MIT License](LICENSE). The optional
Decision 2.0 model is downloaded separately and is governed by its own Apache 2.0
license and model terms.
