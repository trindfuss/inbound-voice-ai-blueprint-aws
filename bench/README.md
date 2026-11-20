# Latency benchmarks

Four scripts in `scripts/bench/` measure latency on your own deployment. Each writes JSON to `bench/results/`, which is git-ignored and created on the first run, so your runs are never committed by accident. Run every command from the repository root.

| Script | Measures | Needs |
|---|---|---|
| `agent_bench.py` | The agent stage alone: fixed caller scenarios through the bridge's `AgentClient` interface, every streamed event timed. Repeatable, no phone | AWS credentials for the account the stack is deployed in |
| `call_latency.py` | Real calls end to end, from the deployed system's logs and the agent's per-model-call timings | Calls made since the `--since` time, and read access to their CloudWatch Logs |
| `slot_model_bench.py` | The model-assisted slot resolver per mode: latency, tokens and cost per 1,000 calls over fixed synthetic answers | AWS credentials, and Bedrock model-invocation logging that is off or doesn't log text (see "Logging guard" below) |
| `handshake_timer.py` | The WebSocket handshake to the AgentCore Runtime, on new and on pre-warmed sessions (`bridgeHosting=agentcore` only) | AWS credentials, and `infra/cdk-outputs.json` or `--runtime-arn` |

## Prerequisites

- `uv sync` has run, and the AWS profile you use can call the services below.
- Deploy with `--outputs-file cdk-outputs.json` from `infra/`, so the scripts find the stack's outputs (`GatewayUrl`, `MemoryId`, `RuntimeArn`, `RuntimeLogGroup`). `agent_bench.py` and `call_latency.py` read `infra/cdk-outputs.json` by default; `handshake_timer.py` takes `--cdk-outputs` or `--runtime-arn`.
- Permissions for the local user:
  - `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on the inference profile and its foundation models;
  - `bedrock:GetModelInvocationLoggingConfiguration`, for the logging guard in `slot_model_bench.py`;
  - `bedrock-agentcore:InvokeGateway` on the Gateway (its inbound auth is IAM, SigV4);
  - `bedrock-agentcore:CreateEvent`, `GetEvent` and `ListEvents` on the Memory, and use of the Memory's KMS key;
  - read access to the bridge's and the front door's CloudWatch log groups, for `call_latency.py`.

The scripts call real AWS services and bill for what they use. Every utterance and tool result they send is synthetic and fictional.

## Agent benchmark

```bash
uv run python scripts/bench/agent_bench.py --runs 8                               # warm, like production
uv run python scripts/bench/agent_bench.py --label strands-cold --cold            # no warm-up
uv run python scripts/bench/agent_bench.py --compare bench/results/A.json bench/results/B.json
```

`agent_bench.py` runs a real `StrandsAgentClient` in your process, against the deployed Gateway and Memory. It needs no bridge deploy. Prompt caching is a flag (`--prompt-caching on|off`, default `off`): the client is local, so the stack's `promptCaching` setting only matters for real calls.

What it reads from your stack:

- `infra/cdk-outputs.json` for the Gateway URL and the Memory ID, or `--gateway-url` and `--memory-id`.
- The model settings (model ID, max tokens, reasoning) from the synthesized `infra/cdk.out/OwlBankBridge.template.json`, written by your last `cdk synth` or `cdk deploy` from this checkout (`-c` flags included). Without that file it falls back to the defaults in `infra/owlbank_infra/config.py` and `cdk.json`'s context. `--model-id`, `--max-tokens` and `--thinking` override them, and the result file records which source was used.

Scenarios (one step is one caller utterance; tool results are stubs from the fictional fixtures):

| Scenario | Steps |
|---|---|
| `greeting` | "Hi, what can you help me with today?" (no tool) |
| `faq_gateway` | "What are your branch hours on Saturday?" (a Gateway tool, or a clarifying question) |
| `transactions_inline` | Recent transactions: an inline tool hop, then the answer |
| `balance_then_bargein` | Balance (hand-off to the front door), then the caller's words over the read-out arrive with the result and lead to transactions |
| `payment_stepup` | Payment: `AUTH_LEVEL_INSUFFICIENT`, step-up hand-off, retry hand-off to the scripted confirmation, then "transfer me to an agent" |

`--scenarios` picks a subset. `--cold` skips the warm-up. `--tracing on|off` (default `off`) installs the bridge's telemetry in the process, so you can compare a run with call observability against one without.

### Reading the results

`response_s` is what the caller waits for in a step: the time to the first words the bridge would send or, for a step that ends in a scripted step, the time to that hand-off. Stub tool results take no time; a real `bank_tools` call typically adds 0.1 to 0.3 s.

Each step's audio is modelled from its measured timings and the masking settings (`data/policy.json` and the en-US catalog), with no AWS call:

- `first_audio`: the first thing the caller hears (the timed acknowledgement, a masked tool call's phrase, or the answer).
- `max_silence`: the longest stretch with nothing heard.
- `phrase_delay`: how long a ready answer waited behind phrases still playing (0 if none).

`--masking-model v2` (default) schedules each tool phrase after the tool's delay and the acknowledgement's estimated end, and speaks it only if the answer isn't ready by then. `--masking-model legacy` speaks a tool phrase the moment its call arrives. Phrase durations are `0.3 s + characters / chars_per_s` for the longest phrase of each pool, so the estimates tend to be conservative.

Re-compute a saved run with other masking settings, offline:

```bash
uv run python scripts/bench/agent_bench.py --resummarize bench/results/FILE.json \
    --after-ms 2000 --tool-after-ms 600 --tool-delay list_recent_transactions=300
uv run python scripts/bench/agent_bench.py --resummarize bench/results/FILE.json \
    --sweep after_ms=1500,2000,2500 tool_after_ms=0,300,600,900
```

`--sweep` prints one row per combination (other values from the flags or the policy): longest silence p50 and p90, first audio p50, phrase delay p50 and p90, and the share of steps with an acknowledgement and with a tool phrase. Delays are 0 to 10,000 ms; `after_ms=0` turns the acknowledgement off.

To compare two or three runs, for example caching off and on, run `agent_bench.py` once per setting and pass the files to `--compare`:

```bash
uv run python scripts/bench/agent_bench.py --label nocache --runs 8
uv run python scripts/bench/agent_bench.py --label cache --prompt-caching on --runs 8
uv run python scripts/bench/agent_bench.py --compare bench/results/nocache-<time>.json bench/results/cache-<time>.json
```

`--compare` shows each step's p50 and p90 per file and the gain of each run over the one before it, then `overall` rows: per-hop p50 and p90, model latency and time to first byte per model call, overhead per model call, cache hit ratio and token totals. A value a file doesn't have shows `-`. Compare runs taken with the same prompt and model settings.

Fields in a result file:

| Field | What |
|---|---|
| `backend` | `backend`, and for Strands `prompt_caching`, `model_id`, `max_tokens`, `thinking`, `model_settings_source`, `gateway_url`, `memory_id`, `memory_actor_id`, `session_prefix`, `packages` (`strands-agents`, `bedrock-agentcore`, `mcp-proxy-for-aws` versions) |
| `raw[].steps[].hops[].usage` | `TurnEnd` usage: `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens` |
| `raw[].steps[].hops[].model_calls` | Each `strands_model_call` line of the hop (`model_ttfb_s`, `model_latency_s`, the four token counts, `stop_reason`) |
| `raw[].history_builds` | The `agent_history_new` / `agent_history_reload` lines of the run (reason, `duration_ms`, outcome) |
| `summary.<step>.{input,output,cache_read,cache_write}_tokens_mean` | Tokens per step (all its hops), mean over the runs |
| `overall` | Over every hop: `hops`, `hop_end_p50`/`p90`, `model_calls`, `model_latency_p50`/`p90`, `model_ttfb_p50`/`p90`, `overhead_per_model_call_p50`/`mean` ((hop time - its model time) / its model calls), `usage` totals, `cache_hit_ratio` (cache reads / (uncached input + reads + writes)), `history_builds` counts |

Benchmark sessions are Memory sessions named `bench-<28 hex>`, under the actor `owlbank-voice-strands`, so they are easy to tell from calls (`CA...`). The bench deletes none of their events: they expire with the Memory resource's event expiry, like a call's. A run of 8 repeats over 5 scenarios writes 40 short sessions.

### Adding an architecture

Implement `tac_bridge.agent_client.AgentClient` (one async-generator method, `turn`), add a factory to `AGENTS` in `agent_bench.py` (it returns the client, an optional `warm(session_id)`, sync or async, and optionally a dict recorded as the result's `backend`), and run with `--agent <name>`. The scenarios, the timing and the summary stay the same.

## Real calls

```bash
uv run python scripts/bench/call_latency.py --since <YYYY-MM-DDTHH:MM:SSZ> --label baseline
```

Place some calls first, then set `--since` to a time before them. The script reports turn time by kind (no tool, inline tool, hand-off), model time to first byte and latency per model call, overhead per model call, the Orchestrator conversation wait at segment start, and front-door time by route. It reads no caller or AI words.

Its model timings come from the bridge's `strands_model_call` lines. The summary has a `strands` section: model calls, model latency and time to first byte, overhead per model call ((turn time - model time) / model calls), cache read and write token totals and the cache hit ratio, and the `agent_history_reload` count, reasons, errors and durations, plus the `agent_history_new` count.

### Per hosting

```bash
uv run python scripts/bench/call_latency.py --since <YYYY-MM-DDTHH:MM:SSZ> --label hostings
uv run python scripts/bench/call_latency.py --since <YYYY-MM-DDTHH:MM:SSZ> --label hostings \
    --runtime-log-group /aws/bedrock-agentcore/runtimes/<runtime id>-DEFAULT
```

The bridge lines are read from `/owlbank/ecs/tac-bridge` (Fargate) and, when there is one, the Runtime log group: `--runtime-log-group`, else the `RuntimeLogGroup` output in `infra/cdk-outputs.json`. A log group that doesn't exist is skipped. Each call's `hosting` is the front door's `relay_minted` `hosting` field, else the bridge's `ws_session_start` line's, else the log group its lines came from, else `fargate`.

`summary.hostings.<hosting>` repeats the whole summary over that hosting's calls and adds:

| Field | What |
|---|---|
| `connects`, `unpaired_connects` | Connects (one per `relay_minted`) that reached a bridge session, and those that didn't |
| `connect_to_first_word_s.{all,warm,cold,first,reentry}` | `relay_minted` to the first `relay_speaker_event` `agentSpeaking on` of the session it opened, or to that session's `ws_session_start` when it logged no speaker event (`first_word_source` counts which). **Cold**: the process that served the session (its log stream: a Fargate task or a Runtime microVM) logged a start-up line (`startup_complete` or `agent_settings`) after `relay_minted`, so the connect waited for a new process; otherwise warm. `first` is a call's first connect, `reentry` any later one |
| `session_history` | The bridge's `agent_session_start` `history` per connect: `new`, `cached` (same process), `reload` (another process) |
| `dial_to_first_word_s` | The front door's first line for the call to its first connect's first word |
| `prewarm` (`agentcore`) | The warmer's `runtime_prewarm` outcomes and its `ok` durations (a cold boot as the warmer saw it) |
| `prewarm_hit_rate` (`agentcore`) | Share of first connects that were warm |
| `cold_start_s` (`agentcore`) | The warmer's `ok` durations, and the cold-minus-warm p50 of connect-to-first-word |

`summary.hosting_difference` (when both hostings have calls) is `agentcore` minus `fargate` for each p50 and p90 of `turn_s`, `connect_to_first_word_s`, `dial_to_first_word_s`, `model_latency_s` and `model_ttfb_s` (positive: `agentcore` is slower). There is no pass or fail bar.

## Handshake timer

```bash
uv run python scripts/bench/handshake_timer.py --cdk-outputs infra/cdk-outputs.json          # dry run
uv run python scripts/bench/handshake_timer.py --apply --cdk-outputs infra/cdk-outputs.json \
    --output bench/results/handshake.json
```

For `bridgeHosting=agentcore`. It times the WebSocket handshake to the Runtime on new sessions (each boots a microVM) and on warm ones, `--count` of each (default 10), and reports p50 and p90. The Runtime is the stack's `RuntimeArn` output unless `--runtime-arn` names one. It signs the URL with your own AWS session and writes no call state. It is a dry run unless you pass `--apply`.

## Slot model bench

`slot_model_bench.py` measures the model a slot answer reaches: latency, tokens per call and cost per 1,000 calls. Every utterance is synthetic and fictional: 44 fixed cases, five slot types (`digits`, `yes_no`, `choice`, `amount`, `date`) in en-US and es-US, in five categories (`clean`, `messy`, `self_corrected`, `choice`, `off_script`).

```bash
uv run python scripts/bench/slot_model_bench.py --list                                  # the cases, offline
uv run python scripts/bench/slot_model_bench.py --profile <profile> --repeats 3         # every mode
uv run python scripts/bench/slot_model_bench.py --profile <profile> --modes assisted --slot-types choice
```

A run calls Bedrock in your account. Filters: `--modes`, `--slot-types`, `--langs`, `--categories`. `--warmup N` (default 1) makes N unrecorded model calls per mode first, to open the connection.

**The resolver per mode.** Each case goes through the slot engine (`owlbank.slots.engine.resolve`) with `BedrockResolver` and the front door's client settings (`connect_timeout` and `read_timeout` equal to the resolver budget, `DEFAULT_RESOLVER_BUDGET_MS`, one attempt), once per mode:

- `deterministic`: the engine gets no resolver. The run makes no model call and reports 0.
- `assisted`: an answer reaches the model only when the deterministic stages leave it open, as on a call.
- `generative`: the effective mode, as `owlbank.slots.modes` decides it with `generative_enabled` on. That is `generative` on a non-sensitive choice slot and `assisted` everywhere else. When the grounded answer needs a follow-up question (an inferred or ambiguous choice), the model's wording is checked with `generative.fallback_reason`. The run reports the fallback rate by reason.

The resolver sees what it sees on a call: the caller's words, unredacted, for a sensitive slot too (`ssn4` here, with fictional digits).

**Logging guard.** Before any model call the bench reads `GetModelInvocationLoggingConfiguration` in `--region`. That Region's configuration logs every call through the `us.` profile, wherever it runs. The bench refuses to run (exit 2) while it delivers text, or doesn't say whether it does. In a trusted dev sandbox, `--accept-text-logging` runs anyway with a warning, and the synthetic utterances are then logged.

**Result file:** `bench/results/<label>-<UTC>.json`, default label `slot-resolver`. It holds case IDs, kinds, model outcomes, token counts and latencies, never the words: no utterance, question, label, model wording or value.

| Field | What |
|---|---|
| `price` | The price used, with its source and dates |
| `logging_check` | Each guard Region's verdict: `off`, `on_no_text` |
| `cases`, `repeats`, `warmup` | The run's size |
| `latency_stop` | `threshold_ms` (1000), `p90_ms` per mode, `over`, `stop` and `message` |
| `modes.<mode>.adapter` | `mode`, `model_id`, `budget_ms`, `max_tokens` |
| `modes.<mode>.summary` | `overall`, `by_slot_type`, `by_lang` and `by_category`, each with `answers`, `model_calls`, `errors` (bench failures, not model outcomes), `latency_p50_ms`, `latency_p90_ms`, `latency_max_ms`, `server_latency_p50_ms`, `server_latency_p90_ms` (Bedrock's `metrics.latencyMs`), `input_tokens_mean` (uncached input only: with the prompt cache, Bedrock reports cached tokens separately), `output_tokens_mean`, `cost_per_1000_usd` (from uncached input and output only, so it slightly understates the cost of cache reads and writes), `model_outcomes` (`ok`, `timeout`, `error`, `refusal`, `malformed`, `ungrounded`), `fallback` (`accepted` or the reason: `missing`, `lang`, `placeholder`, `number`, `length`), `fallback_rate`, `kinds` and `correct_rate` |
| `modes.<mode>.samples[]` | One per answer: `case_id`, `slot_type`, `lang`, `category`, `latency_ms`, `input_tokens`, `output_tokens`, `server_latency_ms`, `kind`, `correct`, `error`, `model_calls`, `fallback`, `mode`, `effective_mode` |

Latency is measured around the engine call over the answers that called the model. That covers what the caller waits for: the deterministic stages, the request, Converse and parsing. A timeout counts at the time it took. Tokens come from Converse `usage`, so a call with no response has none. The percentiles are **nearest rank**: each one is one of the samples. `agent_bench.py` and `call_latency.py` interpolate, so compare their percentiles with care.

**Price.** Claude Haiku 5.5 on-demand, Standard (in-Region) rates, which the `us.` geographic profile is billed at, from the `PRICES` table in the script (with its source and date). Check it against the current AWS price list before you rely on the cost. `cost_per_1000_usd` is 1000 times the mean per-call `in x p_in + out x p_out`. A model with no price in `PRICES` is an error, never a zero cost.

**Latency stop.** If any mode's p90 is over 1.0 s, the run prints a `STOP` line to stdout and stderr, records `stop: true` in the `latency_stop` field and exits 3. A resolver that slow adds a noticeable pause to a spoken answer: cover the wait with a catalog phrase and a `<Redirect>` while the resolver runs, or lower the resolver budget, before you go on.
