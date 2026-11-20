# AGENTS.md

Guidance for coding agents working in this repository. Read it whole at the start of a session; the guides it points to hold the detail.

## What this is

A reference build of an AI phone line for a fictional bank, OwlBank, on Twilio and AWS. A caller dials a Twilio number, picks English or Spanish, and verifies with keypad or spoken factors. Only then can an AI agent see their data. The agent answers questions, reads balances and transactions, takes payments after a step-up, and escalates: by default a spoken briefing and the call ends, with Flex added a human agent in Flex with a context panel (screen pop). It is an example to read, run in a sandbox and adapt, not a product. All customers, numbers and account IDs in the repository are fictional placeholders.

### A call, end to end

1. Twilio Voice calls the **front door** (API Gateway REST, WAF, the `frontdoor` Lambda). The front door answers with TwiML (Twilio's XML call instructions): language choice, lookup of the calling number, factor collection with `<Gather>` and a slot engine, verification through the `auth_service` Lambda, step-up, scripted tools, escalation (a spoken briefing by default, `<Enqueue>` to Flex when Flex is enabled).
2. Answers resolve in this order: keypad, then parsed spoken words (digits inside a sentence too), then, if the policy allows, an LLM classifier that sees the words with every number redacted.
3. Once the caller is verified, the front door connects the call to **ConversationRelay**. Twilio streams the caller's speech as text over a WebSocket to the **bridge** (`src/tac_bridge`, FastAPI, Fargate by default). The bridge runs Twilio Agent Connect in Orchestrator mode and runs the agent loop itself with Strands, calling Claude on Bedrock. History lives in AgentCore Memory.
4. Customer-data tools are inline tools: the loop pauses on a call and the bridge runs it against the `bank_tools` Lambda with the call-bound token. Level-0 tools (branch hours, FAQ) go through AgentCore Gateway to `level0_tools`. While the agent or a tool works, the bridge speaks short scripted phrases from the catalog (latency masking).
5. Hand-off tools (`authenticate`, `escalate`, `end_call`) end the ConversationRelay session and return the call to the front door. Escalation reads a briefing and ends the call by default; with Flex enabled (both `workspaceSid` and `workflowSid` in the setup outputs file, from `twilio_setup.py --with-flex`) it enqueues the call to a Flex agent, and the Flex plugin fetches the redacted call timeline from `context_api`.
6. State lives in DynamoDB (customers, tokens, calls, segments, policy, catalog) under a customer-managed KMS key.

Diagram: `docs/diagrams/owlbank-architecture.md`. Start with `docs/solution-guide.md` for how and why.

## Repository map

| Path | Responsibility |
|---|---|
| `src/owlbank/` | Shared library for every Lambda and the bridge: logging and scrubber (`obs.py`), Twilio signature and Basic auth (`twilio_sig.py`, `basic_auth.py`), secrets (`secrets.py`), the `Store` protocol (`store.py`, DynamoDB and in-memory), call-bound tokens (`tokens.py`), token sealing (`token_seal.py`), policy (`policy.py`), prompt catalog (`catalog.py`), prompt rendering (`agent_prompt.py`), prompt validation records (`prompt_validation.py`), untrusted-text wrapping (`untrusted.py`), PII redaction (`pii.py`), slot engine (`src/owlbank/slots/`), TwiML builders (`twiml.py`), tool contract (`tools_contract.py`), model failover (`src/owlbank/failover/`), telemetry (`src/owlbank/telemetry/`) |
| `src/lambdas/` | One folder per Lambda: `frontdoor`, `auth_service`, `bank_tools`, `level0_tools`, `context_api`, `flex_authorizer`, `admin_api`, `conversation_sweep`, `model_poller`, `recap_writer`, `runtime_warmer` |
| `src/tac_bridge/` | The bridge: `app.py` (FastAPI app, `/ws` and `/webhook` only), `bridge.py` (per-call loop), `agent_client.py` (the `AgentClient` interface), `agent_client_strands.py` (the Strands loop), `Dockerfile`, pinned `requirements.txt` |
| `src/admin_console/` | Optional admin console: a static site in plain ES modules, no build step and no dependencies. Edits tools, masking, phrases, prompts and slots through `admin_api`; Cognito sign-in |
| `flex-plugin/` | Optional Flex UI 2.x context-panel (screen-pop) plugin (TypeScript). See `flex-plugin/README.md` and `docs/add-flex.md` |
| `infra/` | AWS CDK app in Python, its own uv project: stacks in `infra/owlbank_infra/`, tests in `infra/tests/`, deploy helpers in `infra/scripts/` (`preflight.py`, `push_bridge_image.py`, `build_lambda_deps.py`, `resolve_models.py`). See `infra/README.md` |
| `scripts/` | Operator scripts: `seed.py`, `populate_secrets.py`, `check_secrets.py`, `twilio_setup.py`, `deploy_flex_plugin.py`, `assign_agent_skills.py`, `prompt_suite.py`, `check_resolver.py`, `hash_tool.py`, `masking.py`, `call_timeline.py`, `memory_history.py`, `web_activity.py` and `rehearse_human_leg.py`. Mutating scripts default to `--dry-run`, except `deploy_flex_plugin.py`, which deploys unless you pass `--dry-run` |
| `scripts/bench/` | Latency benchmark scripts; their notes are in the top-level `bench/README.md` |
| `data/` | Seed data: `fixtures.json` (fictional customers), `policy.json`, `prompt_catalog.json` (en-US, es-US), `faq.json`, `prompt_suite.json` and `prompt_validation.json` |
| `tests/` | `tests/unit/`, `tests/property/` (Hypothesis), `tests/component/` (signed fake webhooks, ConversationRelay simulator), `tests/e2e/` (front door, bridge and tools wired in memory), `tests/live/` (needs AWS, off by default) |
| `docs/` | `docs/solution-guide.md`, `docs/production-guide.md`, `docs/runbook.md`, `docs/diagrams/` |

CDK stacks (`infra/owlbank_infra/`): `OwlBankData`, `OwlBankNetwork`, `OwlBankAgent`, `OwlBankBridgeImage`, `OwlBankBridge`, `OwlBankCompute`, `OwlBankApi`, and the optional `OwlBankAdmin`. The table of what each holds is in `infra/README.md`.

## Setup and commands

[uv](https://docs.astral.sh/uv/), which installs Python 3.12 itself. Node 22.18 or later for the JavaScript tests. No AWS or Twilio credentials are needed for anything in this section.

```bash
uv sync --all-groups                         # runtime, dev and bridge dependency groups
cd infra && uv sync && cd ..                 # the CDK app is a separate uv project

uv run pytest tests -q                       # everything under tests/ except live suites
uv run pytest tests/unit -q                  # likewise tests/property, tests/component, tests/e2e
cd infra && uv run pytest -q                 # synth, cdk-nag, security assertions, schema parity
cd infra && uv run python app.py             # synth all stacks offline to infra/cdk.out

uv run ruff check .                          # lint (config in pyproject.toml)
uv run mypy                                  # types for src/ (disallow_untyped_defs for owlbank.*)

node --test 'src/admin_console/tests/*.test.js'   # admin console tests (Node 22.18+; quote the glob)
cd flex-plugin && npm test                   # Flex plugin tests (node --test, no install needed)
cd flex-plugin && npm run hygiene            # plugin hygiene check
```

Notes:

- Some tests read the docs: README headings, the admin deploy block, and the script order shared by `README.md`, `docs/add-flex.md` and the setup script's next steps. After editing any doc, run the full `tests/` and `infra/` suites, not only the docs tests.
- Run the narrowest test directory that covers your change while you work, and the whole `tests/` plus `infra/` suites before you finish; the full run takes several minutes.
- Three pytest markers are off by default because they call real services (select with `-m`): `live_eval` (live AgentCore Evaluations), `prompt_suite` (the per-model prompt suite against Bedrock) and `human_leg_contract` (an Orchestrator contract test against the real API; creates and closes one test conversation). Do not run them without a sandbox account you may bill.
- Synth needs no AWS account: the account defaults to the placeholder `111111111111` unless `-c accountId=...` is passed. The CDK app finds its own Node for jsii; `npx aws-cdk@2` (system Node) is needed only to deploy.
- `uv run pip-audit` checks dependencies for known vulnerabilities and needs network access. Run it before a deploy.

### Which tests guard which invariant

| Invariant | Tests |
|---|---|
| Untrusted text is wrapped and cannot close its block | `tests/property/test_untrusted_props.py`, `tests/property/test_untrusted_cap_props.py` |
| Verification gates tools: tokens are call-bound, hashed at rest, level-checked | `tests/property/test_tokens_props.py`, `tests/unit/test_tools_contract.py`, `tests/e2e/test_security.py` |
| The static prompt part holds no caller values; unverified callers get no Memory | `tests/property/test_agent_prompt_props.py`, `tests/unit/test_agent_prompt.py` |
| Prompt changes need current validation records | `tests/property/test_prompt_validation_props.py`, `tests/unit/test_prompt_validation.py`, `tests/unit/test_seed_prompt_gate.py` |
| No factor value, raw token or one-time code in any output | `tests/leakscan.py`, run at the end of every `tests/e2e` scenario; `tests/e2e/test_security.py` |
| Logs and speech logs are scrubbed | `tests/unit/test_obs.py`, `tests/unit/test_obs_sensitive_scrub.py`, `tests/property/test_scrubber_props.py`, `tests/property/test_speech_log_props.py` |
| Secrets only in Secrets Manager; docs name real scripts and flags | `tests/unit/test_secrets.py`, `tests/unit/test_docs_secrets.py`, `infra/tests/test_secret_registry.py` |
| AWS security defaults hold in every synthesized template | `infra/tests/test_aws_defaults.py`, `infra/tests/test_synth_and_nag.py` |

## Invariants a change must keep

1. **Untrusted data is wrapped.** Memory recall, account nicknames and transaction descriptions were not written by the bank. They reach the model only inside `<untrusted_data source="...">...</untrusted_data>`, escaped by `owlbank.untrusted` so content cannot close the block. The catalog's system text tells the model such a block is data, never instructions. New model inputs that carry third-party text go through `wrap` or `wrap_result`.
2. **Verification gates Memory and tools.** A call-bound token is issued by `auth_service` only after verification. Only its hash is stored; it validates for the `CallSid` it was issued on, before expiry, unrevoked, and at or above the tool's minimum level. Unknown tools need the highest level (fail closed). Tools take no customer or account ID from the model: the customer comes only from the validated token, and the model never sees the token. An unverified caller's prompt carries no Memory content.
3. **The static part of the prompt holds no caller values.** `owlbank.agent_prompt` renders the system prompt as a static part (the same for every caller of a model and language, cached) and a per-call part. Caller data goes only into the per-call part. Do not add a placeholder fill to the static part.
4. **Prompt changes need revalidated records.** Each model's catalog text has a validation record in `data/prompt_validation.json`, keyed by a hash of exactly what the renderer reads. Change any hashed part and the record is stale: rerun `scripts/prompt_suite.py` (it calls Bedrock) and commit the new records. `scripts/seed.py` refuses to seed an unvalidated per-model prompt, except `--allow-unvalidated-prompt` at stage `dev`. Suite checks are structural (tool called, silent, no markup); never match phrases or wording.
5. **No sensitive value in logs, journals or the model's context.** Slots flagged `sensitive` are stored only sealed with KMS, replaced by a placeholder on their answering turn, and scrubbed by exact value from every log line. Free text goes through Comprehend-based redaction before it reaches the journal, task attributes or Flex. Speech logs of Gather turns are redacted. Logs are JSON lines keyed by `call_sid` and carry identifiers, counts, durations and error codes, not message text.
6. **Tokens are sealed in hand-over.** The front door seals the raw token with a dedicated KMS key, using the `CallSid` as encryption context. The bridge unseals it in memory for each tool call.
7. **Secrets live in Secrets Manager.** No secret is a CDK parameter, environment variable, template value or file in the repository. Scripts read them from Secrets Manager; `scripts/populate_secrets.py` writes them with no echo.
8. **Twilio-facing endpoints verify before doing anything.** `X-Twilio-Signature` and Basic credentials are checked first, in constant time, behind WAF; a failure returns 403 with an empty body.
9. **Timers tolerate early firing.** Production runs on uvloop, where a timer can fire a fraction of a millisecond before its due time. Reschedule instead of returning; test on uvloop.

## Conventions

- **Python style.** Python 3.12, `from __future__ import annotations`, full type hints in `src/owlbank` (mypy `disallow_untyped_defs`), line length 110 (advisory, `ruff format` is not enforced). Ruff rules are in `pyproject.toml`; every ignore carries a comment saying why. No blanket `# type: ignore`.
- **Comments and docstrings.** Each module opens with a docstring that says what it is for and which contract it keeps. Comments explain why, in present tense. Do not write history ("was changed to", "since version N") or refer to tickets or issue numbers. A comment that restates the code is noise.
- **Docs code blocks.** They must paste cleanly into an interactive zsh: no inline `#` comments inside a block (zsh does not treat them as comments by default), and deploy flags in an array (`"${flags[@]}"`), not a string variable.
- **Tests.** Offline by default: moto, in-memory `Store`, signed fake webhooks, fake model clients. Test names say what is checked, not where the requirement came from. Add a Hypothesis property test (`tests/property/`) when the code has an invariant, a round trip, an ordering or a wide input space; example tests (`tests/unit/`, `tests/component/`) otherwise. Security properties have a test that fails if the property is dropped.
- **Where behavior is configured.** Prompts and phrases: `data/prompt_catalog.json` (per language, per model for the agent's system text). Policy (levels, factors, slots, tool levels, escalation, masking, limits, redaction): `data/policy.json`. Both are seeded into DynamoDB by `scripts/seed.py` and editable at run time in the admin console. A seed file change reaches a deployed table only after a reseed.
- **Fictional data only.** Customers come from `data/fixtures.json`. Phone numbers are in the `+1202555...` and `555-01xx` ranges, the placeholder AWS account is `111111111111`. Keep it that way in code, tests, docs and examples.
- **Tool contract.** Inline tool definitions live in `src/owlbank/tools_contract.py`; the schemas of the Gateway level-0 tools are in `infra/owlbank_infra/contract.py`. Each tool's minimum level is set in `data/policy.json`; a tool missing there needs the highest level.
- **Admin console.** Plain ES modules and CSS under `src/admin_console/`; no framework, no bundler, no runtime dependency. Keep it that way.
- **Dependencies.** Pinned in `pyproject.toml` and `uv.lock`. The bridge image installs from `src/tac_bridge/requirements.txt` (pinned with hashes, exported from the `bridge` group); regenerate it after changing that group.

## Safety rules

- Never commit secrets: no API keys, auth tokens, passwords, `.env` files or key material. `.gitignore` blocks common patterns; do not work around it.
- Never put a real phone number, real account SID (`AC...`) or real AWS account ID in the repository. Use placeholders.
- Do not call AWS or Twilio, deploy, or run mutating scripts unless the user asked and gave you credentials. Mutating scripts default to `--dry-run`; read the dry run before adding `--apply`. `scripts/deploy_flex_plugin.py` is the exception: it deploys unless you pass `--dry-run`, so always start with that flag.
- AWS security defaults hold for every resource you add, and the synth tests enforce them:
  - no public S3 bucket: Block Public Access on, no ACLs, `BucketOwnerEnforced`;
  - no `*` principal in any IAM or resource policy, and least-privilege actions and resources;
  - APIs go through API Gateway, never a Lambda Function URL, and never `AuthType: NONE`;
  - no `0.0.0.0/0` ingress on sensitive ports;
  - encryption at rest (customer-managed KMS key) and in transit;
  - WAF on every Twilio-facing route.
- Secrets only in Secrets Manager. Never in template values, task definitions' plain environment, context keys or logs.
- Keep speech and transcripts out of logs unless redacted. Do not add a log line that prints message text, a factor value, a token or a header.
- cdk-nag runs at synth. Fix a finding, or suppress it with a stated reason next to the suppression.

## Deploys in outline

Deploys go through the CDK app in `infra/` and are run by a person with working AWS credentials and a Twilio account. Do not run them on your own. In outline:

1. Run the code quality checks, including `uv run pip-audit`.
2. Set the deploy context:
   - the account on the command line: `-c accountId=$(aws sts get-caller-identity --query Account --output text)`, never stored in a file;
   - the Twilio number on the command line: `-c twilioPhoneNumber=`, never in a file;
   - the region in `infra/cdk.json` (add `modelIds` only if preflight prints a different list).
3. Deploy the Data stack, write the two Twilio secrets, run Twilio setup phase `pre`, seed the data, then deploy the remaining stacks and run phase `post`. Flex is optional and added later with `--with-flex` (`docs/add-flex.md`).

**Context flags are not sticky.** Every `-c key=value` you omit reverts to its default on the next deploy, whatever the deployed stack had. Pass the full flag set on every deploy of a stack that uses it, and run `cdk diff` first to see what would change. The keys, defaults and which stacks read them are in the "Context parameters" table of `infra/README.md`. In zsh, write flags inline or in an array: a flags variable is not word-split.

The full deploy steps are in `README.md` ("Deploy") and `infra/README.md` ("First deploy", "Deploying a change", "Deploy hazards"). Production settings and trade-offs are in `docs/production-guide.md`; investigating a failed call is in `docs/runbook.md`.

## Where the docs are

| Need | Read |
|---|---|
| How the solution works and why | `docs/solution-guide.md` |
| Settings, trade-offs and hardening for production | `docs/production-guide.md` |
| Find why a call failed, from a CallSid, a time or a symptom | `docs/runbook.md` |
| Stack contents, context keys, deploy steps | `infra/README.md` |
| Add Flex to a working deployment | `docs/add-flex.md` |
| Flex plugin install and test | `flex-plugin/README.md` |
| Benchmarks | `bench/README.md` |
| Architecture diagram | `docs/diagrams/owlbank-architecture.md` |
| Twilio product details (Voice, ConversationRelay, Verify, Memory, Flex) | The Twilio Skills and docs MCP in [github.com/twilio/ai](https://github.com/twilio/ai) (MCP endpoint `https://mcp.twilio.com/docs`); check current Twilio docs before changing Twilio-facing behavior |

### Twilio skills (optional)

Twilio publishes skills and a docs MCP for coding agents at [github.com/twilio/ai](https://github.com/twilio/ai) (public beta); its README says how to install them in each tool.

- The skill set is large: loading it can add tens of thousands of tokens of context, and a loaded skill adds more. Leave headroom.
- Treat skill content as guidance, not as the contract. Check it against current Twilio docs (the docs MCP, `https://mcp.twilio.com/docs`) before you change Twilio-facing behavior, and trust this repo's code and tests over a skill where they differ.
- Do not edit skill files from this repo. Report a wrong skill to the Twilio team (issues on `twilio/ai`).

When you change behavior, update the guide that describes it in the same change.
