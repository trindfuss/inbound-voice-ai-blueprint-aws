# OwlBank inbound voice blueprint

A reference build of an AI phone line for a fictional bank, OwlBank. Callers reach a Twilio number, choose English or Spanish, and verify themselves with keypad or spoken factors before an AI agent can see any of their data. The agent answers questions, reads balances and transactions, takes payments after a step-up, and escalates. By default an escalation reads the caller a short briefing of the context an agent would have received and ends the call. With Flex added ([docs/add-flex.md](docs/add-flex.md)) it hands the call to a human agent with a context panel (screen pop).

**What it covers.** The bridge runs the agent loop itself on Strands (an open-source agent framework). Around it:

- Latency masking: per-tool delays and phrase pools with enumerated-argument templates, all editable in DynamoDB.
- Caller-silence handling: repeat the last message up to twice, then escalate or hang up.
- Outcomes that can't be cut off, and early finish for spoken digits.
- A 3 s digit speech timeout, with cut-short answers asked again at once.
- Yes/no phrases resolved without the classifier.
- Orchestrator webhook de-duplication and hardened front-door request parsing.
- An admin console (Cognito sign-in; edits tools, masking and phrases, with an audit log).

The silence watch follows ConversationRelay's speaker events: it pauses while the caller speaks and counts from the end of the agent's speech. Architecture: `docs/diagrams/owlbank-architecture.md`; settings and their trade-offs: `docs/production-guide.md`. `infra/README.md` has the deploy runbook, and `docs/runbook.md` how to investigate a failed call. Security: [SECURITY.md](SECURITY.md). License: MIT (see [LICENSE](LICENSE)).

**New to the solution? Start with the [solution guide](docs/solution-guide.md)**: how it works and why, from the architecture to identity, tools, latency, escalation and the admin console.

**Terms used below.** ANI is the caller's number as the carrier reports it. ConversationRelay is Twilio's service that streams a call's speech to your server as text over a WebSocket. Conversation Orchestrator and Memory are Twilio services that hold a call's conversation and its history. AgentCore Memory and Gateway are Amazon Bedrock AgentCore services for the agent's short-term history and its shared tools. The slot engine is the front door's code that collects one answer at a time (a factor, a card, a yes or no). A step-up is the extra verification (more factors) a caller must pass before a sensitive action such as a payment. Level-0 tools (branch hours, FAQ) need no verification. The demo is US-only: US ZIP and SSN-4 factors and `+1` numbers. Inline tools run in the bridge; handoff tools end the AI session and return the call to the front door. Each is explained in [solution guide](docs/solution-guide.md) section 2, "Components".

## Architecture in brief

- A **deterministic front door** (`frontdoor` Lambda behind API Gateway REST and WAF) answers Twilio Voice webhooks. It handles language choice, ANI lookup, factor collection with `<Gather>` and a slot engine, verification through the `auth_service` Lambda, step-up, deterministic tools and escalation: a spoken briefing by default, or `<Enqueue>` to Flex when Flex is enabled. Answers are resolved by keypad first, then by parsing the spoken words (including digits inside a sentence), and only then by an LLM classifier that sees the words with every number redacted; the one-time code is sent in the background while the caller hears "Got it." (production guide, "How the IVR collects answers").
- Once the caller is verified, the front door connects the call to **ConversationRelay**. Twilio streams the caller's speech as text over a WebSocket to the **TAC bridge** (Twilio Agent Connect; `src/tac_bridge`, FastAPI on ECS Fargate behind CloudFront with a VPC origin to an internal ALB (no domain), or a public HTTPS ALB (custom domain), with WAF in both), which runs Twilio Agent Connect v2.5.0 in Orchestrator mode. A second hosting option runs the same bridge in an AgentCore Runtime (`bridgeHosting=agentcore`). Fargate stays the default for security (WAF and the Twilio signature on the WebSocket), and `agentcore` is an opt-in, for example for cheaper low-volume testing (production guide, "Hosting options for the bridge"). While the agent or a tool works, the bridge speaks short scripted phrases from the catalog (latency masking), and it adds the bank's local date and time to the agent's prompt on every turn.
- The bridge runs the **agent loop itself with Strands**, calling Claude on Bedrock directly, with the call's history in AgentCore Memory. Customer-data tools are inline tools: the loop pauses on each call and the bridge runs it against the `bank_tools` Lambda with the call token; level-0 tools (branch hours, FAQ) go through AgentCore Gateway to `level0_tools`. Handoff tools (`authenticate`, `escalate`, `end_call`) end the ConversationRelay session and return the call to the front door, which comes back to the agent afterwards.
- State lives in DynamoDB tables (customers, tokens, calls, the segment log, policy, the prompt catalog and a few more), encrypted with a customer-managed KMS key. A `context_api` Lambda behind a Flex token authorizer serves the redacted call timeline to the Flex plugin; it is deployed with or without Flex, and serves data only with Flex (503 otherwise).

Diagram: [docs/diagrams/owlbank-architecture.md](docs/diagrams/owlbank-architecture.md). How it works and why: the [solution guide](docs/solution-guide.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/owlbank/` | Shared library used by every Lambda and the bridge: logging (`obs.py`: JSON lines, `call_sid` correlation, scrubber, `@instrumented`, `timed_call`), Twilio signature and Basic auth checks, secrets, the `Store` protocol (DynamoDB and in-memory), call-bound tokens, token sealing, policy, prompt catalog, segment log, call state, result envelope, slot engine (`slots/`), TwiML (Twilio's XML call instructions) builders, tool contract, task attributes |
| `src/lambdas/` | One folder per Lambda: `frontdoor`, `auth_service`, `bank_tools`, `level0_tools`, `context_api`, `flex_authorizer`, `admin_api`, `conversation_sweep`, `model_poller`, `recap_writer`, `runtime_warmer` |
| `src/admin_console/` | Optional admin console: a static site (plain ES modules, no build step) served by the `OwlBankAdmin` stack |
| `src/tac_bridge/` | The TAC bridge: `app.py` (FastAPI app with TAC's `/ws` and `/webhook` only), `bridge.py` (per-call loop), `agent_client.py` (the `AgentClient` interface), `agent_client_strands.py` (`StrandsAgentClient`, the Strands loop), `Dockerfile`, pinned `requirements.txt` |
| `infra/` | AWS CDK app (Python, its own uv project): eight stacks (`OwlBankData`, `OwlBankNetwork`, `OwlBankAgent`, `OwlBankBridgeImage`, `OwlBankBridge`, `OwlBankCompute`, `OwlBankApi` and the optional `OwlBankAdmin`), cdk-nag; deploy helpers in `infra/scripts/` (`preflight.py`, `push_bridge_image.py`, `build_lambda_deps.py`, `resolve_models.py`). See [infra/README.md](infra/README.md) |
| `flex-plugin/` | Optional Flex UI 2.x plugin, the context panel (screen pop). See [flex-plugin/README.md](flex-plugin/README.md) and [docs/add-flex.md](docs/add-flex.md) |
| `scripts/` | Operator scripts: `seed.py`, `populate_secrets.py`, `check_secrets.py`, `twilio_setup.py`, `deploy_flex_plugin.py`, `assign_agent_skills.py`, `prompt_suite.py`, `check_resolver.py`, `hash_tool.py`, `masking.py`, `call_timeline.py`, `memory_history.py`, `web_activity.py` and `rehearse_human_leg.py`; the latency benchmark scripts are in `scripts/bench/` and their notes in [bench/README.md](bench/README.md). Mutating scripts default to `--dry-run`, except `deploy_flex_plugin.py`, which deploys unless you pass `--dry-run` |
| `data/` | Seed data: `fixtures.json` (fictional customers, 555-01xx numbers), `policy.json`, `prompt_catalog.json` (en-US, es-US), `faq.json` |
| `tests/` | `unit/`, `property/` (Hypothesis), `component/` (signed fake webhooks, ConversationRelay simulator), `e2e/` (front door, bridge and tools wired in memory), `live/` (needs AWS, off by default) |
| `docs/` | [solution guide](docs/solution-guide.md) (start here), [production guide](docs/production-guide.md), [runbook](docs/runbook.md) (find why a call failed from a CallSid, a time or a symptom), [adding Flex](docs/add-flex.md), diagrams |
| `LICENSE`, `SECURITY.md`, `AGENTS.md`, `CLAUDE.md` | The MIT license, why it is not production-ready, how to report a problem and what the example protects, and a map of the code and its invariants for coding agents ([AGENTS.md](AGENTS.md); `CLAUDE.md` imports it for Claude Code) |

## The path in brief

Do these in this order. Each is a section below.

1. **Local setup**: install the tools and run the offline tests. No account is needed.
2. **Twilio account**: accept the AI/ML addendum, buy a number, create an API key. Flex is not needed.
3. **AWS**: a profile that can deploy, Bedrock model access, CDK bootstrap.
4. **First deploy** (steps 1 to 13): Data stack, the two Twilio secrets, Twilio setup (`pre`), seed, the other stacks, checks, Twilio setup (`post`), then the test calls.

The steps assume macOS (Apple silicon), `us-east-1` and Finch as the container client. This guide uses Finch on macOS (AWS's open-source container client); elsewhere Docker works, with `CDK_DOCKER` unset. With the Twilio and AWS accounts ready, expect about 2.5 to 3 hours to a first call, most of it the CloudFront and bridge deploy and the approval prompts. From scratch (accounts, number registration, model access) plan a day; Twilio's approval time for a subaccount number is not known in advance.

## Local setup

Install these first and check each:

| Tool | Needed for | Check |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Everything in Python. It installs Python 3.12 itself | `uv --version` |
| Node 22.18 or later | The JavaScript tests, and `npx` runs the CDK CLI at deploy time | `node --version`. With Homebrew `node@22` is keg-only: put it first on the PATH (`export PATH="/opt/homebrew/opt/node@22/bin:$PATH"`). If `test_flex_plugin_hygiene` is reported as skipped, Node is missing from the PATH |
| AWS CLI v2 | Credentials and the account check | `aws --version` |
| A container client (Finch or Docker) | Only when `OwlBankBridge` deploys (it builds the bridge image); not for the layer build | `finch --version` or `docker --version` |

Node 20, the Twilio CLI and the Flex plugin tooling are needed only if you add Flex ([docs/add-flex.md](docs/add-flex.md)).

```bash
uv sync --all-groups
(cd infra && uv sync)
```

The first command installs the runtime dependencies plus the dev group (pytest, hypothesis, moto, httpx) and the bridge group (TAC, boto3>=1.43.99). The CDK app is a separate uv project, so it needs its own sync.

No AWS or Twilio credentials are needed for anything in this section or the next.

## Tests

All tests in this section run offline (the `live/` suites and three pytest markers that call real services, `live_eval`, `prompt_suite` and `human_leg_contract`, are off by default; `pyproject.toml` defines the markers).

```bash
uv run pytest tests -q
uv run pytest tests/unit -q
uv run pytest tests/property -q
uv run pytest tests/component -q
uv run pytest tests/e2e -q
(cd infra && uv run pytest -q)
(cd infra && uv run python app.py)
```

The first command runs everything below except infra. Then, in order: unit tests; property-based tests (the design's correctness properties and the log scrubber); front door and bridge component tests; offline end-to-end journeys (the security suite and the log scan at DEBUG). In `infra/`, `pytest` covers synth, cdk-nag, policy assertions, schema parity and logging, and `app.py` only synthesizes, to `infra/cdk.out`.

## Code quality checks

Four tools, configured in `pyproject.toml` (the `[tool.ruff]`, `[tool.mypy]` and `[tool.bandit]` sections):

```bash
uv run ruff check .
uv run mypy
uv run bandit -r src scripts -q
uv run pip-audit
```

In order: lint (pyflakes errors, import order, bugbear, security rules (S), simplifications); types for `src/`; security static analysis of the repository's code; known vulnerabilities in the installed dependencies.

- `ruff` covers undefined names, unused imports and variables, bugbear pitfalls, hardcoded secrets and unsafe calls, and import order. Every ignore in `pyproject.toml` has a comment saying why; `src/` and `scripts/` have no wholesale ignores. `ruff format` is not enforced.
- `mypy` checks `src/` only; the tests are not type-checked. There is no blanket `# type: ignore`.
- `bandit` scans `src/` and `scripts/`; add `-ll` to show medium and high severity only. At `-ll` it is clean. Its low-severity output includes the `x-flex-token` header name flagged as a "password" (B105), a false positive.
- `pip-audit` needs network access to PyPI. Run it before every deploy and record the result with the deploy notes, because new advisories appear over time.
- `ruff check infra` is not part of the checks.

## Security model in brief

- **Twilio-facing HTTP endpoints** check `X-Twilio-Signature` and HTTP Basic credentials (constant time) before anything else, behind WAF and API Gateway throttling. A failure returns 403 with an empty body. **Twilio's challenge flow:** for a webhook URL with `user:pass@`, Twilio first sends the request with no `Authorization` header and repeats it with `Authorization: Basic` only after a `401` with `WWW-Authenticate`, so a request without the header gets that 401 (from the API WAF rule or the handler, never a 403) and is logged as `twilio_basic_challenge`, not as an auth failure. The bridge's status callback URL has no credentials. **The Basic credentials appear in the number's webhook URLs**, so anyone who can read the number's settings in the Twilio Console can see them. They are a shared key, not the authentication boundary: the Twilio signature is. Rotate them after a screenshot or screen share that showed the URL ("Rotating the webhook Basic password" below). The bridge's `/ws` checks the signature on the upgrade request, then a one-time nonce; Basic auth there isn't possible, because ConversationRelay rejects credentials in a `wss://` URL (error 64101). With `bridgeHosting=agentcore` the WebSocket has no WAF or Twilio signature: a SigV4 pre-signed URL (at most 300 s), the CallSid binding and the nonce replace them.
- **Call-bound tokens.** `auth_service` issues a token after verification. Only its SHA-256 hash is stored, and it validates only for the `CallSid` it was issued on, before expiry, unrevoked, and at or above the tool's minimum level. Unknown tools need the highest level (fail closed).
- **Sealed hand-over.** The front door seals the raw token with a dedicated KMS key, using the `CallSid` as encryption context, into call state. The bridge unseals it in memory for each tool call. A sealed value copied to another call won't decrypt. A factor held for shared-phone disambiguation is sealed the same way.
- **No customer IDs in tool arguments.** Tools take no customer or account ID from the model; the customer comes only from the validated token. The model never sees the token.
- **No sensitive values in logs, journals or the AI's context.** A slot flagged `sensitive` (SSN last-4 and the one-time code in the demo seed; production should flag every factor) is stored only sealed with KMS in call state, replaced by `[<slot>]` on its answering turn, and scrubbed by exact value from every log line. The demo seed leaves ZIP unflagged, so ZIP is logged and reaches the AI. Segment, call-state, task-attribute and HandoffData writers run a redaction guard. Logs are JSON lines keyed by `call_sid` (`owlbank.obs`), carry identifiers, counts, durations and error codes but no message text, and pass a pattern scrubber (phone numbers, long digit runs, token-like strings, auth headers, URL queries). Set the level with the CDK context value `logLevel` (`OWLBANK_LOG_LEVEL` at run time). To find why a call failed, see [docs/runbook.md](docs/runbook.md).
- **Secrets** live only in Secrets Manager, under the customer-managed KMS key (CMK). CDK creates the Twilio secrets and the Context API key's secret: Secrets Manager generates the webhook Basic password, and you write the Twilio API key and Auth Token once after the Data stack (console, CLI, `scripts/populate_secrets.py` or your own pipeline; `scripts/check_secrets.py` checks them). No secret is a CDK parameter, environment variable or file in the repo.
- **Infra checks:** no `*` principal in any Allow (the only `*` principals are in three Deny statements: TLS-only on the ALB log bucket and the alarm topic, and the token key's crypto Deny), no Lambda Function URLs, no public S3, CMK encryption on every table and secret, and WAF on every Twilio-facing route (tested at synth). The one exception is the `agentcore` WebSocket, whose endpoint AWS owns (with the controls above).
- **Twilio credentials.** Every component except the Context API holds a Standard API key with full access to the Twilio account; only the Context API runs on a Restricted key. See "Twilio credentials" below for the credential table and the reasons.

Why this is not a production-ready solution, how to report a problem, and what the example does and doesn't protect: [SECURITY.md](SECURITY.md). Production hardening beyond the demo is in [docs/production-guide.md](docs/production-guide.md).

## Set up your Twilio account

Do these in the Twilio account before the first deploy, in this order. Use a dedicated subaccount for the blueprint if you can (item 2). The default path needs no Flex account. Use an upgraded (paid) Twilio account: trial accounts play a preamble before the call connects and can call only verified numbers. Check Twilio's current pricing pages for Conversation Orchestrator, Memory, ConversationRelay and Verify, and that each is available to your account, before you buy anything.

| # | What | Who | Where | Cost |
|---|---|---|---|---|
| 1 | **Accept the Predictive and Generative AI/ML Features Addendum** (once per account) | The account owner (it is a legal acceptance) | Console: **Explore products**, **Orchestration**, **Conversation Orchestrator**, **Create a Conversation configuration**, then tick the checkbox on the AI/ML Features page | Free |
| 2 | **A dedicated subaccount (recommended)** | The account owner | Console: **Account**, **Subaccounts**, **Create new Subaccount** | Free |
| 3 | **A voice-capable phone number** | The developer | Console, switched to the subaccount: **Phone Numbers**, **Buy a number**, with the Voice capability; pass it as `twilioPhoneNumber` and to `twilio_setup.py --phone-number` | Monthly number fee plus per-minute voice charges |
| 4 | **Verify SMS prerequisites** | The developer, with the deploying bank's compliance team | Console: **Messaging**, **Settings**, **Geo permissions** (enable the countries the one-time code goes to), and the SMS geo permissions of Verify itself (check both in the Console); optionally request a custom template (see below) | Per successful verification, plus the SMS |
| 5 | **The API key and Auth Token** | The developer | See the credential table below | Free |

**1. The addendum.** The Memory and Conversation Orchestrator APIs refuse every call with error 20003 ("Terms of service have not been accepted") until the account owner accepts the addendum, which also covers ConversationRelay and `<Say>`. The Console path runs through "Create a Conversation configuration" because that is where Twilio shows the checkbox. Acceptance is all you need: it is saved on its own, so tick the box and leave the page without creating a configuration. Do not create one by hand: the blueprint's configuration deliberately has no voice capture rules, and `twilio_setup.py --phase pre` creates the real one. Wait 1 to 2 minutes after accepting. To check it without a failing call, run `uv run python scripts/twilio_setup.py --check --phase pre --account-sid AC... --cdk-outputs infra/cdk-outputs.json` (possible only after deploy step 6, because it reads the API key from Secrets Manager): it reads the Memory Store list, which Twilio refuses with 20003 until the addendum is accepted, and prints that under "resources not readable".

**2. The subaccount.** Twilio signs webhooks with the Auth Token, which grants full account access; a subaccount contains it. `twilio_setup.py --apply` refuses an account that isn't an active subaccount, unless you pass `--dedicated-main-account`. Using the main account with that flag is a supported route for a demo; what it gives up is the containment: the Auth Token and the Standard key then have full access to everything in the main account, so use it only for an account the blueprint has to itself.

**3. The number.** (Numbers are written in E.164 form, such as `+14155550123`.) Buy it in the Console with the Voice capability, in the account the blueprint runs in. A number on a subaccount needs the parent account's Primary Customer Profile in Trust Hub first, so check that the parent has one before you buy, and switch to the subaccount before buying (the number, the API key and the Auth Token must all belong to the same account). How long Twilio takes to approve a profile is not known in advance. You pass the number to the deploy as `-c twilioPhoneNumber=+1...` and to `twilio_setup.py --phone-number`; the setup script points its voice webhook at the deployment in phase `post`.

**4. Verify.** You don't create the Verify service by hand: `twilio_setup.py --phase pre` finds one by friendly name (`--verify-friendly-name`, default "OwlBank") or creates it (6-digit code, 10-minute validity, do-not-share warning on) and records its SID in its `--output` file, which CDK reads as part of `twilioOutputs`. The code goes only to a number on file for the verified customer: the calling number, or a number the caller keyed in (only after ZIP and SSN-4). Have your compliance team confirm the text's wording before go-live; otherwise the call goes to a human (see "One-time codes" in the [production guide](docs/production-guide.md)). The text carries Twilio's do-not-share warning by default. For the full fraud-warning wording, request a custom template from Twilio using the English and Spanish text in [docs/verify-sms-template.md](docs/verify-sms-template.md), then pass its SID: `twilio_setup.py --verify-template-sid HJ...`. Twilio must enable custom templates on your account.

**5. The API key and Auth Token.** Create them in the Console, in the account the blueprint runs in, as described under "Twilio credentials" below.

**Flex (optional, later).** Flex is billable and is not part of the default path. Without it an escalation reads a briefing and ends the call; nothing is queued or transferred. To hand escalations to human agents, add Flex afterwards: [docs/add-flex.md](docs/add-flex.md).

### Twilio credentials

Every Twilio credential the deployment uses. Secret values go into AWS Secrets Manager (Amazon's secret store, which the Data stack creates) only; none is a CDK parameter, environment variable or file in the repository.

| Credential | Created by | Type and permissions | Secrets Manager secret | Turned on by | Read by |
|---|---|---|---|---|---|
| API key (key SID, secret and account SID) | The developer, in the Console | Standard API key: full access to the account | `owlbank/twilio/api-key` | Always | Front door, `auth_service`, `flex_authorizer`, `recap_writer`, `conversation_sweep` (Flex only), the bridge, and the setup scripts |
| Auth Token | The developer, in the Console | The account's Auth Token: full access, and the key Twilio signs webhooks with | `owlbank/twilio/auth-token` | Always | Front door and bridge (`X-Twilio-Signature` checks); `twilio_setup.py` (to create the Restricted key) |
| Webhook Basic credentials | Secrets Manager generates them at the first deploy | Not a Twilio credential: the user and password in the webhook URLs, visible to anyone who can read the number's settings. A shared key; the signature is the authentication boundary | `owlbank/twilio/basic-auth` | Always | Front door; `twilio_setup.py --phase post` (writes them into the webhook URLs) |
| Context API key | `twilio_setup.py --phase pre`, using the Auth Token | Restricted API key with two permissions: TaskRouter `workspaces.tasks/read` and `workspaces.tasks.reservations/list` | `owlbank/twilio/context-api-key` | `contextApiRestrictedKey` (on by default; `off` makes the Context API use the Standard key) | `context_api` |

**Every component except the Context API holds a Standard key**, which has full access to the Twilio account. Twilio offers no Restricted-key permissions for Memory, Orchestrator, call transcription or Flex token validation (`Tokens/validate` refuses even a key with no permissions, so the Flex authorizer needs a Standard key too). The Context API only reads TaskRouter tasks and reservations, so it is the one component on a Restricted key. A deployer who wants revocation and attribution per component can create a separate Standard key for each component and write each into its own secret, at the price of more secrets, grants and rotation; a leaked Standard key is full power wherever it sits.

**Create the API key and read the Auth Token (Console).** Open **Account**, **API keys & tokens**. Choose **Create API key**, give it a friendly name, set the key type to **Standard**, and save. The Console shows the key SID (`SK` followed by 32 hex characters) and its secret once: copy the secret now. The same page shows the account's **Auth Token** under **Live credentials**. The **Account SID** (`AC` followed by 32 hex characters) of the account the blueprint runs in is shown in the Console's Account Info panel. Until deploy step 6, keep the key secret and the Auth Token in a password manager (not in a file in the repository). The Auth Token is needed because Twilio signs every webhook with it (the front door and bridge check the signature), and because `twilio_setup.py` uses it to create the Restricted key and to check the account; `populate_secrets.py` asks for it at step 6. You write the key and the Auth Token into Secrets Manager once, after the Data stack is deployed (deploy step 6); the JSON shapes, the Console path, the CLI form that reads from a file and the pipeline's IAM needs are in [infra/README.md](infra/README.md#twilio-secrets-the-manual-step). The Restricted Context API key needs no Console steps: the setup script creates it and writes it straight to its secret without printing it. If you prefer to create it by hand, choose **Restricted** as the key type, grant exactly the two permissions above and write it to `owlbank/twilio/context-api-key`.

**A Standard key is enough for the setup script and the warm-up.** The Accounts resource refuses a Standard key (Twilio error 70004), so the script checks the account with the Auth Token instead, and the front door's start-up warm-up reads only resources a Standard key can read. You do not need a main-account key for either.

### Rotating the webhook Basic password

Rotate after a screenshot or screen share that showed a webhook URL. The previous password stays accepted for an overlap (8 hours by default, `--overlap-hours`), so calls in progress are not dropped:

1. `uv run python scripts/populate_secrets.py --cdk-outputs infra/cdk-outputs.json --new-basic-password --apply` writes the new password and prints the deadline for step 3.
2. Wait 60 seconds, the secret's cache lifetime.
3. Before that deadline, run `uv run python scripts/twilio_setup.py --phase post --account-sid AC... --phone-number <twilio-number> --cdk-outputs infra/cdk-outputs.json --apply` so the number's webhooks carry the new password. With Flex, add `--with-flex --taskrouter-events` to this command; otherwise TaskRouter's event callback URL keeps the old password and the events are refused after the overlap.

Do not rotate twice inside one overlap: the second rotation drops the oldest password, so calls started before the first rotation can lose callbacks. [docs/runbook.md](docs/runbook.md) has the full procedure.

## AWS prerequisites

1. **An AWS account and Region** (the examples use `us-east-1`) with Bedrock model access enabled for the models you configure (`modelIds`); `preflight.py` checks it. By default the agent uses Claude Haiku 5.5, Claude Haiku 4.5 and Claude Sonnet 5.5 through the `us.` inference profiles, and the slot resolver uses `us.anthropic.claude-haiku-5-5`. Enable them in the Console under **Bedrock**, **Model access** (Anthropic models ask once for a use-case form); preflight sends a one-token test request to each.
2. **An AWS CLI profile** that can deploy the stacks.
   - **Use it from the environment.** Export `AWS_PROFILE=<profile>` (and `AWS_DEFAULT_REGION=us-east-1` if the profile has no Region) in **every** shell you use for the deploy steps. `populate_secrets.py`, `check_secrets.py`, `twilio_setup.py`, `seed.py`, `masking.py` and `web_activity.py` have no `--profile` flag, and the first-deploy `cdk` and `aws` commands carry none, so they read the environment. `preflight.py`, `check_resolver.py`, `call_timeline.py` and `push_bridge_image.py` also take `--profile`: it is optional once `AWS_PROFILE` is exported, and the examples below show it. Verify with `aws sts get-caller-identity`. An assumed role or credentials in the environment work too.
   - **Create one with IAM Identity Center (SSO).** Run `aws configure sso`. It asks, with example answers: *SSO session name* (`owlbank`), *SSO start URL* (your organization's portal address), *SSO region* (the Region of your Identity Center, for example `us-east-1`), *SSO registration scopes* (accept the default `sso:account:access`). It then opens a browser to sign in, lists the accounts and roles you can use (pick the account and role), and asks for the *default client Region* (`us-east-1`), *output format* (`json`) and *profile name* (`<profile>`). Check that the profile exists with `aws configure list-profiles`, then that it works with `aws sts get-caller-identity --profile <profile>`.
   - **Expiry.** Temporary credentials expire, and a long deploy can outlast them. When a command reports expired credentials, renew them the way your organization does and re-export if needed.
   - **Roles.** Use the least-privileged role that works. `cdk deploy` needs a role that can pass the CloudFormation execution role (`iam:PassRole`) and create IAM roles. A power-user role is enough for `cdk diff`, the scripts and the read-only checks.
3. **AWS account limits.**
   - **Region.** Without a custom domain use `us-east-1`: the CloudFront mode needs it. The admin console needs `us-east-1` in every mode, because its web ACL lives there (see [infra/README.md](infra/README.md), "Two exposure modes" and the `adminConsole` row).
   - **Service Quotas.** Check Lambda concurrent executions (provisioned concurrency counts against it; if the quota is low, request an increase or pass `-c lambdaProvisionedConcurrency=0`) and Bedrock tokens per minute for the default models.
   - **CloudFront.** A new account may need verification before it can create distributions.
4. **CDK bootstrap** of the account and Region (deploy step 5).
5. **uv and Node 22.18 or later** (see "Local setup"). The CDK CLI runs through `npx aws-cdk@2`, so deploys need Node. If a newer CDK CLI misbehaves, pin a version that worked (`npx aws-cdk@<version> ...`; `npx aws-cdk@2 --version` prints the current one).
6. **A container build tool for the bridge image**, needed when `OwlBankBridge` deploys: Finch (`brew install finch`, `finch vm init`, then `export CDK_DOCKER=finch`) or Docker. The bridge image is `linux/arm64`, so the build host must be arm64 (Apple silicon, Graviton) or have arm64 emulation. On an Intel Mac or x86 Linux use Docker with buildx and install the emulation once with `docker run --privileged --rm tonistiigi/binfmt --install arm64`; the build is then slower. Docker Desktop's license may need a paid plan at a larger company.
7. **For production only:** a domain and an ACM certificate (`domainName` and `certificateArn` in `infra/cdk.json`). A domain isn't required for dev and test: without `domainName` (the default) the stack runs in the CloudFront mode, where the bridge WebSocket is served on a `*.cloudfront.net` name with a valid certificate, through a CloudFront VPC origin to an internal ALB, and the front-door API on API Gateway's default `execute-api` name. Two trade-offs, both fixed by a domain: the CloudFront-to-ALB hop is plain HTTP inside AWS (for dev and test only: `stage=prod` without a domain fails synth), and `execute-api` stays on (compensated by the signature check, Basic auth, WAF and throttling). With `domainName` and `certificateArn` you get the custom-domain mode: HTTPS end to end, `execute-api` off. See the [production guide](docs/production-guide.md).
8. **Your own phone number** for seeding (`--developer-phone`), dev and test stages only.

**Idle cost.** The deployment bills while idle, whether or not anyone calls: the NAT gateway, Lambda provisioned concurrency, the ALB, the Fargate task, the WAF web ACLs, CloudFront, the KMS keys and secrets, and the model poller. Twilio charges for the number and for usage. Figures for provisioned concurrency and the hosting choice are in the [production guide](docs/production-guide.md) ("What provisioned concurrency costs", "Hosting options for the bridge"); check the AWS and Twilio pricing pages for the rest. Tear down when you are done ("Clean up").

## Deploy

### First deploy

Run these steps against the AWS and Twilio accounts you set up above. Details, context keys and the reasons for the order are in [infra/README.md](infra/README.md#first-deploy); what to deploy for each kind of later change is in its "Deploying a change", and the call script in its [Test call script](infra/README.md#test-call-script). Commands run from the repository root unless a `cd infra` says otherwise.

**Context flags are not remembered.** Every `cdk` command takes its `-c` flags again; a flag you leave out reverts to its default (see step 2).

**Why the order is not "deploy, then configure Twilio".** The bridge container calls Twilio at startup, so it needs real secret values and the Twilio Conversation Configuration ID. But the webhooks need the API and relay domains, which exist only after the deploy. So Twilio setup is split in two (`--phase pre` and `--phase post`), and the secrets are written after the Data stack and before phase pre and any stack that reads them.

1. **AWS credentials.** Check them with `aws sts get-caller-identity` (see "AWS prerequisites").
2. **Context.** Decide where each setting lives (details: [infra/README.md](infra/README.md#first-deploy), steps 2 and 5).
   - **Command line only, never in a file.** `-c accountId=$(aws sts get-caller-identity --query Account --output text)` on every command, and `-c twilioPhoneNumber=+1...` on every deploy that includes `OwlBankBridge` (or the bridge's tasks fail at start-up).
   - **In `infra/cdk.json`.** `region`; `domainName` and `certificateArn` only for the custom-domain mode. Leave `modelIds`, `bridgeDesiredCount` and `wafManagedRuleMode` as shipped (`cdk.json` sets no `modelIds`; step 3 says when to add it).
   - **If your organization's SCP (service control policy) denies `s3:PutBucketPublicAccessBlock`**, also pass `-c s3PublicAccessBlockMode=account` on every deploy (leaving it out fails the Bridge stack and rolls it back), and add one flag to the bootstrap (step 5). You are in that case when the bootstrap fails with AccessDenied on `PutBucketPublicAccessBlock`.
3. **Preflight (read-only).** `(cd infra && uv run python scripts/preflight.py --profile <profile> --stage dev --slot-mode-ceiling <ceiling>)` (`<ceiling>` is the `slotModeCeiling` you will deploy: `assisted` by default; the choice of `deterministic` is explained below). Also run the code quality checks above, in particular `uv run pip-audit`, before each deploy; for a dev deploy a pip-audit finding is informational: record it with the deploy notes. Some checks depend on settings that may belong to someone else:
   - **Bedrock invocation logging is account-wide.** If it already delivers text in the Region, preflight fails, and the setting belongs to whoever enabled it: ask them, and don't change it. Then either rerun with `--accept-text-logging` (here and on `check_resolver.py`; dev and test only, and callers' test speech is then logged, so use fictional values in every test call), or deploy with `--slot-mode-ceiling deterministic` (on `preflight.py` and as `-c slotModeCeiling=deterministic`), which turns off all the front door's model calls, the escalation briefing included, so the caller hears a fixed line instead.
   - **Model access.** A model-access failure can clear once the model has been used: rerun preflight once before changing anything. If preflight prints `deploy with: -c modelIds=[...]` and the list differs from the default, add `"modelIds"` to `infra/cdk.json` (or pass the flag on every Bridge and Compute deploy).
   - **Account-wide S3 Block Public Access and CloudTrail.** A FAIL on account-level S3 Block Public Access, or a WARN that no CloudTrail trail logs management write events, is also an account setting that may belong to someone else: ask the account owner, and do not change account-wide settings in a shared account. The preflight's fix line shows the command if you do own the account. In dev, the CloudTrail WARN does not fail the run (it only means the alarm on Bedrock invocation logging cannot fire). The Block Public Access FAIL does: resolve it with the account owner before you deploy (`-c s3PublicAccessBlockMode=account` is for organizations whose SCP denies the per-bucket setting, step 2, and it relies on the account-level setting being on).
4. **Build the Lambda layer** (`(cd infra && uv run python scripts/build_lambda_deps.py)`). This needs no container client. It worked when `infra/build/lambda-deps/python/twilio/` exists; without that step the layer ships empty and the Lambdas fail at run time importing `twilio`. **Start the container client** (Finch or Docker, see "AWS prerequisites") before step 10.
5. **Bootstrap, then deploy the Data stack only.**
   - **5a. Bootstrap.** `(cd infra && npx aws-cdk@2 bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/<region> -c twilioOutputs=)`, where `<region>` is `us-east-1` unless you set `domainName` (then the Region in `infra/cdk.json`). `npx` asks once to install the CDK CLI: answer `y`. If the SCP case in step 2 applies, add `-c s3PublicAccessBlockMode=account --public-access-block-configuration false`. If a failed bootstrap left the `CDKToolkit` stack in `ROLLBACK_COMPLETE`, delete that stack in CloudFormation (it holds nothing yet) and run the bootstrap again.
   - **5b. Deploy the Data stack.** `(cd infra && npx aws-cdk@2 deploy OwlBankData -c accountId=$(aws sts get-caller-identity --query Account --output text) -c twilioOutputs= --outputs-file cdk-outputs.json)` (plus the SCP flag, if you need it); it stops at a `(y/n)` approval prompt. Secrets Manager generates the webhook Basic password here; the two Twilio secrets hold a placeholder until step 6.
   - **Why `-c twilioOutputs=`.** `cdk.json` names `twilio.context.json`, which step 7 creates; until then pass an empty value (on every bootstrap and on this deploy) or synth fails with `twilioOutputs file not found`.
6. **Write the two Twilio secrets, then check them** (the manual step; shapes and alternatives in [infra/README.md](infra/README.md#twilio-secrets-the-manual-step)). Dry run first, then add `--apply`; the script asks for each of the four values with no echo, in this order: Account SID, API key SID, API key secret, Auth Token (the prompts read `twilio_api_key account_sid:`, `twilio_api_key api_key:`, `twilio_api_key api_secret:` and `twilio_auth_token auth_token:`). It asks also in the dry run, which checks their format and writes nothing: `uv run python scripts/populate_secrets.py --cdk-outputs infra/cdk-outputs.json`. Then `uv run python scripts/check_secrets.py --cdk-outputs infra/cdk-outputs.json` must exit 0.
7. **Twilio setup, phase pre (no domain needed).** Dry run first, then add `--apply`: `uv run python scripts/twilio_setup.py --phase pre --account-sid AC... --cdk-outputs infra/cdk-outputs.json` (add `--dedicated-main-account` on a main account).
   - It creates the Verify service, Memory Store, Conversation Configuration and the Context API's Restricted key, and writes `infra/twilio.context.json`.
   - It makes no Flex call and creates no queues or workflow; those come only with `--with-flex` ([docs/add-flex.md](docs/add-flex.md)).
   - The dry run makes no Twilio call, so it shows the plan and proves nothing about the account.
   - To read what Twilio holds today (account status, the key's account, existing resources) without changing anything, add `--check` (exit 0 when everything is fine, 3 when it found something to fix or a secret could not be read, 2 on a bad argument, 1 on an unexpected error).
   - Then run `check_secrets.py ... --strict`, which must exit 0 with `context_api_key` now `ok`. When it finishes, `twilio_setup.py` prints the next command to run.
8. **Point the stacks at the file and check.** `infra/cdk.json` already names `twilio.context.json`, so from here on deploy without the `-c twilioOutputs=` override. Run `(cd infra && uv run python scripts/preflight.py --profile <profile> --stage dev --before-bridge --slot-mode-ceiling <ceiling>)` (`<ceiling>` is the value you chose in step 3, `assisted` by default; add `--accept-text-logging` if step 3 needed it); it must pass.
9. **Seed the demo data, before the other stacks** (the bridge crashes at start-up on empty tables): `uv run python scripts/seed.py --stage dev --cdk-outputs infra/cdk-outputs.json --developer-phone <your-mobile> --developer-factor zip=NNNNN` (the dry run), then the same with `--apply`. `<your-mobile>` is your own mobile number in E.164 form, the phone you will call from; set `AWS_DEFAULT_REGION` if the profile has none. The demo customer's ZIP is 94105 unless you pass your own with `--developer-factor zip=...`; it is hashed at seed time and never printed. A plain seed writes the fixtures only if they are absent, so a `--developer-factor` you add later needs `--reseed` as well.
10. **Deploy the remaining stacks, in three commands.** Order matters because Compute depends on Bridge, Api on Compute, and Admin on Api. **The Api stack replaces the account's per-Region API Gateway CloudWatch logging role, and `destroy` removes it: do not deploy in an account whose other APIs rely on that role.** Put the flags in an array once (a plain variable would reach the CDK as a single argument and the context would be lost) and run all three commands from `infra/` in the same shell. Replace `<twilio-number>` with the Twilio number you bought (E.164, the same one you pass to `twilio_setup.py --phone-number`) and `<admin-email>` with the address of the console's first user:

    ```bash
    export CDK_DOCKER=finch
    cd infra
    flags=(
      -c "accountId=$(aws sts get-caller-identity --query Account --output text)"
      -c "twilioPhoneNumber=<twilio-number>"
      -c "adminEmail=<admin-email>"
    )
    ```

    Set `CDK_DOCKER=finch` only with Finch; with Docker leave it unset. If you need any of these, add them as extra lines inside the parentheses: `-c s3PublicAccessBlockMode=account` (your organization's SCP denies the per-bucket setting), `-c lambdaProvisionedConcurrency=0` (a low Lambda concurrency quota), `-c adminConsole=false` (then leave `OwlBankAdmin` out of command 3), `-c slotModeCeiling=deterministic` (if step 3 chose it).

    Run `diff` for each group first and read what it prints; the deploy then stops to ask for approval of IAM and security-group changes.
    1. `npx aws-cdk@2 diff OwlBankData OwlBankAgent OwlBankNetwork OwlBankBridgeImage --exclusively "${flags[@]}"`, then the same with `deploy` and `--outputs-file cdk-outputs-1.json`.
    2. `npx aws-cdk@2 diff OwlBankBridge --exclusively "${flags[@]}"`, then `deploy` with `--outputs-file cdk-outputs-2.json`. The bridge image is built from source as part of this stack; the first CloudFront distribution takes 15 minutes or more.
    3. `npx aws-cdk@2 diff OwlBankCompute OwlBankApi OwlBankAdmin --exclusively "${flags[@]}"`, then `deploy` with `--outputs-file cdk-outputs-3.json`. In `dev` and `test` the optional `OwlBankAdmin` stack is part of the app; `adminEmail` creates the console's first user. With `-c adminConsole=false` the stack does not exist, so leave it out of the command.

    **The approval prompt** is `Do you wish to deploy these changes? (y/n)`. It can scroll away behind the image build output, which makes the command look hung: search the terminal for `(y/n)`. In each table check that no principal is `*`, that no security group opens a sensitive port to `0.0.0.0/0` (an `Everyone (IPv4)` rule for outbound HTTPS 443 on the bridge tasks is expected), and that new roles are limited to their own resources. Expected: the Data update adds a recap-writer statement to the token key policy (limited to the call's encryption context); Agent adds the AgentCore gateway role; Network a flow-log role; Bridge the security groups (the ALB takes port 80 only from the CloudFront origin-facing prefix list); Api an account-level API Gateway logging role; Admin two logging-only helper roles.

    Each `--outputs-file` replaces its file, so merge the three into the one the scripts read: `uv run python -c "import json,sys; d={}; [d.update(json.load(open(f))) for f in sys.argv[1:]]; json.dump(d, open('cdk-outputs.json', 'w'), indent=2)" cdk-outputs.json cdk-outputs-1.json cdk-outputs-2.json cdk-outputs-3.json` (run from `infra/`; the first argument keeps step 5's output), then run `cd ..` to return to the repository root. Open **`ConsoleUrl`** to reach the admin console; **`InternalApiEndpoint`** is the console's own API and not a page. The first admin user receives an email with a temporary password, valid for 3 days. Custom-domain mode only: point DNS, `voice.<domain>` to `ApiDomainTarget` and `relay.<domain>` to `AlbDnsName`.
11. **Post-deploy checks.** `uv run python scripts/check_secrets.py --cdk-outputs infra/cdk-outputs.json --strict` again, then `(cd infra && uv run python scripts/preflight.py --profile <profile> --stage dev --post-deploy --slot-mode-ceiling <ceiling>)` (`<ceiling>` as in step 8; with `--accept-text-logging` if step 3 needed it). A pass means: every S3 bucket of the `OwlBank*` stacks and of `CDKToolkit` has Block Public Access on, `BucketOwnerEnforced` and a non-public policy, and Access Analyzer has no active findings on the blueprint's own resources. In a shared account, findings on other resources are printed as a note with their count and do not fail the run; `--strict-account` fails on any finding.
12. **Live resolver check** (when `slotModeCeiling` isn't `deterministic`): `uv run python scripts/check_resolver.py --profile <profile>` (add `--accept-text-logging` if step 3 needed it). It makes one untimed warm-up call, then one real Bedrock call per resolver tool, and fails if Bedrock refuses a request or a call exceeds the resolver budget (2500 ms by default). A pass is one PASS line per tool.
13. **Twilio setup, phase post (the webhooks).** No redeploy follows.
   - **Read first.** `uv run python scripts/twilio_setup.py --phase post --account-sid AC... --phone-number +1... --cdk-outputs infra/cdk-outputs.json --check` is read-only. It shows the number's current webhooks with credentials masked, and any voice application on the number, which would override the voice URL.
   - **Dry run, then apply.** Run the same command without `--check` (the dry run proves only that the plan builds), then add `--apply`. It prints the number's previous webhooks (masked) before replacing them, points the number's voice webhook at the API (with the Basic credentials) and sets the Conversation Configuration's status callbacks to the bridge. On a main account add `--dedicated-main-account`.
   - **Rotate after a screen share.** The webhook URLs show the Basic credentials in the Console: rotate the password after a screen share ("Rotating the webhook Basic password" above).

**Test calls (step 14).** Call the number from the `--developer-phone` number you gave the seed (another number takes the keyed-in phone path), as the fictional customer Thomas Doe: ZIP 94105 (or the one you passed to the seed) and SSN last four 1111. The full script, with what you hear on every call and what to do when one fails, is the [Test call script](infra/README.md#test-call-script).
- **Call 1:** ask for your checking balance. Only the ZIP question comes first.
- **Call 2:** pay 25 dollars to your Red Card from checking. The payment needs the step-up: the SSN last four, then a code texted to your phone.
- **Call 3:** ask for a person. Without Flex you hear a briefing and the call ends; nothing is queued or transferred.

Then sign in to the admin console (`ConsoleUrl`). The first-deploy checks that offline tests can't cover (WebSocket idle timeout, end-to-end TLS, the origin header and the WAF count-mode review) are in [First deploy](infra/README.md#first-deploy), step 14. Read the WAF logs, then consider `wafManagedRuleMode=block` ([production guide](docs/production-guide.md), section 3).

**Expected warnings.** The first two are harmless: the bridge image build prints `useradd warning: app's uid 10001 is greater than SYS_UID_MAX 999` (the image uses a high non-root user ID on purpose); every CDK command prints `aws-cdk-lib.CfnResource#addDependency is deprecated`. A front-door `twilio_connection_warm_failed` WARNING at a cold start is harmless unless its status is 401 or 403, which means the API key in Secrets Manager is wrong (re-run `check_secrets.py`); another cause, such as a network blip at the cold start, can be ignored.

**Deploy hazards** (more in [infra/README.md](infra/README.md#deploy-hazards)):

- The bridge deploys before the Compute stack exists (the second command of step 10), so its tool calls fail until Compute is deployed. Nothing calls it in this order, because the number is pointed at the API only in step 13.
- A replaced CloudFront distribution gets a new domain. Compute reads it from an SSM parameter at start-up, so redeploy `OwlBankCompute` after the Bridge, then re-run `twilio_setup.py --phase post`.
- Do not rename `bank_tools` or `level0_tools`: their fixed function names make a replacement fail.
- The CloudFront mode's plain-HTTP hop (CloudFront to ALB on 80) is for **dev and test only**; `stage=prod` without `domainName` fails synth.

**The AgentCore hosting (opt-in).** The switch, the rollback and their hazards (an image push and digest for every bridge change; a rollback from VPC mode deploys Compute and Api without waiting for the Runtime's security group) are in the AgentCore section of [infra/README.md](infra/README.md).

## Clean up

1. **Stop calls reaching the stack.** In Twilio, point the number's voice and status webhooks elsewhere, or release the number.
2. **Destroy the stacks.** In `infra/`, run `npx aws-cdk@2 destroy --all` with the same `-c` values you deployed with (they are not remembered): at least `-c accountId=...` and `-c twilioPhoneNumber=...`, plus `-c s3PublicAccessBlockMode=account` if you deployed with it, plus `-c bridgeHosting=agentcore -c bridgeImageDigest=...` if you used AgentCore. The app has to synthesize before it can destroy. Save any logs you need first: outside `stage=prod`, log groups are destroyed with their stacks. To remove only the admin console, run `npx aws-cdk@2 destroy OwlBankAdmin` ([infra/README.md](infra/README.md#the-admin-console-owlbankadmin)). Destroying the Api stack also removes the account's per-Region API Gateway CloudWatch logging role that it replaced at deploy; other APIs in the account that relied on it lose their logging.
3. **AgentCore in VPC mode.** Deleting the Runtime can time out, and a retry can then fail on its security group until AgentCore releases its network interfaces, which has taken up to about 1.5 hours. Wait, then run the destroy command again ([infra/README.md](infra/README.md), AgentCore hazards).
4. **Once every stack is gone, delete what they leave behind.** These resources are retained on stack delete in every stage, so remove them yourself if you don't need them:
   - the DynamoDB tables (the core tables are named `OwlBankData-*Table*`, generated by CloudFormation; of the retained tables only `owlbank-<stage>-modelstats` has a fixed name, and the admin console's `owlbank-<stage>-admin-audit` is retained only at `stage=prod`);
   - the secrets under `owlbank/` (the Twilio API key and Basic-auth credentials are among them). Delete them with `--force-delete-without-recovery`, or they sit in a recovery window;
   - the bridge image repository `owlbank-<stage>-bridge` in ECR;
   - the `CDKToolkit` bootstrap stack, its assets bucket and its ECR repository for the bridge images, if nothing else in the account and Region uses the bootstrap;
   - the local files `infra/cdk-outputs*.json` and `infra/twilio.context.json`;
   - on Fargate hosting, the ALB access-log bucket. Its name is generated, so find it by the `aws:cloudformation:stack-name` tag `OwlBankBridge`, then empty and delete it;
   - the two KMS keys, `alias/owlbank-data` and `alias/owlbank-token-sealing`. Schedule their deletion last, after the tables, secrets and repository they encrypt are gone. Deletion has a waiting period of 7 to 30 days, and the data is unrecoverable afterwards.
5. **Remove the Twilio side** in the Console. None of it is in the CDK app: the Verify service, the Memory Store, the Conversation Configuration, the API keys (yours and the restricted Context API key that `twilio_setup.py` created), and, if you added Flex, the TaskRouter queues and workflow and the Flex plugin release. Cancel Flex if you enabled it only for this test. Then rotate the Auth Token. The webhook Basic password needs no step of its own: once the number points elsewhere (step 1 above) and the secrets are deleted, it protects nothing.

The production guide lists what to settle before using the solution beyond a demo ([docs/production-guide.md](docs/production-guide.md)).

## Where to go next

- [Solution guide](docs/solution-guide.md): how it works and why.
- [Production guide](docs/production-guide.md): settings, trade-offs and hardening.
- [Runbook](docs/runbook.md): find why a call failed, from a CallSid, a time or a symptom.
- [infra/README.md](infra/README.md): stacks, context keys, deploy and change procedures.
- [Add Flex](docs/add-flex.md): hand escalations to human agents, with the context panel.
- [Latency benchmarks](bench/README.md) and the [Flex plugin](flex-plugin/README.md).

## License

The solution is released under the MIT license: see [LICENSE](LICENSE). The admin console's fonts carry their own license notes. For the security status and how to report a problem, see [SECURITY.md](SECURITY.md). 
