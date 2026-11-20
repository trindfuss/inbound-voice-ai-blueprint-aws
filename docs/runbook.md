# OwlBank runbook: diagnosing calls and operating the deployment

For an engineer on call. Most diagnosis works from logs and consoles; the remedies use the scripts in this repository (`scripts/`, `infra/scripts/`) and CDK deploys.

**Contents.** Finding the cause: "What you have", the event reference ("Event reference: masking, silence, keypad and agent loop", "Keyed-in phone number", "Slot settings and sensitive values", "Slot modes and the resolver"), "The Bedrock logging alarm", "Saved queries", then start from a call ("From a CallSid"), a time ("From a time") or a symptom ("From a symptom"). Operating: "Secrets: checking and rotating", "Raising the level for an investigation", "Tuning latency masking", "Prompts and the seed", "Using the admin console", "Keypad shortcuts in the AI session". Features: "The agent loop (Strands)", "Escalation without Flex: the briefing", "Model failover", "Validating a prompt", "Call observability", "AgentCore hosting: investigating a call", "The conversation sweep". Demos (`dev` and `test` only): "Demoing the web-activity offer" and the demo script in "Keypad shortcuts in the AI session". Last: "What is never in the logs".

**Terms.**
- **Front door**: the Lambda behind the API that answers Twilio's voice webhooks and runs the scripted steps (lookup, language, verification, hand-offs). Log `component` `frontdoor`.
- **Bridge**: the service that runs the AI session and the agent loop, on Fargate or as an AgentCore Runtime (`component` `bridge`).
- **TAC** (Twilio Agent Connect): the SDK in the bridge that speaks ConversationRelay and Orchestrator and gives access to Twilio Memory. **ConversationRelay** carries the call's speech to and from the bridge as text; **Orchestrator** (Conversation Orchestrator) keeps the call's conversation records.
- **Segment log**: the `segments` table, the record of what the caller heard and what the call did.
- **Memory**: this runbook always says which. **AgentCore Memory** holds the agent's message history. **Twilio Memory** holds a customer's profile (traits, summaries, observations). An **AWS profile** (`<profile>`) is an AWS CLI profile.

**Conventions.**
- Commands run from the repository root unless a `cd infra` or "from `infra`" says otherwise. Preflight is `infra/scripts/preflight.py`, run from `infra`: `cd infra && uv run python scripts/preflight.py --profile <profile> --stage <stage>`.
- `infra/cdk-outputs.json` is written by a deploy run from `infra` with `--outputs-file cdk-outputs.json` (README, "Deploy"). Scripts take it as `--cdk-outputs infra/cdk-outputs.json`.
- Deploy commands run from `infra`. Add `--profile <profile>`, `-c accountId=$(aws sts get-caller-identity --profile <profile> --query Account --output text)`, `-c twilioPhoneNumber=+1...` and every other non-default `-c` value your deployment uses: the CDK keeps no context between runs, and an omitted key goes back to its default. Set `CDK_DOCKER=finch` (or use Docker) for a deploy that builds the bridge image, and add `--exclusively` to deploy only the named stacks. Examples below show only the flag being changed.

## What you have

| Source | Holds | Key |
|---|---|---|
| CloudWatch Logs, `/owlbank/lambda/<function>` (one group per Lambda function; list them with `aws logs describe-log-groups --log-group-name-prefix /owlbank/lambda/`) and `/owlbank/ecs/tac-bridge` | One JSON line per event from every component | `call_sid` |
| With `bridgeHosting=agentcore`: the Runtime log group `/aws/bedrock-agentcore/runtimes/<RuntimeId>-DEFAULT` (the `RuntimeLogGroup` output of `OwlBankBridge`) instead of `/owlbank/ecs/tac-bridge`, and the warmer's `/owlbank/lambda/runtime_warmer` | The same bridge lines, one log stream per microVM; the warmer's pre-warm lines | `call_sid` (see "AgentCore hosting: investigating a call") |
| Saved Logs Insights queries, folder `OwlBank/` | The queries below | n/a |
| X-Ray (Lambdas: active tracing) | Timing and errors of one invocation and its AWS calls | `trace_id` (a field on the Lambda lines) |
| With `callObservability=on`: traces in `/owlbank/traces/bridge` (the bridge) and `aws/spans` (the front door), and CloudWatch GenAI Observability | One span per front-door webhook and per bridge turn, model call and tool call, in time order across the call; no caller or AI words with `traceContent=off` ("Call observability" below) | `session.id` = `call_sid` = the CallSid |
| The segment log (`segments` table) | The record of one call: prompts the caller heard, how each slot answer resolved (`slot_resolution`), verification outcomes, tool calls, handoffs, escalation | `call_sid` |
| Twilio Console, Monitor, Logs, Calls (and Debugger) | What Twilio did: webhooks, status, errors 11200 and 64xxx | the Call SID |
| AgentCore Memory (short-term), the call's session | The agent's real message history: every caller message, tool call, tool result and reply the model saw (model timings are in the bridge's `strands_model_call` lines, not here) | `sessionId` = the call SID, `actorId` = `owlbank-voice-strands` (see "The agent loop (Strands)"). Memory `owlbank_voice_stm-*` |
| The console audit table, `owlbank-<stage>-admin-audit` (stack `OwlBankAdmin`) | Every change made in the admin console, and each change of `acceptSpokenFactors` seen at an admin API cold start: who, when, which item and field, old and new values, and whether it was a security change | `pk` = the item (`policy#tool#<name>`, `policy#masking`, `policy#slot#<name>`, `policy#slot_resolution`, `policy#redaction`, `catalog#<key>`, `user#<username>`, `deploy#accept_spoken_factors`) |
| Twilio Conversation Orchestrator API, `GET https://conversations.twilio.com/v2/Conversations?channelId=<CallSid>` | The call's conversation records and their status (with `reuseConversation`, there should be exactly one) | the Call SID |

Without `callObservability=on` there are no agent traces: use the bridge's `turn` and `strands_model_call` lines and the AgentCore Memory history below. For what is never logged, see "What is never in the logs" at the end.

Every line has `ts`, `level`, `component` (the function name, such as `frontdoor`, `auth_service`, `bank_tools`, `level0_tools`, `context_api` or `flex_authorizer`, or `bridge`), `event`, and, when known, `call_sid`, `request_id` and `trace_id`. The Lambda runtime adds its own `timestamp`, `level`, `message` and `requestId`.

Events you will use, by area. Events with their own table below are only named here.

- **Every request and session:** `invocation_start`; `invocation_end` (`outcome` is `ok`, `degraded` or `error`; `duration_ms`; `result_code`); `invocation_error` (`error_class`, `error_message`, `stack_trace`); `dependency_call` (`dependency`, `operation`, `outcome`, `duration_ms`, `error_class`, `error_code`); `ws_session_start`; `ws_session_end` (`reason`, counts); `turn` (`turn`, `duration_ms`, `stop_reason`, `tools`, token counts).
- **Escalation:** `escalation_requested` (from the bridge); `escalation` (without Flex, the default: the front door read the briefing and ended the call, `destination` `briefing`; with Flex it enqueued the call and logs `queue=`).
- **Slot answers:** `factor_answer`, one per slot answer, with `factor`, `input_source` (`dtmf`, `speech` or `none`), `digit_count`, `speech_present`, `result_kind` and `tries`; the other fields are in the `factor_answer` rows of "Slot modes and the resolver". The segment log has the matching `slot_resolution` event. The tables below add fields to it; each row says which.
- **Agent text:** `ai_text_suppressed` (the bridge dropped text the model wrote alongside a tool call: `chars` only); `agent_settings` (bridge start-up: `thinking`, `prompt_caching`); `agent_text_release` (bridge start-up: `mode`, `stream` or `message`); `ai_text_spoken_before_tool` (WARNING: the model started a tool call after some of its text had been sent to the caller, `stream` mode only; `chars`, `tool`, `hop`, `turn`, never the text; also counted as `text_before_tool` on the `turn` line); `agent_speech_started` (`turn`, `ms_after_first_text`: ConversationRelay's `agentSpeaking on` came that long after the turn's first chunk was sent; once per turn; with `turn.wait_ms` it splits the caller's wait into the bridge's part and text-to-speech's part).
- **Conversations:** `extra_conversations_closed` (WARNING: the bridge found more than one ACTIVE Orchestrator conversation for the call and closed the extras).
- **Masking and hand-off:** `mask_spoken` and `mask_skipped` (latency masking, see "Event reference: masking, silence, keypad and agent loop" below); `handoff_hold` (how long the bridge held a hand-off for its phrase or for text it had already sent: `wait_ms`, `end_mode`, `prefetched`, `source` `mask` after a phrase, else `released_text`, and `basis`, the estimate that set the hold: `watch`, `hop_text` or `phrase`); `prefetch_unused` (the bridge's balance prefetch failed or was late: `cause`); front door `prefetch_used` (`age_s`) and `prefetch_ignored` (`reason`); `step_up_shortcut` (the bridge started a step-up for a tool refused only for the caller's level, with no model call: `tool`, `level`, `min_level`); `caller_words_dropped` (WARNING: words the caller spoke over a read-out could not be sent to the agent because the agent started a hand-off first; `chars` only).
- **One-time code sent in the background:** `otp_send_async` (front door, the send was started and the caller heard "Got it."); `otp_send_async_unavailable` (WARNING: the asynchronous invoke could not be started, so the code was sent synchronously); `otp_send_recorded` / `otp_send_not_recorded` (auth service, ERROR on the second: the outcome could not be written to call state); `otp_send_ready` (front door, `status` `sent`, `escalate` or `error`, `wait_ms` since the send started); `otp_send_timeout` (WARNING, `wait_ms`, `decision` `give_up`: no outcome within 2.5 s; the caller heard the error path).
- **Welcome offer:** `welcome_built` and `welcome_used` (see "Demoing the web-activity offer").
- **Audit and errors:** audit lines (`token_issued`, `verification_failed`, `rate_limited`, `verification_risk`, `otp_unavailable`, ...) carry `audit` and `event` with the same name. Error lines carry a `decision` (`retry`, `fallback_*`, `escalate`, `return_unavailable`, `ignored_because_*`): what the code did about the error.

**Stack traces.** A caught error (one the code handled and carried on from) logs `error_class`, `error_code`, `decision` and a `stack` field: the frames (file, line, function and source line) and the class names of the exception and of what caused it, oldest first. It never has the exception message, so to find the cause read the frames and the `error_code`, and open the code at the last frame. An unhandled error (`invocation_error`) has `stack_trace` (the same format) and a scrubbed `error_message`. A failed `dependency_call` has a `stack` too. Some expected outcomes log no stack on purpose: Verify check 404 (no pending verification), rejected Flex tokens, bad auth headers and unparseable input. `stack` is capped at 8 KB, so in Logs Insights use `display @timestamp, event, error_class, error_code, decision, stack` and expand the row.

## Event reference: masking, silence, keypad and agent loop

This table covers latency masking, the bridge's silence, keypad and hang-up handling, the front door's Gather events and the agent loop. None of them carries caller or AI text, digits or tool arguments.

| Event | From | Fields | How to read it |
|---|---|---|---|
| `mask_spoken` | bridge | `key`, `tool`, `kind` (`timer` or `tool`), `turn`, `ms_since_turn_start`, `after_ms`, `source` | A masking phrase was sent. `source` says where the delay came from: `tool` (the tool's own `mask.after_ms`), `default` (`masking.tool_after_ms`, or `masking.after_ms` for the acknowledgement) or `handoff` (a hand-off tool, spoken at once). `ms_since_turn_start` much larger than `after_ms` means the phrase waited for the acknowledgement to end |
| `mask_skipped` | bridge | `tool`, `key`, `reason`, `turn`, `ms_since_turn_start`, and `after_ms`, `source` once it was scheduled | A tool phrase not spoken. `reason`: `answer_started` (the answer came first: the delay is long enough, or longer than needed), `tool_off` (the tool has no `mask`), `turn_phrase_used` (the turn already had its tool phrase, or another masked tool came first in the same message), `superseded_by_handoff` (a hand-off replaced a pending phrase), `turn_ended` (the turn ended while the phrase was waiting), `no_renderable_phrase` (no phrase in the pool could be filled; see `mask_phrase_unrenderable`) |
| `mask_phrase_unrenderable` | bridge | ERROR: `key`, `tool`, `language`, `pool`, `decision` `no_phrase` | No phrase in the tool's pool could be rendered for this call (an unknown placeholder, a missing word in the call's language). Fix the catalog text or add the `arg.*` word |
| `policy field <field> is invalid` | bridge | ERROR, the field name | A masking value in the policy table is of the wrong type or out of range (the field's default is used), or a field is unknown (ignored). An absent field takes its default with no ERROR. Fix it in the console or with `scripts/masking.py` |
| `prompt_repeat` | bridge (also in the segment log) | `repeat`, `max_repeats`, `what`, `chars`, `timeout_ms`, `waited_ms` | The caller was silent for `waited_ms` after the last message, so it was sent again. `waited_ms` is `greeting_silence_timeout_ms` for the first AI session's greeting, else `timeout_ms` (`silence_timeout_ms`). `what` is the catalog key, `agent` (the agent's words) or `greeting` (the front door's welcome greeting) |
| `silence_reset` | bridge | `kind` (`prompt`, `dtmf`, `interrupt`), `repeats` | Caller input after one or more repeats: the count starts again. `repeats` leaves out a collided repeat (next row), so it is absent when the only repeat collided |
| `prompt_repeat_collided` | bridge | `repeat`, `what`, `ms_after_repeat`, `grace_ms` (500), `cause` (`interrupt` or `client_speaking`) | An `interrupt` or a `clientSpeaking on` came within `grace_ms` of a repeat being sent: the caller was already speaking when the repeat went out. That repeat doesn't count. Normal now and then, at the start of an answer; often on one call means the silence timeout is shorter than the caller's pause before answering (raise `silence_timeout_ms`) |
| `silence_action` | bridge | `action` (`escalate` or `hangup`), `repeats`, `timeout_ms`, `what` | The caller stayed silent after the last repeat. `escalate` is followed by `escalation_requested` with `reason` `no_input` and `source` `bridge`; `hangup` by the front door's `no_input_hangup` |
| `end_call_requested` | bridge | `level` | The model called `end_call`; the bridge handed off with reason `caller_done` (segment log `handoff` with `tool` `end_call`, `outcome` `requested`) |
| `end_call_refused` | bridge | `cause` (`disabled` or `pending`), and for `pending` the step: `tool`, `result` or `handoff` | `end_call` was refused. `disabled`: the policy switch `end_call.enabled` is off (the model got `TOOL_DISABLED`). `pending`: a step was waiting (`CALL_IN_PROGRESS`), so the call went on |
| `caller_done_hangup` | front door | `source` `frontdoor`, `level_reached` (absent when call state couldn't be read) | The front door said `common.goodbye` for a hand-off with reason `caller_done` and hung up. Not an escalation: no `escalation` line follows |
| `hangup_without_call_state` | front door | `decision` | A hang-up hand-off (`caller_done` or `no_input_hangup`) found no call state. The goodbye is said in the default language and the call still hangs up; a read that failed with an error logs `hangup without call state` instead |
| `verification_closed` | bridge | `tool` | The login failed or was declined earlier in this call, and the model called `authenticate` (refused `VERIFICATION_CLOSED`) or a tool above the caller's level (its `AUTH_LEVEL_INSUFFICIENT` refusal says `"verification": "closed"`). The front door's segment log `verification` event with `outcome` `login_closed` is the same refusal made there |
| `relay_speaker_event` | bridge app | `recognised`; recognised: `speaker_event` (`clientSpeaking` or `agentSpeaking`), `value` (`on` or `off`); unrecognised: `type`, `keys` (key names only) | A ConversationRelay speaker event (the front door sets `events="speaker-events"`). Never passed to TAC. The caller speaking pauses the silence wait; the agent's `off` starts it. `recognised` false means the message shape isn't the one the bridge expects: the session falls back to estimate timing; report the `type` and `keys` |
| `silence_timing` | bridge | `mode` (`speaker_events` or `estimate`) | Once per session: `speaker_events` at the session's first recognised speaker event, `estimate` at the end of a session that had none (the wait counted from estimated speech ends). `estimate` on every session after a deploy means the speaker events aren't arriving or aren't recognised |
| `silence_skipped` | bridge | `kind`, `cause` (`no_session` or `input_or_turn`) | A silence expiry that did nothing, because the session was gone or input or a turn arrived at the same moment. Normal |
| `no_input_hangup` | front door | `source` `frontdoor`, `level_reached` (absent when call state couldn't be read) | The front door said `common.goodbye_no_input` and hung up (`silence_action` = `hangup`) |
| `relay_interrupt` | bridge | `playing`, `ms_played` | ConversationRelay reported the caller cut in. `playing` is what the session sent last: a catalog key, `agent`, `greeting` or `unknown`; `ms_played` is ConversationRelay's `durationUntilInterruptMs` |
| `factor_partial` | front door | `slot`, `slot_type` (`digits` or `choice`), `chars`, `ms_since_gather`, `outcome` | A partial speech result for a digit factor, or for a choice Gather (the card question and the follow-ups that list cards). `outcome`: `incomplete` (for a choice also: no card, more than one, a yes or no word, or only the model could resolve it), `ambiguous` (two values of the right length), `complete` (the Gather was claimed and ended early), `already_claimed`, `stale` (not the current Gather); the phone slot also: `settling` (a complete reading waits `settle_ms` for a newer partial), `superseded` (a newer partial, by `SequenceNumber`, owns the decision; also an older one delivered late), `unsequenced` (no valid `SequenceNumber`: never claims) |
| `factor_early` | front door | `slot`, `slot_type`, `ms_since_gather`, `ms_since_partial`, `speech_timeout_ms`, `digits_present` | The early redirect arrived and its digits went to verification (for a choice, the card was taken, `match` `exact`, `source` `early`, and read back). `ms_since_partial` is how long the REST update took to take effect; `speech_timeout_ms` is roughly the wait saved |
| `factor_replay` | front door | `via` (`early` or `final`), `state` | A second request for a Gather already answered (Twilio's own final webhook after an early finish, or a retry); it got the same TwiML (Twilio's XML call instructions), with no second verification |
| `factor_early_update_failed` | front door | `slot`, `decision` `gather_finishes_on_its_own`, `error_class`, `error_code` | The REST call update failed; the Gather ended on its own timeout as before. Repeated failures: check the Twilio API key secret and Twilio's status |
| `factor_answer` | front door | adds `gather_ms`, and `short_digits` | With the last `factor_partial`'s `ms_since_gather`, the difference is Twilio's end-of-speech wait. `short_digits: true` (with `result_kind` `unclear` and `input_source` `none`; in the log line only; the segment log's `slot_resolution` has `source` `short_digits`): the words were only number and carrier words with too few digits, so the caller was asked again at once without a model call. It means Twilio ended the Gather while the caller was still giving digits: they paused longer than `speechTimeout` |
| `twilio_connection_warm_failed` | front door | WARNING: `error_class` (`HttpStatus` for a non-2xx answer), `http_status`, `decision` `ignored` | At INIT the one account read that opens the Twilio connection failed. Harmless: the first early finish opens the connection itself (about 1 s slower). Every environment logging it with 401/403: the API key can't read its account. `twilio_connection_warmed` (INFO, `ms`) is the success line |
| `twilio_client_init_failed` | front door | WARNING: `error_class`, `error_code`, `decision` `build_on_first_use`, `stack` | At INIT (provisioned concurrency) the Twilio REST client for early finish couldn't be built, usually the API key secret unreadable. The first early finish builds it instead and pays about 1 s more. Every new environment logging it: check the secret and the front door's permission to read it; a following `factor_early_update_failed` means early finish isn't working at all |
| `frontdoor: malformed request` | front door | WARNING: `reason` `malformed_body`, `error_class`, `route`, `decision` `reject_400` | A request with the right Basic credentials whose body couldn't be read as a UTF-8 form; it got a generic 400. The body is never logged. Twilio's webhooks are plain forms, so this points at a client that isn't Twilio or a proxy that altered the body. Not counted as an auth failure |
| `webhook_duplicate` | bridge | `token_ref`, `decision` `ignored` | Twilio delivered an Orchestrator webhook to `/webhook` again with an `I-Twilio-Idempotency-Token` this task had seen in the last 10 minutes; it was answered 200 and not passed to TAC. `token_ref` is the first 8 hex characters of the token's SHA-256, so two lines with the same `token_ref` are the same delivery. Occasional repeats are Twilio retrying; many mean the bridge answers `/webhook` too slowly or with errors |
| `strands_model_call` | bridge | `call`, `model_ttfb_s`, `model_latency_s`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `stop_reason` | One line per model call. With `promptCaching=on`, `cache_read_tokens` above 0 on later calls means the cache works (production guide, "What the LLM sees, and never sees", under "Prompt caching") |
| `agent_history_new` / `agent_history_reload` | bridge | `reason`, `duration_ms`, `outcome`, `messages`, `interrupted` | The call's agent was started empty (`new_session`) or rebuilt from AgentCore Memory (`owner_changed`, `cache_miss`, `build_failed`, `unknown_tool_use_recovery`). WARNING with `outcome` `error` if AgentCore Memory couldn't be read |

## Keyed-in phone number (`keyed_lookup`)

When the caller ID matches no customer, or is withheld, the front door asks for the account's phone number (policy switch `ani_trust.ask_phone_on_no_match`). Neither event carries the number, digits or caller words.

| Event | From | Fields | How to read it |
|---|---|---|---|
| `keyed_lookup` | front door (log line and segment log event) | `outcome`, `count` | One per keyed number looked up, or per give-up. `outcome`: `match` (`count` 1: login goes on with the untrusted factors), `multiple` (`count` is the number of candidates), `no_match` (`count` 0: asked once more, then the no-match handling), `error` (the lookup failed twice: apology, then the `lookup_error` fallback), `unclear` (no usable ten digits: the no-match handling). `count` is 0 for every outcome but `match` and `multiple` |
| `keyed_number_unavailable` | front door | ERROR: `decision` `apology_and_escalate` | The sealed number could not be read back when a later step (ZIP and SSN, the code, the check) needed it, so the caller was handed to a human. The call never falls back to the calling number. Check the call's other lines by `call_sid` for a sensitive-store or KMS error |

To find the calls: filter `keyed_lookup` and group by `outcome`. A rising `no_match` or `unclear` share means callers don't know which number is on file, or speech recognition struggles with digits (see `factor_answer`). Task attribute `identity_source` is `keyed_phone` on the Flex task of an escalated call. To turn the step off, see the production guide, "Configuring caller-ID trust".

## Slot settings and sensitive values

This section covers the model-assisted slots' per-slot `sensitive`, `keypad_only` and `barge_in` settings and the `acceptSpokenFactors` ceiling (production guide, "How the IVR collects answers"). None of these lines carries a value, the caller's words or a sealed blob: slot names, setting names, booleans and counts only.

| Event | From | Fields | How to read it |
|---|---|---|---|
| `slot_input` | front door | `acceptSpokenFactors`, `slots` (per slot: `keypad_only`, `barge_in`, both effective) | Once per cold start: what the Gathers will use. A factor showing `keypad_only: true` with its own setting off means `acceptSpokenFactors` is off. `slot_input_failed` (caught error, `decision` `ignored`): the line couldn't be built; the call path is unaffected |
| `verification_result` | front door (also in the segment log as `verification`) | `outcome` (`verified`, `not_verified`, `factor_held`, `ambiguous`, `declined`, `login_closed`, or the auth service's status such as `locked`), `factors` (names), `level`, or `count` (candidates) for `ambiguous` | One per verification check. Read it with `factor_answer`: a complete `digit_count` followed by `not_verified` is a wrong value, not a hearing problem. `declined`: the caller declined a factor (spoken or `#` alone); no check ran, no attempt counted, `factors` names the factor only. `login_closed`: a second login was refused after a failed or declined one |
| `policy slot field <field> is invalid (<reason>)` | front door, bridge (each policy load) | ERROR, `decision` `slot_fail_closed`, the item and field (`slot#ssn4.keypad_only`), never a value | The table's slot item is wrong. A non-boolean setting loads as `true` (fail closed). `no item; loaded fail closed (sensitive)` for a factor: the table has no item for the factor (not seeded), so it counts as sensitive until the seed writes it |
| `sensitive_scrub_skipped` | front door, bridge | INFO: `slot`, `reason`, `decision` `sealed_not_registered` | A sensitive value was sealed but not registered for the log scrub, once per slot per call in each process. `reason` `too_short`: under 4 characters. `reason` `slot_type`: a `yes_no` or `choice` answer. Either way the answering turn still carries the placeholder; the scrub of later log lines doesn't cover that value |
| `sensitive_seal_refused` | front door | ERROR: `field` `sealed_sensitive`, `slot`, `reason` `limit`, `limit` (16), `decision` `registered_in_process_only` | The call already holds 16 sealed values. This one is scrubbed only by the process that heard it; another front-door instance or the bridge won't scrub it. Shouldn't happen with the seed's slots; a repeat points at a loop asking the same slot |
| `sensitive_unseal_failed` | front door, bridge | ERROR (caught): `operation` `restore_sensitive`, `error_class`, `decision` `withhold_texts` | A sealed value couldn't be unsealed (KMS error, wrong key or context). The call is **unverifiable** in that process: its AI turns are recorded in the segment log as `[WITHHELD]` and its escalation summary is the catalog's `escalation.summary_withheld`. What the agent receives is unchanged. Check the KMS key and the process's `kms:Decrypt` permission |
| `admin_deploy_setting` | admin API | `setting` `acceptSpokenFactors`, `before`, `after`, `security_change`, `outcome` (`recorded`, `unchanged` at DEBUG, `lost_race`) | At cold start the admin API compared `acceptSpokenFactors` with the last recorded value. `recorded` wrote a `deploy_setting` audit entry (below). An ERROR `admin api: deploy setting not recorded` (`decision` `continue`) means it couldn't; the start went ahead |

**The `deploy_setting` audit entry.** `acceptSpokenFactors` is set by the deploy, not the console, so the admin API records each change in the audit table under `pk` = `deploy#accept_spoken_factors`: `action` `deploy_setting`, actor `{"sub": "deploy"}`, the old and new value (`old` is `null` the first time), and `security_change: true` when it turned on. One entry per change, even when two cold starts race. It is written when the admin API next starts, not at the moment of the deploy, and only if `OwlBankAdmin` is deployed. In the console: Audit, item `deploy#accept_spoken_factors`.

**Changing a slot setting.** Console, Slots: one switch per setting; each save writes that field alone with the row's `rev` (409: someone saved first; reload). Turning `sensitive` or `keypad_only` off asks for confirmation, and any change to either is a security change in the audit log (`pk` = `policy#slot#<name>`). The page shows the read-only `acceptSpokenFactors` and an "Effective: keypad only" badge where it forces a factor to the keypad, and warns on every factor that isn't sensitive. It takes effect within 60 s. A reseed keeps the table's values unless run with `--overwrite-slots` ("Prompts and the seed", "What a reseed keeps and overwrites").

**What the AI got for a sensitive slot.** The agent's history in AgentCore Memory shows the answering turn with `[<slot name>]` in place of the answer ("what's my balance? my last four are [ssn4]"). The value itself should appear nowhere: not in AgentCore Memory, the segment log or the logs (a value under 4 characters isn't scrubbed from later log lines; see `sensitive_scrub_skipped`). A non-sensitive slot's words (ZIP in the seed) appear unchanged.

## Slot modes and the resolver

This section covers the model-assisted slots: the `deterministic`, `assisted` and `generative` modes, the `slotModeCeiling` deploy setting, the resolver, corrections, the card question and the redaction switch (production guide, "How the IVR collects answers" and "Configuring redaction"). None of these lines carries the caller's words, the resolver's input or output, or a sensitive value: slot, flow and setting names, modes, kinds, counts, durations and error codes only.

| Event | From | Fields | How to read it |
|---|---|---|---|
| `slot_modes` | front door | `ceiling`, `default_mode`, `flows`, `generative_enabled`, `slots` (per slot and flow: `configured`, `source`, `effective`) | Once per cold start: the modes in force. `source` is `slot`, `flow`, `default` or `invalid`. To check what a call ran in, read its `factor_answer` lines instead (the modes are frozen when each flow starts) |
| `slot_mode_capped` | front door | `scope` (`default`, `flow`, `slot`), `target`, `configured`, `effective`, `ceiling`, `reason`, `flows` | Once per cold start for each setting above what runs. `reason`: `ceiling` (above `slotModeCeiling`), `generative_disabled` (`generative` without `generative_enabled`), `type` or `sensitive` (`generative` on a non-choice or sensitive slot runs `assisted`). Expected, not an error |
| `slot_mode_invalid` | front door | ERROR: `item`, `field`, `policy_version`, `decision` `deterministic` | A mode value in the policy table is missing, unknown or of the wrong type, so that slot runs `deterministic` (no model call; the call goes on). Once per process per item, field and policy version. **Alarm `SlotModeInvalid`** (threshold 1 in 5 minutes, alarm topic). Fix the field in the console's Slots view, or reseed. A table with no `slot_resolution` item (field `default_mode`) logs it until it is seeded |
| `resolver_call` | front door | `slot`, `flow`, `mode`, `latency_ms`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `kind` | One model call that returned a tool call. `input_tokens` counts only the tokens not read from or written to the prompt cache (the fixed instructions; production guide, "What the LLM sees, and never sees", under "Prompt caching"); the total sent is the sum of the three. Expect `cache_write_tokens` on the first call after a quiet 5 minutes and `cache_read_tokens` after that; both null or 0 on every call means caching isn't taking effect, which costs latency, not correctness. `kind` is the model's, before grounding: a value it proposed may still be refused (`model` `ungrounded` on `factor_answer`) |
| `resolver_timeout` | front door | WARNING: `slot`, `flow`, `mode`, `elapsed_ms`, `budget_ms`, `reason` (`deadline` or `no_worker`), `decision` `fallback_none` | The call didn't finish within `resolver_budget_ms` (2500 by default; set in the console's `slot_resolution` card). The caller was asked again with the catalog template, using a try. `no_worker`: both resolver workers were still busy with abandoned calls. Many of them: compare `dependency_call` `bedrock` durations and Bedrock throttling for the profile; raising `resolver_budget_ms` (console, `slot_resolution`) lengthens the silence callers hear |
| `resolver_error` | front door | WARNING or caught error: `slot`, `flow`, `mode`, `reason` (`error`, `refusal`, `malformed`), `decision` `fallback_none`; `error_class`, `error_code` and `stack` for `error`; `stop_reason` or `tool_calls` for the others | The call failed, refused, or returned something off-schema; treated as `none` (asked again, one try). `error` with `AccessDeniedException`: model access in one of the profile's Regions, or the front door role's `SlotResolver` statement; `ThrottlingException`: the Bedrock quota for the profile. Many `refusal` or `malformed`: check the model (`OWLBANK_RESOLVER_MODEL_ID`) against the bench |
| `factor_answer` | front door | adds `flow`, `configured_mode`, `effective_mode`, `mode_source`, `ceiling`, `capped`, `match`, `source`, `model`, and `value` for a non-sensitive slot only | One per slot answer. `match` (`exact`, `inferred`, `ambiguous`, `none`, `correction`, `intent_switch`, ...), `source` (`dtmf`, `speech`, `alias`, `ordinal`, `early`, `command`, `model`, `short_digits`, `none`), `model` (`not_called`, `ok`, `timeout`, `error`, `refusal`, `malformed`, `ungrounded`). `effective_mode` `deterministic` with `mode_source` `invalid`: see `slot_mode_invalid` |
| `factor_answer` `timings_ms` | front door | `{name: ms}`: `state_load`, `policy`, `catalog`, `resolve`, `render`, `state_save`, and `speech_redact` with `speechLogRedaction` on (names and integers only, never values) | Where the answer request's own time went. Spans nest and repeat: a name used twice is summed, and `resolve` includes any catalog or resolver time inside it, so the entries don't add up to `invocation_end` `duration_ms`. `policy` or `catalog` in the hundreds of ms: that request paid a cache reload (look for the `reload:*` line next to it). `resolve` large in `assisted`: the resolver (`resolver_call` `latency_ms`). `state_load` or `state_save` large: DynamoDB on `calls` (`dependency_call`). Silence the caller heard after an answer is Twilio's end-of-speech wait plus this request: see "Silence after the SSN or the code" |
| `factor_answer` `speech` | front door | the turn's transcript, only with `speechLog` on (the default) and a spoken answer | What Twilio transcribed for this Gather (`SpeechResult`). A sensitive slot's value is its placeholder (`[ssn4]`, `[otp]`; the same text the agent gets); a non-sensitive slot's turn is as said. No field for keypad input (`digit_count` instead) or with `speechLog=off`. With `speechLogRedaction=on` the text is also masked by the transcript redaction (`[NAME]`); `[withheld]` means the PII check failed for that turn (look for the `PII check unavailable` WARNING with `surface` `speech_log`). Use it for "the caller said the right thing but was asked again": compare `speech` with `match` and `source`. Never on `resolver_call` or in the segment log |
| `dependency_call` `reload:policy`, `reload:catalog` | front door, bridge, context API, admin API | `dependency` `dynamodb`, `operation`, `outcome` (`ok`, `slow`), `duration_ms`, `items` | One line per cache reload, on the first request after the 60 s TTL: the policy in one `GetItem` and one `BatchGetItem`, the catalog in one paginated `Scan`. INFO, or WARNING with `outcome` `slow` over 500 ms. A failed reload logs as any failed `dependency_call` and keeps the last good copy (or the bundled seed). `AccessDeniedException` on `reload:catalog`: the role lacks its `CatalogScan` statement (`dynamodb:Scan` on the catalog table); on the admin API's `reload:policy`, its `PolicyBatchRead` |
| `slot_preload` | front door | `slots`, `texts`, `languages`, `duration_ms` | Once per cold start: the slot code, the slot table and the `option.*` and `slot.*` catalog texts were loaded, so the first card answer pays no import or first read. `slot_preload_failed` (caught error, `decision` `ignored`): the reads happen on first use instead; the call path is unaffected |
| `slot_resolution` | front door, segment log only | `flow`, `slot`, the same mode fields, `match`, `source`, `model`, `asked` (`confirm_digits`, `confirm_option`, `disambiguate`, `not_available` or `none`), `answer` (`yes`, `no`, `none`), `correction`, `correction_of`, `tries`, and `value` for a non-sensitive slot only | The segment log's record of each answer. An applied correction writes its own event, with `correction: true` and `correction_of` = the slot being asked when the caller corrected |
| `generative_fallback` | front door | `slot`, `flow`, `reason`, `decision` `template`, `asked` | A generated question wasn't spoken; the catalog template was. `reason`: `missing` (no sentence), `lang` (another language), `placeholder` (an option not allowed here, or a stray brace), `number` (a digit or a number word in the call's language; Spanish "un" and "una" are allowed, "uno" and ordinals are not), `length` (over 200 characters, after healing), `budget` (the resolver ran out of time), and `twiml_invalid` (ERROR, with `template` in place of `asked`: the rendered response didn't parse as XML, so the template was spoken; report it with the call SID). Invisible and control characters are removed before the check (healing), so they never cause a fallback. Only with `slotModeCeiling=generative` and `generative_enabled`. The segment log's `prompt` event of a generated sentence has `generated: true` |
| `transcript_redaction_off` | bridge, front door, context API | WARNING: `component`, `decision` `pass_through` | Once per process: the `redaction` setting has `enabled: false`, so no Comprehend call masks AI turns, summaries or the timeline. Sensitive slot values are still replaced and scrubbed. If it shouldn't be off: the console's Redaction view (admins), or `scripts/seed.py --overwrite-redaction` |

**Changing a mode.** Console, Slots: a slot's `mode` and `confirm`, or the `slot_resolution` card (`default_mode`, the flow modes, `resolver_budget_ms`; `generative_enabled` for admins only). Before a mode change is saved, the console shows which flows and slots would then call a model; moving away from `deterministic` is a security change in the audit log (`policy#slot#<name>`, `policy#slot_resolution`). A change applies from the next flow that starts, within 60 s. The ceiling is a deploy setting: `slotModeCeiling` on `OwlBankCompute` and `OwlBankAdmin`.

## The Bedrock logging alarm

A message on the alarm topic from the EventBridge rule `owlbank-<stage>-bedrock-logging-change` (stack `OwlBankData`) means someone called `PutModelInvocationLoggingConfiguration` in the deployment Region (the rule watches only the deployment Region). The message is the CloudTrail event: `userIdentity` says who, `awsRegion` where, and `requestParameters.loggingConfig` whether text is delivered. While logging delivers text, every resolver prompt, with the caller's spoken digits in it (SSN last four and codes included), is written to the logging destination. What to do:
1. **Run preflight** with the deployed ceiling, from `infra`: `uv run python scripts/preflight.py --profile <profile> --stage <stage> --slot-mode-ceiling <ceiling>`. The "Bedrock invocation logging" lines say, per Region, whether text is delivered, and print the fix.
2. **If it FAILs, delete the logging configuration** in each Region it names, with the account owner's approval: `aws bedrock delete-model-invocation-logging-configuration --region <region> --profile <profile>`. If the logging is needed for something else, keep it and turn text delivery off instead (`put-model-invocation-logging-configuration` with the same `loggingConfig` and `textDataDeliveryEnabled` false). Run preflight again until it passes.
3. If neither can be done at once, stop the prompts: deploy `OwlBankCompute` and `OwlBankAdmin` with `-c slotModeCeiling=deterministic` and the usual `-c` values (no resolver call, and no Bedrock permission on the front door).
4. Whichever you chose, prompts logged while text delivery was on hold callers' digits: treat the destination (its S3 bucket or log group) as customer data, and delete those objects under your data policy.

A logging change made on purpose (for example a test with a synthetic prompt) fires it too.

**CloudTrail.** The alarm needs a trail that is logging management write events in the Region. Preflight checks it on every run: a WARN in `dev` and `test`, a FAIL at `stage=prod`.

## Saved queries

CloudWatch, Logs, Logs Insights, Queries, folder `OwlBank/`:

- **Trace a call by call SID** searches all the `/owlbank/*` application log groups, sorted by time. Replace the `call_sid` parameter with the SID in double quotes, for example `"CA0123456789abcdef0123456789abcdef"`. Set the time range to cover the call (plus a few minutes).
- **Errors by component** counts `ERROR` and `WARNING` lines and `outcome = "error"` by component and event.
- **Slowest invocations and dependency calls**: the top 50 by `duration_ms`. Dependency calls slower than 500 ms are logged at WARNING, so they show even at INFO.
- **Auth failures**: rejected Twilio requests, failed and rate-limited verifications, risk escalations and codes sent to untrusted caller IDs, in 5-minute bins.
- **Escalations by reason**: how many calls went to a human, by reason and source.
- **Call timeline** (only with `callObservability=on`): every span of one call in start order, from the traces log group and `aws/spans`. Parameter `session_id`, the CallSid in double quotes. See "Call observability" below, and `scripts/call_timeline.py` for the gaps between spans.

## From a CallSid

1. Run **Trace a call by call SID**. Read it top to bottom: `frontdoor` `invocation_start` for `incoming`, then lookup, language, factors (`auth_service`), `ws_session_start` (the bridge), one `turn` line per prompt, handoffs back to the front door, `escalation` or the status callback at the end.
2. Find the first line with `level` `WARNING` or `ERROR`, or `outcome` `degraded` or `error`. Its `decision` says what the code did next. The line before it is usually a `dependency_call` with an `error_code` (a DynamoDB, Lambda, Bedrock, KMS, Comprehend, Verify or AgentCore code).
3. With `callObservability=on`, the whole call's spans, front door and bridge, are one query away ("Call observability" below). For a Lambda line, take `trace_id` and open it in X-Ray (CloudWatch, X-Ray traces) to see the AWS calls and where the time went. For the agent, read the bridge's `turn` and `strands_model_call` lines, and the session's history in AgentCore Memory (the `list-events` command in "The agent loop (Strands)"). It shows exactly what the model was given, which is the fastest way to explain an odd answer. It contains the caller's words; treat it as customer data.
4. For what the caller heard and what the call did (no caller or AI words, by design), read the segment log: DynamoDB, table `segments`, partition key `call_sid`. After an escalation the Flex screen pop reads the same segment log through the Context API: `GET /context/{journal_ref}?call_sid=CA...` with a valid `X-Flex-Token` (the `journal_ref` is in the task attributes and in the `calls` item's `escalation`). Without a Flex token, read the table.
5. For the Twilio side, open the call in Twilio Console, Monitor, Logs, Calls: its webhook requests and responses, status, duration and the Debugger alerts. A front-door `twilio_auth_failed` line with no matching call in the trace usually means Twilio's request was rejected (signature, Basic auth or route), not that the code failed. A `twilio_basic_challenge` line (INFO) is normal: it is the first request of every webhook, answered `401` so Twilio repeats it with its credentials. If a call plays "application error" and the Console shows a `401` or `403` on the voice URL, check that the repeat arrived (`twilio_signature_form` follows the challenge) and the WAF logs for a block on the repeat.

## From a time

Run **Errors by component** over that window and look at the biggest counts. Then run **Slowest invocations and dependency calls** to see whether a dependency was slow (DynamoDB, Bedrock, AgentCore Memory, the Gateway). Pick a `call_sid` from a bad line and use "From a CallSid".

## From a symptom

| Symptom | Look for |
|---|---|
| **The caller heard an apology** | `handler error` (front door, `decision` `apology_and_escalate`), or a bridge `turn` with `outcome` `error` or `escalated`, or `agent turn failed` with `decision` `retry` then `escalate`. The `error_class` and `error_code` give the cause: a throttled model (`ThrottlingException`), a model stream error, a missing catalog text |
| **A request for a person got a read-out and the call ended** | By design without Flex: the call was not queued or transferred (see "Escalation without Flex: the briefing") |
| **Escalated unexpectedly** | The `escalation` line gives `reason` and `source`. `verification_risk`: an untrusted caller ID hit the attempt cap or needed a code before login (`verification_risk` audit line, `cause`). `otp_unavailable`: no number to text (`otp_unavailable` audit, `reason`). `unable_to_resolve`: an error path (see the apology row). `escalation_requested` from the bridge with `source` `agent` means the model chose to escalate |
| **A payment did not go through** | `bank_tools` `invocation_end` for `make_payment` (`result_code`), the `customer_write` audit line (written only on success), `token_rejected` (`code` says why: `EXPIRED`, `WRONG_CALL`, `AUTH_LEVEL_INSUFFICIENT`), and bridge lines `bank_tools call failed` (`attempt`, `decision`). A payment carries the model's tool-use ID as its idempotency key, so a retry cannot move money twice. `idempotent_replay` shows a retry that returned the first answer. A confirmation flow ending in `CANCELLED_BY_CALLER` is in the segment log (`confirm_answer`) |
| **An alarm fired** | Lambda errors or throttles: **Errors by component**, then the function's `invocation_error` lines (stack trace). `FrontdoorRejected` or `BridgeWsRejected`: **Auth failures**, `reason` says signature, basic, nonce or route. `OtpSentUntrustedAni`: the `otp_sent_untrusted_ani` line and the `otp_sent` audit line for the same `call_sid`. A code sent on a keyed-in number's call doesn't raise it: it logs `otp_sent_keyed_phone`, counted as `OtpSentKeyedPhone` with no alarm (watch it for a sudden rise). `SlotModeInvalid`: the `slot_mode_invalid` line names the item and field. A notification from the `bedrock-logging-change` rule: "The Bedrock logging alarm". 5xx or unhealthy hosts: the bridge group, `ws_session_end` with `reason` `error`, and `relay session failed` |
| **Slow answers** | **Slowest invocations and dependency calls**; the bridge `turn` `duration_ms`. Compare it with the `strands_model_call` lines of the turn (`model_ttfb_s`, `model_latency_s`, one per model call): the difference is the loop's own time and the tools. A turn with a tool call has two model calls. Silence right after a scripted step (after "you're verified" or a balance read-out): the time from `ws_session_start` to TAC's "Started VOICE conversation" line is Orchestrator creating a conversation (several seconds when `reuseConversation` is off) |
| **"Something went wrong on our end" right after returning from a scripted step** | Segment log `handoff` with `outcome` `relay_failed` and `error_code` `64105`; bridge line from TAC "Expected exactly 1 conversation for call_sid ..., but found N after 20 attempts". **N = 0**: Orchestrator didn't create the conversation within about 30 s (TAC polls for it; the bridge raises the poll to about 30 s). **N > 1**: should not happen; look for `extra_conversations_closed` and check that the front door and the bridge have the same `OWLBANK_REUSE_CONVERSATION` (they differ during a deploy of one of the two stacks). List the call's conversations with the Orchestrator API (sources table) |
| **`agent turn failed` with "Invalid signature in thinking block"** | The model request lacks `block_binding` `prefix_mismatch_behavior: drop_block`, which `thinking: adaptive` sends: the entry's `thinking` is `default`. The bridge's start-up line `agent_settings` shows it (`thinking`), as does `thinking` in `OWLBANK_MODELS` in the bridge task definition. Deploy `OwlBankBridge` without `thinking: default` in `modelParams` |
| **Silence after the SSN or the code** | Two possible causes: the speech model waiting for the caller to finish, or the front door's own work. Take the `factor_answer` line for the factor and the front-door `invocation_end` lines with `op` `factor` around it, then follow the `input_source`.<br><br>**`input_source` `speech`:** the gather ends `speechTimeout` seconds (3 by default) after the caller stops, or earlier with `earlyDigits` (a `factor_partial` with `outcome` `complete`, then `factor_early`). That wait is before the front door's webhook is called. `factor_answer` `gather_ms` minus the last `factor_partial` `ms_since_gather` is Twilio's end-of-speech wait.<br>- No `factor_partial` lines at all: the slot is keypad-only (`slot_input`), `earlyDigits` is off, or the partial requests aren't reaching the API (check the API WAF log for `/voice/partial`).<br>- `factor_early_update_failed`: the REST update failed and the Gather finished on its own.<br><br>**`input_source` `dtmf`** with the full `digit_count`: the gather ended on the last key, so any silence is the front door's. Its work is the `invocation_end` `duration_ms` of that `factor` request (the auth service's check).<br><br>**After the SSN in a step-up:** "Got it." should come at once (`otp_send_async`), then a second `factor` request (the redirect) whose `duration_ms` includes waiting for the code send: `otp_send_ready` `wait_ms` (2.5 s is the cap, then `otp_send_timeout`). No `otp_send_async` but `otp_send_async_unavailable`: the send was synchronous, so the first request's duration includes it.<br><br>**After the code:** the step-up finishes and the call returns to ConversationRelay: see the "Slow answers" row for the reconnect |
| **Silence after the caller names a card** | The card Gather's `factor_partial` lines (`slot_type` `choice`): `complete` then `factor_early` means it finished early, and the remaining wait is the front door's request; only `incomplete` means the partials never held exactly one of the caller's cards by name, alias or ordinal (a phrase only the model could resolve waits for the final result). Then the `factor_answer` `timings_ms` of that request: `policy` or `catalog` large is a cache reload (`reload:*`), `resolve` large is the resolver. No `factor_partial` at all: the card slot is keypad-only |
| **The caller was asked again for digits on the first try** | `factor_partial` lines for that Gather: `outcome` `incomplete` only, never `complete`, means Twilio's final result came before all the digits were heard. Then `factor_answer`: `short_digits: true` means the caller paused longer than `speechTimeout` mid-answer (3 s by default; a 1 s timeout cuts answers off after one digit); `result_kind` `unclear` with `speech_present` and no `short_digits` means other words came with the digits and the deterministic stages couldn't take them: in `deterministic` that is a re-ask, and in `assisted` the resolver had the words (`model` says what it returned: `ungrounded` is a value the caller didn't say, `timeout` the `resolver_budget_ms` budget, 2.5 s by default). No `factor_partial` at all: early digits is off or partials aren't arriving (see "Silence after the SSN or the code"). Raise `speechTimeout` (Compute deploy) if callers often pause this long |
| **The caller entered the right digits but verification failed** | `factor_answer` lines for the call: `input_source` `speech` at a keypad-only slot (the slot's `keypad_only`, or a factor with `acceptSpokenFactors` off; the caller heard "please use your keypad"), `digit_count` short of the factor's length (input ended early), or `result_kind` `unclear`. Two retries per factor, then the step fails. If `factor_answer` shows a complete value, follow `auth_service` `verification_failed` (audit) instead |
| **A masking phrase was cut off, or a phrase plays twice** | `mask_spoken` lines for the turn (at most one `timer` and one `tool`) and `handoff_hold` (`wait_ms`). `immediate` is the seeded default because ConversationRelay finishes speaking the words it already has after `end`. A cut-off on a hand-off means ConversationRelay cut speech at `end`: set `end_mode=wait` with `scripts/masking.py set --masking end_mode=wait`. With `wait`, a cut-off can also mean `chars_per_s` is too high for the voice: lower it (console or `scripts/masking.py`) |
| **No tool phrase where one was expected** | `mask_skipped` for the turn gives the `reason` (table above). `answer_started` is the phrase working as intended (the answer was faster than the delay). `tool_off`: the tool has no `mask`. `no_renderable_phrase`: read the `mask_phrase_unrenderable` ERROR and fix the catalog. Gateway tools (`faq_lookup`, `get_branch_hours`) have no `mask` in `policy.json`, so only the acknowledgement covers them |
| **A key press did nothing, or the hint played instead of the action** | "Keypad shortcuts in the AI session" below: `ai_keypad state=` at start-up, then the call's `keypad_action`, `keypad_ignored` and `keypad_action_failed` lines. A hint after several keys is the multi-digit rule (`keys` above 1), not a fault |
| **The caller went quiet and was transferred or hung up on** | `prompt_repeat` (the bridge repeated the last message), then `silence_action`. `escalate`: `escalation_requested` with `reason` `no_input` and an `escalation` line from the front door. `hangup`: the front door's `no_input_hangup`. The segment log has the `prompt_repeat` events and the `handoff`. Count the time from the last message: `timeout_ms` per repeat (`waited_ms` for the greeting) from the end of the agent's speech, which is the `agentSpeaking` `off` line (`relay_speaker_event`) when the session's `silence_timing` is `speaker_events`, else the estimated speaking time. A caller who was speaking shows `clientSpeaking` `on` lines; with speaker events nothing fires between `on` and `off`. A caller cut off mid-sentence with `silence_timing` `estimate`: the speaker events didn't arrive or weren't recognised (look for `relay_speaker_event` with `recognised` false). A caller who was speaking but not heard shows no `silence_reset`: check the ConversationRelay transcription in the Twilio call log. To change the timing or the action: the `silence_*` masking fields |
| **An outcome or the read-out after it was cut off** | `relay_interrupt` lines: `playing` and `ms_played`. Payment and transfer results, cancellations and failures are spoken as plain `<Say>` before ConversationRelay connects and can't be interrupted, so a `relay_interrupt` after one has `playing` `greeting`: the caller cut in on the follow-up ("Anything else?"). A balance read-out is still interruptible by design |
| **The AI said something wrong or ignored a request** | Read the session's history in AgentCore Memory ("From a CallSid", step 3). Words the caller spoke over a scripted read-out arrive as an ordinary caller message right after the read-out's tool result (which is recorded, never answered: the model answers only the words), not inside the result. If they are missing from the history, look for `caller_words_dropped`: the agent started a hand-off on the result instead |

## Escalation without Flex: the briefing

Without Flex (the default), a request for a person ends in a read-out, not a handoff: the caller hears a lead-in, one generated sentence and the goodbye, and the call ends. Nothing is queued or transferred. It demonstrates a handoff and is not one (production guide, "Escalation without Flex"). The one model call uses the resolver model; its prompt is the catalog entry `agent.briefing.system`.

- **The caller hears silence while it is made**, up to the budget: `OWLBANK_BRIEFING_BUDGET_MS` on the front door, fixed at 4000 ms by the CDK app (there is no context key; a hand edit of the Lambda environment is reverted by the next Compute deploy). The agent's summary check and the transcript read share at most 1.2 s of it (and at most 40% of it); the model gets the rest.
- **The briefing model's system prompt** is the catalog entry `agent.briefing.system`. Any seed writes it, but a text edited in the console is kept (and listed) unless you pass `--overwrite-prompts`.
- **The fixed line is read** (the same facts from catalog labels, no model text) when the model fails, times out, or its output fails the structural checks (no digit, at most 30 words, no markup, one line), and always when the slot-mode ceiling is `deterministic` (no model call is made). A first escalation after an idle period can read the fixed line because the model call is cold. A missing input (agent summary, transcript) is left out. With less than 300 ms of the budget left, no model call is made.
- **Timing:** read `latency_ms` and `transcript_ms` on `briefing_generated` for your own numbers.

| Event | Fields | Meaning |
|---|---|---|
| `briefing_generated` | INFO: `source` (`model` or `fallback`), `words`, `latency_ms`, `transcript_ms`, `has_summary`, `has_transcript` | The read-out was built. Never the text. `fallback` with `has_summary` false is the normal degraded case |
| `briefing_timeout`, `briefing_error` | WARNING | The model call missed the budget or failed; the fixed line was read |
| `briefing_skipped` | WARNING: `budget_ms`, `decision` `fixed_line` | Too little of the budget was left (or it is set very low): no model call |
| `briefing_unparseable` | WARNING | The generated clause would not parse as TwiML; the fixed line was read |
| `briefing_summary_failed`, `briefing_transcript_failed` | WARNING: `decision` `without_summary` or `without_transcript` | One input was dropped; the briefing went on without it |

To judge the budget, filter `briefing_generated` and read the share with `source` `fallback` and the `latency_ms` spread. The summary and the scrubbed transcript tail reach the Bedrock invocation log when that log is on (see "The Bedrock logging alarm").

**Queue wait.** The queue-wait metric (`handoff.queue`) needs TaskRouter events and is off without Flex.

**Expected warnings** at the first deploy and in a quiet account: a front-door `twilio_connection_warm_failed` WARNING at a cold start is harmless unless its status is 401 or 403, which means the API key in Secrets Manager is wrong (re-run `check_secrets.py`); another cause, such as a network blip at the cold start, can be ignored (the first early finish opens the connection itself, about 1 s slower).

## Secrets: checking and rotating

The secrets (`owlbank/twilio/api-key`, `owlbank/twilio/auth-token`, `owlbank/twilio/basic-auth`, and, with the default `contextApiRestrictedKey=on`, the Context API's restricted key `context_api_key`) live only in Secrets Manager. CDK creates them, Secrets Manager generates the Basic password, and the operator or the bank's pipeline writes the Twilio ones (the restricted key is written by `twilio_setup.py --phase pre`) (steps in [infra/README.md](../infra/README.md#twilio-secrets-the-manual-step)). Components only read them.

- **Check them** (read-only, no value printed; exit 0 only if every secret is set, well formed and under the customer-managed key; a restricted key still holding its placeholder shows `PENDING` and passes unless you add `--strict`): `uv run python scripts/check_secrets.py --cdk-outputs infra/cdk-outputs.json`. A runtime component that finds a placeholder or a missing field fails closed and logs the secret and the field, never the value.
- **The Basic credentials are visible.** They sit in the number's webhook URLs, so anyone who can read the number's settings (or a screenshot or screen share of them) can read them. They are a shared key, not the authentication boundary: the Twilio signature is. Rotate after any exposure.
- **Rotate the Basic password (no downtime):**
  1. Dry run: `uv run python scripts/populate_secrets.py --cdk-outputs infra/cdk-outputs.json --new-basic-password`. Then run it again with `--apply`. It writes a new generated password and keeps the old one as the previous password for `--overlap-hours` (default 8: 2 to apply, 4 for the longest call, 2 margin), counted from the write. It prints the deadline for step 3. Neither password is printed.
  2. Wait 60 s, the secret's cache lifetime. A callback that reaches an instance whose cache predates the write can get a 401 for at most that long.
  3. Before the deadline, put the new password into the webhook URLs: `uv run python scripts/twilio_setup.py --phase post --account-sid AC... --phone-number +1... --cdk-outputs infra/cdk-outputs.json --apply` (the number, the Orchestrator callbacks and, with Flex, the TaskRouter callbacks). With Flex, add `--with-flex --taskrouter-events` to this command: otherwise TaskRouter's event callback URL keeps the old password and the events are refused after the overlap. Until then Twilio sends the old password, and TwiML already returned to a call in progress carries it; the front door accepts it during the overlap and checks the Twilio signature against the URL as Twilio signed it.
  4. A request that passed with the previous password logs `twilio_basic_previous_accepted` (call-keyed, no password): when it stops appearing, nothing uses the old one. To end the overlap early: `populate_secrets.py --cdk-outputs infra/cdk-outputs.json --clear-previous-basic-password --apply` (the previous password stays accepted up to 60 s more). `check_secrets.py` reports an active previous password and its deadline.
  - **Don't rotate twice inside one overlap.** The script warns but doesn't stop you: the second rotation drops the oldest password (at most two are accepted), so calls started before the first rotation can then lose callbacks.
  - After the overlap, the previous password is refused with the same 401 or 403 as any wrong one. The bridge doesn't read this secret.
- **A Standard API key has a limit.** It can't read the Accounts resource or create keys (the Accounts read fails with error 70004), so `twilio_setup.py` reads the Auth Token secret for those two requests and uses the key for everything else. The Auth Token secret must therefore be written before any `twilio_setup.py` run, `pre` or `post`.
- **Rotate the Twilio API key and Auth Token** as in the [production guide](production-guide.md#twilio-secrets-writers-rotation-and-the-subaccount). The bridge reads both only when it starts, so the new values reach it differently on each hosting:
  - **Fargate:** force a new deployment of the bridge service (find the names with `aws ecs list-clusters`, then `aws ecs list-services --cluster <cluster>`; the `OwlBankBridge-Cluster...` and `OwlBankBridge-Service...` ones): `aws ecs update-service --cluster <cluster> --service <service> --force-new-deployment --profile <profile>`, then `aws ecs wait services-stable --cluster <cluster> --services <service> --profile <profile>`.
  - **`bridgeHosting=agentcore`:** there is no service to redeploy. Each call's Runtime session reads both secrets when it starts, so new calls get the new values and calls in progress keep the old ones. Wait until the calls started before the write have ended, make a test call, and only then revoke the old key or token (at least 300 s after the write, for the Lambdas' cache).
- **Who wrote a secret:** CloudTrail, event `PutSecretValue` on the secret's ARN, shows the caller.

## Raising the level for an investigation

`logLevel` is a CDK context value (`DEBUG`, `INFO`, `WARN` or `ERROR`; anything else falls back to INFO). It sets the Lambda JSON log levels and `OWLBANK_LOG_LEVEL` on every Lambda and the bridge.

```
cd infra
npx aws-cdk@2 deploy --exclusively OwlBankCompute OwlBankBridge -c logLevel=DEBUG
npx aws-cdk@2 deploy --exclusively OwlBankCompute OwlBankBridge -c logLevel=INFO
```

The first command raises the level, the second puts it back (or omit the flag). Add your usual flags to both (Conventions).

Pass the same other `-c` values your deployment uses (or set `logLevel` in `cdk.json` context). The Lambda change is a configuration update; the bridge's change replaces its tasks (a rolling deploy with draining, see the production guide). At DEBUG you also get every `dependency_call` that succeeded (each DynamoDB read and write) and each swallowed-error line marked `ignored_because_*`. It is a lot more volume: put it back when you are done. DEBUG never turns on `botocore` or `urllib3` logging, whatever the level.

## Tuning latency masking

Two ways to change the masking values in the live policy table, with no seed, build or deploy: the admin console (Tools and Masking views, below), or `scripts/masking.py` from a shell with AWS access. Running bridges pick up a change within 60 s, on the next turn. A reseed keeps these values ("Prompts and the seed", "What a reseed keeps and overwrites"), so tuning isn't lost when demo data is reset.

```
# what is set: policy-wide values and every tool's mask, each marked as the table's or the default
uv run python scripts/masking.py show --stage dev --cdk-outputs infra/cdk-outputs.json [--tool get_balance]

# one tool: phrase delay, hand-off hold, phrase key (dry run: prints old -> new; add --apply to write)
uv run python scripts/masking.py set --stage dev --cdk-outputs infra/cdk-outputs.json --tool get_balance --after-ms 900
#   --after-ms default / --handoff-wait-ms default   use the policy-wide value again
#   --phrase-key mask.<key>                          must exist in the catalog in every language
#   --phrase-key off                                 no phrase for this tool

# policy-wide values (FIELD=default removes the field, so the bridge uses its built-in default)
uv run python scripts/masking.py set --stage dev --cdk-outputs infra/cdk-outputs.json --masking tool_after_ms=500 --masking after_ms=2500
```

Fields: `after_ms`, `tool_after_ms`, `handoff_wait_ms`, `silence_timeout_ms` (integers 0–10,000 ms), `silence_max_repeats` (0–5, default 2), `chars_per_s` (8–30), `end_mode` (`wait` or `immediate`), `silence_action` (`escalate` or `hangup`). Each change is checked with the bridge's own rules before anything is written; an invalid value is refused with the field and the reason (exit 2). Only the named fields change. The write is conditional on the item not having changed since the script read it (on its `rev` when it has one, else on its whole content); if it did, nothing is written (exit 3): run the command again. The script prints values and table names only, never caller data.

To judge a change, read the bridge's `mask_spoken` and `mask_skipped` lines for the tool (table above). Many `answer_started` skips for a tool mean its answer usually comes before the delay ends: the delay is doing its job. A phrase that plays and then makes a ready answer wait behind it means the delay is too short for that tool: raise its `after_ms` so the phrase plays only when the tool is slow, or turn the phrase off. A caller left in silence before a slow tool's answer means the opposite: lower it. A Logs Insights query over `/owlbank/ecs/tac-bridge`:

```
fields @timestamp, event, tool, reason, after_ms, source, ms_since_turn_start
| filter event in ["mask_spoken", "mask_skipped"]
| stats count() by tool, event, reason, source
```

A phrase's wording or a tool's pool is catalog text: the console's Prompts view, or `data/prompt_catalog.json` and a reseed.

## Prompts and the seed

**What a reseed keeps and overwrites.**

| Stored setting | A reseed | Overwrite with |
|---|---|---|
| Slot items (the Slots view) | Keeps the table's values | `--overwrite-slots` |
| Masking values (`scripts/masking.py`, the Masking view) | Keeps the table's values | `--overwrite-masking` |
| The `redaction` item | Keeps it and prints each field that differs from `data/policy.json` | `--overwrite-redaction` (logged at WARNING, `outcome` `overwrite`; check the dry run first) |
| Catalog prompts | Keeps those edited in the console; see below | `--overwrite-prompts` |
| The rest of each tool's item (levels, confirmation modes) | Rewrites it from `data/`, so copy console edits of these into `data/` before the next reseed | n/a |

Each catalog item the seed writes stores a hash of the text it wrote (`seed_hash`; the console keeps it on an edit or restore and never shows it). A reseed compares the table's text with that hash:

- **Not edited in the console** (text matches the hash): it takes the file's text when `data/prompt_catalog.json` changed it, so a plain reseed picks up text changes in the file.
- **Edited in the console** (text no longer matches the hash) and different from the file: kept, and printed as key and language (never the text), in a dry run and an apply.
- **No hash** and text different from the file: kept and listed as "no hash": the seed can't tell a console edit from seeded text. If the text already equals the file, it just gets its hash, with no flag and no `rev` change.
- **A key or language new in the file** is always added. Keys added in the console (`mask.*`, `arg.*`, `option.*`) stay in the catalog unless `--overwrite-prompts`.

`--overwrite-prompts` writes the file's text (and its hash) over all kept prompts, those with no hash included, and prints what it replaced (logged at WARNING, `step` `prompts`, `outcome` `overwrite`). Run the dry run first and copy any console edit you want to keep into the file.

**Turning redaction back on.** `scripts/seed.py --overwrite-redaction` writes the file's setting; use it to turn redaction back on from the file, after checking the dry run. Restore from the console's Audit view is blocked for redaction entries: make the change again in the Redaction view.

## Using the admin console

Open the `ConsoleUrl` output of `OwlBankAdmin` and sign in (Cognito managed login; TOTP MFA at `stage=prod`). Deploy and first sign-in: `infra/README.md`, "The admin console".

- **Tune a tool.** Tools, the tool, then `after_ms` or `handoff_wait_ms` (or "use the default"), the phrase key, or masking off. Save. A level or confirmation-mode change asks for confirmation and is audited as a security change.
- **Policy-wide values**, the silence fields included: Masking.
- **Phrases and words:** Prompts (prefix `mask.` or `arg.`). The placeholder picker lists the tool's enumerated arguments; the preview shows every phrase for every value and language, and each acknowledgement followed by each tool phrase. A phrase the bridge couldn't render isn't saved.
- **It takes effect** within 60 s, on the next turn of each call (the bridge's policy cache). Make a test call a minute after saving.
- **Undo:** Audit, find the change, Restore. It writes the old value back through the same checks. Restore is blocked for redaction entries ("Prompts and the seed").
- **A 409 on save** (the item changed since you loaded it): someone else saved first, or a reseed ran. Reload and make the change again.
- **Who changed what:** Audit in the console, or the audit table (`owlbank-<stage>-admin-audit`, `pk` = the item, `sk` = the time). Entries expire after 90 days (`dev`, `test`) or 400 days (`prod`).
- **Users** (admins only): Users view. Create (email and group: `admin`, `admin-editor` or `admin-viewer`; Cognito emails a temporary password valid 3 days), change group, disable or enable, resend the invitation (only before the first sign-in), delete. The last enabled admin can't be removed, and an admin can't delete themselves. If no admin is left who can sign in, redeploy `OwlBankAdmin` with `-c adminEmail=<email>` (a new address creates a new admin); there is no break-glass script.
- **Errors:** 401 means the session expired (sign in again); 403 means your group can't do that; 503 `not_seeded` means the policy and catalog tables are empty (run the seed).
- **The console's own logs:** `/owlbank/lambda/admin_api` (API access logs: `/owlbank/admin/apigw-access-logs`): `admin_audit_write_failed` (ERROR) is a user change that stood but has no audit entry. The API access log shows when it was made; note it in your own change record.

## Keypad shortcuts in the AI session (demo and troubleshooting)

The demo script below is for `dev` and `test`; the log lines and the checks after the table apply to any stage.

Keys during the AI conversation (production guide, "Keypad shortcuts in the AI session"): 1 yes, 2 no, 0 a human agent, 9 repeat, any other key a hint. None of these lines carries a digit: only the action and a key count.

**In the demo call script.** On any AI-session call (the keys aren't announced to callers, so say them to the audience):

1. At the HELOC welcome ("Would you like help with our home equity line of credit?"), press **1**. This step needs the web-activity setup ("Demoing the web-activity offer"); without it, skip to step 2. The agent takes it as "yes" once the caller has stopped pressing keys for the digit-collection window (1.5 s).
2. Press **9** after any agent message: the same message plays again, with no pause for the model.
3. Enter **five digits** (for example 1 2 3 4 5): the only result is the hint, "Please say what you need, or press 0 for an agent." A number that starts with 1 is not taken as yes.
4. Press **2** at a yes or no question: the agent takes it as "no".
5. Press **0** (with Flex; otherwise the call ends with the briefing): the call goes to the OwlBank Support queue (reason `caller_requested_human`) and the Flex screen pop shows "The caller pressed 0 to speak with an agent."

**Troubleshooting: the log lines.**

| Log line (bridge) | Fields | What it means |
|---|---|---|
| `ai_keypad` | `state` (`on` or `off`) | At start-up: whether this bridge handles keys. A bad `OWLBANK_AI_KEYPAD` stops the start-up instead |
| `keypad_action` | `action` (`yes`, `no`, `agent`, `repeat`, `hint`), `keys` (a count) | A key entry was acted on. `hint` with `keys` above 1 is a multi-digit entry. A `keypad_action` with no effect after it: look for `keypad_action_failed` |
| `keypad_ignored` | `cause` (`no_session`, `session_not_ready`, `handoff_pending`, `prompt`, `session_end`), `decision` `dropped` | Keys dropped on purpose, nothing spoken: no session (or it ended or restarted), TAC's session never came within `session_wait_s` of the socket opening (keys pressed during the welcome are held until it does, normally a few seconds), a hand-off pending (or just sent), the caller spoke before the key acted (one turn, the spoken one), or the socket closed with a key not yet acted on. `handoff_pending` may be followed by `handoff_end_resent` |
| `keypad_action_skipped` | `action` `repeat`, `cause` `nothing_spoken` | 9 pressed before the session had said anything. Normal |
| `keypad_action_failed` | `action`, `error_class`, `decision` `call_unchanged` | The turn, repeat or hint failed and the call went on as it was. A failed 0 is ERROR and falls back to the error escalation (`escalation` lines follow) |
| `keypad_phrase_missing` | `key`, `language`, `decision` | The catalog has no `keypad.yes`, `keypad.no`, `keypad.hint` or `keypad.agent_summary` in this call's language. Yes and no fall back to the English word; the hint is skipped; the 0 summary is a fixed English text. Reseed the catalog |
| `handoff_end_resent` | `cause` `keypad` or `interrupt` | A key or an interrupt cancelled the turn that was handing off before TAC sent `end`, so the bridge sent it, once. Rare; the call still reaches the front door's hand-off |

**A key does nothing.** Check in this order:

1. **The setting on either stack.** `ai_keypad state=` in the bridge's start-up line (Logs Insights on the bridge group, filter `ai_keypad`): `off` means the bridge drops keys. For the front door, check `OWLBANK_AI_KEYPAD` on the front-door function (absent or `on` adds `dtmfDetection="true"` to the AI session's `<ConversationRelay>`; `off` leaves the TwiML unchanged). On the bridge (task definition, or the Runtime's environment), absent or empty also means on; `off` turns it off.
2. **A mismatched deploy.** `OwlBankCompute` and `OwlBankBridge` deployed from different settings or at different times: the front door on and the bridge old gives keys that interrupt the agent and do nothing. Deploy both with the same `-c aiKeypadShortcuts=` (and the usual flags) and re-check the start-up line. A bridge task that was already running keeps its old setting until it is replaced (force a new deployment).
3. **A hand-off in progress.** `keypad_ignored` with `cause` `handoff_pending`: the agent had just called `escalate` or a payment tool, or the session was ending. By design.
4. **No `keypad_action` and no `keypad_ignored` at all** with the setting on at both ends: the key frames aren't reaching the bridge. Look at the bridge's `relay_interrupt` lines for the call (a key pressed during the agent's speech shows as a `relay_interrupt` and, one digit-collection window (1.5 s) later, its `keypad_action`), and check that the caller's phone sends tones (some VoIP apps don't).

## The agent loop (Strands)

The bridge runs the agent loop itself with Strands and calls Claude on Bedrock directly; it is the only agent loop (the managed AgentCore harness is not used, see the production guide's "Why not the AgentCore harness?"). Its model settings are CDK context on `OwlBankBridge`: the model list `modelIds` and `modelParams` (per entry: `thinking` `adaptive`, `off` or `default`, `max_tokens`, `text_release`; see "Model failover" below) and `promptCaching` (every entry). The bridge logs `agent_settings` (`thinking`, `prompt_caching`, `models`) and `models_resolved` at start-up.

**History.** The call's history is in AgentCore Memory (`owlbank_voice_stm-*`), `actorId` `owlbank-voice-strands`, `sessionId` the call SID: Strands session events, with the messages (agent `owlbank_voice`) and the agent's state, including a pending inline tool call, written once per invocation, at its end.

Read a call's history in AgentCore Memory (find the memory id with `aws bedrock-agentcore-control list-memories`; it starts with `owlbank_voice_stm`): `aws bedrock-agentcore list-events --memory-id <owlbank_voice_stm-...> --session-id <CallSid> --actor-id owlbank-voice-strands --include-payloads` (Administrator or a role with `bedrock-agentcore:ListEvents`). It holds the caller's words; treat it as customer data. Model timings are not in it: use the `strands_model_call` lines instead (one per model call, with tokens and cache counts), next to the bridge's `turn` line.

**How the bridge picks the history.** On each session it reads `agent_owner` from call state: no owner and no sign of an earlier AI session gives an empty agent (`agent_history_new`, reason `new_session`); this task gives the cached agent (no line); any other task, or an earlier session with no owner, gives a rebuild from Memory (`agent_history_reload`). A rebuild that fails is a retryable `history_unavailable` error, never an empty history. A tool result for a tool use the agent never made rebuilds once (`unknown_tool_use_recovery`) before the turn fails with `unknown_tool_use`.

**Changing the model settings.** Deploy `OwlBankBridge` with the new value, for example `-c 'modelParams={"<model id>":{"thinking":"off","text_release":"message"}}'`, plus the usual flags (Conventions). On Fargate the ECS deployment is rolling, so calls in progress finish on the old tasks.

**Streaming mode (`text_release` in `modelParams`).** `stream` (default) sends the agent's text as the model writes it; `message` holds each model message until it ends and drops text that shares a message with a tool call. The bridge logs its mode at start-up (`agent_text_release`). To switch, change `text_release` to `message` or `stream` in the entry's `modelParams`, as above. Turning reasoning off (`thinking` `off`) needs `text_release` `message` for the same entry: synth fails otherwise, naming the entry. Any other value fails synth. Count the rare late tool call with `ai_text_spoken_before_tool`: a few mean the model narrated before a tool; many mean check the entry's `thinking` and the prompt, or switch to `message`. A hand-off that follows such text logs `handoff_hold` with `source` `released_text`.

## Model failover: which model answered, forcing the preferred model, the latency drill

The bridge keeps an ordered list of models (`modelIds`; the default list is in the production guide, "Model failover") and moves a turn to the next when the first is slow or failing; the production guide's "Model failover" explains the rules. Nothing here prints caller text: every line carries model IDs, numbers and codes only.

**Which model a call used.**
- The bridge's `strands_model_call` line (one per model call, with the call's `call_sid`) has `model_id`, `outcome` (`ok` or `error`), `model_ttfb_s`, the token counts and, for a call abandoned before any output, `cause` (`deadline` or the error class). The turn's `ai_turn` event in the segment log (`segments` table) carries `model`. With `callObservability=on` the time-to-first-token and latency metrics have a `model_id` attribute.
- Logs Insights, in the bridge group (`/owlbank/ecs/tac-bridge`, or the Runtime group): `filter event = "strands_model_call" and call_sid = "<CallSid>" | fields ts, model_id, outcome, cause, model_ttfb_s | sort ts`. A `model_failover` line for that call (`turn`, `from_model`, `to_model`) shows each switch.
- To see the current state: `filter event = "model_failover" or event = "model_degraded" or event = "model_unavailable" or event = "model_recovered" | sort ts desc` over the bridge group and `/owlbank/lambda/model_poller`. The metric `ActiveRank` above 0 (namespace `OwlBank/<stage>/Model`) says a task is off the preferred model.

**What each `model_*` line means.**

| Event | Level | Where | Meaning |
|---|---|---|---|
| `models_resolved` | INFO (WARNING with a `global.` entry) | bridge, at start-up | The list the task runs (`models`, `preferred`, `entries`), `failover` and `fleet` on or off, and `global_entries`: any `global.` profile, which may be processed outside the geography |
| `model_fleet` | INFO | bridge, at start-up | `state=on` or `off` (`reason` `single_entry` or `no_table`): whether the task reports to and reads the fleet table |
| `model_failover` | INFO (`cause=degraded`), WARNING (`deadline`, `error`) | bridge | A turn moved from `from_model` to `to_model`. `degraded`: the turn started on the next model because the verdict had marked the first. `deadline`: the first model gave no output within its first-token deadline, and the turn was re-issued. `error`: it failed before any output (`error_code`). `model_calls` counts the turn's model calls so far |
| `model_degraded` | WARNING | poller (fleet verdict) and bridge (`scope=process`) | An entry tripped: `p50_ms`, `p90_ms`, the error rate and `samples` (or `miss_run`) that did it |
| `model_recovered` | INFO | poller | A degraded entry passed the recovery rule (`pings` it was judged on) and resumes its rank |
| `model_unavailable` | ERROR | poller and bridge | One unambiguous error (invalid identifier, not found, access denied): `error_code`, `preferred`. Fires the alarms. The entry is out of rotation until a recheck succeeds |
| `model_available` | INFO (WARNING `cause=unconfirmed`) | poller and bridge (`scope=process`) | An unavailable entry answered the recheck, or this process's local mark was cleared (`cause` `fleet`, or `unconfirmed`: no fleet view confirmed the mark within `unavailable_recheck_s`, so check the poller) |
| `model_recheck_failed` | INFO | poller | The 1-token recheck of an unavailable entry failed again (`cause`, `error_code`); it stays out |
| `model_pings` | INFO | poller | One cycle's recovery pings to a degraded entry: `sent`, `ok`, `missed`, `errors`, `p90_ms`. None are sent while every entry is healthy |
| `model_state_written` | INFO | poller | The verdict record was written (`changed`, `report_every`: the sampling rate, 1 in k) |
| `model_request_error` | WARNING | bridge | The model rejected the request (`error_code`); never changes a model's state. A request bug, not an outage |
| `model_state_realigned` | WARNING | poller | The stored verdicts were matched to a changed list (a deploy changed `modelIds`); normal once after such a deploy |
| `model_state_conflict` | WARNING | poller | Another invocation wrote a newer verdict first; this one kept it |
| `model_state_unreadable`, `model_samples_unreadable`, `model_state_write_failed` | WARNING | poller | The table could not be read or written; the last verdict stays and, if the heartbeat stops for 3 minutes, `ModelStateStaleAlarm` fires and bridges use their own picture |
| `model_state_malformed`, `model_samples_malformed` | WARNING | poller | An item could not be parsed (and was skipped, or the state restarted healthy) |
| `model_poller_cycle_skipped` | WARNING | poller | A cycle was skipped for lack of time in the invocation (`reason=time`) |
| `model_failover_settings_unreadable` | WARNING | poller | The `model_failover` policy section could not be read; the poller keeps the last settings (a bridge with an invalid section uses the defaults and logs an ERROR naming the field) |

**Alarms.** `ModelUnavailable` and `PreferredModelUnavailable`: find `model_unavailable` (the model and `error_code`), run preflight (see Conventions; it tests each entry and proposes a replacement of the same series), fix the entry or the access, deploy `OwlBankBridge` and `OwlBankCompute` with the new `modelIds` (then reseed at once: "Validating a prompt", "Required step"), or wait for the 15-minute recheck if the cause was fixed outside (a grant, an agreement). `ModelActiveRankAlarm`: a task has been off the preferred model for 15 minutes: look for `model_degraded` and `model_pings` (are the pings recovering it?) and the Bedrock status. `ModelAllEntriesDownAlarm`: no usable entry; calls still try the last entry, which has no deadline. `ModelStateStaleAlarm`: the poller has not written for 3 minutes: its Errors and logs; bridges are on their own picture meanwhile.

**Force the preferred model, or turn failover off.** There is no per-call switch. To make every turn use the preferred model with no deadline, no switching, no poller and no table access, deploy with `-c modelFailover=off` on `OwlBankData`, `OwlBankCompute` and `OwlBankBridge`, with the usual flags (`-c twilioPhoneNumber=...`, and every other non-default value: an omitted key goes back to its default). Only the first `modelIds` entry then reaches IAM and `OWLBANK_MODELS`. The `modelstats` table stays, unused. To go back, deploy again with `modelFailover=on`. To keep failover but stop a flapping or too-eager threshold, raise `trip_ttft_s` and `deadline_s` in the `model_failover` section of `data/policy.json` and reseed (no deploy; the bridge and poller read it within about a minute); keep `recover_ttft_s` below `trip_ttft_s`. A degraded verdict clears itself through the recovery pings; there is no command to clear it.

**Drill: inject first-token latency (non-prod).** To see failover work on a real call, delay the preferred model's first token: deploy `OwlBankBridge` and `OwlBankCompute` with `-c failoverTest=6` (seconds, above 0, at most 30; needs `modelFailover=on` and two or more entries; refused at `stage=prod`). It sets `OWLBANK_FAILOVER_TEST` on the bridge and the poller, so the preferred entry's first output is held back by that many seconds, in calls and in pings. Choose a delay above the entry's first-token deadline (`deadline_s` in the `model_failover` section of `data/policy.json`). Every caller's first model call on the stage is delayed while the setting is on, so tell anyone else using the stage, and use synthetic callers only. Then:
1. Call and speak. The first model call logs `strands_model_call` with `cause=deadline`, followed by `model_failover` (`cause=deadline`) and a call on the second model; the caller hears the answer a little later, no apology.
2. Keep calling for a minute or two: the fleet verdict flips (`model_degraded` from the poller), turns start on the second model (`model_failover` with `cause=degraded`), `ActiveRank` is 1, and `model_pings` shows 3 pings per cycle to the delayed entry, none to callers.
3. Remove the setting: deploy again without `failoverTest`. Pings pass, and after at least 10 samples over 2 minutes and the dwell time (3 minutes) `model_recovered` follows and turns return to the preferred model.
Do not leave `failoverTest` set after the drill.

## Validating a prompt (the per-model prompt suite)

Each model in the failover list has its own system prompt, and a model may be deployed only with a prompt that passed the suite (production guide, "Each model has its own system prompt"). This section is how to run the suite, read a failure and handle the checks that gate a deploy and a reseed.

**Run the suite.** From the repository root, with a profile that may invoke the models on Bedrock (for example a role with `bedrock:InvokeModel`). It uses fixture data only; no caller data is involved.

```
AWS_PROFILE=<profile with Bedrock invoke> uv run python scripts/prompt_suite.py                       # every model, every case, both languages, 5 runs
AWS_PROFILE=<profile> uv run python scripts/prompt_suite.py --model haiku-4-5                         # one model (repeat --model for more)
AWS_PROFILE=<profile> uv run python scripts/prompt_suite.py --model haiku-4-5 --case <case-id> --lang en-US --no-write
AWS_PROFILE=<profile> uv run python scripts/prompt_suite.py --models "$OWLBANK_MODELS"                # the list you deploy (the `OWLBANK_MODELS` value in the bridge task definition), with its own parameters
```

- `--model` takes the key (`haiku-5-5`, `haiku-4-5`, `sonnet-5-5`), `--case` a case ID, `--lang` a language; each repeats. `--runs`, `--concurrency` and `--attempts` change the runs per case, the runs in flight and the tries on a throttle.
- Without `--models` it uses the default list with each model's default parameters. If you deploy with other parameters (`promptCaching=off`, `modelParams`), pass the deployed list with `--models`, or the record will not match what preflight compares.
- It writes `data/prompt_validation.json` records only for a model and language that passed every run of every case. A `--case` subset, fewer `--runs` than the case file's 5, and `--no-write` write nothing (the last line says why). `--model` and `--lang` write records for what they ran.
- A full run of the default list is about 420 runs and about $1.70 before cache reads, at us-east-1 on-demand input prices (production guide). Use `--case` and `--no-write` while iterating on a text, then one full run.
- Exit code: 0 every model and language passed, 1 something failed, 2 the suite stopped on an error (no record).

**Read a failure.** After the table of passes, each failed run is one line:

```
FAIL <case-id> en-US haiku-4-5: speaks | silent=True stop=end_turn tools=[list_recent_transactions({...})] text=''
```

That is the case, the language, the model, the failed checks (`silent`, `speaks`, `tool`, `no_tool`, `no_text_before_tool`, `no_markdown`, `no_markup`), then what the model did: whether it wrote `<silent/>`, the stop reason, the tools it called with their arguments, and the text the caller would have heard. Read the pattern across the five runs before editing the prompt: 5 of 5 is a missing or contradicting rule, 1 of 5 is usually a wording that leaves the model a choice. A `stop=max_tokens` is a failed run, not an error: in a call the bridge would escalate, so the answer (often reasoning written as text) is too long for the entry's `max_tokens`.

**Throttle or stopping error.** A throttle or a server error (5xx) is retried with backoff (`--attempts`, default 6) and never counts as a failed run. If the run still fails after the tries, or fails for another reason (access denied, a model not available, a request the model rejects), the suite prints `STOPPED: <model> <case> <language>: <error code> (after retries | not retried, attempt n); no record written` on stderr and exits 2. Fix the access or the model ID, or lower `--concurrency` if throttling persists, and run again.

**Preflight's prompt lines.** From `infra`, `uv run python scripts/preflight.py --profile <profile> --stage dev` prints one line per model in the list to deploy and per catalog language, for example `Model prompt us.anthropic.claude-haiku-4-5 (en-US)`. The prompt check reads the repository only (no AWS call; preflight's other checks do call AWS) and PASSes when a record matches the prompt, the case file, full runs and the entry's parameters. A FAIL names the reason:

| Message | What to do |
|---|---|
| no validation record for `<model>` (`<lang>`) | Run the suite for that model; commit the record with the prompt |
| the record's prompt hash is stale | The prompt or a shared block changed after the run: run the suite again |
| the record's cases are stale | `data/prompt_suite.json` changed: run the suite again for every model |
| the record is not a full run | A record from fewer runs: run the suite without `--case` or `--runs` |
| model parameters changed since validation | The entry's reasoning, text release or max tokens differ from the validated run: validate with the parameters you deploy (`--models`), or deploy the validated ones |
| no `agent.system.<model>` text in the catalog | Add the model's prompt in both languages, then validate it |

Preflight must see the same parameters you deploy with: `--model-params '{"<profile id>": {...}}'` for the `modelParams` you pass; without it it compares the defaults. A record covers thinking, text release and max tokens; prompt caching (`promptCaching` or a `prompt_caching` override) is not part of it, because caching changes cost and latency, not the answer. `--no-resolve-models` skips the live model check and checks the prompts of the `--list-model` entries (or the default list); `--data-dir` points at another `data/`. The resolver never proposes a replacement model that has no validated prompt in every language.

**The seed refuses an unvalidated prompt.** `scripts/seed.py` stops (exit 2) when any `agent.system.*` text in the catalog has no current record, and names each model and language: `prompt_catalog.json: agent prompt(s) without a current validation record: ...`. Run the suite and commit the record. In dev only, `--allow-unvalidated-prompt` seeds the text anyway (the plan lists each unvalidated text, and a WARNING is logged); at any other stage the flag is refused. Use it to try a draft prompt on a dev call, and not for a prompt that goes further.

**Required step: reseed right after deploying `OwlBankBridge`.** When a deploy of `OwlBankBridge` changes the model list, or the catalog gains new `agent.system.<model>` keys, reseed straight after the deploy, before any call:

```
uv run python scripts/seed.py --stage <stage> --cdk-outputs <outputs file> --apply     # with your usual seed flags
```

A bridge whose preferred model's prompt key is not in the table cannot build the agent's prompt, and **every AI turn escalates to a human** until the reseed lands. There is no fallback to an older prompt. Order of a change that adds or swaps a model: preflight (prompts PASS), deploy `OwlBankBridge` (and `OwlBankCompute`) with the new `modelIds`, reseed at once, then one test call. A change to a prompt text alone needs no deploy: a plain reseed writes it, unless that prompt was edited in the console or has no seed hash, which the reseed keeps and lists; then reseed with `--overwrite-prompts` (see "Prompts and the seed").

## Call observability (traces, call timeline, alarms)

Applies when the stack was deployed with `-c callObservability=on` (off by default; production guide, "Call observability: traces, metrics and alarms"). With it off there are no traces: use the log lines above.

**Find a call's trace from its CallSid.** The CallSid is the session ID: every span carries `session.id` and `call_sid` with it.
1. **The call timeline.** CloudWatch, Logs Insights, saved query `OwlBank/Call timeline` (it searches `/owlbank/traces/bridge` and `aws/spans` together). Set the parameter `session_id` to the CallSid in double quotes and the time range to cover the call. Every span of the call comes back in start order: `name`, `service` (`owlbank-<stage>-frontdoor` or `-bridge`), `duration_ms`, `start_ms`, `end_ms`, `prev_end_ms`, `traceId`. Logs Insights can't compute the gap between spans, so use the script for that.
2. **`scripts/call_timeline.py`.** From the repository root, with an AWS profile for the stack's account:

   ```
   uv run python scripts/call_timeline.py --profile <profile> --call-sid CA0123...
   ```

   It runs the saved query, then prints one line per span: the offset from the call's first span, the span's name and service, its duration and the **gap before it**. A positive gap is time outside the blueprint's code, labelled `twilio-side` (speech recognition, playback, waiting for the caller); a zero or negative gap is a span nested in or overlapping an earlier one, labelled `nested`. The totals are the span count, the call's wall time, the time in the blueprint's code and the Twilio-side time. `--hours N` (default 24) or `--since <ISO time>` sets how far back to search, `--json` gives machine-readable output, and a bench session ID works too. It prints names, timings and IDs only, never span attributes.
3. **GenAI Observability.** CloudWatch, GenAI Observability, the agent `owlbank-<stage>-bridge`, Sessions: look the CallSid up. Each bridge turn is a trace, with its model calls and tool calls. With prompt caching on, the token counts shown here leave out cached tokens; the totals are on `ws_session_end` and in the `bedrock-agentcore` metrics.
4. **Transaction Search**, filtering on the `session.id` attribute, shows the same spans one at a time.

The spans hold identifiers, timings and counts, no words (with `traceContent=off`). To see why an answer was odd, read the AgentCore Memory history ("The agent loop (Strands)").

**If the call isn't there.**
- Is `callObservability` on for the stage that took the call? The bridge logs `telemetry_started` (with `trace_content` and `sample_rate`) at start-up; `telemetry_refused` or `telemetry_install_failed` means it runs untraced, and the line says why.
- `traceSampleRate` below 1.0: the call may be sampled out. The decision comes from the CallSid, so the front door and bridge agree and a call has all its spans or none. The logs and metrics are still there.
- The time range: the query only searches the range you set (the script, `--hours`).
- Front-door spans go over UDP to the Lambda X-Ray daemon and can be dropped without notice; the webhook's log lines are the record. A `trace_export_failed` line (at most one a minute) on the bridge means spans were dropped after a bounded queue, with the error class and the count.
- Transaction Search off: preflight (from `infra`) fails on it with the commands (account-wide; the production guide says what it changes).
- A change to the X-Ray grant or a new role can take about a minute to apply: wait before retrying.

**When an alarm fires.** The alarms go to the alarm topic; their names contain `CallObs`. An `...Anomaly` alarm is outside the learned band for its metric (it needs about two weeks of data to learn); the `...Ceiling` alarm is the fixed backstop (`ttftCeilingSeconds`, `turnWaitCeilingSeconds`, `silenceCeilingSeconds`, `tokensPerCallCeiling`). Find an affected call in the metric's time window and read its timeline. For the tool error-rate alarm, list the failures: Logs Insights on the bridge group, `filter event = "tool_result" and outcome = "error"`, then `stats count(*) by tool, error_code`. Only actual failures count: a business answer, a refused call, and a call cancelled by a hang-up or barge-in don't. Don't use Strands' own `strands.tool.error_count` for this: it counts every interrupt and hand-off as an error.

**Changing `traceContent` in a non-prod stage, and back.** `off` (default) | `redacted` | `full`; `full` is refused at `stage=prod` (synth fails). It is a Bridge setting: it changes the bridge's environment and the traces log group's retention, and the front door never carries text. From `infra`, with a container runtime available to CDK (`CDK_DOCKER=finch` if you use Finch):

```
npx aws-cdk@2 deploy OwlBankBridge --exclusively --profile <profile> \
  -c accountId=$(aws sts get-caller-identity --profile <profile> --query Account --output text) \
  -c twilioPhoneNumber=+1... \
  -c callObservability=on -c traceContent=full -c traceContentRetentionDays=1
```

Pass every other value you deploy with (an omitted key goes back to its default, including `callObservability`, which would remove the feature) and the usual `twilioPhoneNumber`. Synth prints a warning that the agent's text is exported. The ECS deployment is rolling (about 3 to 4 minutes). Check: the new task logs `telemetry_started` with `trace_content` = the new value, and no `telemetry_refused`. Make test calls with synthetic callers only, and with Bedrock model invocation logging off (preflight checks it).

**Going back.** Spans already exported keep their text until the group's retention expires, and switching to `off` sets the retention back to the stage's log retention (1 month in dev and test), which is longer than `traceContentRetentionDays` (default 7). So go back in steps:

1. Deploy again with the current `traceContent` value and `-c traceContentRetentionDays=1` (the command above).
2. Wait for the text to expire (about a day).
3. Deploy with `-c traceContent=off` (or leave `traceContent` out).
4. Check that a new call's spans hold no text, and that the TracesKey and `TraceReader` role are unchanged.

At `stage=prod`, `redacted` is your compliance owner's decision; the same steps apply.

**TaskRouter events and the status callback.** `call.end` needs the number's status callback, which `scripts/twilio_setup.py` sets with `callObservability=on`. The queue wait after a hand-off (`handoff.queue`) needs Flex (it is off without it) and the workspace's event callback: `uv run python scripts/twilio_setup.py --phase post --with-flex --taskrouter-events --account-sid AC... --phone-number +1... --cdk-outputs infra/cdk-outputs.json --apply`. The callback is **workspace-wide** (Flex or another integration may use it), so the step sets it only when it's empty or already this front door's, and otherwise reports the current host and path and stops. On a main (non-sub) account that only the blueprint uses, add `--dedicated-main-account`, as for the other phases.

## AgentCore hosting: investigating a call

Applies when the bridge runs as an AgentCore Runtime (`bridgeHosting=agentcore`; `infra/README.md`, "Two bridge hostings"). The front door, the auth service, the tools and the segment log are the same as on Fargate; what differs is where the bridge's lines are, how a call reaches the bridge, and how its session starts and ends.

**Which hosting served a call.** The front door's `relay_minted` line (one per connect) and the bridge's `startup_complete`, `ws_session_start` and `ws_session_end` lines carry `hosting` (`fargate` or `agentcore`). `OwlBankBridge`'s `BridgeHosting` output says `agentcore`; Fargate has no such output.

**Find the call's session.** The Runtime session ID **is** the call SID (`CA...`), and so is the AgentCore Memory session ID. The bridge lines are in the Runtime log group (the `RuntimeLogGroup` output; the name is `/aws/bedrock-agentcore/runtimes/<RuntimeId>-DEFAULT`, where the `RuntimeId` starts with `owlbank_<stage>_bridge`). Each microVM writes its own log stream, so the streams a call's lines come from tell you whether a re-entry reached the same microVM. A process's first lines are its start-up (`startup_complete`, `agent_settings`, and `ws_signature_check` with `state` `off`, `reason` `agentcore_hosting`: there is no Twilio signature on this hosting, the binding and nonce checks below replace it).

**Read its logs.** Run **Trace a call by call SID** with the Runtime log group selected as well as the `/owlbank/*` groups (on this hosting the saved queries include the Runtime group and the warmer's `/owlbank/lambda/runtime_warmer`). Or Logs Insights over those groups:

```
fields @timestamp, @logStream, component, event, outcome, reason, hosting, duration_ms, history
| filter call_sid = "CA0123456789abcdef0123456789abcdef"
| sort @timestamp asc
```

A normal call reads: front door `incoming` → warmer `runtime_prewarm` `ok` (while the caller is in the language menu and verification) → front door `relay_minted` (`hosting` `agentcore`) → bridge `ws_session_start`, `agent_session_start` (`history` `new`) and the turns → hand-offs and re-entries (a new `relay_minted` and `ws_session_start` each; `history` `cached` on the same microVM) → front door `runtime_session_stopped` (`outcome` `ok`) at the end.

**Events on this hosting** (none carries a URL, nonce, credential or event body):

| Event | From | Fields | How to read it |
|---|---|---|---|
| `relay_minted` | front door | `hosting` | A WebSocket URL was minted for a connect (pre-signed, valid at most 300 s) and `relay_minted_at` stored. The start of the connect-to-first-word measure in the latency benchmark (`scripts/bench/call_latency.py`) |
| `relay_presign_failed` | front door | ERROR: `error_class`, `hosting`, `decision` `apology_and_escalate` | The front door couldn't get the presign role's credentials or sign the URL; the caller heard the apology and went to a human. Check that the front door may assume `owlbank-<stage>-relay-presign` (STS errors in the line's `error_class`) |
| `runtime_prewarm` | warmer | `duration_ms`, `outcome` (`ok` or `in_progress`) | The warmer's `InvokeAgentRuntime` on the call's session returned: `ok` when the microVM answered, `in_progress` when AgentCore was still provisioning it (not a failure). On a session that attaches to one of AgentCore's spare microVMs it is short; a new microVM, as in a burst beyond the pool, takes the boot (production guide: p50 9.4 s with 10 sessions opened at once, beyond the pool) |
| `runtime_prewarm_failed` | warmer | WARNING: `error_class`, `code`, `decision` `continue_cold` | The pre-warm failed; the call continues and its first connect waits for the boot. A throttling or quota `code` fires the pre-warm alarm |
| `runtime_prewarm_rejected` | warmer | `reason` `bad_call_sid` | The event had no valid call SID; nothing was invoked |
| `runtime_prewarm_dispatch_failed` | front door | `error_class`, `error_code`, `stack` | The front door couldn't start the warmer (Event invoke); ignored, the call starts cold |
| `twilio_auth_failed` | bridge | `component` `bridge`, `reason` `binding` or `nonce` | A socket was closed before TAC saw it. `binding`: the Runtime session-ID header, the `setup` frame's `callSid` and the TwiML `customParameters.callSid` differ, or the process is already pinned to another session; nothing was written, so the call's nonce is not used up. `nonce`: `customParameters.n` was missing, wrong or already used. Both count in the bridge's auth-rejected alarm and in **Auth failures** |
| `agent_session_start` | bridge | `history` (`new`, `cached`, `reload`) | `cached` on a re-entry: the same microVM within the idle timeout. `reload` (with `agent_history_reload` `reason` `owner_changed`): a new microVM, history rebuilt from AgentCore Memory, as after a Fargate task change; expected after a return from a queue or a scripted step longer than `agentcoreIdleTimeoutSeconds` |
| `relay_reconnect` | front door | `reason` `stream_limit`, `stream_reconnects`, `hosting` | AgentCore closed the WebSocket at its 60-minute cap (about 60 min 57 s, close code 1008, which Twilio reports to `connect-action` only as `failed` + ErrorCode 64105). The front door reconnected silently with a fresh URL and nonce: no apology, not counted as a relay failure, at most 4 times per call.<br>A 64105 failure **before** 3,540 s (59 minutes) since `relay_minted_at` is an ordinary drop: the apology, one reconnect, then a human |
| `runtime_session_stopped` | front door | `outcome` (`ok`, `not_found`, `error:<class>`), `duration_ms`; on an error WARNING with `error_code`, `decision` `idle_timeout_stops_it` | `StopRuntimeSession` at the call's end status, exactly once per call (a claim item `<CallSid>#runtime-stop` in `calls` stops repeats). `not_found`: the session had already ended. An error: the session stays until the idle timeout |
| `orchestrator_event_dropped` | front door (`POST /voice/orchestrator`) | `reason` `no_relay`, `event_type`, `lag_ms` | An Orchestrator status callback, checked (WAF Basic challenge, Basic, signature) and acknowledged with 200, not forwarded: no event is needed during a call on this hosting. No `call_sid`. A `twilio_auth_failed` on this route with `reason` `basic` or `signature` is a rejected callback |
| `relay_parameter_unavailable` | front door, warmer | ERROR: `parameter`, `reason`, `error_class`, `code`, `decision` `fail_closed` | At start-up the environment couldn't read `/owlbank/<stage>/bridge/relay` (the Runtime ARN here): every request in that environment fails. Check that the Bridge stack is deployed and the parameter exists |
| `startup_complete` | bridge | `hosting` | A bridge process finished its start-up and is ready (before it listens, so `/ping` is healthy only then). In the Runtime group, one per microVM. AgentCore keeps spare microVMs started, so most of these lines belong to no call: a fresh one starts after a session is stopped, refilling the pool. Only one in the call's own log stream between a `relay_minted` and the next `ws_session_start` means that connect waited for a cold microVM |

**Harmless on this hosting.** A uvicorn `Invalid HTTP request received` WARNING at the moment a session attaches to a spare microVM is the platform's probe. A call stopped right after it ended (`runtime_session_stopped`) can lose its last bridge lines, for example the `turn` and `ws_session_end` of a caller who hung up during a hand-off; the front door's lines are complete.

**Symptoms on this hosting.**

| Symptom | Look for |
|---|---|
| Silence or a delay when the AI should first speak | Was the session warm? A `runtime_prewarm` `ok` well before `relay_minted` means it was; `runtime_prewarm_failed` or no warmer line means the connect waited for the boot (a `startup_complete` in a new log stream between `relay_minted` and `ws_session_start`) |
| "Something went wrong on our end", then a human, at connect | `relay_presign_failed`, or a bridge `twilio_auth_failed` (`binding` or `nonce`) for the call, or Twilio `connect-action` `failed` with 64105 and no bridge line (a refused connect: AgentCore quota or throttling, `AWS/Bedrock-AgentCore` `Throttles` and the alarms) |
| A long call heard the apology near the hour | No `relay_reconnect reason=stream_limit`: the drop came before 3,540 s since the last `relay_minted`, so it was an ordinary drop; look for the bridge's `ws_session_end` `reason` and the Twilio ErrorCode |
| Sessions still running after calls ended | `runtime_session_stopped` with `outcome` `error:<class>`, or none (no end status reached the front door). The idle timeout ends them; to end one now, see below |
| An alarm fired | The warmer's pre-warm throttle alarm: `runtime_prewarm_failed` `code`. The Runtime's throttles, system errors or active sessions near `agentcoreSessionQuota`: the `AWS/Bedrock-AgentCore` metrics for the Runtime, then `preflight.py --agentcore` for the quotas (Service Quotas) |

**Stop a stuck session.** With a role that may call `bedrock-agentcore:StopRuntimeSession` on the Runtime (Administrator; the front door's role has it):

```
aws bedrock-agentcore stop-runtime-session --profile <profile> \
  --agent-runtime-arn <RuntimeArn output> --runtime-session-id <CallSid>
```

It ends that call's microVM: a caller still on that call loses the AI session (the front door's drop path: apology, one reconnect on a new microVM with history from AgentCore Memory, then a human). Use it only for a call that has ended.

**Compare the hostings.** `scripts/bench/call_latency.py` groups calls by `hosting` and reads the bridge lines from the Runtime log group (from the `RuntimeLogGroup` output in `infra/cdk-outputs.json`, or `--runtime-log-group`):

```
uv run python scripts/bench/call_latency.py --since <ISO time> --label hostings \
  --runtime-log-group /aws/bedrock-agentcore/runtimes/<RuntimeId>-DEFAULT
```

Its `hostings` section repeats the summary per hosting and adds connect-to-first-word (warm and cold, first connect and re-entry), dial-to-first-word, the warmer's outcomes and durations, the pre-warm hit rate and the cold-start time; `hosting_difference` is `agentcore` minus `fargate`. The benchmark writes its results to a git-ignored folder (`bench/README.md`). Run it **before** a switch back to Fargate: outside `stage=prod` the Runtime log group is deleted with the Runtime.

**Switch back to Fargate** (run the benchmark above first). One deploy of Bridge, Compute and Api without `bridgeHosting`, then `scripts/twilio_setup.py --phase post` with that deploy's outputs: `infra/README.md`, "Deploying `bridgeHosting=agentcore`, and switching back", and its hazards (calls in progress may drop; a new CloudFront domain). In VPC mode the Bridge update then sticks in `UPDATE_COMPLETE_CLEANUP_IN_PROGRESS` for hours, because AgentCore's `agentic_ai` network interfaces hold the Runtime's security group, and until Compute is deployed every call fails closed (the front door still has the Runtime ARN). Don't wait:

1. Deploy `OwlBankCompute`, then `OwlBankApi`, from the synthesized assembly: `npx aws-cdk@2 deploy <stack> --app cdk.out --exclusively`, where `cdk.out` is the assembly of the earlier synth or deploy.
2. Run `twilio_setup.py --phase post`; its read-back re-reads up to four times while Twilio catches up.
3. Later, once `aws ec2 describe-network-interfaces --filters Name=group-id,Values=<security group id>` shows none, delete the orphaned security group.

## The conversation sweep (human leg)

With Flex only (without it, Orchestrator's own status timeouts close the conversation), the conversation sweep Lambda runs every 10 minutes and closes the Orchestrator conversation of an escalated call whose close signal was lost (production guide, "The human leg: transcript, Memory and retention", item "The sweep"). It reads the `calls` table's `sweep-due` index (escalated calls whose conversation is not closed, oldest due first), so it needs no Orchestrator configuration and keeps no cursor item. A call is in the index only if it can have a conversation (an AI session ran, or a conversation id is recorded) and has had fewer than 6 futile tries in a row (`escalation.sweep_retries`). A call escalated before any AI session (for example on a verification risk, or a caller who asked for a person straight away) has no conversation and is never swept.

| Event | Level | Fields | Meaning and action |
|---|---|---|---|
| `conversation_sweep` | INFO, WARNING when cut short | `decision` (`done` or `continue_next_run`), `due`, `skipped`, `statuses`, `closed`, `rescheduled`, `unknown`, `out_of_time` | One per run, counts only. `due`: calls due at the start; `skipped`: stale index entries (closed, expired or not due on the fresh read); `statuses`: Calls API reads; `rescheduled`: live, ended within 5 minutes, or close not done; `unknown`: a call with no usable status (due again in 10 minutes, one retry counted), every call of a `BatchGetItem` chunk that failed (left due), a reschedule or retry whose write failed (left due), or one that wrote nothing because the fresh read decided otherwise (for example the conversation closed meanwhile). `continue_next_run` on several runs in a row means more due calls than one run can handle: look at `due` and the Calls API latency |
| `conversation_sweep_query_failed` | ERROR | `operation` `calls.query` | The index query failed; the run raised, so the Lambda's Errors metric and alarm see it. Check the role's `dynamodb:Query` grant on `calls/index/sweep-due` and the table's status |
| `conversation_sweep_state_failed`, `conversation_sweep_state_unreadable` | WARNING | `call_sid` (the second only) | A batch read failed, or one call-state item did not parse; those calls stay due for the next run |
| `conversation_sweep_status_failed` | WARNING | `call_sid`, `http_status` or `reason` | The Calls API read failed; the call is unknown and is looked at again in 10 minutes (never closed on an unknown status) |
| `conversation_sweep_reschedule_failed` | WARNING | `call_sid` | Writing the next due time failed; the call stays due and comes first next run |

How the sweep treats a call that it can't close:

- **Unknown or failing:** a call that stays unknown (for example the Calls API answers 404, or an ended status with no end time), or whose close keeps failing, is tried about every 10 minutes and goes to the back of the queue each time, so it never crowds out other calls.
- **Leaving the index:** each such try counts in `escalation.sweep_retries`. After 6 in a row (about an hour) the call leaves the index for good and Orchestrator's timeout is the backstop.
- **Resets:** a live answer, or an end within the last 5 minutes, resets the count to 0, so a caller in a long queue is never dropped because the Calls API failed now and then.
- **Early due times:** retry and live due times are written 60 s early, so a run that starts a little early on the schedule still finds them due.
- **`unknown` in the summary** counts these calls and failed writes.
- **To see what is due:** query the `sweep-due` index of the `calls` table with `sweep_open = "open"` and `sweep_due_at <= <epoch now>`. A call's own `escalation.sweep_next_at` is its next look, set by the sweep only and dropped when `/voice/status` records `call_ended_at`.

## Demoing the web-activity offer (`web_activity.py`, `memory_history.py`)

Demo only: `dev` and `test` stages, with fictional data.

If the verified caller looked at a product on the website within `webActivityMaxAgeHours` (default 24 hours) and the product is on the bank's list (policy `web_offers`), the welcome offers it. In production a feed such as Segment writes the traits; for a demo, `scripts/web_activity.py` writes them to the developer-phone customer's Twilio Memory profile (the customer the seed gives your phone number, found by `memory_email`; dev or test only, `prod` is refused). `scripts/memory_history.py` writes a past WhatsApp chat (a summary and an observation) to the same profile. Both print that these are demo data. Run them from the repo root, with AWS credentials that can read the Twilio API key secret (they find it through `--cdk-outputs infra/cdk-outputs.json`, or `--secret twilio_api_key=<ARN>`) and the output file of `twilio_setup.py` (`infra/twilio.context.json`).

**Set up once**

1. The Disputes transfer part of this demo needs Flex ([docs/add-flex.md](add-flex.md)); without it call 1 ends with the spoken briefing. With Flex: `uv run python scripts/twilio_setup.py --phase pre --with-flex --cdk-outputs infra/cdk-outputs.json` (plan) then the same with `--account-sid AC... --apply` (add `--dedicated-main-account` on a main account used only by the blueprint, as for the other phases): creates the OwlBank Disputes queue and the workflow filter for `card_dispute`; the TaskRouter steps need no domain, and existing resources are left as they are. It also defines the agent skills (`en`, `es`, `fraud`, `disputes`) in Flex, so the Add-skill dropdown is populated; it assigns them to no one.
2. **Flex only:** give the demo agent the routing skill `disputes`. Without it a disputes task never reaches the agent. To give an agent the skills: in Flex admin, Users, edit the user, Add skill (the dropdown lists `en`, `es`, `fraud` and `disputes` once `twilio_setup.py --phase pre` has run); or `uv run python scripts/assign_agent_skills.py --email agent@example.com --cdk-outputs infra/cdk-outputs.json` (dry run; add `--apply` to write; `--email` is repeatable and required, there is no all-workers mode). A Flex user who has never signed in to the agent desktop has no worker yet, so sign in once first.
3. `uv run python scripts/seed.py --stage dev --reseed --developer-phone <your phone> --developer-factor zip=<your ZIP> --apply` (with the usual `--cdk-outputs`): seeds the offer list, its phrases and the `card_dispute` reason, and restores the fixtures. Without `--developer-factor` the seeded ZIP is 94105.

**On the day**

1. **Run `web_activity.py` on the day of the call.** `--apply` stamps `lastViewedAt` with now, and the default window is 24 hours; the next day `--check` says "too old" and the welcome has no offer.
2. `uv run python scripts/web_activity.py` prints the plan and writes nothing. Add `--product "auto loan" --views 1` to try another product (default `HELOC`, 3 views). The product must match a value in `web_offers` exactly (ignoring case and spacing): `HELOC` and `home-equity` for the HELOC offer, `Auto loan` and `auto-loans` for the auto loan offer. Anything else gets the usual welcome.
3. `uv run python scripts/web_activity.py --apply` makes sure the `WebActivity` group exists, writes the traits, then reads the profile back for up to `--wait` seconds (default 30) and says how long they took to show. Re-running with the same `--viewed-at` changes nothing; without it the time is now, so a re-run refreshes it.
4. `uv run python scripts/web_activity.py --check` is read-only. It runs the bridge's own recall and prints the text the agent would get, or why `WebActivity` doesn't come back (exit 4). If the deployment sets `memoryTraitGroups` or `webActivityMaxAgeHours`, pass the same values with `--trait-groups` and `--max-age-hours`. It shows the profile traits, not observations or summaries, and it doesn't check the offer list.
5. `uv run python scripts/memory_history.py` prints the plan; `--apply` writes the card-dispute preset (a summary and an observation about the Example Airlines charge on the customer's card, `source` `owlbank-demo`) and reads back for up to `--wait` seconds. The profile must exist (exit 3 otherwise). `--summary` and `--observation` take any text, and `--channel` the chat's channel (default `whatsapp`); with only one of the two, only that one is written.
6. **Demo call.** Call the bank number from the developer phone and verify as the customer.
   - **Call 1.** The welcome offers the product: "Hi <first name>. I am the OwlBank virtual agent. Would you like help with our home equity line of credit?" Answer: "No, I'm calling about a charge on my card." The agent doesn't repeat the offer, knows about the WhatsApp report, confirms it and transfers to OwlBank Disputes. With Flex, the task is in the Disputes queue and the screen pop shows the summary; without it the call ends with the briefing.
   - **Call 2.** Same setup, answer "yes" to the offer: the agent offers a HELOC specialist or asks what you want to know, and doesn't mention the chat.
   - Before verification the agent knows nothing about the visit or the chat.
7. **Afterwards (optional).** The chat can stay for repeated demos. `uv run python scripts/memory_history.py --clear --apply` deletes the summaries and observations whose source is `owlbank-demo` (others are left alone). `uv run python scripts/web_activity.py --clear --apply` sets every `WebActivity` trait to null and reads back until the group is gone or empty. The next call gets the usual welcome.

**When there is no offer**

Look first at the two log lines of the call (Logs Insights on the auth service and front door log groups, filter on the `call_sid`): `welcome_built` (`reason`, `duration_ms`, `trait_count`) and `welcome_used` (`outcome`, and `reason` for `fallback`).

| `welcome_built` reason | What it means |
|---|---|
| `matched` | The product is on the list. If the call had no offer, read `welcome_used` |
| `no_web_activity` | No `lastViewedProduct` came back: none written, older than the window (run `web_activity.py --apply` again on the day), or `WebActivity` is not on `memoryTraitGroups` |
| `no_match` | The product isn't a listed value. Compare it with `web_offers` (exact match after case and spacing); add the variant to the list and reseed |
| `no_first_name` | The customer record has no first name, so the welcome would read "Hi ." |
| `no_profile` | No Twilio Memory profile for the customer's `memory_email` (or two profiles share it, or no store ID is set). Run `web_activity.py --check` |
| `memory_timeout` | Twilio Memory didn't answer within the 1 s limit. Try again |
| `memory_error` | A Twilio Memory call failed, or the store is missing. Check the store ID in the deploy and `--check` |
| (no line at all) | The prepare step wasn't started: the last factor wasn't asked on this call, the caller ID matched several customers or none, or `WebActivity` is off the allow-list |

| `welcome_used` outcome | What it means |
|---|---|
| `made` | The offer was spoken |
| `not_ready` | Nothing was prepared when the caller finished verification. Check `duration_ms` of `welcome_built` against the time the caller took on the last factor, and whether the line exists |
| `fallback` | The prepared result wasn't a match (`reason` says why, same list as above), or it couldn't be opened (`reason` `unusable`, or `error`) |
| `mismatch` | It was prepared for another customer or language than the one verified. The usual welcome is spoken |
| `no_phrase` | The call's language has no `offer.<id>` or `greeting.offer`, or the filled text isn't one clean question. Check the catalog in that language |

| Other checks | What it means |
|---|---|
| `--apply` ends with a timeout (exit 3) | Twilio Memory queues trait writes and the traits weren't on the profile in time. Run `--check` a little later, or `--apply --wait 60`. A re-run is safe. A new profile can read "not found" for a moment after the first write; that counts as part of the wait |
| `--check` says the group is missing or the store doesn't accept it | The store doesn't have the `WebActivity` group. `--apply` creates it; `twilio_setup.py` also does, as its last step, and only warns if it can't. Re-run either |
| `--check` says the trait is too old | `lastViewedAt` is outside the window. Run `--apply` again on the day (or, if you changed `webActivityMaxAgeHours`, pass the same value to `--check`) |
| `--check` says the group is not on the allow-list | `memoryTraitGroups` doesn't list `WebActivity`. Deploy with it, or leave it out on purpose to turn the offer off |
| The call has an offer but the agent doesn't know about the chat | The summary or observation isn't on the profile the call used (the one keyed on the customer's email). Run `memory_history.py` again and wait for it to read back. The agent only has what the Recall returns: up to 20 observations and 5 summaries |
| The call reaches Support, not Disputes (Flex only) | The agent chose another reason (read the `escalate` call in the segment log), or the workflow has no `card_dispute` filter (run `twilio_setup.py --phase pre --with-flex` again) |
| The task is in the Disputes queue but nobody gets it (Flex only) | No worker has the `disputes` routing skill. Add it in Flex |
| Two profiles for one email | The script reports an error and stops. Merge them in Twilio Memory first |

## What is never in the logs

Look elsewhere for these:

- Sensitive slot values (SSN last four, one-time codes; ZIP too once it is flagged `sensitive`), account answers, confirmation codes, call tokens (raw or sealed), secrets, the HTTP Basic credentials and the relay nonce (any URL query string).
- Full phone numbers (at most the last four digits, `+*******1234`), and any run of six or more digits.
- What the caller said and what the AI said, and what the slot resolver was sent or returned (except `factor_answer.speech`, below): no message text, no tool arguments, no tool results, no escalation summaries, no account numbers, balances or amounts. For the words, use the Twilio call recording if you have turned it on, or the call's history in AgentCore Memory (access through IAM).
- Request and response bodies of AWS and Twilio calls. A failed call logs its error class and error code, never its message.

What is there: the call SID, the customer ID in audit lines, tool names, factor names, route names, slot modes, counts, durations, error classes and codes, and decisions, plus a non-sensitive slot's value (ZIP in the seed, the card) in `factor_answer` and `slot_resolution`. With `speechLog` on (the default), `factor_answer.speech` holds each Gather turn's transcript, a sensitive slot's value as its placeholder, and free text said at a non-sensitive prompt as said unless `speechLogRedaction` is on (production guide, "The speech log of Gather turns").

The scrubber masks phone numbers, long digit runs, token-like strings, Authorization and Basic values, and query strings. It is pattern-based, so it cannot catch a name or an address in free text. That is why free text is never logged.
