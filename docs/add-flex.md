# Add Flex

The default deployment has no Flex. When a caller asks for a person, the front door reads a short briefing of the context an agent would have received and ends the call; nothing is queued or transferred. This page moves a working deployment to the Flex path: escalations go to a TaskRouter queue, a human agent accepts the call in Flex, and a context panel (screen pop) shows the caller, the reason, the AI summary and the redacted call timeline.

Flex is billable. Check Twilio's current pricing before you start.

## What changes

| | Without Flex (default) | With Flex |
|---|---|---|
| Escalation | A spoken briefing, then the call ends | `<Enqueue>` to the TaskRouter workflow; a Flex agent takes the call |
| Setup output | `infra/twilio.context.json` has no `workspaceSid`, `workflowSid` or `queueSids` | It has all of them |
| Flex mode | Off | On when the file has both `workspaceSid` and `workflowSid`. Exactly one of them fails synth with a message to re-run setup with `--with-flex` |
| `humanLegConversation` (on by default, effective only in Flex mode) | Off | On: adds `POST /voice/taskrouter` and `POST /voice/transcription`, and keeps the Orchestrator conversation through the human leg |
| Queue-wait metric of call observability (`handoff.queue`) | Off (it needs TaskRouter events) | On when `callObservability=on` (it also adds `POST /voice/taskrouter`; step 3 points the workspace's events at it) |
| Context API | Deployed, answers 503 | Serves the redacted timeline to the plugin |

Nothing else about the call path changes.

## Prerequisites

- **A Flex account for the account the blueprint runs in.** Either Flex is already activated in the Console, or `--create-flex` creates it (billable). If Twilio refuses activation because the parent is a dedicated Flex account, run the blueprint in an account that can have its own Flex; the existing deployment keeps working without Flex.
- **Tools for the plugin:** Node 20 (the Flex Plugins tooling does not run on newer Node; keep Node 22.18 or later for the repository's tests and switch per shell), the Twilio CLI and the Flex Plugins CLI plugin (`twilio plugins:install @twilio-labs/plugin-flex`). With Homebrew: `brew install node@20 twilio/brew/twilio`. `node@20` is keg-only, so put `/opt/homebrew/opt/node@20/bin` first on the PATH for the plugin steps.
- **A Twilio CLI profile** for the target account: `twilio login` asks for the Account SID, an API key SID and its secret. A Standard key works. If you did not keep the secret of the key you made for the blueprint, create a second Standard key. Name the profile with letters, digits, `.`, `_` or `-` only: `deploy_flex_plugin.py` refuses a name with spaces, and the CLI has no rename command.
- **`AWS_PROFILE`** exported in the shell for steps 1 to 5. The setup script reads Secrets Manager, the deploy uses your AWS credentials, and the plugin dry run in step 5 reads the Context API URL from the `OwlBankApi` stack outputs (not with the Twilio profile); or pass `--api-url`.

## Steps

Run them from the repository root unless stated.

1. **Set up Flex in Twilio.** Dry run first, then add `--apply`. Add `--create-flex` only if the account has no Flex yet:
   `uv run python scripts/twilio_setup.py --phase pre --with-flex --account-sid AC... --cdk-outputs infra/cdk-outputs.json`
   Ignore the `next:` hint it prints, which is the first-deploy one: continue with step 2 below.
   It finds the Flex TaskRouter workspace ("Flex Task Assignment", or "Flex Task Assignment for GO..." when Flex was created through the API), creates the queues and the workflow, defines the agent skills (`en`, `es`, `fraud`, `disputes`) and writes `workspaceSid`, `workflowSid` and `queueSids` into `infra/twilio.context.json`. What exists already is found, not created again. Add `--dedicated-main-account` on a main account.
2. **Redeploy with the same flags.** Context flags are not remembered, so pass every `-c` flag you used for the first deploy. In a new shell, export `AWS_PROFILE` and define the `flags` array again, as in step 10 of the root README (account, Twilio number, admin email, and any optional flag such as `s3PublicAccessBlockMode`). Use an administrator role: Compute gains the conversation sweep's role and schedule. From `infra/`, read what the diff prints; the stacks that read the outputs file change, so expect `OwlBankCompute` and `OwlBankApi` (the briefing turns off, and the workflow, the human-leg conversation and the TaskRouter route turn on):
   `npx aws-cdk@2 diff OwlBankCompute OwlBankApi --exclusively "${flags[@]}"`
   Then deploy them, with their own outputs file (the option replaces the file it names; you need not merge it, and the later steps keep using `infra/cdk-outputs.json`), and read the approval table before you answer `y`:
   `npx aws-cdk@2 deploy OwlBankCompute OwlBankApi --exclusively "${flags[@]}" --outputs-file cdk-outputs-flex.json`
3. **Point the TaskRouter events at the front door.** First read what Twilio holds today, including the Flex workspace, with `--check`:
   `uv run python scripts/twilio_setup.py --phase post --with-flex --taskrouter-events --account-sid AC... --phone-number +1... --cdk-outputs infra/cdk-outputs.json --check`
   Then re-run the same command without `--check` (dry run), then add `--apply`. It sets the workspace's TaskRouter event callback only when it is empty or already this front door's; any other value is reported and left alone.
4. **Assign agent skills.** The queues route on worker skills, so without them no escalation reaches anyone. The agent must have signed in to Flex once. Dry run, then `--apply`:
   `uv run python scripts/assign_agent_skills.py --email agent@example.com --cdk-outputs infra/cdk-outputs.json`
   Or use Flex admin, Users, Add skill. Delete any worker that has the skills but never signs in: it can be offered calls and never answer them.
5. **Install the plugin.** First check what is already released on the Flex instance, with `--json`: the plain listings print nothing when run non-interactively, which looks like "no plugins".
   `twilio flex:plugins:list-releases --profile <twilio-cli-profile> --json`
   A release replaces the set of active plugins, so read what the newest release holds. Then run
   `uv run python scripts/deploy_flex_plugin.py --profile <twilio-cli-profile> --dry-run`, and again without `--dry-run`. The script runs `npm ci`, the build, the deploy and the release, and reads the released version back. The Flex CLI, not the script, assigns the version on `deploy --major` and rewrites `flex-plugin/package.json`, so the dry run names none. A re-run is safe: a version that is already released is reported, not an error. The manual route and details: [flex-plugin/README.md](../flex-plugin/README.md).

## Verify

1. `uv run python scripts/twilio_setup.py --check --phase pre --with-flex --account-sid AC... --cdk-outputs infra/cdk-outputs.json` reports the Flex workspace and exits 0.
2. Open the agent desktop at <https://flex.twilio.com/agent-desktop/> and set yourself Available. Hard-reload after a release.
3. Call the number, verify, and ask for a person. The call now goes to Flex instead of ending with the briefing. Accept the task.
4. **The panel is on the task's Info tab**, in the left-hand task panel. The task opens on the Call tab, so click **Info**. The large right-hand pane may show a "localhost refused to connect" error: it belongs to a CRM container already configured on the Flex instance and is not this feature.
5. In the browser's network tab the Context API call goes to your API domain with an `Origin: https://flex.twilio.com` header and returns 200.

## Going back

Remove `workspaceSid`, `workflowSid` and `queueSids` from `infra/twilio.context.json`, redeploy the same stacks with the same flags, and escalations read the briefing again. Also clear the workspace's TaskRouter Event Callback URL in the Console (it still points at the front door). The queues, workflow and plugin stay in Twilio until you remove them in the Console.
