# OwlBank Flex plugin (context panel, or screen pop)

Optional: the blueprint runs without Flex and without this plugin (an escalation then reads a short briefing and ends the call). To move to Flex, follow [docs/add-flex.md](../docs/add-flex.md), which uses this plugin.

A minimal Flex UI 2.x plugin. When an agent gets a task the OwlBank AI escalated (a task with `journal_ref` and `call_sid` attributes), it adds a panel to the task's **Info** tab (in the left-hand task panel) showing:

- the caller name, escalation reason, verified level and language, from the task attributes
- the AI summary, labelled "AI-generated summary"
- after the agent accepts the task, the redacted call timeline from the Context API (`GET /context/{journal_ref}?call_sid=CA…&task_sid=WT…`), sent with the agent's Flex token in `X-Flex-Token`

Every value is rendered as React text, so attribute values (including the model's summary) can't inject markup. The panel is a React component rather than a Handlebars override of `manager.strings.TaskInfoPanelContent`, because the timeline needs a fetch and React escapes all text by default.

It also removes the Country line from Flex's own "Customer context" (in `TaskInfoPanelContent` and `SupervisorTaskInfoPanelContent`) for every task. Twilio sets `caller_country` from the calling number when the call is enqueued, and task attributes can't override it. It can be wrong for a withheld caller ID.

## Files

| File | Purpose |
|---|---|
| `src/index.ts` | Plugin entry point (`loadPlugin`) |
| `src/OwlBankContextPlugin.tsx` | Adds the panel to `TaskInfoPanel.Content` for OwlBank tasks |
| `src/components/OwlBankContextPanel.tsx` | The panel; fetches the timeline once the task is `accepted` or `wrapping` |
| `src/taskInfoTemplate.ts` | `withoutCountry`: drops the Country line from Flex's task info templates |
| `src/contextApi.ts` | Context API client and response types |
| `src/taskAttributes.ts` | Task attribute types and labels; `identityNote` ("Identified by a keyed-in number" when `identity_source` is `keyed_phone`) |
| `src/__tests__/` | `npm test` (`node --test`, Node's TypeScript type stripping; no install needed) |
| `.env.example` | `FLEX_APP_OWLBANK_CONTEXT_API_URL`, the API's https base URL |
| `scripts/check-hygiene.mjs` | `npm run hygiene`: the plugin hygiene check |

## Build and deploy

`scripts/deploy_flex_plugin.py` in the repository root runs the install, build, deploy and release with the Twilio CLI (`--dry-run` prints the plan; it needs `--api-url` or the `OwlBankApi` stack outputs, read with your AWS credentials, so set `AWS_PROFILE` for the dry run); the steps below are the manual route. The Flex CLI, not the script, assigns the plugin version on `deploy --major` and rewrites `package.json`, so the dry run names no version and a fresh clone's `4.0.0` is not the number Flex releases. Notes:

- **You need two Node versions.** Node 20 for install, build and deploy: the Flex Plugins tooling (7.1.x) doesn't run on newer Node. Node 22.18 or later for `npm test` (the test commands use quoted globs, which Node 20 does not expand): the tests are TypeScript run with Node's type stripping, and on Node 20 `node --test` finds none. Switch with nvm, fnm or your package manager (on macOS with Homebrew, for example, `brew install node@20` is keg-only: prefix commands with `PATH="/opt/homebrew/opt/node@20/bin:$PATH"`).
- **webpack 4 is pinned** with `overrides` in `package.json`: npm otherwise installs webpack 5 for the loaders' loose peer ranges, and the build fails with `Cannot find module 'webpack/lib/LoaderTargetPlugin'`. `package-lock.json` is committed, so `npm ci` (used by `scripts/deploy_flex_plugin.py`) installs the same tree.

1. Install the Twilio CLI and the Flex Plugins CLI (Flex UI 2.x needs Plugins CLI 4.1.3 or later):
   `twilio plugins:install @twilio-labs/plugin-flex`
2. Log in to the target account with a CLI profile: `twilio login`, then `twilio profiles:use <name>` (without an active profile the plugins commands stop with "no profile set"). `twilio login` asks for the Account SID, an API key SID and the key's secret: a Standard key works. Name the profile with letters, digits, `.`, `_` or `-` and no spaces; `deploy_flex_plugin.py` refuses anything else.
3. In this directory: `npm ci`. It installs exactly the tree in `package-lock.json`; `npm install` may change the lock file.
4. Copy `.env.example` to `.env` and set `FLEX_APP_OWLBANK_CONTEXT_API_URL` to the Context API's https base URL (the custom domain, or the `execute-api` URL with its stage in the no-domain mode).
5. Optional: run locally with `twilio flex:plugins:start` (not needed to deploy). The CLI generates `public/appConfig.js` on first run if it's missing.
6. Deploy: `twilio flex:plugins:deploy --major --changelog "OwlBank screen pop" --description "OwlBank AI escalation context"`. The Flex CLI rewrites the version in `flex-plugin/package.json` on deploy: restore it afterwards with `git checkout flex-plugin/package.json` (or commit nothing from it).
7. Check what is already released, with `--json` (the plain listings print nothing when run non-interactively): `twilio flex:plugins:list-releases --json`. A release replaces the set of active plugins, so read what the newest release holds. Then release the version the deploy printed (`--major` bumps it each time; the Plugins API keeps the count, not `package.json`): `twilio flex:plugins:release --plugin plugin-owlbank-context@<version> --name "OwlBank screen pop <version>" --description "OwlBank AI escalation context"`. Release names should differ per release.
8. Test: open the agent desktop at <https://flex.twilio.com/agent-desktop/> and set yourself Available (hard-reload after a release), escalate a demo call, accept the task, and open its **Info** tab: the panel and timeline are there. The large right-hand pane may show a "localhost refused to connect" error; it belongs to a CRM container already configured on the Flex instance and is not this plugin. In the browser's network tab, the Context API call should go to the configured domain with an `Origin: https://flex.twilio.com` header and a 200 response.

## Back-end prerequisites

- **Context API** (`src/lambdas/context_api`) behind API Gateway REST, `GET /context/{journal_ref}`, with WAF and the `flex_authorizer` REQUEST authorizer (identity source `method.request.header.X-Flex-Token`).
- **CORS:** only `https://flex.twilio.com`. If agents use a self-hosted Flex UI or a different Flex origin, update `ALLOWED_ORIGIN` in the Lambda.
- **The `calls` record** must hold the escalation's `journal_ref` in `escalation.journal_ref`, the same value the front door writes to the task attributes.

## Sources

References (Twilio docs):

- Flex Plugins CLI and Flex UI 2.x plugin creation: <https://www.twilio.com/docs/flex/developer/plugins/cli>, <https://www.twilio.com/docs/flex/developer/plugins/react-versions> (`twilio flex:plugins:create … --flexui2`, React 17.0.2)
- Updating plugins for Flex UI 2.x (dependencies, `@twilio/flex-ui` ^2 as a dev dependency): <https://www.twilio.com/docs/flex/developer/ui/migration-guide/migration-guide-update-plugins>
- Plugin structure (`FlexPlugin`, `init(flex, manager)`): <https://www.twilio.com/docs/flex/quickstart/getting-started-plugin>
- Adding components to the task info panel (`TaskInfoPanel.Content.add`): <https://www.twilio.com/docs/flex/developer/ui/add-components-flex-ui>
- Validating the Flex token server-side (`POST https://iam.twilio.com/v1/Accounts/{AccountSid}/Tokens/validate`, response `valid`, `identity`, `roles`, `worker_sid`; the Twilio Flex Token Validator for Node): <https://www.twilio.com/docs/flex/developer/insights/playback-recordings-custom-storage>
- Flex user tokens (JWE, default TTL one hour): <https://www.twilio.com/docs/flex/developer/user-management/flex-token-api>

## Known limits

- The timeline shows the language choice as `language.select` and the front door's hand-off as an empty `Handoff:`.
- The plugin reads the Flex token from `manager.store.getState().flex.session.ssoTokenPayload.token`; this path isn't documented for Flex UI 2.x.
