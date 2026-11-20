# Security policy

## This is not a production-ready solution

This blueprint is not a production-ready solution. It deploys with fictional data and a mock bank. Its defaults are demo defaults, and some controls are documented but not built, including validated Transport Layer Security (TLS) from the load balancer to the bridge task (that hop is encrypted with a self-signed certificate that nothing checks, so the task is not authenticated), destination-based egress restrictions and a load test. Before you use any of it with real customers, review it against your own security, compliance and legal requirements, including Payment Card Industry (PCI) scope, call recording and AI disclosure wording, one-time-code consent and data retention. Treat the code as a starting point for that review, not a substitute for it.

The code is checked with static analysis (bandit, cdk-nag) and security tests. That lowers the risk of the example. It does not make the example safe to run with real customers, and it is not a substitute for your own review.

## Reporting a problem

This is example code, not a supported product. It has no security SLA and no private reporting channel. If you find a security problem in this repository, open a GitHub issue that describes the file or component, the steps to reproduce it and the impact you expect. Don't include real customer data, real phone numbers or credentials.

To report a vulnerability in a Twilio product or service itself, use Twilio's [Vulnerability Disclosure Program](https://www.twilio.com/en-us/security/vulnerability-disclosure-program).

## Scope

In scope:

- The code in this repository: the front door, the bridge, the Lambdas, the admin console and its API, the Flex plugin, the CDK app and the scripts.
- Insecure patterns in this example that a reader might copy into their own deployment, even when this repository's own configuration is safe.

Out of scope:

- Twilio products and services themselves. Report these through Twilio's Vulnerability Disclosure Program, linked above.
- AWS services themselves. Report these to AWS through its [vulnerability reporting process](https://aws.amazon.com/security/vulnerability-reporting/).
- Findings that need a deployment you configured differently from the documented defaults and that don't show a flaw in this code.
- The fictional customers, phone numbers and account IDs used as placeholders.
- Denial of service by traffic volume, and automated scanner output without a demonstrated impact.

## Security model

You deploy this solution into your own AWS and Twilio accounts, and you are responsible for it there. The guides give the detail: [production guide, network exposure and endpoint security](docs/production-guide.md#3-network-exposure-and-endpoint-security), [production guide, data protection and retention](docs/production-guide.md#4-data-protection-and-retention) and [solution guide](docs/solution-guide.md).

### Trust boundaries

- **Twilio to the front door.** Every voice and status webhook is signed, and the front door checks the signature (and the WAF rate-limits the routes) before it acts. The caller's words and the caller ID are untrusted input.
- **The caller to the agent.** What a caller says reaches the model as untrusted text. Tools that touch an account are gated on verification level in code, outside the model, and the front door's scripted steps (login, step-up, payment confirmation) run outside the model.
- **The bridge to AWS and Twilio.** The bridge and Lambdas run with task and function roles scoped to the resources they use. Memory, the call journal and the secrets are encrypted with customer-managed keys.
- **The admin console.** Sign-in is through Cognito, with membership of the `admin` group for administrator routes. Every write is audited.
- **Flex to the Context API.** The Context API checks each request per task and returns allow-listed fields with digits masked.

### What you must do

- Deploy into your own AWS account and Twilio account. Nothing in this repository is shared infrastructure.
- Keep every secret in AWS Secrets Manager. The deploy and setup scripts write their values; never put them in templates, context flags, environment files or source control.
- Deploy the stacks with an administrator profile: CDK needs it to create the IAM roles and policies. Use a least-privilege profile for day-to-day work, diffs and scripts.
- Review the IAM policies, the WAF settings (a rule in count mode logs but doesn't block) and the exposure mode before production, and set the account-level controls that back them: S3 Block Public Access and IAM Access Analyzer.
- Run the deploy-time checks (`infra/scripts/preflight.py`, `scripts/check_secrets.py --strict`) and read what they report.
- Decide your own retention and redaction settings; the defaults are examples.

### Known limits

These are accepted in this example. Know them before you copy it.

- **The admin group check is cached for 60 seconds.** Removing a user from the `admin` group, or disabling the user, takes effect within about 60 seconds on administrator routes. The user's ID token itself stays valid until it expires.
- **The load balancer to task hop uses a self-signed certificate that nothing validates.** On Fargate the Application Load Balancer talks HTTPS to the task on port 8080 inside the VPC. The task makes its certificate in memory at start-up, and the load balancer does not validate target certificates. So the hop is encrypted but the task isn't authenticated. With `bridgeHosting=agentcore` the Runtime fronts the container over HTTP, unchanged. In the no-domain (CloudFront) mode, the CloudFront to load balancer hop is HTTP inside AWS's network; use the custom-domain mode for production.
- **The spoken login ZIP is kept in the call journal.** The spoken value of a login factor (the ZIP code) is kept out of the logs, but it is still in the `slot_resolution` event of the call journal in DynamoDB, which the Context API, the recap writer and the human-leg close read. Set the journal's retention to what you need, and restrict who can read the table.
- **Every component except the Context API holds a Standard Twilio API key.** The setup script creates a Restricted key for the Context API alone. The other components, including the front door, the bridge and the Flex authorizer, read one Standard API key, because Twilio has no restricted permissions for Memory, Orchestrator, call transcription or Flex token validation (`Tokens/validate` refuses even a key with no permissions). The front door and the bridge also hold the Auth Token. A leak from any of them is a leak of that key, and a Standard key has full access to the Twilio account. See the credential table in [README.md](README.md#set-up-your-twilio-account), keep the key's secret in Secrets Manager, and rotate it on a schedule.
