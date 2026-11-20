// Parity helper for tests/property/test_admin_prompts_props.py reads a
// JSON array of {key, text, current} on stdin and prints, for each, whether the console's
// checkPromptTexts accepts the text (one language; current null means no same-set rule; an
// agent.* key has its placeholder allow-list either way).
import { readFileSync } from "node:fs";

import { checkPromptTexts } from "../../src/admin_console/lib/validate.js";

const cases = JSON.parse(readFileSync(0, "utf8"));
const out = cases.map(({ key, text, current }) =>
  checkPromptTexts(key, { "en-US": text }, ["en-US"], current === null ? null : { "en-US": current }).length === 0);
process.stdout.write(JSON.stringify(out));
