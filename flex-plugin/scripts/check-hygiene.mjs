// Plugin hygiene: the sources and config hold no secret or token-like value,
// and the only environment variable read is FLEX_APP_OWLBANK_CONTEXT_API_URL. Run: npm run hygiene
import { readdirSync, readFileSync, statSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const ALLOWED_ENV = 'FLEX_APP_OWLBANK_CONTEXT_API_URL';
const SECRET_PATTERNS = [
  [/\b(?:AC|SK|WS|WT|CA|PN|WK)[0-9a-f]{32}\b/i, 'Twilio SID-like value'],
  [/\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\./, 'JWT-like token'],
  [/\bAKIA[0-9A-Z]{16}\b/, 'AWS access key id'],
  [/-----BEGIN [A-Z ]*PRIVATE KEY-----/, 'private key'],
  [/\b[0-9a-f]{32}\b/i, '32-hex value (auth token-like)'],
  [/\b(?:auth[_-]?token|api[_-]?secret|secret|password|bearer)\s*[:=]\s*['"][^'"]{8,}['"]/i, 'credential assignment'],
];
const SKIP_DIRS = new Set(['node_modules', 'build', 'dist', 'scripts']);
const FILES = ['package.json', 'tsconfig.json', '.env', '.env.example'];

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) {
      if (!SKIP_DIRS.has(name)) yield* walk(path);
    } else if (/\.(ts|tsx|js|jsx|json)$/.test(name)) {
      yield path;
    }
  }
}

const files = [...walk(join(root, 'src')), ...FILES.map((f) => join(root, f)).filter(existsSync)];
const problems = [];
for (const file of files) {
  const text = readFileSync(file, 'utf8');
  for (const [re, what] of SECRET_PATTERNS) {
    if (re.test(text)) problems.push(`${file}: ${what}`);
  }
  for (const m of text.matchAll(/process\.env\.([A-Za-z0-9_]+)/g)) {
    if (m[1] !== ALLOWED_ENV) problems.push(`${file}: reads process.env.${m[1]}`);
  }
  if (/dangerouslySetInnerHTML|\.innerHTML\s*=/.test(text)) problems.push(`${file}: raw HTML injection`);
  if (file.endsWith('.env') || file.endsWith('.env.example')) {
    for (const line of text.split('\n')) {
      const t = line.trim();
      if (t && !t.startsWith('#') && t.split('=')[0] !== ALLOWED_ENV) problems.push(`${file}: variable other than ${ALLOWED_ENV}`);
    }
  }
}
if (problems.length) {
  console.error('Plugin hygiene check failed:\n' + problems.map((p) => '  ' + p).join('\n'));
  process.exit(1);
}
console.log(`Plugin hygiene ok (${files.length} files)`);
