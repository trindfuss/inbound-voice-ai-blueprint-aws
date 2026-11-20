// Keyed-in phone login: the panel's identity note. Runs with `npm test` (node --test,
// Node's built-in TypeScript type stripping; no dependencies needed).
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { KEYED_PHONE_NOTE, identityNote } from '../taskAttributes.ts';
import type { OwlBankTaskAttributes } from '../taskAttributes.ts';

test('keyed_phone shows the keyed-in note', () => {
  assert.equal(identityNote({ identity_source: 'keyed_phone' }), 'Identified by a keyed-in number');
});

test('the note carries no digits', () => {
  assert.doesNotMatch(KEYED_PHONE_NOTE, /[0-9]/);
});

test('ani, missing or unknown identity_source shows nothing', () => {
  const cases = [
    { identity_source: 'ani' },
    {},
    { identity_source: 'KEYED_PHONE' },
    { identity_source: '+14155550123' },
  ] as unknown as OwlBankTaskAttributes[];
  for (const attrs of cases) {
    assert.equal(identityNote(attrs), null, JSON.stringify(attrs));
  }
});
