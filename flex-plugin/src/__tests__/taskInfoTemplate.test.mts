// Flex's "Customer context" without the Country line. Runs with `npm test`.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { withoutCountry } from '../taskInfoTemplate.ts';

// Flex UI 2.18.1's default TaskInfoPanelContent, Customer context part.
const CUSTOMER_CONTEXT = `<p>Customer context</p>
<dl>
    <div><dt>Customer name / {{customerContext}}:&nbsp;</dt><dd data-testid="customer-name-number">{{task.defaultFrom}}</dd></div>
    {{#if task.attributes.caller_country}}<div><dt>Country:&nbsp;</dt><dd data-testid="customer-country">{{task.attributes.caller_country}}</dd></div>{{/if}}
</dl>
{{#if to}}
    <p>Conversation Participants</p>
{{/if}}`;

test('the Country line is gone and the rest stays', () => {
  const out = withoutCountry(CUSTOMER_CONTEXT);
  assert.doesNotMatch(out, /caller_country|Country:/);
  assert.equal(out, CUSTOMER_CONTEXT.replace(/ {4}\{\{#if task\.attributes\.caller_country[^\n]*\n/, ''));
  assert.match(out, /\{\{#if to\}\}\n {4}<p>Conversation Participants<\/p>\n\{\{\/if\}\}$/);
});

test('a template without a Country line is unchanged', () => {
  const t = '<p>Task context</p>\n{{#if task.attributes.subject}}x{{/if}}\n';
  assert.equal(withoutCountry(t), t);
});
