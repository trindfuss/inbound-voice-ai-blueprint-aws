import React, { useEffect, useState } from 'react';
import { withTaskContext } from '@twilio/flex-ui';
import type { ITask, Manager } from '@twilio/flex-ui';

import { CallContext, ContextApiError, TimelineEntry, fetchCallContext } from '../contextApi';
import { LANGUAGE_LABELS, REASON_LABELS, attributesOf, identityNote } from '../taskAttributes';

interface Props {
  manager: Manager;
  task?: ITask;
}

/** Task states in which the agent has accepted the task. */
const ACCEPTED_STATES = new Set(['accepted', 'wrapping']);

type Load =
  | { state: 'idle' }
  | { state: 'loading' }
  | { state: 'loaded'; context: CallContext }
  | { state: 'error'; message: string };

function describe(entry: TimelineEntry): string {
  switch (entry.type) {
    case 'prompt':
      // The language choice is journaled as a prompt with the chosen language in `outcome`.
      if (entry.prompt_key === 'language.select' && !entry.rendered_text) {
        return `Language chosen: ${LANGUAGE_LABELS[entry.outcome || ''] || entry.outcome || 'default'}`;
      }
      return entry.rendered_text || entry.prompt_key || 'Prompt played';
    case 'factor_asked':
      return `Asked for ${entry.factor || 'a verification factor'}`;
    case 'verification':
      return `Verification: ${entry.outcome || 'unknown'}` +
        (entry.level !== undefined ? ` (level ${entry.level})` : '');
    case 'tool_call':
      return `Tool called: ${entry.tool}`;
    case 'tool_result':
      return `Tool result: ${entry.tool || ''} ${entry.outcome || ''}`.trim();
    case 'handoff':
      // The front door's hand-off to a person carries neither (only `to`, which isn't returned).
      if (!entry.outcome && !entry.tool) return 'Handed to a specialist';
      return `Handoff: ${entry.outcome || ''} ${entry.tool || ''}`.trim();
    case 'escalation':
      return `Escalated: ${REASON_LABELS[entry.reason || ''] || entry.reason || ''}` +
        (entry.queue ? ` to ${entry.queue}` : '');
    case 'lookup':
      return `Caller lookup: ${entry.outcome || ''}`;
    case 'call_start':
      return 'Call started';
    case 'call_end':
      return 'Call ended';
    default:
      return entry.type;
  }
}

const styles: Record<string, React.CSSProperties> = {
  root: { padding: '12px 0', fontSize: 13, lineHeight: 1.4 },
  heading: { fontWeight: 700, fontSize: 14, margin: '0 0 8px' },
  row: { margin: '2px 0' },
  label: { fontWeight: 600, marginRight: 4 },
  summary: { background: 'rgba(0,0,0,0.04)', borderLeft: '3px solid #f2be5a', padding: '6px 8px', margin: '6px 0' },
  aiTag: { fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: '#8a5a00' },
  list: { margin: '4px 0 0', paddingLeft: 18 },
  muted: { color: '#606b85' },
};

/**
 * Everything shown is rendered as React text, so attribute values (including the
 * model-written summary) can never inject markup.
 */
function OwlBankContextPanel({ manager, task }: Props): JSX.Element | null {
  const attrs = attributesOf(task);
  const accepted = !!task && ACCEPTED_STATES.has(String(task.status));
  const taskSid = task?.taskSid;
  const [load, setLoad] = useState<Load>({ state: 'idle' });

  useEffect(() => {
    if (!accepted || !attrs.journal_ref || !attrs.call_sid || !taskSid) {
      return undefined;
    }
    const abort = new AbortController();
    setLoad({ state: 'loading' });
    fetchCallContext(manager, attrs.journal_ref, attrs.call_sid, taskSid, abort.signal)
      .then((context) => setLoad({ state: 'loaded', context }))
      .catch((err: unknown) => {
        if (abort.signal.aborted) return;
        const message = err instanceof ContextApiError && err.status === 404
          ? 'No call history found for this task.'
          : 'Call history is unavailable right now.';
        setLoad({ state: 'error', message });
      });
    return () => abort.abort();
  }, [accepted, attrs.journal_ref, attrs.call_sid, taskSid, manager]);

  if (!task) return null;
  const reason = attrs.escalation_reason || '';
  const language = attrs.language || '';
  const level = load.state === 'loaded' ? load.context.verified_level : attrs.auth_level;
  const note = identityNote(attrs);

  return (
    <div style={styles.root} data-testid="owlbank-context-panel">
      <hr />
      <p style={styles.heading}>OwlBank AI escalation</p>
      <p style={styles.row}>
        <span style={styles.label}>Caller:</span>
        {attrs.name || <span style={styles.muted}>Not identified</span>}
      </p>
      {note && <p style={styles.row} data-testid="owlbank-identity-note">{note}</p>}
      <p style={styles.row}>
        <span style={styles.label}>Reason:</span>
        {REASON_LABELS[reason] || reason || 'Not given'}
      </p>
      <p style={styles.row}>
        <span style={styles.label}>Verified level:</span>
        {level !== undefined && level !== null ? String(level) : 'Unknown'}
      </p>
      <p style={styles.row}>
        <span style={styles.label}>Language:</span>
        {LANGUAGE_LABELS[language] || language || 'Unknown'}
      </p>
      <div style={styles.summary}>
        <div style={styles.aiTag}>AI-generated summary</div>
        <div>{attrs.ai_summary || <span style={styles.muted}>No summary</span>}</div>
      </div>
      <p style={{ ...styles.heading, marginTop: 10 }}>Call timeline</p>
      {!accepted && <p style={styles.muted}>Accept the task to load the call history.</p>}
      {load.state === 'loading' && <p style={styles.muted}>Loading…</p>}
      {load.state === 'error' && <p style={styles.muted}>{load.message}</p>}
      {load.state === 'loaded' && (
        load.context.timeline.length === 0
          ? <p style={styles.muted}>No events recorded.</p>
          : (
            <ol style={styles.list}>
              {load.context.timeline.map((entry) => (
                <li key={entry.seq}>{describe(entry)}</li>
              ))}
            </ol>
          )
      )}
    </div>
  );
}

export default withTaskContext(OwlBankContextPanel);
