import type { ITask } from '@twilio/flex-ui';

/** The task attributes the OwlBank front door writes (owlbank.task_attributes). */
export interface OwlBankTaskAttributes {
  type?: string;
  name?: string | null;
  call_sid?: string;
  language?: string;
  escalation_reason?: string;
  ai_summary?: string;
  auth_level?: number;
  /** How identity was claimed: the calling number, or a number the caller keyed in. Never digits. */
  identity_source?: 'ani' | 'keyed_phone';
  journal_ref?: string | null;
  last_tool_attempted?: string | null;
  escalated_at?: string;
  test?: boolean;
}

export function attributesOf(task?: ITask): OwlBankTaskAttributes {
  return ((task && task.attributes) || {}) as OwlBankTaskAttributes;
}

export function hasOwlBankContext(task?: ITask): boolean {
  const attrs = attributesOf(task);
  return typeof attrs.journal_ref === 'string' && typeof attrs.call_sid === 'string';
}

/** Human labels for the escalation reasons (owlbank.tools_contract). */
export const REASON_LABELS: Record<string, string> = {
  caller_requested_human: 'Caller asked for a person',
  unable_to_resolve: 'AI could not resolve the request',
  step_up_failed: 'Step-up verification failed',
  fraud_suspected: 'Suspected fraud',
  card_dispute: 'Disputed or unrecognised charge',
  complaint: 'Complaint',
  vulnerable_customer: 'Possibly vulnerable customer',
  verification_risk: 'Verification risk (repeated untrusted attempts)',
  otp_unavailable: 'One-time code could not be sent to the calling number',
  confirm: 'Confirmation step',
};

export const LANGUAGE_LABELS: Record<string, string> = {
  'en-US': 'English',
  'es-US': 'Spanish',
};

/** Shown when the caller was identified by a keyed-in number; no digits. */
export const KEYED_PHONE_NOTE = 'Identified by a keyed-in number';

/**
 * The identity note for the panel: the keyed-in text only when `identity_source` is
 * `keyed_phone`, so the agent knows caller ID didn't confirm the caller. Otherwise null.
 */
export function identityNote(attrs: OwlBankTaskAttributes): string | null {
  return attrs.identity_source === 'keyed_phone' ? KEYED_PHONE_NOTE : null;
}
