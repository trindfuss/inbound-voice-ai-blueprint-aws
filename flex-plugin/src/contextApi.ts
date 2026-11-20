import type { Manager } from '@twilio/flex-ui';

/** One redacted journal entry, as returned by GET /context/{journal_ref}. */
export interface TimelineEntry {
  seq: number;
  ts?: number;
  type: string;
  prompt_key?: string;
  rendered_text?: string;
  outcome?: string;
  tool?: string;
  factor?: string;
  level?: number;
  reason?: string;
  queue?: string;
}

export interface CallContext {
  journal_ref: string;
  call_sid: string;
  language: string;
  verified_level: number;
  escalation_reason: string | null;
  timeline: TimelineEntry[];
}

/** Set at build time from `.env` (Flex Plugins CLI exposes FLEX_APP_* variables). */
const API_BASE = (process.env.FLEX_APP_OWLBANK_CONTEXT_API_URL || '').replace(/\/+$/, '');

/** The logged-in agent's Flex user token. */
export function flexToken(manager: Manager): string | undefined {
  const state = manager.store.getState() as any;
  return state?.flex?.session?.ssoTokenPayload?.token;
}

export class ContextApiError extends Error {
  constructor(public readonly status: number) {
    super(`Context API returned ${status}`);
  }
}

export async function fetchCallContext(
  manager: Manager,
  journalRef: string,
  callSid: string,
  taskSid: string,
  signal?: AbortSignal,
): Promise<CallContext> {
  if (!API_BASE.startsWith('https://')) {
    throw new Error('FLEX_APP_OWLBANK_CONTEXT_API_URL is not configured (https only)');
  }
  const token = flexToken(manager);
  if (!token) {
    throw new Error('No Flex token in the session');
  }
  const url =
    `${API_BASE}/context/${encodeURIComponent(journalRef)}` +
    `?call_sid=${encodeURIComponent(callSid)}&task_sid=${encodeURIComponent(taskSid)}`;
  const resp = await fetch(url, {
    method: 'GET',
    headers: { 'X-Flex-Token': token },
    credentials: 'omit',
    cache: 'no-store',
    signal,
  });
  if (!resp.ok) {
    throw new ContextApiError(resp.status);
  }
  return (await resp.json()) as CallContext;
}
