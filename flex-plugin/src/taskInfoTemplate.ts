/** Flex's task info templates that show `caller_country` in "Customer context". */
export const COUNTRY_TEMPLATES = ['TaskInfoPanelContent', 'SupervisorTaskInfoPanelContent'] as const;

const COUNTRY_LINE = /[ \t]*\{\{#if task\.attributes\.caller_country\}\}[\s\S]*?\{\{\/if\}\}[ \t]*\r?\n?/g;

/**
 * The template without its Country line. Twilio fills `caller_country` from the calling number,
 * and for a withheld caller ID it is wrong (a US caller shown as TH); task attributes
 * can't override it, so Flex doesn't show it.
 */
export function withoutCountry(template: string): string {
  return template.replace(COUNTRY_LINE, '');
}
