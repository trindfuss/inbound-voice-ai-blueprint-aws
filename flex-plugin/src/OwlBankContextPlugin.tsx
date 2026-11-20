import React from 'react';
import * as Flex from '@twilio/flex-ui';
import { FlexPlugin } from '@twilio/flex-plugin';

import OwlBankContextPanel from './components/OwlBankContextPanel';
import { hasOwlBankContext } from './taskAttributes';
import { COUNTRY_TEMPLATES, withoutCountry } from './taskInfoTemplate';

const PLUGIN_NAME = 'OwlBankContextPlugin';

/**
 * Screen pop for calls the OwlBank AI agent escalated.
 *
 * Adds a panel to the task info panel for tasks that carry a `journal_ref`.
 * The panel shows the task attributes (reason, AI summary, verified level,
 * language, caller name) and, once the agent has accepted the task, fetches
 * the redacted call timeline from the Context API with the agent's Flex token.
 * It also drops Flex's Country line from Customer context (see taskInfoTemplate).
 */
export default class OwlBankContextPlugin extends FlexPlugin {
  constructor() {
    super(PLUGIN_NAME);
  }

  async init(flex: typeof Flex, manager: Flex.Manager): Promise<void> {
    const strings = manager.strings as unknown as Record<string, string>;
    for (const key of COUNTRY_TEMPLATES) {
      if (typeof strings[key] === 'string') strings[key] = withoutCountry(strings[key]);
    }
    flex.TaskInfoPanel.Content.add(
      <OwlBankContextPanel key="owlbank-context-panel" manager={manager} />,
      {
        sortOrder: -1,
        if: (props: { task?: Flex.ITask }) => hasOwlBankContext(props.task),
      },
    );
  }
}
