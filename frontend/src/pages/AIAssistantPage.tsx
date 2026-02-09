/**
 * AIAssistantPage — Route page wrapping the AIAssistant component.
 */

import { type FC } from 'react';
import { AIAssistant } from '@/components/assistant/AIAssistant';

export const AIAssistantPage: FC = () => (
  <div
    className="container mx-auto h-[calc(100vh-8rem)] px-4 py-6 sm:px-6 lg:px-8"
    data-testid="assistant-page"
  >
    <AIAssistant className="h-full" />
  </div>
);

AIAssistantPage.displayName = 'AIAssistantPage';
