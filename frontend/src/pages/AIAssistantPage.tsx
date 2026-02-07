/**
 * AIAssistantPage — Route page wrapping the AIAssistant component.
 */

import { type FC } from 'react';
import { AIAssistant } from '@/components/assistant/AIAssistant';

export const AIAssistantPage: FC = () => (
    <div className="container mx-auto py-6 h-[calc(100vh-8rem)]" data-testid="assistant-page">
        <AIAssistant className="h-full" />
    </div>
);

AIAssistantPage.displayName = 'AIAssistantPage';
