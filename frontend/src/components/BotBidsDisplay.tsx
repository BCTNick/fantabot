import React from 'react';
import { Card } from './UI';
import type { BotBidResult } from '../types';

interface BotBidsDisplayProps {
  bids: BotBidResult[];
  show: boolean;
}

export const BotBidsDisplay: React.FC<BotBidsDisplayProps> = ({ bids, show }) => {
  if (!show || !bids.length) return null;

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'bid': return '💰';
      case 'pass': return '❌';
      case 'cannot_bid': return '🚫';
      default: return '❓';
    }
  };

  const getActionText = (bid: BotBidResult) => {
    switch (bid.action) {
      case 'bid': return `Ha offerto €${bid.amount}`;
      case 'pass': return 'Ha passato';
      case 'cannot_bid': return `Non può offrire (${bid.reason})`;
      default: return 'Azione sconosciuta';
    }
  };

  const getActionColor = (action: string) => {
    switch (action) {
      case 'bid': return 'text-green-600 bg-green-50 border-green-200';
      case 'pass': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'cannot_bid': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  return (
    <Card title="🤖 Ultime Azioni Bot" className="mb-6">
      <div className="space-y-3">
        {bids.map((bid, index) => (
          <div 
            key={`${bid.agent_id}-${index}`}
            className={`p-3 rounded-lg border ${getActionColor(bid.action)}`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <span className="text-xl">{getActionIcon(bid.action)}</span>
                <div>
                  <div className="font-medium">{bid.agent_id}</div>
                  <div className="text-sm opacity-75">{getActionText(bid)}</div>
                </div>
              </div>
              {bid.action === 'bid' && (
                <div className="text-lg font-bold">
                  €{bid.amount}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};
