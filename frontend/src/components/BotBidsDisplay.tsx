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
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {bids.map((bid, index) => (
          <div 
            key={`${bid.agent_id}-${index}`}
            className={`p-4 rounded-lg border-2 transition-all duration-200 hover:shadow-md ${getActionColor(bid.action)}`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">{getActionIcon(bid.action)}</span>
                <div>
                  <div className="font-bold text-lg">{bid.agent_id}</div>
                  <div className="text-sm opacity-75">{getActionText(bid)}</div>
                </div>
              </div>
              {bid.action === 'bid' && (
                <div className="text-xl font-bold bg-white px-3 py-1 rounded-full shadow-sm">
                  €{bid.amount}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      
      {/* Summary */}
      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <div className="flex justify-between items-center text-sm">
          <span className="text-gray-600">
            📊 {bids.filter(b => b.action === 'bid').length} offerte, 
            {bids.filter(b => b.action === 'pass').length} passate
          </span>
          <span className="text-gray-500">
            ⏱️ Ultimo aggiornamento: ora
          </span>
        </div>
      </div>
    </Card>
  );
};
