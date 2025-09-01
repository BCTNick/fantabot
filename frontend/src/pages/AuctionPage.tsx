import React, { useState } from 'react';
import { BotBidsDisplay } from '../components/BotBidsDisplay';
import { NextPlayerCard } from '../components/auction/NextPlayerCard';
import { AgentsStatusCard } from '../components/auction/AgentsStatusCard';
import type { AuctionStatus, BotBidResult, BotBidsResponse } from '../types';
import { CurrentPlayerCard } from '../components/auction/CurrentPlayerCard';

interface AuctionPageProps {
  auctionStatus: AuctionStatus;
  onStartNextPlayer: (roleFilter?: string) => Promise<unknown>;
  onMakeBid: (agentId: string, amount: number) => Promise<unknown>;
  onProcessBotBids: () => Promise<BotBidsResponse>;
  onFinalize: () => Promise<unknown>;
  loading: boolean;
}

export const AuctionPage: React.FC<AuctionPageProps> = ({
  auctionStatus,
  onStartNextPlayer,
  onMakeBid,
  onProcessBotBids,
  onFinalize,
  loading,
}) => {
  const [bidAmount, setBidAmount] = useState<number>(1);
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [roleFilter, setRoleFilter] = useState<string>('');
  const [lastBotBids, setLastBotBids] = useState<BotBidResult[]>([]);

  // Aggiorna l'importo dell'offerta quando cambia il giocatore corrente
  React.useEffect(() => {
    if (auctionStatus.current_player) {
      const minBid = auctionStatus.current_player.current_price + 1;
      setBidAmount(minBid);
    }
  }, [auctionStatus.current_player]);

  const handleStartNext = async () => {
    setLastBotBids([]); // Reset bot bids when starting new player
    await onStartNextPlayer(roleFilter || undefined);
  };

  const handleBid = async (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedAgent && bidAmount > 0) {
      await onMakeBid(selectedAgent, bidAmount);
      // L'importo verrà aggiornato automaticamente dall'useEffect quando cambia current_price
    }
  };

  const handleProcessBotBids = async () => {
    const result = await onProcessBotBids();
    if (result && result.success && result.bids) {
      setLastBotBids(result.bids);
    }
  };

  const humanAgents = auctionStatus.agents.filter(a => a.type === 'HumanAgent');

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200 mb-6">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-gray-800">
              🏆 Asta in Corso
            </h2>
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-green-100 text-green-800">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
              {auctionStatus.state === 'player_auction' ? 'Asta Giocatore' : 'In Attesa'}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4">
        {/* Main Layout - 2 Columns */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          
          {/* Left Column - Current Player & Bidding */}
          <div className="xl:col-span-2 space-y-6">
            
            {/* Bot Bids Display */}
            <BotBidsDisplay bids={lastBotBids} show={lastBotBids.length > 0} />

            {/* Current Player or Next Player Selection */}
            {auctionStatus.current_player ? (
              <CurrentPlayerCard
                player={auctionStatus.current_player}
                humanAgents={humanAgents}
                selectedAgent={selectedAgent}
                setSelectedAgent={setSelectedAgent}
                bidAmount={bidAmount}
                setBidAmount={setBidAmount}
                onBid={handleBid}
                onProcessBotBids={handleProcessBotBids}
                onFinalize={onFinalize}
                loading={loading}
              />
            ) : (
              <NextPlayerCard
                roleFilter={roleFilter}
                setRoleFilter={setRoleFilter}
                onStartNext={handleStartNext}
                loading={loading}
              />
            )}
          </div>

          {/* Right Column - Agents Status */}
          <div className="space-y-6">
            <AgentsStatusCard agents={auctionStatus.agents} />
          </div>
        </div>
      </div>
    </div>
  );
};
