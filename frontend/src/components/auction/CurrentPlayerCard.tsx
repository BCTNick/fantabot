import React, { useState } from 'react';
import { Card, Button, Input } from '../UI';
import type { CurrentPlayer, Agent } from '../../types';

interface CurrentPlayerCardProps {
  player: CurrentPlayer;
  humanAgents: Agent[];
  selectedAgent: string;
  setSelectedAgent: (agent: string) => void;
  bidAmount: number;
  setBidAmount: (amount: number) => void;
  onBid: (e: React.FormEvent) => Promise<void>;
  onProcessBotBids: () => void;
  onFinalize: () => void;
  loading: boolean;
}

const ROLE_COLORS = {
  GK: 'bg-yellow-100 text-yellow-800 border-yellow-300',
  DEF: 'bg-blue-100 text-blue-800 border-blue-300',
  MID: 'bg-green-100 text-green-800 border-green-300',
  ATT: 'bg-red-100 text-red-800 border-red-300',
};

const ROLE_NAMES = {
  GK: 'Portiere',
  DEF: 'Difensore', 
  MID: 'Centrocampista',
  ATT: 'Attaccante',
};

const getRoleIcon = (role: string) => {
  switch (role) {
    case 'GK': return '🥅';
    case 'DEF': return '🛡️';
    case 'MID': return '⚽';
    case 'ATT': return '🎯';
    default: return '👤';
  }
};

export const CurrentPlayerCard: React.FC<CurrentPlayerCardProps> = ({
  player,
  humanAgents,
  selectedAgent,
  setSelectedAgent,
  bidAmount,
  setBidAmount,
  onBid,
  onProcessBotBids,
  onFinalize,
  loading,
}) => {
  const [processingBots, setProcessingBots] = useState(false);

  const handleBid = async (e: React.FormEvent) => {
    setProcessingBots(true);
    try {
      await onBid(e);
    } finally {
      setProcessingBots(false);
    }
  };

  const isCurrentlyBidding = loading || processingBots;

  return (
    <Card>
      {/* Player Header */}
      <div className="text-center mb-6">
        <div className="text-8xl mb-4">
          {getRoleIcon(player.role)}
        </div>
        <h3 className="text-3xl font-bold text-gray-800 mb-2">
          {player.name}
        </h3>
        <p className="text-lg text-gray-600 mb-3">{player.team}</p>
        <span className={`inline-block px-4 py-2 rounded-full text-sm font-medium border ${ROLE_COLORS[player.role as keyof typeof ROLE_COLORS]}`}>
          {ROLE_NAMES[player.role as keyof typeof ROLE_NAMES]}
        </span>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg text-center border border-blue-200">
          <div className="text-2xl font-bold text-blue-600 mb-1">
            {player.evaluation}
          </div>
          <div className="text-sm font-medium text-blue-800">Valutazione</div>
        </div>
        <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg text-center border border-green-200">
          <div className="text-2xl font-bold text-green-600 mb-1">
            €{player.current_price}
          </div>
          <div className="text-sm font-medium text-green-800">Offerta Attuale</div>
        </div>
      </div>

      {/* Current Bid Info */}
      {player.highest_bidder && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-yellow-800">Offerta più alta:</div>
              <div className="text-lg font-bold text-yellow-900">
                {player.highest_bidder} - €{player.current_price}
              </div>
            </div>
            <div className="text-3xl">🔥</div>
          </div>
        </div>
      )}

      {/* Bidding Form */}
      <div className="space-y-4">
        <h4 className="text-lg font-semibold text-gray-700">Fai un'Offerta</h4>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Seleziona il tuo Agente
          </label>
          <select
            value={selectedAgent}
            onChange={(e) => setSelectedAgent(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            disabled={isCurrentlyBidding}
          >
            <option value="">Scegli un agente...</option>
            {humanAgents.map((agent) => (
              <option key={agent.id} value={agent.id}>
                {agent.id} (Crediti: €{agent.credits})
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Importo Offerta
          </label>
          <Input
            type="number"
            value={bidAmount}
            onChange={(e) => setBidAmount(Number(e.target.value))}
            min={player.current_price + 1}
            placeholder={`Minimo €${player.current_price + 1}`}
            disabled={isCurrentlyBidding}
            className="text-lg"
          />
        </div>

        <form onSubmit={handleBid} className="space-y-4">
          <Button 
            type="submit" 
            disabled={isCurrentlyBidding || !selectedAgent || bidAmount <= player.current_price}
            className="w-full py-3 text-lg font-semibold"
          >
            {processingBots ? (
              <div className="flex items-center justify-center space-x-2">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                <span>Elaborando offerte bot...</span>
              </div>
            ) : loading ? 'Caricamento...' : 'Fai Offerta'}
          </Button>
        </form>

        <div className="border-t pt-4 space-y-2">
          <Button 
            onClick={onProcessBotBids} 
            disabled={isCurrentlyBidding}
            variant="secondary"
            className="w-full"
          >
            {loading ? 'Elaborando...' : 'Elabora Offerte Bot'}
          </Button>
          
          <Button 
            onClick={onFinalize} 
            disabled={isCurrentlyBidding}
            variant="secondary"
            className="w-full"
          >
            {loading ? 'Finalizzando...' : 'Finalizza Aggiudicazione'}
          </Button>
        </div>

        {processingBots && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
              <div>
                <div className="text-sm font-medium text-blue-700">
                  Elaborazione in corso...
                </div>
                <div className="text-xs text-blue-600">
                  Gli agenti bot stanno valutando le loro offerte
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};
