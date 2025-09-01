import React, { useState } from 'react';
import { Card, Button, Input } from '../components/UI';
import { BotBidsDisplay } from '../components/BotBidsDisplay';
import type { AuctionStatus, BotBidResult, BotBidsResponse } from '../types';

interface AuctionPageProps {
  auctionStatus: AuctionStatus;
  onStartNextPlayer: (roleFilter?: string) => Promise<unknown>;
  onMakeBid: (agentId: string, amount: number) => Promise<unknown>;
  onProcessBotBids: () => Promise<BotBidsResponse>;
  onFinalize: () => Promise<unknown>;
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

export const AuctionPage: React.FC<AuctionPageProps> = ({
  auctionStatus,
  onStartNextPlayer,
  onMakeBid,
  onProcessBotBids,
  onFinalize,
  loading,
}) => {
  const [bidAmount, setBidAmount] = useState<number>(0);
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [roleFilter, setRoleFilter] = useState<string>('');
  const [lastBotBids, setLastBotBids] = useState<BotBidResult[]>([]);

  const handleStartNext = async () => {
    setLastBotBids([]); // Reset bot bids when starting new player
    await onStartNextPlayer(roleFilter || undefined);
  };

  const handleBid = async (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedAgent && bidAmount > 0) {
      await onMakeBid(selectedAgent, bidAmount);
      setBidAmount(auctionStatus.current_player?.current_price || 0);
    }
  };

  const handleProcessBotBids = async () => {
    const result = await onProcessBotBids();
    if (result && result.success && result.bids) {
      setLastBotBids(result.bids);
    }
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

  const getAgentTypeIcon = (type: string) => {
    switch (type) {
      case 'HumanAgent': return '👤';
      case 'CapAgent': return '🤖';
      case 'DynamicCapAgent': return '🧠';
      case 'RandomAgent': return '🎲';
      case 'RLDeepAgent': return '🤯';
      default: return '🤖';
    }
  };

  const humanAgents = auctionStatus.agents.filter(a => a.type === 'HumanAgent');

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header Status */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-800 mb-2">
          🏆 Asta in Corso
        </h2>
        <div className="inline-flex items-center px-4 py-2 rounded-full bg-green-100 text-green-800">
          <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
          Stato: {auctionStatus.state === 'player_auction' ? 'Asta Giocatore' : 'In Attesa'}
        </div>
      </div>

      {/* Bot Bids Display */}
      <BotBidsDisplay bids={lastBotBids} show={lastBotBids.length > 0} />

      {/* Current Player Card */}
      {auctionStatus.current_player ? (
        <Card title="🔥 Giocatore in Asta">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="text-center">
                <div className="text-6xl mb-2">
                  {getRoleIcon(auctionStatus.current_player.role)}
                </div>
                <h3 className="text-2xl font-bold text-gray-800">
                  {auctionStatus.current_player.name}
                </h3>
                <p className="text-gray-600">{auctionStatus.current_player.team}</p>
                <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium border ${ROLE_COLORS[auctionStatus.current_player.role]}`}>
                  {ROLE_NAMES[auctionStatus.current_player.role]}
                </span>
              </div>
              
              <div className="grid grid-cols-2 gap-4 text-center">
                <div className="bg-blue-50 p-3 rounded">
                  <div className="text-xl font-bold text-blue-600">
                    {auctionStatus.current_player.evaluation}
                  </div>
                  <div className="text-sm text-blue-800">Valutazione</div>
                </div>
                <div className="bg-green-50 p-3 rounded">
                  <div className="text-xl font-bold text-green-600">
                    €{auctionStatus.current_player.current_price}
                  </div>
                  <div className="text-sm text-green-800">Offerta Attuale</div>
                </div>
              </div>
              
              {auctionStatus.current_player.highest_bidder && (
                <div className="text-center p-3 bg-yellow-50 rounded border border-yellow-200">
                  <span className="text-yellow-800 font-medium">
                    🏆 Miglior offerta di: {auctionStatus.current_player.highest_bidder}
                  </span>
                </div>
              )}
            </div>

            {/* Bid Form */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800">💰 Fai un'offerta</h4>
              
              {humanAgents.length > 0 ? (
                <form onSubmit={handleBid} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Seleziona Agente
                    </label>
                    <select
                      value={selectedAgent}
                      onChange={(e) => setSelectedAgent(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-accent focus:border-accent"
                      required
                    >
                      <option value="">Scegli un agente...</option>
                      {humanAgents.map((agent) => (
                        <option key={agent.id} value={agent.id}>
                          👤 {agent.id} (€{agent.credits})
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  <Input
                    label="Importo Offerta (€)"
                    type="number"
                    min={auctionStatus.current_player.current_price + 1}
                    value={bidAmount}
                    onChange={(e) => setBidAmount(parseInt(e.target.value))}
                    placeholder="Inserisci importo..."
                    required
                  />
                  
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      onClick={() => setBidAmount(auctionStatus.current_player!.current_price + 5)}
                    >
                      +5€
                    </Button>
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      onClick={() => setBidAmount(auctionStatus.current_player!.current_price + 10)}
                    >
                      +10€
                    </Button>
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      onClick={() => setBidAmount(auctionStatus.current_player!.current_price + 25)}
                    >
                      +25€
                    </Button>
                  </div>
                  
                  <Button
                    type="submit"
                    variant="success"
                    className="w-full"
                    loading={loading}
                    disabled={!selectedAgent || bidAmount <= auctionStatus.current_player.current_price}
                  >
                    🚀 Fai Offerta
                  </Button>
                </form>
              ) : (
                <div className="text-center p-4 bg-gray-50 rounded">
                  <p className="text-gray-600">Nessun giocatore umano per fare offerte</p>
                  <p className="text-sm text-gray-500">Solo bot partecipano a questa asta</p>
                </div>
              )}
              
              <div className="grid grid-cols-1 gap-2">
                <Button
                  type="button"
                  variant="primary"
                  className="w-full"
                  onClick={handleProcessBotBids}
                  loading={loading}
                >
                  🤖 Elabora Offerte Bot
                </Button>
                
                <Button
                  type="button"
                  variant="warning"
                  className="w-full"
                  onClick={onFinalize}
                  loading={loading}
                >
                  ✅ Finalizza Asta
                </Button>
              </div>
            </div>
          </div>
        </Card>
      ) : (
        /* No Current Player */
        <Card title="🎯 Controlla Prossimo Giocatore">
          <div className="text-center space-y-4">
            <p className="text-gray-600">Nessun giocatore in asta al momento</p>
            
            <div className="flex flex-wrap justify-center gap-2 mb-4">
              <button
                onClick={() => setRoleFilter('')}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                  roleFilter === '' 
                    ? 'bg-accent text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Tutti
              </button>
              {(['GK', 'DEF', 'MID', 'ATT'] as const).map((role) => (
                <button
                  key={role}
                  onClick={() => setRoleFilter(role)}
                  className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                    roleFilter === role 
                      ? 'bg-accent text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  {getRoleIcon(role)} {ROLE_NAMES[role]}
                </button>
              ))}
            </div>
            
            <Button
              variant="primary"
              size="lg"
              onClick={handleStartNext}
              loading={loading}
            >
              🎲 Inizia Prossimo Giocatore
              {roleFilter && ` (${ROLE_NAMES[roleFilter as keyof typeof ROLE_NAMES]})`}
            </Button>
          </div>
        </Card>
      )}

      {/* Agents Status */}
      <Card title="👥 Stato Partecipanti">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {auctionStatus.agents.map((agent) => (
            <div key={agent.id} className="p-4 bg-gray-50 rounded-lg border">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center">
                  <span className="text-xl mr-2">
                    {getAgentTypeIcon(agent.type)}
                  </span>
                  <span className="font-semibold text-gray-800">{agent.id}</span>
                </div>
                <span className="text-sm text-gray-500">{agent.type.replace('Agent', '')}</span>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Crediti:</span>
                  <span className="font-medium text-green-600">€{agent.credits}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Giocatori:</span>
                  <span className="font-medium">{agent.squad_size}</span>
                </div>
                <div className="grid grid-cols-4 gap-1 text-xs">
                  <div className="text-center">
                    <div className="font-medium">{agent.squad_gk}</div>
                    <div className="text-gray-500">GK</div>
                  </div>
                  <div className="text-center">
                    <div className="font-medium">{agent.squad_def}</div>
                    <div className="text-gray-500">DEF</div>
                  </div>
                  <div className="text-center">
                    <div className="font-medium">{agent.squad_mid}</div>
                    <div className="text-gray-500">MID</div>
                  </div>
                  <div className="text-center">
                    <div className="font-medium">{agent.squad_att}</div>
                    <div className="text-gray-500">ATT</div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};
