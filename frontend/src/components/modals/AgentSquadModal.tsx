import React, { useState, useEffect } from 'react';

interface Player {
  name: string;
  role: 'GK' | 'DEF' | 'MID' | 'ATT';
  team: string;
  evaluation: number;
  standardized_evaluation?: number;
  final_cost: number;
}

interface SquadData {
  success: boolean;
  agent_id: string;
  agent_type: string;
  squad_total: Player[];
  squad_by_role: {
    GK: Player[];
    DEF: Player[];
    MID: Player[];
    ATT: Player[];
  };
  metrics: {
    total_evaluation: number;
    total_standardized: number;
    bestxi_evaluation: number;
    bestxi_standardized: number;
    credits_remaining: number;
    credits_spent: number;
    total_players: number;
  };
}

interface AgentSquadModalProps {
  agentId: string;
  agentType: string;
  isOpen: boolean;
  onClose: () => void;
}

const roleNames = {
  GK: '🥅 Portieri',
  DEF: '🛡️ Difensori', 
  MID: '⚽ Centrocampisti',
  ATT: '🎯 Attaccanti'
};

const roleColors = {
  GK: 'bg-yellow-50 border-yellow-200',
  DEF: 'bg-blue-50 border-blue-200',
  MID: 'bg-green-50 border-green-200',
  ATT: 'bg-red-50 border-red-200'
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

export const AgentSquadModal: React.FC<AgentSquadModalProps> = ({
  agentId,
  agentType,
  isOpen,
  onClose
}) => {
  const [squadData, setSquadData] = useState<SquadData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSquadData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`http://localhost:8081/api/agent/${agentId}/squad`);
        const data = await response.json();
        
        if (data.success) {
          setSquadData(data);
        } else {
          setError(data.error || 'Errore nel caricamento della rosa');
        }
      } catch (err) {
        setError('Errore di connessione');
        console.error('Error fetching squad data:', err);
      } finally {
        setLoading(false);
      }
    };

    if (isOpen && agentId) {
      fetchSquadData();
    }
  }, [isOpen, agentId]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden">
        {/* Header - più compatto */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="text-2xl">{getAgentTypeIcon(agentType)}</span>
              <div>
                <h2 className="text-xl font-bold">{agentId}</h2>
                <p className="text-blue-100 text-sm">{agentType.replace('Agent', '')}</p>
              </div>
              {/* Metrics inline nell'header */}
              {squadData && (
                <div className="flex items-center space-x-4 ml-6">
                  <div className="text-center">
                    <div className="text-lg font-bold">{squadData.metrics.total_players}</div>
                    <div className="text-xs text-blue-100">Giocatori</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold">€{squadData.metrics.credits_remaining}</div>
                    <div className="text-xs text-blue-100">Crediti</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold">{Math.round(squadData.metrics.total_evaluation)}</div>
                    <div className="text-xs text-blue-100">Valutazione</div>
                  </div>
                </div>
              )}
            </div>
            <button
              onClick={onClose}
              className="text-white hover:text-gray-300 transition-colors p-1 rounded-full hover:bg-white hover:bg-opacity-20"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content - layout a griglia compatto */}
        <div className="p-4 overflow-y-auto max-h-[calc(90vh-100px)]">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className="ml-2 text-gray-600">Caricamento...</span>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700 text-sm">
              <div className="flex items-center">
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {error}
              </div>
            </div>
          )}

          {squadData && !loading && !error && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Squad by Role - layout compatto */}
              {Object.entries(squadData.squad_by_role).map(([role, players]) => (
                <div key={role} className={`rounded-lg border ${roleColors[role as keyof typeof roleColors]} p-3`}>
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-bold text-gray-800">
                      {roleNames[role as keyof typeof roleNames]}
                    </h3>
                    <span className="text-xs bg-white px-2 py-1 rounded-full font-medium">
                      {players.length}
                    </span>
                  </div>
                  
                  {players.length === 0 ? (
                    <div className="text-gray-500 italic text-center py-2 text-xs">
                      Nessun giocatore
                    </div>
                  ) : (
                    <div className="space-y-1">
                      {players
                        .sort((a, b) => b.evaluation - a.evaluation)
                        .map((player, index) => (
                          <div key={index} className="bg-white rounded p-2 shadow-sm border text-xs">
                            <div className="flex justify-between items-center">
                              <div className="flex-1 min-w-0">
                                <div className="font-medium text-gray-800 truncate">{player.name}</div>
                                <div className="text-gray-500 flex items-center space-x-2">
                                  <span>{player.team}</span>
                                  <span>•</span>
                                  <span>{player.evaluation.toFixed(1)}</span>
                                </div>
                              </div>
                              <div className="text-right ml-2">
                                <div className="font-bold text-green-600">€{player.final_cost}</div>
                              </div>
                            </div>
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Best XI Info compatto in fondo se disponibile */}
          {squadData && squadData.metrics.bestxi_evaluation > 0 && (
            <div className="mt-4 bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-lg">⭐</span>
                  <span className="font-medium text-yellow-800">Best XI</span>
                </div>
                <div className="flex items-center space-x-4 text-sm">
                  <div className="text-center">
                    <div className="font-bold text-yellow-700">{Math.round(squadData.metrics.bestxi_evaluation)}</div>
                    <div className="text-xs text-yellow-600">Valutazione</div>
                  </div>
                  <div className="text-center">
                    <div className="font-bold text-yellow-700">{Math.round(squadData.metrics.bestxi_standardized)}</div>
                    <div className="text-xs text-yellow-600">Standardizzata</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
