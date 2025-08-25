import React, { useState, useEffect, useCallback } from 'react';

interface LogEntry {
  timestamp: number;
  level: string;
  message: string;
  player?: string;
  price?: number;
  highest_bidder?: string;
}

interface CurrentPlayer {
  name: string;
  role: string;
  evaluation: number;
}

interface LiveState {
  state: string;
  current_player?: CurrentPlayer;
  current_price: number;
  highest_bidder?: string;
  waiting_for_human_bid: boolean;
  valid_human_agents: Array<{ id: string; name: string }>;
  logs: LogEntry[];
}

interface LiveAuctionProps {
  auctionId: string;
  isAuctionRunning: boolean;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
}

const LiveAuction: React.FC<LiveAuctionProps> = ({
  auctionId,
  isAuctionRunning,
  onStart,
  onStop,
  onReset,
}) => {
  const [liveState, setLiveState] = useState<LiveState | null>(null);
  const [auctionLogs, setAuctionLogs] = useState<LogEntry[]>([]);
  const [pollingInterval, setPollingInterval] = useState<number | null>(null);
  const [bidAmount, setBidAmount] = useState<string>('');
  const [selectedAgent, setSelectedAgent] = useState<string>('');

  // Cleanup polling interval on unmount
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  const pollAuctionState = useCallback(async () => {
    try {
      const response = await fetch(`http://localhost:8080/api/auction/${auctionId}/live`);
      const data = await response.json();
      
      setLiveState(data);
      
      // If auction is completed, stop polling
      if (data.state === 'completed' || data.state === 'error') {
        onStop();
        if (pollingInterval) {
          clearInterval(pollingInterval);
          setPollingInterval(null);
        }
      }
      
      // Update logs
      if (data.logs) {
        setAuctionLogs(prev => {
          const newLogs = [...prev];
          data.logs.forEach((log: LogEntry) => {
            if (!newLogs.find(l => l.timestamp === log.timestamp)) {
              newLogs.push(log);
            }
          });
          return newLogs.sort((a, b) => a.timestamp - b.timestamp);
        });
      }
      
    } catch (err) {
      console.error('Error polling auction state:', err);
    }
  }, [auctionId, pollingInterval, onStop]);

  const startPolling = useCallback(() => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }
    
    const interval = setInterval(pollAuctionState, 1000); // Poll every second
    setPollingInterval(interval);
  }, [pollingInterval, pollAuctionState]);

  const submitHumanBid = async () => {
    if (!selectedAgent || !bidAmount) return;
    
    try {
      const response = await fetch(`http://localhost:8080/api/auction/${auctionId}/bid`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          agent_id: selectedAgent,
          bid_amount: parseInt(bidAmount)
        }),
      });

      const data = await response.json();
      
      if (data.success) {
        setBidAmount('');
        setSelectedAgent('');
        // Poll immediately to get updated state
        pollAuctionState();
      } else {
        console.error(data.error || 'Errore nell\'invio dell\'offerta');
      }
    } catch (err) {
      console.error('Errore nell\'invio dell\'offerta', err);
    }
  };

  const submitHumanPass = async () => {
    try {
      const response = await fetch(`http://localhost:8080/api/auction/${auctionId}/pass`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();
      
      if (data.success) {
        setBidAmount('');
        setSelectedAgent('');
        // Poll immediately to get updated state
        pollAuctionState();
      } else {
        console.error(data.error || 'Errore nell\'invio del pass');
      }
    } catch (err) {
      console.error('Errore nell\'invio del pass', err);
    }
  };

  // Start polling when auction is running
  useEffect(() => {
    if (isAuctionRunning && !pollingInterval) {
      startPolling();
    } else if (!isAuctionRunning && pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
  }, [isAuctionRunning, pollingInterval, startPolling]);

  const handleReset = () => {
    setLiveState(null);
    setAuctionLogs([]);
    setBidAmount('');
    setSelectedAgent('');
    if (pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
    onReset();
  };

  // Auto-scroll logs to bottom - ENABLED for logs container only
  const logsEndRef = React.useRef<HTMLDivElement>(null);
  const logsContainerRef = React.useRef<HTMLDivElement>(null);
  
  React.useEffect(() => {
    // Scroll only the logs container, not the whole page
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [auctionLogs]);

  const getStateColor = (state: string) => {
    const colors: { [key: string]: string } = {
      'running': 'bg-green-100 text-green-800',
      'waiting_for_human': 'bg-yellow-100 text-yellow-800',
      'completed': 'bg-blue-100 text-blue-800',
      'error': 'bg-red-100 text-red-800'
    };
    return colors[state] || 'bg-gray-100 text-gray-800';
  };

  const getLogIcon = (message: string) => {
    if (message.includes('💰')) return '💰';
    if (message.includes('❌')) return '❌';
    if (message.includes('🏆')) return '🏆';
    if (message.includes('✅')) return '✅';
    if (message.includes('🏈')) return '🏈';
    if (message.includes('⏳')) return '⏳';
    if (message.includes('🚀')) return '🚀';
    if (message.includes('🏁')) return '🏁';
    return '📝';
  };

  const getAgentName = (agentId: string | number) => {
    // Converte sempre a stringa
    const agentIdStr = String(agentId);
    
    // Se abbiamo la lista degli agenti umani validi, cerchiamo il nome
    if (liveState?.valid_human_agents) {
      const agent = liveState.valid_human_agents.find(a => a.id === agentIdStr);
      if (agent) return agent.name;
    }
    
    // Fallback: usa direttamente l'ID come nome
    // Se è un numero, lo ritorna come stringa
    // Se contiene underscore, li sostituisce con spazi e capitalizza
    if (/^\d+$/.test(agentIdStr)) {
      return `Agente ${agentIdStr}`;
    }
    
    return agentIdStr
      .replace(/^(agent_|human_)/i, '') // Rimuovi prefissi
      .replace(/_/g, ' ') // Sostituisci underscore con spazi
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1)) // Capitalizza
      .join(' ');
  };

  return (
    <div className=" text-black min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">🏆 Asta Live</h1>
              <p className="text-sm text-gray-700 font-mono">{auctionId}</p>
            </div>
            <div className="flex items-center space-x-3">
              {liveState && (
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStateColor(liveState.state)}`}>
                  {liveState.state}
                </span>
              )}
              {!isAuctionRunning ? (
                <button
                  onClick={onStart}
                  className="text-black px-4 py-2 bg-green-600  rounded-lg hover:bg-green-700 transition-colors"
                >
                  🚀 Avvia Asta
                </button>
              ) : (
                <button
                  onClick={onStop}
                  className="px-4 py-2 text-black bg-red-600 rounded-lg hover:bg-red-700 transition-colors"
                >
                  ⏹️ Stop
                </button>
              )}
              <button
                onClick={handleReset}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                🔄 Reset
              </button>
            </div>
          </div>
        </div>

        <div className="flex flex-col lg:flex-row gap-8 min-h-[calc(100vh-12rem)]">
          {/* Left Column - Main Auction Info */}
          <div className="flex-1 lg:w-1/2 space-y-6">
            {/* Current Player Card */}
            {liveState?.current_player && (
              <div className="bg-gradient-to-br from-white to-blue-50 rounded-xl shadow-lg border border-blue-200 overflow-hidden">
                <div className="bg-gradient-to-r from-blue-600 to-blue-700 p-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-xl font-bold text-white">🏈 Giocatore in Asta</h2>
                    <span className="px-4 py-2 rounded-full text-sm font-bold bg-white shadow-sm text-gray-800">
                      {liveState.current_player.role}
                    </span>
                  </div>
                </div>
                
                <div className="p-6">
                  <div className="text-center mb-6">
                    <h3 className="text-3xl font-bold text-gray-900 mb-2">{liveState.current_player.name}</h3>
                    <div className="inline-flex items-center px-4 py-2 bg-yellow-100 rounded-full">
                      <span className="text-sm text-yellow-800 font-medium">
                        ⭐ Valutazione: {liveState.current_player.evaluation}
                      </span>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="bg-gradient-to-br from-green-50 to-green-100 p-5 rounded-xl border border-green-200">
                      <div className="text-center">
                        <p className="text-sm text-green-700 font-medium mb-1">💰 Prezzo Attuale</p>
                        <p className="text-3xl font-bold text-green-600">€{liveState.current_price}</p>
                      </div>
                    </div>
                    {liveState.highest_bidder && (
                      <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-5 rounded-xl border border-purple-200">
                        <div className="text-center">
                          <p className="text-sm text-purple-700 font-medium mb-1">🏆 Miglior Offerente</p>
                          <p className="text-lg font-bold text-purple-600 truncate">{getAgentName(liveState.highest_bidder)}</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Bidding Section */}
            {liveState?.waiting_for_human_bid && (
              <div className="bg-gradient-to-r from-amber-50 via-yellow-50 to-orange-50 rounded-xl shadow-lg border-2 border-amber-300 overflow-hidden">
                <div className="bg-gradient-to-r from-amber-500 to-orange-500 p-4">
                  <h3 className="text-xl font-bold text-white flex items-center">
                    💰 È il tuo turno! Fai la tua offerta
                  </h3>
                </div>
                
                <div className="p-6">
                  <div className="mb-6">
                    <p className="text-sm font-semibold text-gray-700 mb-3">Agenti disponibili per l'offerta:</p>
                    <div className="flex flex-wrap gap-2">
                      {liveState.valid_human_agents.map(agent => (
                        <button 
                          key={agent.id} 
                          onClick={() => setSelectedAgent(agent.id)}
                          className={`px-4 py-2 rounded-full text-sm font-medium border transition-all duration-200 hover:scale-105 hover:shadow-md ${
                            selectedAgent === agent.id 
                              ? 'bg-blue-600 text-black border-blue-600 shadow-lg' 
                              : 'bg-blue-100 text-blue-800 border-blue-200 hover:bg-blue-200'
                          }`}
                        >
                          {agent.name}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Seleziona Agente</label>
                        <select 
                          value={selectedAgent} 
                          onChange={(e) => setSelectedAgent(e.target.value)}
                          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent bg-white shadow-sm"
                        >
                          <option value="">Scegli il tuo agente...</option>
                          {liveState.valid_human_agents.map(agent => (
                            <option key={agent.id} value={agent.id}>{agent.name}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Importo Offerta</label>
                        <input
                          type="number"
                          value={bidAmount}
                          onChange={(e) => setBidAmount(e.target.value)}
                          placeholder={`Minimo: €${liveState.current_price + 1}`}
                          min={liveState.current_price + 1}
                          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent shadow-sm"
                        />
                      </div>
                    </div>
                    
                    <div className="flex flex-col sm:flex-row gap-3 pt-2">
                      <button 
                        onClick={submitHumanBid}
                        disabled={!selectedAgent || !bidAmount || parseInt(bidAmount) <= liveState.current_price}
                        className="flex-1 px-6 py-3 bg-gradient-to-r from-green-600 to-green-700 text-white rounded-lg hover:from-green-700 hover:to-green-800 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-200 font-semibold shadow-lg"
                      >
                        🎯 Offri €{bidAmount || '---'}
                      </button>
                      
                      <button 
                        onClick={submitHumanPass}
                        className="px-6 py-3 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-lg hover:from-gray-700 hover:to-gray-800 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-200 font-semibold shadow-lg"
                      >
                        ⏭️ Passa
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Status when not waiting for human input */}
            {liveState && !liveState.waiting_for_human_bid && liveState.current_player && (
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border-2 border-blue-300 p-6 shadow-lg">
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-3 border-blue-600 mr-4"></div>
                  <div className="text-center">
                    <p className="text-blue-800 font-bold text-lg">🤖 Agenti in azione...</p>
                    <p className="text-blue-600 text-sm mt-1">Gli agenti automatici stanno valutando le loro offerte</p>
                  </div>
                </div>
              </div>
            )}

            {/* Quick Stats Summary */}
            {liveState && (
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                  📊 Statistiche Asta
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">Stato</p>
                    <p className="font-bold text-gray-900 capitalize">{liveState.state}</p>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">Log Eventi</p>
                    <p className="font-bold text-gray-900">{auctionLogs.length}</p>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">Agenti Umani</p>
                    <p className="font-bold text-gray-900">{liveState.valid_human_agents?.length || 0}</p>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">Attesa Input</p>
                    <p className="font-bold text-gray-900">{liveState.waiting_for_human_bid ? '✅' : '❌'}</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Logs */}
          <div className="flex-1 lg:w-1/2">
            <div className="h-min bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden h-full">
              <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-4">
                <h3 className="text-xl font-bold text-white flex items-center">
                  📋 Log Live dell'Asta
                  <span className="ml-auto bg-white text-black bg-opacity-20 px-3 py-1 rounded-full text-sm">
                    {auctionLogs.length} eventi
                  </span>
                </h3>
              </div>
              
              <div ref={logsContainerRef} className="h-[600px] overflow-y-auto">
                {auctionLogs.length === 0 ? (
                  <div className="flex items-center justify-center h-full text-gray-700 p-8">
                    <div className="text-center">
                      <div className="text-6xl mb-4">🏁</div>
                      <p className="text-lg font-medium">In attesa di eventi...</p>
                      <p className="text-sm mt-2">I log dell'asta appariranno qui in tempo reale</p>
                    </div>
                  </div>
                ) : (
                  <div className="p-4 space-y-3">
                    {auctionLogs.map((log, index) => (
                      <div key={index} className="group hover:bg-gray-50 transition-colors duration-200 rounded-lg p-3 border border-gray-100">
                        <div className="flex items-start space-x-3">
                          <div className="flex-shrink-0 text-2xl group-hover:scale-110 transition-transform duration-200">
                            {getLogIcon(log.message)}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs font-medium text-gray-700 bg-gray-100 px-2 py-1 rounded">
                                {new Date(log.timestamp * 1000).toLocaleTimeString('it-IT')}
                              </span>
                              {log.level && (
                                <span className={`text-xs font-bold px-2 py-1 rounded ${
                                  log.level === 'ERROR' ? 'bg-red-100 text-red-800' :
                                  log.level === 'WARNING' ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-blue-100 text-blue-800'
                                }`}>
                                  {log.level}
                                </span>
                              )}
                            </div>
                            <p className="text-sm text-gray-900 leading-relaxed break-words">
                              {log.message.replace(/[🏆💰❌✅🏈⏳🚀🏁📝]/gu, '').trim()}
                            </p>
                            {(log.player || log.price) && (
                              <div className="mt-2 flex items-center space-x-4 text-xs text-gray-600">
                                {log.player && (
                                  <span className="flex items-center">
                                    <span className="w-2 h-2 bg-blue-500 rounded-full mr-1"></span>
                                    {log.player}
                                  </span>
                                )}
                                {log.price && (
                                  <span className="flex items-center">
                                    <span className="w-2 h-2 bg-green-500 rounded-full mr-1"></span>
                                    €{log.price}
                                  </span>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                    <div ref={logsEndRef} />
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveAuction;
