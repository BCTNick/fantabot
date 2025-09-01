import React, { useState, useEffect } from 'react';
import { Card, Button } from '../components/UI';
import { apiClient } from '../services/api';
import type { AuctionResultsResponse, AuctionResult } from '../types';

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

interface ResultsPageProps {
  onNewAuction: () => void;
}

export const ResultsPage: React.FC<ResultsPageProps> = ({ onNewAuction }) => {
  const [results, setResults] = useState<AuctionResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

  useEffect(() => {
    loadResults();
  }, []);

  const loadResults = async () => {
    try {
      setLoading(true);
      const response: AuctionResultsResponse = await apiClient.getAuctionResults();
      if (response.success) {
        setResults(response.results);
        setError(null);
      } else {
        setError('Errore nel caricamento dei risultati');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore sconosciuto');
    } finally {
      setLoading(false);
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

  const getAgentTypeIcon = (agentId: string) => {
    if (agentId.toLowerCase().includes('player') || agentId.toLowerCase().includes('giocatore')) {
      return '👤';
    }
    return '🤖';
  };

  const sortedResults = [...results].sort((a, b) => 
    b.metrics.bestxi_standardized - a.metrics.bestxi_standardized
  );

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center p-8">
        <div className="text-red-600 mb-4">❌ {error}</div>
        <Button onClick={loadResults} variant="primary">
          🔄 Riprova
        </Button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-800 mb-2">
          🏆 Risultati Finali dell'Asta
        </h2>
        <p className="text-gray-600">
          Ecco come sono andate le squadre partecipanti
        </p>
      </div>

      {/* Rankings */}
      <Card title="🥇 Classifica per Miglior 11">
        <div className="space-y-4">
          {sortedResults.map((result, index) => (
            <div
              key={result.agent_id}
              className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                selectedAgent === result.agent_id
                  ? 'border-accent bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              } ${index === 0 ? 'bg-gradient-to-r from-yellow-50 to-yellow-100 border-yellow-300' : ''}`}
              onClick={() => setSelectedAgent(selectedAgent === result.agent_id ? null : result.agent_id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="text-2xl font-bold text-gray-500">
                    #{index + 1}
                  </div>
                  <div className="text-2xl">
                    {getAgentTypeIcon(result.agent_id)}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800">
                      {result.agent_id}
                      {index === 0 && <span className="ml-2">🏆</span>}
                    </h3>
                    <div className="flex space-x-4 text-sm text-gray-600">
                      <span>Giocatori: {result.squad.length}</span>
                      <span>Spesi: €{result.metrics.credits_spent}</span>
                      <span>Rimasti: €{result.metrics.credits_remaining}</span>
                    </div>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-xl font-bold text-accent">
                    {result.metrics.bestxi_standardized.toFixed(3)}
                  </div>
                  <div className="text-sm text-gray-500">Best XI Score</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Detailed Squad View */}
      {selectedAgent && (
        <Card title={`👥 Rosa di ${selectedAgent}`}>
          {(() => {
            const agentResult = results.find(r => r.agent_id === selectedAgent);
            if (!agentResult) return null;

            const squadByRole = {
              GK: agentResult.squad.filter(p => p.role === 'GK'),
              DEF: agentResult.squad.filter(p => p.role === 'DEF'),
              MID: agentResult.squad.filter(p => p.role === 'MID'),
              ATT: agentResult.squad.filter(p => p.role === 'ATT'),
            };

            return (
              <div className="space-y-6">
                {/* Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-blue-50 p-4 rounded text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {agentResult.metrics.total_evaluation.toFixed(0)}
                    </div>
                    <div className="text-sm text-blue-800">Valutazione Totale</div>
                  </div>
                  <div className="bg-green-50 p-4 rounded text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {agentResult.metrics.bestxi_evaluation.toFixed(0)}
                    </div>
                    <div className="text-sm text-green-800">Best XI</div>
                  </div>
                  <div className="bg-yellow-50 p-4 rounded text-center">
                    <div className="text-2xl font-bold text-yellow-600">
                      €{agentResult.metrics.credits_spent}
                    </div>
                    <div className="text-sm text-yellow-800">Spesi</div>
                  </div>
                  <div className="bg-purple-50 p-4 rounded text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      €{agentResult.metrics.credits_remaining}
                    </div>
                    <div className="text-sm text-purple-800">Rimasti</div>
                  </div>
                </div>

                {/* Squad by Role */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {(['GK', 'DEF', 'MID', 'ATT'] as const).map((role) => (
                    <div key={role} className="space-y-3">
                      <h4 className="text-lg font-semibold text-gray-800 flex items-center">
                        {getRoleIcon(role)} {ROLE_NAMES[role]} ({squadByRole[role].length})
                      </h4>
                      <div className="space-y-2">
                        {squadByRole[role]
                          .sort((a, b) => b.evaluation - a.evaluation)
                          .map((player, idx) => (
                          <div
                            key={`${player.name}-${idx}`}
                            className={`p-3 rounded border ${ROLE_COLORS[role]}`}
                          >
                            <div className="font-medium">{player.name}</div>
                            <div className="text-sm opacity-75">{player.team}</div>
                            <div className="flex justify-between text-sm">
                              <span>Val: {player.evaluation}</span>
                              <span>€{player.final_cost}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}
        </Card>
      )}

      {/* Actions */}
      <div className="flex justify-center space-x-4">
        <Button onClick={loadResults} variant="secondary">
          🔄 Aggiorna Risultati
        </Button>
        <Button onClick={onNewAuction} variant="primary" size="lg">
          🚀 Nuova Asta
        </Button>
      </div>
    </div>
  );
};
