import React from 'react';
import { Card } from '../UI';
import type { Agent } from '../../types';

interface AgentsStatusCardProps {
  agents: Agent[];
}

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

const getAgentTypeColor = (type: string) => {
  switch (type) {
    case 'HumanAgent': return 'bg-blue-50 border-blue-200 text-blue-800';
    case 'CapAgent': return 'bg-gray-50 border-gray-200 text-gray-800';
    case 'DynamicCapAgent': return 'bg-purple-50 border-purple-200 text-purple-800';
    case 'RandomAgent': return 'bg-yellow-50 border-yellow-200 text-yellow-800';
    case 'RLDeepAgent': return 'bg-red-50 border-red-200 text-red-800';
    default: return 'bg-gray-50 border-gray-200 text-gray-800';
  }
};

const getCreditsColor = (credits: number, total: number = 1000) => {
  const percentage = (credits / total) * 100;
  if (percentage > 75) return 'text-green-600 bg-green-50';
  if (percentage > 50) return 'text-yellow-600 bg-yellow-50';
  if (percentage > 25) return 'text-orange-600 bg-orange-50';
  return 'text-red-600 bg-red-50';
};

export const AgentsStatusCard: React.FC<AgentsStatusCardProps> = ({ agents }) => {
  // Ordina gli agenti per crediti rimanenti (dal più alto al più basso)
  const sortedAgents = [...agents].sort((a, b) => b.credits - a.credits);

  return (
    <Card title="👥 Stato Partecipanti">
      <div className="space-y-4">
        {sortedAgents.map((agent, index) => {
          const totalSlots = agent.squad_gk + agent.squad_def + agent.squad_mid + agent.squad_att;
          const isTopPerformer = index === 0;
          
          return (
            <div 
              key={agent.id} 
              className={`p-4 rounded-lg border-2 transition-all duration-200 hover:shadow-md ${
                getAgentTypeColor(agent.type)
              } ${isTopPerformer ? 'ring-2 ring-yellow-400' : ''}`}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center">
                  <span className="text-2xl mr-3">
                    {getAgentTypeIcon(agent.type)}
                    {isTopPerformer && <span className="ml-1">👑</span>}
                  </span>
                  <div>
                    <span className="font-bold text-lg">{agent.id}</span>
                    <div className="text-xs opacity-75">
                      {agent.type.replace('Agent', '')}
                    </div>
                  </div>
                </div>
                
                {/* Rank Badge */}
                <div className="text-right">
                  <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-bold ${
                    isTopPerformer 
                      ? 'bg-yellow-100 text-yellow-800' 
                      : 'bg-gray-100 text-gray-600'
                  }`}>
                    #{index + 1}
                  </div>
                </div>
              </div>
              
              {/* Credits */}
              <div className="mb-3">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium">💰 Crediti</span>
                  <span className={`font-bold text-lg px-2 py-1 rounded ${
                    getCreditsColor(agent.credits)
                  }`}>
                    €{agent.credits}
                  </span>
                </div>
                
                {/* Credits Bar */}
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(agent.credits / 1000) * 100}%` }}
                  ></div>
                </div>
              </div>
              
              {/* Squad Info */}
              <div className="grid grid-cols-2 gap-3 mb-3">
                <div className="text-center">
                  <div className="text-lg font-bold">{totalSlots}</div>
                  <div className="text-xs text-gray-600">Giocatori Totali</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold">{1000 - agent.credits}</div>
                  <div className="text-xs text-gray-600">€ Spesi</div>
                </div>
              </div>
              
              {/* Squad Composition */}
              <div className="grid grid-cols-4 gap-1">
                <div className="text-center p-2 bg-yellow-100 rounded text-xs">
                  <div className="font-bold text-yellow-800">{agent.squad_gk}</div>
                  <div className="text-yellow-600">🥅 GK</div>
                </div>
                <div className="text-center p-2 bg-blue-100 rounded text-xs">
                  <div className="font-bold text-blue-800">{agent.squad_def}</div>
                  <div className="text-blue-600">🛡️ DEF</div>
                </div>
                <div className="text-center p-2 bg-green-100 rounded text-xs">
                  <div className="font-bold text-green-800">{agent.squad_mid}</div>
                  <div className="text-green-600">⚽ MID</div>
                </div>
                <div className="text-center p-2 bg-red-100 rounded text-xs">
                  <div className="font-bold text-red-800">{agent.squad_att}</div>
                  <div className="text-red-600">🎯 ATT</div>
                </div>
              </div>
            </div>
          );
        })}
        
        {/* Summary */}
        <div className="mt-6 p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg border">
          <h4 className="font-bold text-gray-800 mb-2">📊 Riepilogo</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Partecipanti:</span>
              <span className="font-bold ml-2">{agents.length}</span>
            </div>
            <div>
              <span className="text-gray-600">Crediti medi:</span>
              <span className="font-bold ml-2">
                €{Math.round(agents.reduce((sum, agent) => sum + agent.credits, 0) / agents.length)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};
