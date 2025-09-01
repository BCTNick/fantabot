import React, { useState } from 'react';
import { Card, Button, Input, Select } from '../components/UI';
import type { AgentConfig, AuctionConfig, CreateAuctionRequest } from '../types';

interface CreateAuctionPageProps {
  onAuctionCreated: (config: CreateAuctionRequest) => Promise<boolean>;
  loading: boolean;
}

const agentTypeOptions = [
  { value: 'human', label: 'Giocatore Umano' },
  { value: 'cap', label: 'Bot Strategico' },
  { value: 'dynamic_cap', label: 'Bot Dinamico' },
  { value: 'random', label: 'Bot Casuale' },
  { value: 'rl_deep', label: 'Bot AI Avanzato' },
];

export const CreateAuctionPage: React.FC<CreateAuctionPageProps> = ({ 
  onAuctionCreated, 
  loading 
}) => {
  const [agents, setAgents] = useState<AgentConfig[]>([
    { type: 'human', id: 'giocatore1' }
  ]);
  
  const [config, setConfig] = useState<AuctionConfig>({
    initial_credits: 1000,
    slots_gk: 3,
    slots_def: 8,
    slots_mid: 8,
    slots_att: 6,
  });

  const addAgent = () => {
    const newAgent: AgentConfig = {
      type: 'cap',
      id: `bot${agents.length + 1}`,
    };
    setAgents([...agents, newAgent]);
  };

  const removeAgent = (index: number) => {
    if (agents.length > 1) {
      setAgents(agents.filter((_, i) => i !== index));
    }
  };

  const updateAgent = (index: number, field: keyof AgentConfig, value: string) => {
    const updatedAgents = [...agents];
    updatedAgents[index] = { ...updatedAgents[index], [field]: value };
    setAgents(updatedAgents);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const auctionRequest: CreateAuctionRequest = {
      agents,
      config,
    };
    
    await onAuctionCreated(auctionRequest);
  };

  const getRoleEmoji = (type: string) => {
    switch (type) {
      case 'human': return '👤';
      case 'cap': return '🤖';
      case 'dynamic_cap': return '🧠';
      case 'random': return '🎲';
      case 'rl_deep': return '🤯';
      default: return '🤖';
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8 text-center">
        <h2 className="text-3xl font-bold text-gray-800 mb-2">
          🏆 Crea Nuova Asta
        </h2>
        <p className="text-gray-600">
          Configura i partecipanti e le impostazioni dell'asta
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Configurazione Squadre */}
        <Card title="🏟️ Configurazione Squadre">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <Input
              label="Portieri (GK)"
              type="number"
              min="1"
              max="5"
              value={config.slots_gk}
              onChange={(e) => setConfig({ ...config, slots_gk: parseInt(e.target.value) })}
            />
            <Input
              label="Difensori (DEF)"
              type="number"
              min="1"
              max="15"
              value={config.slots_def}
              onChange={(e) => setConfig({ ...config, slots_def: parseInt(e.target.value) })}
            />
            <Input
              label="Centrocampisti (MID)"
              type="number"
              min="1"
              max="15"
              value={config.slots_mid}
              onChange={(e) => setConfig({ ...config, slots_mid: parseInt(e.target.value) })}
            />
            <Input
              label="Attaccanti (ATT)"
              type="number"
              min="1"
              max="10"
              value={config.slots_att}
              onChange={(e) => setConfig({ ...config, slots_att: parseInt(e.target.value) })}
            />
          </div>
          
          <Input
            label="💰 Crediti Iniziali"
            type="number"
            min="500"
            max="2000"
            step="50"
            value={config.initial_credits}
            onChange={(e) => setConfig({ ...config, initial_credits: parseInt(e.target.value) })}
          />
        </Card>

        {/* Partecipanti */}
        <Card title="👥 Partecipanti">
          <div className="space-y-4">
            {agents.map((agent, index) => (
              <div key={index} className="flex items-end gap-4 p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl">
                  {getRoleEmoji(agent.type)}
                </div>
                
                <div className="flex-1">
                  <Input
                    label="Nome ID"
                    value={agent.id}
                    onChange={(e) => updateAgent(index, 'id', e.target.value)}
                    placeholder="es. giocatore1, bot1..."
                  />
                </div>
                
                <div className="flex-1">
                  <Select
                    label="Tipo"
                    value={agent.type}
                    onChange={(e) => updateAgent(index, 'type', e.target.value as AgentConfig['type'])}
                    options={agentTypeOptions}
                  />
                </div>
                
                <Button
                  type="button"
                  variant="danger"
                  size="sm"
                  onClick={() => removeAgent(index)}
                  disabled={agents.length === 1}
                  className="mb-4"
                >
                  🗑️
                </Button>
              </div>
            ))}
            
            <Button
              type="button"
              variant="secondary"
              onClick={addAgent}
              className="w-full"
              disabled={agents.length >= 8}
            >
              ➕ Aggiungi Partecipante
            </Button>
          </div>
        </Card>

        {/* Riepilogo */}
        <Card title="📊 Riepilogo">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div className="bg-blue-50 p-3 rounded">
              <div className="text-2xl font-bold text-blue-600">{agents.length}</div>
              <div className="text-sm text-blue-800">Partecipanti</div>
            </div>
            <div className="bg-green-50 p-3 rounded">
              <div className="text-2xl font-bold text-green-600">
                {config.slots_gk + config.slots_def + config.slots_mid + config.slots_att}
              </div>
              <div className="text-sm text-green-800">Giocatori per squadra</div>
            </div>
            <div className="bg-yellow-50 p-3 rounded">
              <div className="text-2xl font-bold text-yellow-600">{config.initial_credits}</div>
              <div className="text-sm text-yellow-800">Crediti iniziali</div>
            </div>
            <div className="bg-purple-50 p-3 rounded">
              <div className="text-2xl font-bold text-purple-600">
                {agents.filter(a => a.type === 'human').length}
              </div>
              <div className="text-sm text-purple-800">Giocatori umani</div>
            </div>
          </div>
        </Card>

        {/* Azioni */}
        <div className="flex justify-center">
          <Button
            type="submit"
            variant="success"
            size="lg"
            loading={loading}
            className="px-12"
          >
            🚀 Avvia Asta
          </Button>
        </div>
      </form>
    </div>
  );
};
