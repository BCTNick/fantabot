import { useState, useEffect } from 'react'
import './App.css'
import LiveAuction from './components/LiveAuction'

interface AgentConfig {
  name: string;
  type: string;
  strategy?: string;
}

interface AuctionConfig {
  num_agents: number;
  initial_budget: number;
  auction_type: string;
  num_players?: number;
  agents: AgentConfig[];
}

interface AgentType {
  type: string;
  name: string;
  description: string;
  parameters: Array<{
    name: string;
    type: string;
    options?: string[];
    default?: string;
  }>;
}

interface AuctionResult {
  auction_id: string;
  status: string;
  config: AuctionConfig;
}


function App() {
  const [agentTypes, setAgentTypes] = useState<AgentType[]>([]);
  const [auctionConfig, setAuctionConfig] = useState<AuctionConfig>({
    num_agents: 8,
    initial_budget: 500,
    auction_type: 'mixed',
    num_players: 25,
    agents: []
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AuctionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAuctionRunning, setIsAuctionRunning] = useState(false);

  useEffect(() => {
    fetchAgentTypes();
  }, []);

  const fetchAgentTypes = async () => {
    try {
      const response = await fetch('http://localhost:8080/api/agents/types');
      const data = await response.json();
      setAgentTypes(data.agent_types);
      
      // Initialize with default agents (include human agent)
      const defaultAgents: AgentConfig[] = [
        { name: 'Cucco', type: 'human' },
        { name: 'Michele', type: 'human' },
        { name: 'Andrea', type: 'human' },
        { name: 'Domenico', type: 'human' },
        { name: 'Luca', type: 'human' },
        { name: 'Franco', type: 'human' },
        { name: 'NOI', type: 'cap', strategy: 'aggressive' },
      ];
      
      setAuctionConfig(prev => ({ ...prev, agents: defaultAgents }));
    } catch (err) {
      setError('Errore nel caricamento dei tipi di agenti');
      console.error(err);
    }
  };

  const updateAgent = (index: number, field: keyof AgentConfig, value: string) => {
    const newAgents = [...auctionConfig.agents];
    newAgents[index] = { ...newAgents[index], [field]: value };
    setAuctionConfig(prev => ({ ...prev, agents: newAgents }));
  };

  const addAgent = () => {
    const newAgent: AgentConfig = {
      name: `Agent ${auctionConfig.agents.length + 1}`,
      type: 'random'
    };
    setAuctionConfig(prev => ({
      ...prev,
      agents: [...prev.agents, newAgent],
      num_agents: prev.agents.length + 1
    }));
  };

  const removeAgent = (index: number) => {
    const newAgents = auctionConfig.agents.filter((_, i) => i !== index);
    setAuctionConfig(prev => ({
      ...prev,
      agents: newAgents,
      num_agents: newAgents.length
    }));
  };

  const createAuction = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8080/api/auction/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(auctionConfig),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore nella creazione dell\'asta');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const startAuction = async () => {
    if (!result) return;

    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8080/api/auction/${result.auction_id}/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Auction started:', data);
      
      // Start auction live view
      setIsAuctionRunning(true);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore nell\'avvio dell\'asta');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="">
      <main>
        {!result ? (
          <div className="auction-creator">
            <h2 className='text-3xl font-bold'>Configura la tua Asta</h2>
            <div className='grid grid-cols-2 gap-4'>

            <div className="config-section">
              <h3>Parametri Generali</h3>
              <div className="form-grid">
                <div className="form-group">
                  <label>Budget Iniziale (€)</label>
                  <input
                    type="number"
                    value={auctionConfig.initial_budget}
                    onChange={(e) => setAuctionConfig(prev => ({ 
                      ...prev, 
                      initial_budget: parseInt(e.target.value) || 500 
                    }))}
                    min="100"
                    max="1000"
                  />
                </div>
                
                <div className="form-group">
                  <label>Tipo di Asta</label>
                  <select
                    value={auctionConfig.auction_type}
                    onChange={(e) => setAuctionConfig(prev => ({ 
                      ...prev, 
                      auction_type: e.target.value 
                    }))}
                  >
                    <option value="mixed">Mista (tutti i ruoli)</option>
                    <option value="by_role">Per ruolo</option>
                  </select>
                </div>
                
                <div className="form-group">
                  <label>Numero di Giocatori</label>
                  <input
                    type="number"
                    value={auctionConfig.num_players}
                    onChange={(e) => setAuctionConfig(prev => ({ 
                      ...prev, 
                      num_players: parseInt(e.target.value) || 25 
                    }))}
                    min="10"
                    max="100"
                  />
                </div>
              </div>
            </div>

            <div className="config-section">
              <h3>Agenti ({auctionConfig.agents.length})</h3>
              
              <div className="agents-list">
                {auctionConfig.agents.map((agent, index) => (
                  <div key={index} className="agent-config">
                    <div className="agent-header">
                      <input
                        type="text"
                        value={agent.name}
                        onChange={(e) => updateAgent(index, 'name', e.target.value)}
                        placeholder="Nome agente"
                      />
                      <button 
                        onClick={() => removeAgent(index)}
                        className="remove-btn"
                        disabled={auctionConfig.agents.length <= 2}
                      >
                        ❌
                      </button>
                    </div>
                    
                    <div className="agent-controls">
                      <select
                        value={agent.type}
                        onChange={(e) => updateAgent(index, 'type', e.target.value)}
                      >
                        {agentTypes.map(type => (
                          <option key={type.type} value={type.type}>
                            {type.name}
                          </option>
                        ))}
                      </select>
                      
                      {agent.type === 'cap' && (
                        <select
                          value={agent.strategy || 'conservative'}
                          onChange={(e) => updateAgent(index, 'strategy', e.target.value)}
                        >
                          <option value="conservative">Conservativa</option>
                          <option value="aggressive">Aggressiva</option>
                          <option value="balanced">Bilanciata</option>
                        </select>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              
              <button onClick={addAgent} className="add-agent-btn">
                ➕ Aggiungi Agente
              </button>
            </div>
            </div>


            <div className="actions">
              <button 
                onClick={createAuction} 
                disabled={loading || auctionConfig.agents.length < 2}
                className="create-btn"
              >
                {loading ? 'Creazione...' : 'Crea Asta'}
              </button>
            </div>

            {error && (
              <div className="error">
                ❌ {error}
              </div>
            )}
          </div>
        ) : (
          <LiveAuction
            auctionId={result.auction_id}
            isAuctionRunning={isAuctionRunning}
            onStart={startAuction}
            onStop={() => {
              setIsAuctionRunning(false);
            }}
            onReset={() => {
              setResult(null);
              setIsAuctionRunning(false);
            }}
          />
        )}
      </main>
    </div>
  )
}

export default App
