import React, { useState, useEffect } from 'react';
import { Layout } from './components/Layout';
import { CreateAuctionPage } from './pages/CreateAuctionPage';
import { AuctionPage } from './pages/AuctionPage';
import { ResultsPage } from './pages/ResultsPage';
import { useAuction } from './hooks/useAuction';
import { Button, Card } from './components/UI';
import type { CreateAuctionRequest } from './types';
import './App.css';

type AppState = 'home' | 'create' | 'auction' | 'results';

function App() {
  const [currentPage, setCurrentPage] = useState<AppState>('home');
  const [isInitialized, setIsInitialized] = useState(false);
  const [restoredState, setRestoredState] = useState<string | null>(null);
  const {
    auctionStatus,
    loading,
    error,
    createAuction,
    startNextPlayer,
    makeBid,
    processBotBids,
    finalizeAuction,
    startPlayerAuction,
  } = useAuction();

  // Effetto per ripristinare lo stato della pagina al caricamento
  useEffect(() => {
    if (auctionStatus && !isInitialized) {
      setIsInitialized(true);
      
      console.log('🔄 Ripristino stato asta:', auctionStatus.state);
      
      // Naviga automaticamente alla pagina corretta in base allo stato dell'asta
      switch (auctionStatus.state) {
        case 'created':
          setCurrentPage('auction');
          setRestoredState('Asta creata ripristinata');
          break;
        case 'running':
          setCurrentPage('auction');
          setRestoredState('Asta in corso ripristinata');
          break;
        case 'player_auction':
          setCurrentPage('auction');
          setRestoredState(`Asta giocatore ripristinata: ${auctionStatus.current_player?.name || 'Sconosciuto'}`);
          break;
        case 'completed':
          setCurrentPage('results');
          setRestoredState('Risultati asta ripristinati');
          break;
        case 'not_started':
        default:
          setCurrentPage('home');
          break;
      }
      
      // Rimuovi il messaggio dopo 5 secondi
      if (auctionStatus.state !== 'not_started') {
        setTimeout(() => setRestoredState(null), 5000);
      }
    }
  }, [auctionStatus, isInitialized]);

    const handleCreateAuction = async (config: CreateAuctionRequest): Promise<boolean> => {
    const success = await createAuction(config);
    if (success) {
      setCurrentPage('auction');
    }
    return success;
  };

  const handlePlayerSelect = async (playerName: string) => {
    if (auctionStatus?.state === 'running' || auctionStatus?.state === 'player_auction') {
      try {
        const response = await startPlayerAuction(playerName);
        if (response.success) {
          setCurrentPage('auction');
        }
      } catch (err) {
        console.error('Error starting player auction:', err);
      }
    }
  };

  const handleNewAuction = () => {
    setCurrentPage('create');
  };

  const renderContent = () => {
    // Mostra loading mentre si sta inizializzando
    if (!isInitialized && auctionStatus === null) {
      return (
        <div className="min-h-screen flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Ripristino stato asta...</p>
          </div>
        </div>
      );
    }

    // Auto-navigate based on auction state
    if (auctionStatus?.state === 'completed' && currentPage !== 'results') {
      setCurrentPage('results');
    }

    switch (currentPage) {
      case 'create':
        return (
          <CreateAuctionPage
            onAuctionCreated={handleCreateAuction}
            loading={loading}
          />
        );

      case 'auction':
        if (!auctionStatus || auctionStatus.state === 'not_started') {
          setCurrentPage('home');
          return null;
        }
        return (
          <AuctionPage
            auctionStatus={auctionStatus}
            onStartNextPlayer={startNextPlayer}
            onMakeBid={makeBid}
            onProcessBotBids={processBotBids}
            onFinalize={finalizeAuction}
            loading={loading}
          />
        );

      case 'results':
        return <ResultsPage onNewAuction={handleNewAuction} />;

      default:
        return <HomePage onCreateAuction={() => setCurrentPage('create')} />;
    }
  };

  const canShowSearch = auctionStatus?.state === 'running' || auctionStatus?.state === 'player_auction';

  return (
    <Layout 
      showSearch={canShowSearch}
      onPlayerSelect={handlePlayerSelect}
      searchLoading={loading}
    >
      {/* Messaggio di ripristino stato */}
      {restoredState && (
        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center">
            <span className="text-green-600 mr-2">✅</span>
            <span className="text-green-800 font-medium">{restoredState}</span>
          </div>
        </div>
      )}

      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center">
            <span className="text-red-600 mr-2">⚠️</span>
            <span className="text-red-800">{error}</span>
          </div>
        </div>
      )}

      {renderContent()}
    </Layout>
  );
}

// Home Page Component
const HomePage: React.FC<{ onCreateAuction: () => void }> = ({ onCreateAuction }) => {
  return (
    <div className="max-w-6xl mx-auto">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <div className="mb-8">
          <div className="text-8xl mb-6">⚽</div>
          <h1 className="text-5xl font-bold text-gray-800 mb-4 leading-tight">
            Benvenuto in <span className="text-accent">FantaBot</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Il sistema di asta automatizzato più avanzato per il tuo fantacalcio. 
            Competi contro bot intelligenti in aste realistiche e competitive.
          </p>
        </div>
        
        <Button
          variant="primary"
          size="lg"
          onClick={onCreateAuction}
          className="px-12 py-4 text-xl shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
        >
          � Inizia Nuova Asta
        </Button>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
        <Card className="text-center hover:shadow-lg transition-shadow duration-200">
          <div className="text-6xl mb-6">🤖</div>
          <h3 className="text-xl font-bold mb-4 text-gray-800">Bot Intelligenti</h3>
          <p className="text-gray-600 leading-relaxed">
            Diversi tipi di bot con strategie avanzate: da quelli casuali 
            agli algoritmi di machine learning per aste competitive
          </p>
        </Card>

        <Card className="text-center hover:shadow-lg transition-shadow duration-200">
          <div className="text-6xl mb-6">⚡</div>
          <h3 className="text-xl font-bold mb-4 text-gray-800">Tempo Reale</h3>
          <p className="text-gray-600 leading-relaxed">
            Offerte in tempo reale con aggiornamenti automatici dello stato. 
            Visualizza ogni mossa dei tuoi avversari istantaneamente
          </p>
        </Card>

        <Card className="text-center hover:shadow-lg transition-shadow duration-200">
          <div className="text-6xl mb-6">📊</div>
          <h3 className="text-xl font-bold mb-4 text-gray-800">Analisi Dettagliate</h3>
          <p className="text-gray-600 leading-relaxed">
            Statistiche complete, valutazioni per ogni squadra formata 
            e metriche avanzate per analizzare le performance
          </p>
        </Card>
      </div>

      {/* How it Works */}
      <Card className="mb-16">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">🚀 Come Funziona</h2>
          <p className="text-gray-600">Quattro semplici passi per la tua asta perfetta</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold shadow-lg">
              1
            </div>
            <h4 className="font-bold text-lg mb-2 text-gray-800">Configura</h4>
            <p className="text-sm text-gray-600 leading-relaxed">
              Imposta partecipanti, ruoli, crediti e scegli i tuoi bot preferiti
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-green-600 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold shadow-lg">
              2
            </div>
            <h4 className="font-bold text-lg mb-2 text-gray-800">Asta</h4>
            <p className="text-sm text-gray-600 leading-relaxed">
              Partecipa alle aste giocatore per giocatore con un'interfaccia intuitiva
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-yellow-500 to-yellow-600 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold shadow-lg">
              3
            </div>
            <h4 className="font-bold text-lg mb-2 text-gray-800">Competi</h4>
            <p className="text-sm text-gray-600 leading-relaxed">
              Fai offerte strategiche contro bot intelligenti e altri giocatori
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-purple-600 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold shadow-lg">
              4
            </div>
            <h4 className="font-bold text-lg mb-2 text-gray-800">Analizza</h4>
            <p className="text-sm text-gray-600 leading-relaxed">
              Visualizza risultati dettagliati e confronta le performance delle squadre
            </p>
          </div>
        </div>
      </Card>

      {/* Bot Types */}
      <Card className="mb-16">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">🤖 Tipi di Bot Disponibili</h2>
          <p className="text-gray-600">Scegli tra diverse personalità di asta</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
            <div className="text-3xl mb-2">👤</div>
            <div className="font-bold text-blue-800">Umano</div>
            <div className="text-xs text-blue-600">Controllo manuale</div>
          </div>
          
          <div className="text-center p-4 bg-gray-50 rounded-lg border border-gray-200">
            <div className="text-3xl mb-2">🤖</div>
            <div className="font-bold text-gray-800">Strategico</div>
            <div className="text-xs text-gray-600">Basato sui crediti</div>
          </div>
          
          <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
            <div className="text-3xl mb-2">🧠</div>
            <div className="font-bold text-purple-800">Dinamico</div>
            <div className="text-xs text-purple-600">Strategia adattiva</div>
          </div>
          
          <div className="text-center p-4 bg-yellow-50 rounded-lg border border-yellow-200">
            <div className="text-3xl mb-2">🎲</div>
            <div className="font-bold text-yellow-800">Casuale</div>
            <div className="text-xs text-yellow-600">Offerte random</div>
          </div>
          
          <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
            <div className="text-3xl mb-2">🤯</div>
            <div className="font-bold text-red-800">AI Avanzato</div>
            <div className="text-xs text-red-600">Machine Learning</div>
          </div>
        </div>
      </Card>

      {/* CTA */}
      <div className="text-center bg-gradient-to-r from-accent to-blue-600 text-white rounded-xl p-12">
        <h2 className="text-3xl font-bold mb-4">Pronto per l'Asta?</h2>
        <p className="text-xl mb-8 opacity-90">
          Crea la tua asta personalizzata e scopri chi ha la migliore strategia
        </p>
        <Button
          variant="secondary"
          size="lg"
          onClick={onCreateAuction}
          className="px-8 py-3 bg-white text-accent hover:bg-gray-100 shadow-lg"
        >
          🏆 Inizia Ora
        </Button>
      </div>
    </div>
  );
};

export default App;
