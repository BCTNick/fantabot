import React, { useState } from 'react';
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
  const {
    auctionStatus,
    loading,
    error,
    createAuction,
    startNextPlayer,
    makeBid,
    processBotBids,
    finalizeAuction,
  } = useAuction();

  const handleCreateAuction = async (config: CreateAuctionRequest) => {
    const success = await createAuction(config);
    if (success) {
      setCurrentPage('auction');
    }
    return success;
  };

  const handleNewAuction = () => {
    setCurrentPage('create');
  };

  const renderContent = () => {
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

  return (
    <Layout>
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
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold text-gray-800 mb-4">
          ⚽ Benvenuto in FantaBot
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          Il sistema di asta automatizzato per il tuo fantacalcio
        </p>
        <div className="text-6xl mb-8">🏆</div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <Card className="text-center">
          <div className="text-4xl mb-4">🤖</div>
          <h3 className="text-lg font-semibold mb-2">Bot Intelligenti</h3>
          <p className="text-gray-600">
            Diversi tipi di bot con strategie avanzate per aste competitive
          </p>
        </Card>

        <Card className="text-center">
          <div className="text-4xl mb-4">⚡</div>
          <h3 className="text-lg font-semibold mb-2">Tempo Reale</h3>
          <p className="text-gray-600">
            Offerte in tempo reale con aggiornamenti automatici dello stato
          </p>
        </Card>

        <Card className="text-center">
          <div className="text-4xl mb-4">📊</div>
          <h3 className="text-lg font-semibold mb-2">Analisi Dettagliate</h3>
          <p className="text-gray-600">
            Statistiche complete e valutazioni per ogni squadra formata
          </p>
        </Card>
      </div>

      <Card title="🚀 Come Funziona" className="mb-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="w-12 h-12 bg-accent text-white rounded-full flex items-center justify-center mx-auto mb-2 text-xl font-bold">
              1
            </div>
            <h4 className="font-medium mb-1">Configura</h4>
            <p className="text-sm text-gray-600">Imposta partecipanti e regole</p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-accent text-white rounded-full flex items-center justify-center mx-auto mb-2 text-xl font-bold">
              2
            </div>
            <h4 className="font-medium mb-1">Asta</h4>
            <p className="text-sm text-gray-600">Partecipa alle aste giocatore per giocatore</p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-accent text-white rounded-full flex items-center justify-center mx-auto mb-2 text-xl font-bold">
              3
            </div>
            <h4 className="font-medium mb-1">Competi</h4>
            <p className="text-sm text-gray-600">Fai offerte contro bot intelligenti</p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-accent text-white rounded-full flex items-center justify-center mx-auto mb-2 text-xl font-bold">
              4
            </div>
            <h4 className="font-medium mb-1">Analizza</h4>
            <p className="text-sm text-gray-600">Visualizza risultati e statistiche</p>
          </div>
        </div>
      </Card>

      <div className="text-center">
        <Button
          variant="primary"
          size="lg"
          onClick={onCreateAuction}
          className="px-12 py-4 text-xl"
        >
          🎯 Inizia Nuova Asta
        </Button>
      </div>

      <div className="mt-12 text-center text-gray-500">
        <p className="mb-2">Supporta diversi tipi di agenti:</p>
        <div className="flex justify-center space-x-4 text-sm">
          <span>👤 Umano</span>
          <span>🤖 Strategico</span>
          <span>🧠 Dinamico</span>
          <span>🎲 Casuale</span>
          <span>🤯 AI Avanzato</span>
        </div>
      </div>
    </div>
  );
};

export default App;
