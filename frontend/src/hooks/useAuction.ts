import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../services/api';
import type { AuctionStatus, CreateAuctionRequest, Player, BotBidsResponse } from '../types';

export const useAuction = () => {
  const [auctionStatus, setAuctionStatus] = useState<AuctionStatus | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Polling per aggiornare lo stato dell'asta
  const refreshStatus = useCallback(async () => {
    try {
      const status = await apiClient.getAuctionStatus();
      setAuctionStatus(status);
      setError(null);
      
      // Salva lo stato nel localStorage per ripristino rapido
      if (status && status.state !== 'not_started') {
        localStorage.setItem('fantabot_auction_state', JSON.stringify({
          status,
          timestamp: Date.now()
        }));
      } else {
        localStorage.removeItem('fantabot_auction_state');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore sconosciuto');
    }
  }, []);

  // Carica lo stato dal localStorage (se disponibile e recente)
  const loadStoredState = useCallback(() => {
    try {
      const stored = localStorage.getItem('fantabot_auction_state');
      if (stored) {
        const { status, timestamp } = JSON.parse(stored);
        const now = Date.now();
        const maxAge = 30 * 60 * 1000; // 30 minuti
        
        // Se lo stato salvato è recente, usalo come stato iniziale
        if (now - timestamp < maxAge && status.state !== 'not_started') {
          setAuctionStatus(status);
          return true;
        } else {
          localStorage.removeItem('fantabot_auction_state');
        }
      }
    } catch (err) {
      console.warn('Errore nel caricamento stato salvato:', err);
      localStorage.removeItem('fantabot_auction_state');
    }
    return false;
  }, []);

  // Carica i giocatori disponibili
  const loadPlayers = useCallback(async () => {
    try {
      const response = await apiClient.getPlayers();
      if (response.success) {
        setPlayers(response.players);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore nel caricamento giocatori');
    }
  }, []);

  // Crea una nuova asta
  const createAuction = useCallback(async (config: CreateAuctionRequest) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.createAuction(config);
      if (response.success) {
        await refreshStatus();
        return true;
      } else {
        setError(response.error || 'Errore nella creazione dell\'asta');
        return false;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore nella creazione dell\'asta');
      return false;
    } finally {
      setLoading(false);
    }
  }, [refreshStatus]);

  // Inizia l'asta per il prossimo giocatore
  const startNextPlayer = useCallback(async (roleFilter?: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.startNextPlayer(roleFilter);
      if (response.success) {
        await refreshStatus();
        
        // Se l'asta è iniziata con successo, fai processare subito le offerte dei bot
        if (!response.completed) {
          try {
            await new Promise(resolve => setTimeout(resolve, 500)); // Piccola pausa per stabilizzazione
            const botResponse = await apiClient.processBotBids();
            if (botResponse.success) {
              await refreshStatus(); // Aggiorna lo stato dopo le offerte bot iniziali
            }
          } catch (botErr) {
            console.warn('Errore nel processare offerte bot iniziali:', botErr);
          }
        }
        
        return response;
      } else {
        setError(response.error || 'Errore nell\'avvio asta giocatore');
        return response;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore nell\'avvio asta giocatore');
      return { success: false, error: err instanceof Error ? err.message : 'Errore sconosciuto' };
    } finally {
      setLoading(false);
    }
  }, [refreshStatus]);

  // Fa un'offerta
  const makeBid = useCallback(async (agentId: string, amount: number) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.makeBid({ agent_id: agentId, amount });
      if (response.success) {
        await refreshStatus();
        
        // Automaticamente processa le offerte dei bot dopo ogni offerta umana
        try {
          const botResponse = await apiClient.processBotBids();
          if (botResponse.success) {
            await refreshStatus(); // Aggiorna di nuovo lo stato dopo le offerte bot
            return { 
              ...response, 
              botBids: botResponse as unknown as BotBidsResponse 
            };
          }
        } catch (botErr) {
          console.warn('Errore nel processare offerte bot:', botErr);
          // Non blocchiamo l'offerta umana se i bot falliscono
        }
        
        return response;
      } else {
        setError(response.error || 'Errore nell\'offerta');
        return response;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore nell\'offerta');
      return { success: false, error: err instanceof Error ? err.message : 'Errore sconosciuto' };
    } finally {
      setLoading(false);
    }
  }, [refreshStatus]);

  // Processa le offerte dei bot
  const processBotBids = useCallback(async (): Promise<BotBidsResponse> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.processBotBids();
      if (response.success) {
        await refreshStatus();
        return response as unknown as BotBidsResponse;
      } else {
        setError(response.error || 'Errore nelle offerte bot');
        return response as unknown as BotBidsResponse;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore nelle offerte bot');
      return { 
        success: false, 
        error: err instanceof Error ? err.message : 'Errore sconosciuto',
        bids: [],
        current_price: 0,
        highest_bidder: null
      } as BotBidsResponse;
    } finally {
      setLoading(false);
    }
  }, [refreshStatus]);

  // Finalizza l'asta del giocatore
  const finalizeAuction = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.finalizeAuction();
      if (response.success) {
        await refreshStatus();
        await loadPlayers(); // Ricarica i giocatori per aggiornare lo stato
        return response;
      } else {
        setError(response.error || 'Errore nella finalizzazione');
        return response;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore nella finalizzazione');
      return { success: false, error: err instanceof Error ? err.message : 'Errore sconosciuto' };
    } finally {
      setLoading(false);
    }
  }, [refreshStatus, loadPlayers]);

  // Inizia l'asta per un giocatore specifico
  const startPlayerAuction = useCallback(async (playerName: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.startPlayerAuction(playerName);
      if (response.success) {
        await refreshStatus();
        return response;
      } else {
        setError(response.error || 'Errore nell\'avvio dell\'asta');
        return response;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore nell\'avvio dell\'asta');
      return { success: false, error: err instanceof Error ? err.message : 'Errore sconosciuto' };
    } finally {
      setLoading(false);
    }
  }, [refreshStatus]);

  // Polling automatico dello stato
  useEffect(() => {
    let interval: number;
    
    const startPolling = () => {
      interval = setInterval(refreshStatus, 2000); // Aggiorna ogni 2 secondi
    };

    // Prima prova a caricare lo stato salvato
    loadStoredState();
    
    // Poi carica sempre lo stato aggiornato dal server
    refreshStatus();
    loadPlayers();
    
    // Avvia il polling solo se l'asta è attiva
    if (auctionStatus?.state === 'running' || auctionStatus?.state === 'player_auction') {
      startPolling();
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [refreshStatus, loadPlayers, loadStoredState, auctionStatus?.state]);

  return {
    auctionStatus,
    players,
    loading,
    error,
    createAuction,
    startNextPlayer,
    makeBid,
    processBotBids,
    finalizeAuction,
    startPlayerAuction,
    refreshStatus,
    loadPlayers,
  };
};
