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
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore sconosciuto');
    }
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

  // Polling automatico dello stato
  useEffect(() => {
    let interval: number;
    
    const startPolling = () => {
      interval = setInterval(refreshStatus, 2000); // Aggiorna ogni 2 secondi
    };

    // Carica lo stato iniziale
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
  }, [refreshStatus, loadPlayers, auctionStatus?.state]);

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
    refreshStatus,
    loadPlayers,
  };
};
