import React, { useState, useEffect, useRef } from 'react';
import { Card, Input } from './UI';
import { apiClient } from '../services/api';
import type { SearchPlayer } from '../types';

interface PlayerSearchProps {
  onPlayerSelect: (playerName: string) => void;
  loading?: boolean;
}

const ROLE_COLORS = {
  GK: 'bg-yellow-100 text-yellow-800 border-yellow-300',
  DEF: 'bg-blue-100 text-blue-800 border-blue-300',
  MID: 'bg-green-100 text-green-800 border-green-300',
  ATT: 'bg-red-100 text-red-800 border-red-300',
};

const ROLE_NAMES = {
  GK: 'P',
  DEF: 'D', 
  MID: 'C',
  ATT: 'A',
};

export const PlayerSearch: React.FC<PlayerSearchProps> = ({ onPlayerSelect, loading = false }) => {
  const [query, setQuery] = useState('');
  const [players, setPlayers] = useState<SearchPlayer[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const searchTimeoutRef = useRef<number>();
  const resultsRef = useRef<HTMLDivElement>(null);

  const searchPlayers = async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setPlayers([]);
      setShowResults(false);
      return;
    }

    setIsSearching(true);
    try {
      const response = await apiClient.searchPlayers(searchQuery);
      if (response.success) {
        setPlayers(response.players);
        setShowResults(true);
      }
    } catch (error) {
      console.error('Error searching players:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newQuery = e.target.value;
    setQuery(newQuery);

    // Clear existing timeout
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    // Set new timeout for search
    searchTimeoutRef.current = setTimeout(() => {
      searchPlayers(newQuery);
    }, 300);
  };

  const handlePlayerClick = (player: SearchPlayer) => {
    onPlayerSelect(player.name);
    setQuery('');
    setPlayers([]);
    setShowResults(false);
  };

  const handleClickOutside = (e: MouseEvent) => {
    if (resultsRef.current && !resultsRef.current.contains(e.target as Node)) {
      setShowResults(false);
    }
  };

  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, []);

  return (
    <div className="relative w-full max-w-2xl mx-auto">
      <div className="relative">
        <Input
          type="text"
          value={query}
          onChange={handleInputChange}
          placeholder="Cerca giocatori per nome, squadra o ruolo..."
          disabled={loading}
          className="w-full pl-10 text-black pr-4 py-3 text-lg"
        />
        <div className="absolute left-3 top-1/2 transform -translate-y-1/2">
          {isSearching ? (
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
          ) : (
            <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          )}
        </div>
      </div>

      {showResults && players.length > 0 && (
        <div ref={resultsRef} className="absolute top-full left-0 right-0 z-50 mt-1">
          <Card className="max-h-96 overflow-y-auto shadow-lg border">
            <div className="space-y-2">
              {players.map((player, index) => (
                <button
                  key={`${player.name}-${index}`}
                  onClick={() => handlePlayerClick(player)}
                  disabled={loading}
                  className="w-full p-3 text-left hover:bg-gray-50 rounded-lg transition-colors duration-150 border border-gray-100 hover:border-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-xs font-bold ${ROLE_COLORS[player.role]}`}>
                          {ROLE_NAMES[player.role]}
                        </span>
                        <div>
                          <div className="font-semibold text-gray-900">{player.name}</div>
                          <div className="text-sm text-gray-600">{player.team}</div>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-lg text-blue-600">{player.evaluation}</div>
                      <div className="text-xs text-gray-500">Valutazione</div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </Card>
        </div>
      )}

      {showResults && players.length === 0 && query.trim() && !isSearching && (
        <div ref={resultsRef} className="absolute top-full left-0 right-0 z-50 mt-1">
          <Card className="shadow-lg border">
            <div className="text-center py-4 text-gray-500">
              Nessun giocatore trovato per "{query}"
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};
