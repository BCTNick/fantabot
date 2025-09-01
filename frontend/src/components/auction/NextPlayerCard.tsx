import React from 'react';
import { Card, Button } from '../UI';

interface NextPlayerCardProps {
  roleFilter: string;
  setRoleFilter: (role: string) => void;
  onStartNext: () => void;
  loading: boolean;
}

const ROLE_NAMES = {
  GK: 'Portiere',
  DEF: 'Difensore',
  MID: 'Centrocampista', 
  ATT: 'Attaccante',
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

export const NextPlayerCard: React.FC<NextPlayerCardProps> = ({
  roleFilter,
  setRoleFilter,
  onStartNext,
  loading
}) => {
  return (
    <Card>
      <div className="text-center">
        {/* Icon */}
        <div className="text-8xl mb-6">🎯</div>
        
        {/* Title */}
        <h3 className="text-2xl font-bold text-gray-800 mb-2">
          Seleziona Prossimo Giocatore
        </h3>
        <p className="text-gray-600 mb-8">
          Scegli il ruolo per il prossimo giocatore da mettere all'asta
        </p>

        {/* Role Filter Buttons */}
        <div className="mb-8">
          <h4 className="text-lg font-semibold text-gray-700 mb-4">
            🏈 Filtra per Ruolo
          </h4>
          
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
            <button
              onClick={() => setRoleFilter('')}
              className={`p-4 rounded-lg text-sm font-medium transition-all duration-200 ${
                roleFilter === '' 
                  ? 'bg-accent text-white shadow-lg transform scale-105' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300 hover:shadow-md'
              }`}
            >
              <div className="text-2xl mb-1">🎲</div>
              <div>Tutti</div>
            </button>
            
            {(['GK', 'DEF', 'MID', 'ATT'] as const).map((role) => (
              <button
                key={role}
                onClick={() => setRoleFilter(role)}
                className={`p-4 rounded-lg text-sm font-medium transition-all duration-200 ${
                  roleFilter === role 
                    ? 'bg-accent text-white shadow-lg transform scale-105' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300 hover:shadow-md'
                }`}
              >
                <div className="text-2xl mb-1">{getRoleIcon(role)}</div>
                <div>{ROLE_NAMES[role]}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Current Selection */}
        {roleFilter && (
          <div className="bg-blue-50 p-4 rounded-lg mb-6 border border-blue-200">
            <div className="flex items-center justify-center">
              <span className="text-2xl mr-2">{getRoleIcon(roleFilter)}</span>
              <span className="text-lg font-medium text-blue-800">
                Filtro attivo: {ROLE_NAMES[roleFilter as keyof typeof ROLE_NAMES]}
              </span>
            </div>
          </div>
        )}
        
        {/* Start Button */}
        <Button
          variant="primary"
          size="lg"
          onClick={onStartNext}
          loading={loading}
          className="px-12 py-4 text-xl"
        >
          <span className="mr-2">🚀</span>
          Inizia Prossima Asta
          {roleFilter && (
            <span className="ml-2 text-lg">
              ({ROLE_NAMES[roleFilter as keyof typeof ROLE_NAMES]})
            </span>
          )}
        </Button>

        {/* Help Text */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600">
            💡 <strong>Suggerimento:</strong> Seleziona un ruolo specifico per trovare 
            il miglior giocatore disponibile in quella posizione, oppure scegli "Tutti" 
            per una selezione casuale.
          </p>
        </div>
      </div>
    </Card>
  );
};
