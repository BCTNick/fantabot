import type { 
  AuctionStatus, 
  BotBidsResponse,
  SearchPlayersResponse,
  Player
} from '../types';

// API request/response types
interface ApiResponse {
  success: boolean;
  message?: string;
  error?: string;
}

interface CreateAuctionRequest {
  agents: Array<{
    id: string;
    type: string;
  }>;
  slots: {
    GK: number;
    DEF: number;
    MID: number;
    ATT: number;
  };
  initial_credits: number;
}

interface BidRequest {
  agent_id: string;
  amount: number;
}

interface AuctionResultsResponse {
  success: boolean;
  results?: any[];
  error?: string;
}

const API_BASE_URL = 'http://localhost:8081/api';

class ApiClient {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.request('/health');
  }

  async createAuction(data: CreateAuctionRequest): Promise<ApiResponse> {
    return this.request('/auction/create', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getAuctionStatus(): Promise<AuctionStatus> {
    return this.request('/auction/status');
  }

  async startNextPlayer(roleFilter?: string): Promise<ApiResponse> {
    return this.request('/auction/next-player', {
      method: 'POST',
      body: JSON.stringify(roleFilter ? { role_filter: roleFilter } : {}),
    });
  }

  async makeBid(bid: BidRequest): Promise<ApiResponse> {
    return this.request('/auction/bid', {
      method: 'POST',
      body: JSON.stringify(bid),
    });
  }

  async processBotBids(): Promise<BotBidsResponse> {
    return this.request('/auction/bot-bids', {
      method: 'POST',
    });
  }

  async finalizeAuction(): Promise<ApiResponse> {
    return this.request('/auction/finalize', {
      method: 'POST',
    });
  }

  async getPlayers(): Promise<{ success: boolean; players: Player[]; total: number }> {
    return this.request('/players');
  }

  async searchPlayers(query: string): Promise<SearchPlayersResponse> {
    return this.request(`/players/search?q=${encodeURIComponent(query)}`);
  }

  async startPlayerAuction(playerName: string): Promise<ApiResponse> {
    return this.request('/auction/start-player', {
      method: 'POST',
      body: JSON.stringify({ player_name: playerName }),
    });
  }

  async getAuctionResults(): Promise<AuctionResultsResponse> {
    return this.request('/auction/results');
  }
}

export const apiClient = new ApiClient();
