import type { 
  AuctionStatus, 
  CreateAuctionRequest, 
  ApiResponse, 
  Player, 
  BidRequest,
  AuctionResultsResponse 
} from '../types';

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

  async processBotBids(): Promise<ApiResponse> {
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

  async getAuctionResults(): Promise<AuctionResultsResponse> {
    return this.request('/auction/results');
  }
}

export const apiClient = new ApiClient();
