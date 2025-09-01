// API Types

export interface Player {
  name: string;
  team: string;
  role: 'GK' | 'DEF' | 'MID' | 'ATT';
  evaluation: number;
  standardized_evaluation?: number;
  ranking?: number;
  fantasy_team?: string | null;
  final_cost?: number | null;
}

export interface Agent {
  id: string;
  type: string;
  credits: number;
  squad_size: number;
  squad_gk: number;
  squad_def: number;
  squad_mid: number;
  squad_att: number;
}

export interface CurrentPlayer {
  name: string;
  role: 'GK' | 'DEF' | 'MID' | 'ATT';
  team: string;
  evaluation: number;
  current_price: number;
  highest_bidder: string | null;
}

export interface Slots {
  GK: number;
  DEF: number;
  MID: number;
  ATT: number;
}

export interface AuctionStatus {
  state: 'not_started' | 'created' | 'running' | 'player_auction' | 'completed';
  session_id?: string;
  current_player?: CurrentPlayer | null;
  agents: Agent[];
  slots?: Slots;
  message?: string;
}

export interface AgentConfig {
  type: 'human' | 'cap' | 'dynamic_cap' | 'random' | 'rl_deep';
  id: string;
}

export interface AuctionConfig {
  initial_credits: number;
  slots_gk: number;
  slots_def: number;
  slots_mid: number;
  slots_att: number;
}

export interface CreateAuctionRequest {
  agents: AgentConfig[];
  config: AuctionConfig;
}

export interface ApiResponse {
  success: boolean;
  error?: string;
  [key: string]: unknown;
}

export interface BidRequest {
  agent_id: string;
  amount: number;
}

export interface SquadMetrics {
  total_evaluation: number;
  total_standardized: number;
  bestxi_evaluation: number;
  bestxi_standardized: number;
  credits_remaining: number;
  credits_spent: number;
}

export interface AuctionResult {
  agent_id: string;
  squad: Player[];
  metrics: SquadMetrics;
}

export interface AuctionResultsResponse {
  success: boolean;
  results: AuctionResult[];
}

export interface BotBidResult {
  agent_id: string;
  action: 'bid' | 'pass' | 'cannot_bid';
  amount?: number;
  reason?: string;
}

export interface BotBidsResponse {
  success: boolean;
  bids: BotBidResult[];
  current_price: number;
  highest_bidder: string | null;
}
