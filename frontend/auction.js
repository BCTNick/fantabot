// Fantasy Football Auction - Live Auction JavaScript

class AuctionClient {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.currentState = null;
        this.activityLog = [];
        this.serverPort = 8765; // Default port, will try others if needed
        
        this.initializeElements();
        this.connect();
        this.setupEventListeners();
    }
    
    initializeElements() {
        // Connection status
        this.connectionStatus = document.getElementById('connectionStatus');
        this.statusIndicator = this.connectionStatus.querySelector('.status-indicator');
        this.statusText = this.connectionStatus.querySelector('.status-text');
        
        // Controls
        this.startBtn = document.getElementById('startAuction');
        this.auctionType = document.getElementById('auctionType');
        this.perRuolo = document.getElementById('perRuolo');
        
        // Player display
        this.playerName = document.getElementById('playerName');
        this.playerRole = document.getElementById('playerRole');
        this.playerEvaluation = document.getElementById('playerEvaluation');
        this.playerAvatar = document.getElementById('playerAvatar');
        
        // Auction info
        this.currentPrice = document.getElementById('currentPrice');
        this.highestBidder = document.getElementById('highestBidder');
        
        // Phase indicator
        this.phaseStatus = document.getElementById('phaseStatus');
        this.progressBar = document.getElementById('progressBar');
        
        // Activity feed
        this.activityList = document.getElementById('activityList');
        
        // Sales list
        this.salesList = document.getElementById('salesList');
    }
    
    connect() {
        this.tryConnectToPort(this.serverPort);
    }
    
    tryConnectToPort(port, maxAttempts = 10) {
        try {
            console.log(`Attempting to connect to ws://localhost:${port}`);
            this.ws = new WebSocket(`ws://localhost:${port}`);
            
            this.ws.onopen = () => {
                console.log(`Connected to auction server on port ${port}`);
                this.isConnected = true;
                this.serverPort = port;
                this.updateConnectionStatus('online', `Connected (port ${port})`);
                this.requestState();
            };
            
            this.ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            };
            
            this.ws.onclose = () => {
                console.log('Disconnected from auction server');
                this.isConnected = false;
                this.updateConnectionStatus('offline', 'Disconnected');
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => {
                    if (!this.isConnected) {
                        this.connect();
                    }
                }, 3000);
            };
            
            this.ws.onerror = (error) => {
                console.error(`WebSocket error on port ${port}:`, error);
                
                // Try next port if this one failed and we haven't tried too many
                if (port < this.serverPort + maxAttempts) {
                    setTimeout(() => {
                        this.tryConnectToPort(port + 1, maxAttempts);
                    }, 1000);
                } else {
                    this.updateConnectionStatus('offline', 'Connection Error - Check server');
                }
            };
            
        } catch (error) {
            console.error('Failed to connect:', error);
            this.updateConnectionStatus('offline', 'Failed to Connect');
        }
    }
    
    updateConnectionStatus(status, text) {
        this.statusIndicator.className = `status-indicator ${status}`;
        this.statusText.textContent = text;
        
        // Enable/disable start button based on connection
        this.startBtn.disabled = status !== 'online';
    }
    
    setupEventListeners() {
        this.startBtn.addEventListener('click', () => {
            this.startAuction();
        });
    }
    
    sendMessage(message) {
        if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }
    
    requestState() {
        this.sendMessage({
            type: 'get_state'
        });
    }
    
    startAuction() {
        this.sendMessage({
            type: 'start_auction',
            auction_type: this.auctionType.value,
            per_ruolo: this.perRuolo.checked
        });
        
        this.startBtn.disabled = true;
        this.startBtn.textContent = 'Auction Running...';
    }
    
    handleMessage(message) {
        console.log('Received message:', message);
        
        switch (message.type) {
            case 'state_update':
                this.updateState(message.data);
                break;
            case 'bid_update':
                this.handleBidUpdate(message.data);
                break;
            case 'auction_complete':
                this.handleAuctionComplete(message.data);
                break;
            case 'role_start':
                this.handleRoleStart(message.data);
                break;
            case 'auction_finished':
                this.handleAuctionFinished(message.data);
                break;
        }
    }
    
    updateState(state) {
        this.currentState = state;
        
        // Update phase
        this.updatePhase(state.phase);
        
        // Update current player
        if (state.current_player) {
            this.updateCurrentPlayer(state.current_player, state.current_price, state.highest_bidder);
        }
        
        // Update progress (estimate based on completed auctions)
        this.updateProgress(state);
    }
    
    updatePhase(phase) {
        let statusText = '';
        let progressPercent = 0;
        
        switch (phase) {
            case 'waiting':
                statusText = 'Waiting to start';
                progressPercent = 0;
                break;
            case 'auction':
                statusText = 'Auction in progress';
                progressPercent = 50; // Will be updated by updateProgress
                break;
            case 'completed':
                statusText = 'Auction completed';
                progressPercent = 100;
                this.startBtn.disabled = false;
                this.startBtn.textContent = 'Start New Auction';
                break;
        }
        
        this.phaseStatus.textContent = statusText;
        this.progressBar.style.width = `${progressPercent}%`;
    }
    
    updateCurrentPlayer(player, price, bidder) {
        this.playerName.textContent = player.name;
        this.playerRole.textContent = player.role;
        this.playerEvaluation.textContent = player.evaluation.toFixed(1);
        
        // Update avatar based on role
        const roleEmojis = {
            'GK': '🥅',
            'DEF': '🛡️',
            'MID': '⚽',
            'ATT': '🎯'
        };
        this.playerAvatar.textContent = roleEmojis[player.role] || '⚽';
        
        // Update price and bidder
        this.currentPrice.textContent = price;
        this.highestBidder.textContent = bidder || 'None';
        
        // Add visual feedback for highest bidder
        if (bidder) {
            this.highestBidder.style.color = 'var(--success-color)';
            this.highestBidder.style.fontWeight = '700';
        } else {
            this.highestBidder.style.color = 'var(--text-secondary)';
            this.highestBidder.style.fontWeight = '500';
        }
    }
    
    updateProgress(state) {
        if (state.completed_auctions && state.agents.length > 0) {
            // Estimate total players needed
            const totalSlots = state.agents.length * 10; // Approximate
            const completed = state.completed_auctions.length;
            const progressPercent = Math.min((completed / totalSlots) * 100, 100);
            
            if (state.phase === 'auction') {
                this.progressBar.style.width = `${progressPercent}%`;
            }
        }
    }
    
    handleBidUpdate(data) {
        const timestamp = new Date().toLocaleTimeString();
        let activityText = '';
        let activityClass = '';
        
        switch (data.action) {
            case 'bid':
                activityText = `${data.agent} bids €${data.new_price}`;
                activityClass = 'bid';
                break;
            case 'pass':
                activityText = `${data.agent} passes`;
                activityClass = 'pass';
                break;
            case 'cannot_bid':
                activityText = `${data.agent} cannot bid`;
                activityClass = 'pass';
                break;
        }
        
        this.addActivity(timestamp, activityText, activityClass);
    }
    
    handleAuctionComplete(data) {
        const timestamp = new Date().toLocaleTimeString();
        let resultText = '';
        
        if (data.result === 'sold') {
            resultText = `${data.player} SOLD to ${data.winner} for €${data.final_price}`;
        } else {
            resultText = `${data.player} UNSOLD`;
        }
        
        this.addActivity(timestamp, resultText, 'sale');
        this.addSale(data);
    }
    
    handleRoleStart(data) {
        const timestamp = new Date().toLocaleTimeString();
        const roleNames = {
            'GK': 'Goalkeepers',
            'DEF': 'Defenders', 
            'MID': 'Midfielders',
            'ATT': 'Attackers'
        };
        
        this.addActivity(timestamp, `Starting ${roleNames[data.role]} auctions`, 'role-start');
    }
    
    handleAuctionFinished(data) {
        const timestamp = new Date().toLocaleTimeString();
        this.addActivity(timestamp, data.message, 'finished');
    }
    
    addActivity(timestamp, text, className = '') {
        const activityItem = document.createElement('div');
        activityItem.className = `activity-item ${className} new`;
        
        activityItem.innerHTML = `
            <span class="activity-time">${timestamp}</span>
            <span class="activity-text">${text}</span>
        `;
        
        // Insert at top of list
        const firstChild = this.activityList.firstChild;
        if (firstChild) {
            this.activityList.insertBefore(activityItem, firstChild);
        } else {
            this.activityList.appendChild(activityItem);
        }
        
        // Remove 'new' class after animation
        setTimeout(() => {
            activityItem.classList.remove('new');
        }, 300);
        
        // Limit activity items to prevent performance issues
        const items = this.activityList.children;
        if (items.length > 50) {
            this.activityList.removeChild(items[items.length - 1]);
        }
        
        // Keep first few visible
        this.activityList.scrollTop = 0;
    }
    
    addSale(saleData) {
        const saleItem = document.createElement('div');
        saleItem.className = 'sale-item';
        
        let saleContent = '';
        if (saleData.result === 'sold') {
            saleContent = `
                <div class="sale-text">
                    <strong>${saleData.player}</strong> sold to <strong>${saleData.winner}</strong> for <strong>€${saleData.final_price}</strong>
                </div>
            `;
        } else {
            saleContent = `
                <div class="sale-text">
                    <strong>${saleData.player}</strong> - <span style="color: var(--text-secondary)">UNSOLD</span>
                </div>
            `;
        }
        
        saleItem.innerHTML = saleContent;
        
        // Insert at top of sales list
        const firstSale = this.salesList.firstChild;
        if (firstSale && !firstSale.textContent.includes('No sales yet')) {
            this.salesList.insertBefore(saleItem, firstSale);
        } else {
            // Remove "No sales yet" message
            this.salesList.innerHTML = '';
            this.salesList.appendChild(saleItem);
        }
        
        // Limit sales items
        const sales = this.salesList.children;
        if (sales.length > 20) {
            this.salesList.removeChild(sales[sales.length - 1]);
        }
    }
}

// Initialize auction client when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.auctionClient = new AuctionClient();
});
