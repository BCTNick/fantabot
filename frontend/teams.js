// Fantasy Football Auction - Teams JavaScript

class TeamsClient {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.currentState = null;
        this.serverPort = 8765; // Default port, will try others if needed
        
        this.initializeElements();
        this.connect();
    }
    
    initializeElements() {
        // Connection status
        this.connectionStatus = document.getElementById('connectionStatus');
        this.statusIndicator = this.connectionStatus.querySelector('.status-indicator');
        this.statusText = this.connectionStatus.querySelector('.status-text');
        
        // Teams stats
        this.activeTeams = document.getElementById('activeTeams');
        this.totalSpent = document.getElementById('totalSpent');
        this.playersSold = document.getElementById('playersSold');
        
        // Teams grid
        this.teamsGrid = document.getElementById('teamsGrid');
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
    
    handleMessage(message) {
        console.log('Received message:', message);
        
        switch (message.type) {
            case 'state_update':
                this.updateTeams(message.data);
                break;
            case 'bid_update':
            case 'auction_complete':
                // Refresh teams when auction updates happen
                this.requestState();
                break;
        }
    }
    
    updateTeams(state) {
        this.currentState = state;
        
        if (state.agents) {
            this.updateTeamsStats(state.agents);
            this.renderTeams(state.agents);
        }
    }
    
    updateTeamsStats(agents) {
        const activeTeams = agents.length;
        let totalSpent = 0;
        let playersSold = 0;
        
        agents.forEach(agent => {
            const spent = agent.initial_credits - agent.credits;
            totalSpent += spent;
            playersSold += agent.total_players;
        });
        
        this.activeTeams.textContent = activeTeams;
        this.totalSpent.textContent = `€${totalSpent.toLocaleString()}`;
        this.playersSold.textContent = playersSold;
    }
    
    renderTeams(agents) {
        this.teamsGrid.innerHTML = '';
        
        agents.forEach(agent => {
            const teamCard = this.createTeamCard(agent);
            this.teamsGrid.appendChild(teamCard);
        });
    }
    
    createTeamCard(agent) {
        const card = document.createElement('div');
        card.className = 'team-card';
        
        const spent = agent.initial_credits - agent.credits;
        const squadValue = this.calculateSquadValue(agent.squad);
        
        card.innerHTML = `
            <div class="team-header">
                <div class="team-name">${agent.id}</div>
                <div class="team-stats">
                    <span>Credits: €${agent.credits.toLocaleString()}</span>
                    <span>Spent: €${spent.toLocaleString()}</span>
                    <span>Players: ${agent.total_players}</span>
                </div>
            </div>
            <div class="team-squad">
                ${this.renderSquadByPosition(agent.squad)}
            </div>
        `;
        
        return card;
    }
    
    calculateSquadValue(squad) {
        let totalValue = 0;
        
        Object.values(squad).forEach(players => {
            players.forEach(player => {
                totalValue += player.evaluation || 0;
            });
        });
        
        return totalValue;
    }
    
    renderSquadByPosition(squad) {
        const positions = [
            { key: 'GK', name: 'Goalkeepers', emoji: '🥅' },
            { key: 'DEF', name: 'Defenders', emoji: '🛡️' },
            { key: 'MID', name: 'Midfielders', emoji: '⚽' },
            { key: 'ATT', name: 'Attackers', emoji: '🎯' }
        ];
        
        return positions.map(position => {
            const players = squad[position.key] || [];
            const maxSlots = this.getMaxSlots(position.key);
            
            return `
                <div class="position-section">
                    <div class="position-title">
                        ${position.emoji} ${position.name} (${players.length}/${maxSlots})
                    </div>
                    <div class="players-list">
                        ${this.renderPlayersList(players, maxSlots)}
                    </div>
                </div>
            `;
        }).join('');
    }
    
    getMaxSlots(position) {
        // Default slot configuration - could be dynamic
        const slots = {
            'GK': 1,
            'DEF': 3,
            'MID': 3,
            'ATT': 3
        };
        return slots[position] || 1;
    }
    
    renderPlayersList(players, maxSlots) {
        const playerItems = players.map(player => `
            <div class="player-item">
                <span class="player-item-name">${player.name}</span>
                <span class="player-item-cost">${player.cost}</span>
            </div>
        `);
        
        // Add empty slots
        const emptySlots = maxSlots - players.length;
        for (let i = 0; i < emptySlots; i++) {
            playerItems.push(`
                <div class="player-item">
                    <span class="empty-slot">Empty slot</span>
                </div>
            `);
        }
        
        return playerItems.join('');
    }
}

// Initialize teams client when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.teamsClient = new TeamsClient();
    
    // Refresh teams data every 5 seconds
    setInterval(() => {
        if (window.teamsClient && window.teamsClient.isConnected) {
            window.teamsClient.requestState();
        }
    }, 5000);
});
