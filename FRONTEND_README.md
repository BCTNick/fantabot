# Fantasy Football Auction - Real-Time Frontend

A real-time web interface for the fantasy football auction system with live updates via WebSocket connection.

## Features

### Live Auction Page (`index.html`)
- **Real-time auction display**: Shows current player being auctioned
- **Live bidding updates**: See bids as they happen with agent names and amounts
- **Current price and highest bidder**: Always up-to-date auction state
- **Activity feed**: Live stream of all auction activities (bids, passes, sales)
- **Recent sales**: Summary of completed player auctions
- **Auction controls**: Start auctions with different configurations

### Teams Page (`teams.html`)
- **Live team overview**: See all teams and their current squads
- **Real-time updates**: Team rosters update as players are purchased
- **Position-based display**: Players organized by GK, DEF, MID, ATT
- **Financial tracking**: Credits remaining and total spent per team
- **Squad progress**: Visual indicator of team completion

## Setup Instructions

### 1. Install WebSocket Dependencies

```bash
pip install -r requirements_web.txt
```

### 2. Start the WebSocket Server

```bash
python web_auction_server.py
```

This will:
- Start a WebSocket server on `ws://localhost:8765`
- Load the auction configuration (agents, players, slots)
- Wait for frontend connections

### 3. Open the Frontend

Open `frontend/index.html` in your web browser. The frontend will automatically:
- Connect to the WebSocket server
- Display connection status
- Enable auction controls when connected

## Usage

### Starting an Auction

1. **Open the Live Auction page** (`frontend/index.html`)
2. **Wait for connection** (green indicator in top-right)
3. **Configure auction settings**:
   - **Auction Type**: Random, Alphabetical, or By Evaluation
   - **By Role**: Check to auction by position (GK → DEF → MID → ATT)
4. **Click "Start Auction"** to begin

### Monitoring Progress

- **Live Auction Page**: Watch individual player auctions in real-time
- **Teams Page**: Monitor team building and squad composition
- **Activity Feed**: Follow all auction activities chronologically

### Real-Time Features

- **Instant Updates**: All changes appear immediately across all connected browsers
- **Multi-viewer Support**: Multiple people can watch the same auction
- **Automatic Reconnection**: Frontend reconnects if connection is lost
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

### Backend (Python)
- **`web_auction_server.py`**: WebSocket server with auction logic
- **`WebAuction` class**: Async wrapper around the existing auction system
- **Real-time broadcasting**: Sends updates to all connected clients

### Frontend (JavaScript)
- **`auction.js`**: Live auction page functionality
- **`teams.js`**: Teams overview page functionality
- **WebSocket client**: Handles real-time communication
- **Responsive UI**: Modern CSS with animations and transitions

### Communication Protocol

The frontend and backend communicate via JSON messages:

```javascript
// Client to Server
{
    "type": "start_auction",
    "auction_type": "random",
    "per_ruolo": true
}

// Server to Client
{
    "type": "bid_update",
    "data": {
        "agent": "cap_agent_1",
        "action": "bid",
        "new_price": 15
    }
}
```

## Message Types

### Client → Server
- `start_auction`: Begin auction with specified configuration
- `get_state`: Request current auction state

### Server → Client
- `state_update`: Complete auction state (teams, current player, etc.)
- `bid_update`: Individual bid/pass actions
- `auction_complete`: Player sale completion
- `role_start`: New position category starting
- `auction_finished`: All auctions completed

## Customization

### Adding New Agents
Modify `web_auction_server.py` in the `setup_auction()` function:

```python
agents = [
    CapAgent(agent_id="your_agent", cap_strategy="bestxi_based"),
    # Add more agents here
]
```

### Changing Auction Configuration
Modify slots, initial credits, or player data in `setup_auction()`:

```python
slots = Slots(gk=1, def_=4, mid=4, att=2)  # Custom formation
initial_credits = 1500  # Higher budget
```

### Styling Customization
Edit `frontend/style.css` to change:
- Colors and themes
- Layout and spacing  
- Animations and transitions
- Responsive breakpoints

## Browser Compatibility

- **Chrome/Edge**: Full support
- **Firefox**: Full support
- **Safari**: Full support
- **Mobile browsers**: Responsive design

## Troubleshooting

### Connection Issues
- Ensure the WebSocket server is running (`python web_auction_server.py`)
- Check that port 8765 is not blocked by firewall
- Verify the server address in the JavaScript files

### Performance
- Activity feed is limited to 50 items for performance
- Sales list is limited to 20 items
- Teams page refreshes every 5 seconds

### Data Issues
- Ensure `data/players_list.xlsx` exists and is properly formatted
- Check agent configurations in `web_auction_server.py`
- Verify slot configurations match your league rules

## Development

### Adding New Features
1. **Backend**: Extend `WebAuction` class with new message types
2. **Frontend**: Add corresponding JavaScript handlers
3. **UI**: Update HTML/CSS for new interface elements

### Testing
- Use browser developer tools to monitor WebSocket messages
- Check server console for auction logic debugging
- Test with multiple browser tabs for multi-user simulation

## Future Enhancements

Possible improvements:
- **User authentication**: Multiple human players with accounts
- **Auction history**: Database storage and replay functionality
- **Advanced analytics**: Player value trends and bidding patterns
- **Mobile app**: Native iOS/Android applications
- **Video streaming**: Integrate with video conferencing for remote auctions
