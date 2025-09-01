# API Documentation - Fantasy Football Auction

## Panoramica
Questo server API espone le funzionalità dell'asta di fantacalcio per essere collegato a un'interfaccia web. Tutte le API restituiscono JSON e supportano CORS per l'integrazione web.

## Base URL
```
http://localhost:8081/api
```

## Endpoints

### 1. Health Check
**GET** `/health`

Verifica che il server sia attivo.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-09-01T10:30:00"
}
```

### 2. Create Auction
**POST** `/auction/create`

Crea una nuova asta con la configurazione specificata.

**Request Body:**
```json
{
  "agents": [
    {
      "type": "human",
      "id": "player1"
    },
    {
      "type": "cap",
      "id": "bot1"
    }
  ],
  "config": {
    "initial_credits": 1000,
    "slots_gk": 3,
    "slots_def": 8,
    "slots_mid": 8,
    "slots_att": 6
  }
}
```

**Agent Types:**
- `human`: Giocatore umano (richiede input manuale)
- `cap`: Bot con strategia basata sui crediti
- `dynamic_cap`: Bot con strategia dinamica
- `random`: Bot casuale
- `rl_deep`: Bot con reinforcement learning

**Response:**
```json
{
  "success": true,
  "session_id": "uuid-string",
  "message": "Auction created successfully"
}
```

### 3. Get Auction Status
**GET** `/auction/status`

Ottiene lo stato corrente dell'asta.

**Response:**
```json
{
  "state": "running",
  "session_id": "uuid-string",
  "current_player": {
    "name": "Donnarumma",
    "role": "GK",
    "team": "PSG",
    "evaluation": 95,
    "current_price": 75,
    "highest_bidder": "player1"
  },
  "agents": [
    {
      "id": "player1",
      "type": "HumanAgent",
      "credits": 925,
      "squad_size": 1,
      "squad_gk": 1,
      "squad_def": 0,
      "squad_mid": 0,
      "squad_att": 0
    }
  ],
  "slots": {
    "GK": 3,
    "DEF": 8,
    "MID": 8,
    "ATT": 6
  }
}
```

**States:**
- `not_started`: Nessuna asta creata
- `created`: Asta creata ma non iniziata
- `running`: Asta in corso
- `player_auction`: Asta per giocatore specifico in corso
- `completed`: Asta completata

### 4. Start Next Player Auction
**POST** `/auction/next-player`

Inizia l'asta per il prossimo giocatore disponibile.

**Request Body (optional):**
```json
{
  "role_filter": "GK"
}
```

**Response:**
```json
{
  "success": true,
  "player": {
    "name": "Donnarumma",
    "role": "GK",
    "team": "PSG",
    "evaluation": 95
  }
}
```

Se tutti i giocatori sono stati venduti:
```json
{
  "success": true,
  "completed": true,
  "message": "All players have been auctioned"
}
```

### 5. Make Bid
**POST** `/auction/bid`

Fai un'offerta per il giocatore corrente.

**Request Body:**
```json
{
  "agent_id": "player1",
  "amount": 75
}
```

**Response:**
```json
{
  "success": true,
  "current_price": 75,
  "highest_bidder": "player1"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Invalid bid - insufficient credits or no available slots"
}
```

### 6. Process Bot Bids
**POST** `/auction/bot-bids`

Fa processare le offerte automatiche degli agenti bot per il giocatore corrente.

**Response:**
```json
{
  "success": true,
  "bids": [
    {
      "agent_id": "bot1",
      "action": "bid",
      "amount": 76
    },
    {
      "agent_id": "bot2", 
      "action": "pass"
    },
    {
      "agent_id": "bot3",
      "action": "cannot_bid",
      "reason": "insufficient_credits_or_slots"
    }
  ],
  "current_price": 76,
  "highest_bidder": "bot1"
}
```

**Actions possibili:**
- `bid`: L'agente ha fatto un'offerta
- `pass`: L'agente ha passato
- `cannot_bid`: L'agente non può fare offerte

### 7. Finalize Player Auction
**POST** `/auction/finalize`

Finalizza l'asta del giocatore corrente e assegna il giocatore all'offerente più alto.

**Response (venduto):**
```json
{
  "success": true,
  "sold": true,
  "buyer": "player1",
  "price": 75,
  "player": "Donnarumma"
}
```

**Response (invenduto):**
```json
{
  "success": true,
  "sold": false,
  "player": "Donnarumma"
}
```

### 8. Get All Players
**GET** `/players`

Ottiene la lista di tutti i giocatori disponibili.

**Response:**
```json
{
  "success": true,
  "players": [
    {
      "name": "Donnarumma",
      "team": "PSG",
      "role": "GK",
      "evaluation": 95,
      "standardized_evaluation": 0.95,
      "ranking": 1,
      "fantasy_team": "player1",
      "final_cost": 75
    }
  ],
  "total": 1
}
```

### 9. Get Auction Results
**GET** `/auction/results`

Ottiene i risultati finali dell'asta con le rose complete.

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "agent_id": "player1",
      "squad": [
        {
          "name": "Donnarumma",
          "role": "GK",
          "team": "PSG",
          "evaluation": 95,
          "final_cost": 75
        }
      ],
      "metrics": {
        "total_evaluation": 95,
        "total_standardized": 0.95,
        "bestxi_evaluation": 95,
        "bestxi_standardized": 0.95,
        "credits_remaining": 925,
        "credits_spent": 75
      }
    }
  ]
}
```

## Flusso di lavoro tipico

1. **Creare asta**: `POST /auction/create`
2. **Verificare stato**: `GET /auction/status`
3. **Iniziare asta giocatore**: `POST /auction/next-player`
4. **Fare offerte umane**: `POST /auction/bid` (opzionale)
5. **Processare offerte bot**: `POST /auction/bot-bids`
6. **Ripetere 4-5** fino a quando non ci sono più offerte
7. **Finalizzare vendita**: `POST /auction/finalize`
8. **Ripetere 3-7** per tutti i giocatori
9. **Ottenere risultati**: `GET /auction/results`

## Gestione errori

Tutti gli endpoint restituiscono un oggetto con `success: boolean`. In caso di errore:

```json
{
  "success": false,
  "error": "Descrizione dell'errore"
}
```

I codici di stato HTTP vengono utilizzati appropriatamente:
- `200`: Successo
- `400`: Richiesta non valida
- `500`: Errore interno del server

## Note implementative

- Le aste sono gestite in memoria, non persistenti
- Un'istanza server può gestire una sola asta alla volta
- Per aste multiple simultanee serve un'architettura più complessa
- I bot automatici possono fare offerte durante `start_next_player`
