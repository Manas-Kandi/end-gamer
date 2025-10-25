# Chess Training Visualizer API

FastAPI backend for real-time chess training visualization.

## Overview

This API provides:
- **WebSocket** for real-time training updates
- **REST endpoints** for game history and control
- **Training management** for starting/stopping sessions
- **Metrics tracking** for performance monitoring

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python server.py

# Or with uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Server will be available at `http://localhost:8000`

## API Documentation

### Interactive Docs

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Health Check

```http
GET /
```

**Response:**
```json
{
  "status": "ok",
  "message": "Chess Training Visualizer API"
}
```

### Get Training Status

```http
GET /api/status
```

**Response:**
```json
{
  "is_training": true,
  "metrics": {
    "total_games": 1234,
    "iteration": 12,
    "win_rate": 0.75,
    "draw_rate": 0.85,
    "loss": 1.23,
    "curriculum_level": 1
  },
  "config": {
    "target_games": 100000,
    "mcts_simulations": 400,
    "batch_size": 512
  }
}
```

### Start Training

```http
POST /api/training/start
```

**Request Body:**
```json
{
  "config_path": "configs/default.yaml"
}
```

**Response:**
```json
{
  "status": "started",
  "config": "configs/default.yaml"
}
```

**Error Response:**
```json
{
  "detail": "Training already in progress"
}
```

### Stop Training

```http
POST /api/training/stop
```

**Response:**
```json
{
  "status": "stopped"
}
```

**Error Response:**
```json
{
  "detail": "No training in progress"
}
```

### Get Game History

```http
GET /api/games?limit=50&offset=0
```

**Query Parameters:**
- `limit` (int): Number of games to return (default: 50)
- `offset` (int): Offset for pagination (default: 0)

**Response:**
```json
{
  "games": [
    {
      "id": 123,
      "timestamp": "2024-10-25T12:34:56",
      "moves": ["e2e4", "e7e5", "g1f3"],
      "result": 1,
      "num_moves": 42,
      "positions": ["rnbqkbnr/pppppppp/..."]
    }
  ],
  "total": 1234,
  "limit": 50,
  "offset": 0
}
```

### Get Specific Game

```http
GET /api/games/{game_id}
```

**Path Parameters:**
- `game_id` (int): Game ID

**Response:**
```json
{
  "id": 123,
  "timestamp": "2024-10-25T12:34:56",
  "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
  "result": 1,
  "num_moves": 42,
  "positions": [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
  ]
}
```

### Get Current Metrics

```http
GET /api/metrics
```

**Response:**
```json
{
  "total_games": 1234,
  "iteration": 12,
  "win_rate": 0.75,
  "draw_rate": 0.85,
  "loss": 1.23,
  "curriculum_level": 1
}
```

## WebSocket

### Connect to WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws')

ws.onopen = () => {
  console.log('Connected')
}

ws.onmessage = (event) => {
  const message = JSON.parse(event.data)
  handleMessage(message)
}

ws.onerror = (error) => {
  console.error('WebSocket error:', error)
}

ws.onclose = () => {
  console.log('Disconnected')
}
```

### Message Types

#### Move Update

Sent when a move is made in the current game.

```json
{
  "type": "move",
  "data": {
    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "move_number": 5,
    "move": "e2e4",
    "mcts_stats": {
      "simulations": 400,
      "nodes_explored": 1234,
      "search_time": 0.45
    },
    "evaluation": {
      "value": 0.23,
      "top_move_prob": 0.32
    }
  }
}
```

#### Game Complete

Sent when a game finishes.

```json
{
  "type": "game_complete",
  "data": {
    "id": 123,
    "timestamp": "2024-10-25T12:34:56",
    "moves": ["e2e4", "e7e5", "g1f3"],
    "result": 1,
    "num_moves": 42,
    "positions": ["rnbqkbnr/pppppppp/..."]
  }
}
```

#### Metrics Update

Sent when training metrics are updated.

```json
{
  "type": "metrics",
  "data": {
    "total_games": 1234,
    "iteration": 12,
    "win_rate": 0.75,
    "draw_rate": 0.85,
    "loss": 1.23,
    "curriculum_level": 1
  }
}
```

## Data Models

### Game

```python
{
  "id": int,              # Unique game identifier
  "timestamp": str,       # ISO 8601 timestamp
  "moves": List[str],     # List of moves in UCI format
  "result": float,        # 1.0 (win), 0.0 (draw), -1.0 (loss)
  "num_moves": int,       # Total number of moves
  "positions": List[str]  # List of FEN strings
}
```

### Metrics

```python
{
  "total_games": int,      # Total games played
  "iteration": int,        # Current training iteration
  "win_rate": float,       # Win rate [0.0, 1.0]
  "draw_rate": float,      # Draw rate [0.0, 1.0]
  "loss": float,           # Current training loss
  "curriculum_level": int  # Current curriculum level (0, 1, 2)
}
```

### Position

```python
{
  "fen": str,                    # FEN string
  "turn": str,                   # "white" or "black"
  "legal_moves": List[str],      # Legal moves in UCI format
  "is_check": bool,              # Is current player in check
  "is_checkmate": bool,          # Is position checkmate
  "is_stalemate": bool           # Is position stalemate
}
```

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### HTTP Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## CORS Configuration

The API allows requests from:
- `http://localhost:3000` (Vite dev server)
- `http://localhost:5173` (Alternative Vite port)

To add more origins, edit `server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://your-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Rate Limiting

Currently no rate limiting is implemented. For production:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/games")
@limiter.limit("100/minute")
async def get_games():
    # ...
```

## Authentication

Currently no authentication is required. For production:

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/api/training/start")
async def start_training(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify token
    token = credentials.credentials
    # Validate token
    # ...
```

## Logging

Configure logging in `server.py`:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

## Testing

### Manual Testing

```bash
# Health check
curl http://localhost:8000/

# Get status
curl http://localhost:8000/api/status

# Start training
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/default.yaml"}'

# Get games
curl http://localhost:8000/api/games?limit=10

# Stop training
curl -X POST http://localhost:8000/api/training/stop
```

### WebSocket Testing

```javascript
// test_websocket.js
const WebSocket = require('ws')

const ws = new WebSocket('ws://localhost:8000/ws')

ws.on('open', () => {
  console.log('Connected')
  ws.send('ping')
})

ws.on('message', (data) => {
  console.log('Received:', data.toString())
})

ws.on('close', () => {
  console.log('Disconnected')
})
```

## Deployment

### Development

```bash
python server.py
```

### Production

```bash
# With uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4

# With gunicorn
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker

```bash
# Build image
docker build -t chess-training-api .

# Run container
docker run -p 8000:8000 chess-training-api
```

### Environment Variables

```bash
# .env file
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
LOG_LEVEL=INFO
```

## Performance

### Optimization Tips

1. **Use connection pooling** for database connections
2. **Implement caching** for frequently accessed data
3. **Batch WebSocket messages** to reduce overhead
4. **Use async operations** for I/O-bound tasks
5. **Implement pagination** for large datasets

### Monitoring

```python
from prometheus_client import Counter, Histogram

request_count = Counter('api_requests_total', 'Total API requests')
request_duration = Histogram('api_request_duration_seconds', 'Request duration')

@app.middleware("http")
async def monitor_requests(request, call_next):
    request_count.inc()
    with request_duration.time():
        response = await call_next(request)
    return response
```

## Contributing

1. Follow FastAPI best practices
2. Add type hints to all functions
3. Write docstrings for endpoints
4. Add tests for new features
5. Update this documentation

## License

Same as main project license.
