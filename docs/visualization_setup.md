# Chess Training Visualization Setup Guide

This guide will help you set up the real-time training visualization dashboard.

## Overview

The visualization system consists of two parts:

1. **Backend API** (FastAPI): Serves training data and manages WebSocket connections
2. **Frontend Dashboard** (React): Displays live training, game history, and metrics

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│  - Live Training Board                                   │
│  - Game History Browser                                  │
│  - Metrics Dashboard                                     │
│  - Control Panel                                         │
└────────────────┬────────────────────────────────────────┘
                 │ WebSocket + REST API
                 ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                        │
│  - WebSocket Server (real-time updates)                 │
│  - REST API (game history, control)                     │
│  - Training Orchestrator Integration                    │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              Training Orchestrator                       │
│  - Self-play generation                                 │
│  - Neural network training                              │
│  - Metrics collection                                   │
└─────────────────────────────────────────────────────────┘
```

## Installation

### 1. Backend Setup

Install API dependencies:

```bash
cd api
pip install -r requirements.txt
```

Or add to your main requirements.txt:

```bash
pip install fastapi uvicorn[standard] websockets python-multipart
```

### 2. Frontend Setup

Install Node.js dependencies:

```bash
cd frontend
npm install
```

This will install:
- React 18
- react-chessboard (chess board visualization)
- chess.js (chess logic)
- recharts (charts and graphs)
- axios (HTTP client)

## Running the System

### Option 1: Development Mode (Recommended for Testing)

**Terminal 1 - Start Backend:**

```bash
cd api
python server.py
```

Backend will run on `http://localhost:8000`

**Terminal 2 - Start Frontend:**

```bash
cd frontend
npm run dev
```

Frontend will run on `http://localhost:3000`

**Terminal 3 - Start Training (Optional):**

```bash
python train.py --config configs/default.yaml
```

Or use the Control Panel in the frontend to start training.

### Option 2: Production Mode

**Build Frontend:**

```bash
cd frontend
npm run build
```

**Serve with Backend:**

```bash
cd api
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then serve the built frontend with a web server (nginx, Apache, etc.)

## Usage

### 1. Access the Dashboard

Open your browser and navigate to `http://localhost:3000`

### 2. Start Training

**Option A: Via Control Panel**
1. Click "Control Panel" tab
2. Select configuration (Quick Test, Default, or Full Training)
3. Click "Start Training"

**Option B: Via Command Line**
```bash
python train.py --config configs/default.yaml
```

### 3. Monitor Training

**Live Training Tab:**
- Watch games being played in real-time
- See MCTS statistics (simulations, nodes explored)
- View neural network evaluations
- Monitor current game status

**Game History Tab:**
- Browse all completed games
- Click any game to view details
- Use controls to replay moves
- See game outcomes and statistics

**Metrics Tab:**
- View win rate over time
- Track draw rate progression
- Monitor training loss
- See curriculum level changes

### 4. Control Training

**Control Panel Tab:**
- Start/stop training
- Change configuration
- View system status
- Access quick actions

## Features

### Live Training Board

- **Real-time Updates**: See moves as they happen
- **MCTS Statistics**: Simulations, nodes explored, search time
- **Neural Network Evaluation**: Position value, move probabilities
- **Game Status**: Check, checkmate, stalemate detection

### Game History Browser

- **Pagination**: Browse through thousands of games
- **Game Replay**: Step through moves with controls
- **Move List**: Click any move to jump to that position
- **Game Metadata**: Timestamp, result, move count

### Metrics Dashboard

- **Interactive Charts**: Win rate, draw rate, loss over time
- **Quick Stats**: Total games, iteration, current metrics
- **Curriculum Tracking**: See which difficulty level is active
- **Real-time Updates**: Charts update as training progresses

### Control Panel

- **Training Control**: Start/stop with one click
- **Configuration Selection**: Choose from presets
- **System Status**: API and WebSocket connection status
- **Quick Actions**: Export data, view logs, clear history

## Configuration

### Backend Configuration

Edit `api/server.py` to customize:

```python
# CORS origins (add your frontend URL)
allow_origins=["http://localhost:3000", "http://localhost:5173"]

# WebSocket settings
# Adjust message frequency, buffer size, etc.
```

### Frontend Configuration

Edit `frontend/vite.config.js` for proxy settings:

```javascript
server: {
  port: 3000,
  proxy: {
    '/api': 'http://localhost:8000',
    '/ws': 'ws://localhost:8000'
  }
}
```

### Training Configuration

Create custom configs in `configs/`:

```yaml
# configs/custom.yaml
target_games: 50000
mcts_simulations: 600
num_workers: 8
evaluation_frequency: 500
```

## Integration with Training Code

To enable real-time updates, modify your training orchestrator:

```python
from api.server import manager  # WebSocket manager

class TrainingOrchestrator:
    def __init__(self, config):
        # ... existing code ...
        self.ws_manager = manager
    
    async def broadcast_move(self, move_data):
        """Send move update to connected clients."""
        await self.ws_manager.broadcast({
            'type': 'move',
            'data': move_data
        })
    
    async def broadcast_game_complete(self, game_data):
        """Send game completion to connected clients."""
        await self.ws_manager.broadcast({
            'type': 'game_complete',
            'data': game_data
        })
```

## Troubleshooting

### Backend Issues

**Port Already in Use:**
```bash
# Change port in server.py
uvicorn.run(app, host="0.0.0.0", port=8001)
```

**CORS Errors:**
```python
# Add your frontend URL to CORS origins
allow_origins=["http://localhost:3000", "http://your-domain.com"]
```

**WebSocket Connection Failed:**
- Check firewall settings
- Verify backend is running
- Check browser console for errors

### Frontend Issues

**Dependencies Not Installing:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Board Not Rendering:**
```bash
# Ensure chess.js and react-chessboard are installed
npm install chess.js react-chessboard
```

**WebSocket Disconnecting:**
- Check network stability
- Increase reconnection timeout
- Verify backend WebSocket endpoint

### Performance Issues

**Slow Chart Rendering:**
- Reduce data points in charts
- Implement data sampling
- Use chart virtualization

**High Memory Usage:**
- Limit game history size
- Implement pagination
- Clear old data periodically

**Laggy Board Updates:**
- Throttle WebSocket messages
- Batch updates
- Optimize React rendering

## Advanced Features

### Custom Themes

Edit `frontend/src/index.css`:

```css
:root {
  --bg-primary: #1a1a1a;
  --bg-secondary: #2d2d2d;
  --text-primary: #e0e0e0;
  --accent: #3b82f6;
}
```

### Data Export

Add export functionality:

```javascript
const exportGames = async () => {
  const response = await fetch('/api/games?limit=1000')
  const data = await response.json()
  const blob = new Blob([JSON.stringify(data, null, 2)])
  // Download blob as file
}
```

### Authentication

Add authentication to backend:

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/api/training/start")
async def start_training(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Verify token
    # Start training
```

## Deployment

### Docker Deployment

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  backend:
    build: ./api
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
```

### Cloud Deployment

**Backend (AWS/GCP/Azure):**
1. Deploy FastAPI with uvicorn
2. Use managed WebSocket service
3. Configure load balancer
4. Set up SSL/TLS

**Frontend (Vercel/Netlify):**
1. Build production bundle
2. Deploy to static hosting
3. Configure environment variables
4. Set up custom domain

## Monitoring

### Backend Monitoring

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
```

### Frontend Monitoring

```javascript
// Add error boundary
class ErrorBoundary extends React.Component {
  componentDidCatch(error, errorInfo) {
    console.error('Error:', error, errorInfo)
    // Send to monitoring service
  }
}
```

## Best Practices

1. **Keep WebSocket Messages Small**: Send only necessary data
2. **Implement Reconnection Logic**: Handle network interruptions
3. **Use Pagination**: Don't load all games at once
4. **Cache Data**: Reduce API calls with local caching
5. **Optimize Rendering**: Use React.memo and useMemo
6. **Monitor Performance**: Track render times and memory usage
7. **Handle Errors Gracefully**: Show user-friendly error messages
8. **Test Thoroughly**: Test with different network conditions

## Support

For issues or questions:
- Check the troubleshooting section
- Review browser console for errors
- Check backend logs
- Verify all dependencies are installed
- Ensure ports are not blocked by firewall

## Future Enhancements

- [ ] Multi-user support with authentication
- [ ] Training comparison tools
- [ ] Advanced MCTS tree visualization
- [ ] Position analysis with engine evaluation
- [ ] Training templates and presets
- [ ] Mobile app version
- [ ] Offline mode with local storage
- [ ] Export to PGN format
- [ ] Integration with chess databases
- [ ] Real-time collaboration features
