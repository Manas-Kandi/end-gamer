# Visualization Quick Start Guide

Get up and running with the Chess Training Visualizer in 5 minutes!

## Prerequisites

- Python 3.8+
- Node.js 18+
- npm or yarn

## Installation

### One-Command Setup

```bash
./start_visualization.sh
```

This script will:
1. Install backend dependencies
2. Install frontend dependencies
3. Start the backend API
4. Start the frontend dashboard
5. Open your browser automatically

### Manual Setup

If you prefer to set up manually:

**Backend:**
```bash
cd api
pip install -r requirements.txt
python server.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Access the Dashboard

Open your browser to: **http://localhost:3000**

## Quick Tour

### 1. Live Training Tab 📡

Watch games being played in real-time:
- See the chess board update with each move
- View MCTS statistics (simulations, nodes explored)
- Monitor neural network evaluations
- Track game status (check, checkmate, stalemate)

### 2. Game History Tab 📚

Browse all completed games:
- Click any game card to view details
- Use controls to replay moves: ⏮ ◀ ▶ ⏭
- Click on any move in the list to jump to that position
- See game outcomes and statistics

### 3. Metrics Dashboard Tab 📊

View training progress:
- Win rate over time (line chart)
- Draw rate progression (line chart)
- Training loss curve (line chart)
- Quick stats cards (games, iteration, rates)

### 4. Control Panel Tab ⚙️

Manage training:
- Start/stop training with one click
- Select configuration preset:
  - 🚀 Quick Test (1K games, ~30 min)
  - ⚡ Default (10K games, ~5 hours)
  - 🎯 Full Training (100K games, ~48 hours)
- View system status
- Access quick actions

## Starting Training

### Option 1: From Control Panel (Recommended)

1. Go to Control Panel tab
2. Select a configuration preset
3. Click "▶️ Start Training"
4. Switch to Live Training tab to watch

### Option 2: From Command Line

```bash
python train.py --config configs/default.yaml
```

Then watch progress in the dashboard!

## Common Tasks

### View a Specific Game

1. Go to Game History tab
2. Scroll through game cards
3. Click on any game to view
4. Use replay controls to step through moves

### Monitor Training Progress

1. Go to Metrics Dashboard tab
2. Watch charts update in real-time
3. Check Quick Stats in sidebar
4. View curriculum level progression

### Stop Training

**From Dashboard:**
1. Go to Control Panel tab
2. Click "⏹️ Stop Training"

**From Command Line:**
Press `Ctrl+C` in the terminal running training

## Troubleshooting

### Backend Not Starting

```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Restart backend
cd api && python server.py
```

### Frontend Not Loading

```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### WebSocket Not Connecting

1. Check backend is running: `http://localhost:8000`
2. Check browser console for errors (F12)
3. Verify firewall isn't blocking connections
4. Try refreshing the page

### Board Not Rendering

1. Ensure `react-chessboard` is installed:
   ```bash
   cd frontend
   npm install react-chessboard chess.js
   ```
2. Clear browser cache
3. Restart frontend

## Tips & Tricks

### Keyboard Shortcuts (Game History)

- `←` Previous move
- `→` Next move
- `Home` First move
- `End` Last move

### Performance Tips

1. **Reduce chart data points** if dashboard is slow
2. **Close unused tabs** to save resources
3. **Use Chrome/Edge** for best performance
4. **Limit game history** to recent 1000 games

### Best Practices

1. **Start with Quick Test** to verify setup
2. **Monitor first few games** to ensure training works
3. **Check metrics regularly** to track progress
4. **Save checkpoints frequently** (automatic every 5K games)
5. **Use full training** only when ready for production

## Docker Deployment

For production deployment:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access at: `http://localhost:3000`

## Next Steps

- Read full documentation: `docs/visualization_setup.md`
- Explore training guide: `docs/training_guide.md`
- Check developer guide: `docs/developer_guide.md`
- Review API documentation: `api/README.md`

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review browser console (F12) for errors
3. Check backend logs in terminal
4. Verify all dependencies are installed
5. Try restarting both services

## Features at a Glance

| Feature | Description | Status |
|---------|-------------|--------|
| Live Board | Real-time game visualization | ✅ |
| Game History | Browse and replay games | ✅ |
| Metrics Charts | Win/draw/loss over time | ✅ |
| Control Panel | Start/stop training | ✅ |
| WebSocket | Real-time updates | ✅ |
| REST API | Historical data access | ✅ |
| Docker Support | Easy deployment | ✅ |
| Mobile Responsive | Works on tablets/phones | ✅ |

## Quick Commands Reference

```bash
# Start everything
./start_visualization.sh

# Backend only
cd api && python server.py

# Frontend only
cd frontend && npm run dev

# Build for production
cd frontend && npm run build

# Docker deployment
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop everything
docker-compose down
```

Enjoy visualizing your chess training! 🏆♟️
