# Chess Training Visualizer - Frontend

A real-time visualization dashboard for monitoring chess engine training progress.

## Features

- **Live Training Board**: Watch games being played in real-time with MCTS statistics
- **Game History Browser**: Browse and replay all completed training games
- **Metrics Dashboard**: Real-time charts showing win rate, draw rate, and loss curves
- **Control Panel**: Start/stop training and configure parameters

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Build for Production

```bash
npm run build
npm run preview
```

## Architecture

### Components

- **App.jsx**: Main application with tab navigation and WebSocket management
- **LiveTraining.jsx**: Real-time game visualization with chessboard
- **GameHistory.jsx**: Paginated game browser with replay functionality
- **MetricsDashboard.jsx**: Training metrics and charts
- **ControlPanel.jsx**: Training control and configuration

### State Management

- WebSocket connection for real-time updates
- REST API for historical data and control
- Local state with React hooks

### Styling

- Custom CSS with dark theme
- Responsive design for mobile and desktop
- Smooth animations and transitions

## API Integration

### WebSocket Events

```javascript
{
  type: 'move',
  data: {
    fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    move_number: 5,
    mcts_stats: { simulations: 400, nodes_explored: 1234 },
    evaluation: { value: 0.45, top_move_prob: 0.32 }
  }
}

{
  type: 'game_complete',
  data: {
    id: 123,
    moves: ['e2e4', 'e7e5', ...],
    result: 1,
    num_moves: 42
  }
}

{
  type: 'metrics',
  data: {
    total_games: 1000,
    win_rate: 0.75,
    draw_rate: 0.85,
    loss: 1.23
  }
}
```

### REST Endpoints

- `GET /api/status` - Get training status
- `POST /api/training/start` - Start training
- `POST /api/training/stop` - Stop training
- `GET /api/games?limit=20&offset=0` - Get game history
- `GET /api/games/{id}` - Get specific game
- `GET /api/metrics` - Get current metrics

## Dependencies

- **react**: UI framework
- **react-chessboard**: Chess board visualization
- **chess.js**: Chess logic and move validation
- **recharts**: Charts and graphs
- **axios**: HTTP client

## Development

### Adding New Features

1. Create component in `src/components/`
2. Add styles in corresponding `.css` file
3. Import and use in `App.jsx`
4. Update API integration if needed

### Customization

- Colors: Edit CSS variables in `index.css`
- Layout: Modify grid layouts in component CSS
- Charts: Configure Recharts components in `MetricsDashboard.jsx`

## Troubleshooting

### WebSocket Connection Issues

- Ensure backend is running on port 8000
- Check browser console for connection errors
- Verify CORS settings in backend

### Board Not Rendering

- Check that `react-chessboard` is installed
- Verify FEN string format
- Check browser console for errors

### Slow Performance

- Reduce chart data points
- Implement virtualization for game list
- Optimize WebSocket message frequency

## Future Enhancements

- [ ] Add authentication
- [ ] Implement data export
- [ ] Add training pause/resume
- [ ] Show detailed MCTS tree visualization
- [ ] Add position analysis tools
- [ ] Implement training comparison
- [ ] Add mobile app version
