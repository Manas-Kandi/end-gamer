"""FastAPI server for chess training visualization."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import Config
from src.training.training_orchestrator import TrainingOrchestrator
from src.chess_env.position import Position
import chess

app = FastAPI(title="Chess Training Visualizer API")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
training_state = {
    "is_training": False,
    "orchestrator": None,
    "config": None,
    "current_game": None,
    "game_history": [],
    "metrics": {
        "total_games": 0,
        "iteration": 0,
        "win_rate": 0.0,
        "draw_rate": 0.0,
        "loss": 0.0,
        "curriculum_level": 0
    }
}

# WebSocket connections
active_connections: List[WebSocket] = []


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Chess Training Visualizer API"}


@app.get("/api/status")
async def get_status():
    """Get current training status."""
    return {
        "is_training": training_state["is_training"],
        "metrics": training_state["metrics"],
        "config": training_state["config"].__dict__ if training_state["config"] else None
    }


@app.post("/api/training/start")
async def start_training(config_path: str = "configs/default.yaml"):
    """Start training with specified configuration."""
    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    try:
        # Load configuration
        config = Config.from_yaml(config_path)
        training_state["config"] = config
        
        # Initialize orchestrator
        orchestrator = TrainingOrchestrator(config)
        training_state["orchestrator"] = orchestrator
        training_state["is_training"] = True
        
        # Start training in background
        asyncio.create_task(run_training())
        
        return {"status": "started", "config": config_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/stop")
async def stop_training():
    """Stop current training."""
    if not training_state["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    training_state["is_training"] = False
    return {"status": "stopped"}


@app.get("/api/games")
async def get_games(limit: int = 50, offset: int = 0):
    """Get game history with pagination."""
    games = training_state["game_history"]
    total = len(games)
    
    # Return most recent games first
    games_slice = list(reversed(games))[offset:offset + limit]
    
    return {
        "games": games_slice,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/games/{game_id}")
async def get_game(game_id: int):
    """Get specific game details."""
    if game_id < 0 or game_id >= len(training_state["game_history"]):
        raise HTTPException(status_code=404, detail="Game not found")
    
    return training_state["game_history"][game_id]


@app.get("/api/metrics")
async def get_metrics():
    """Get current training metrics."""
    return training_state["metrics"]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def run_training():
    """Run training loop with real-time updates."""
    orchestrator = training_state["orchestrator"]
    config = training_state["config"]
    
    # Custom callback for game updates
    def game_callback(game_data: Dict):
        """Called after each game completes."""
        # Store game in history
        game_id = len(training_state["game_history"])
        game_record = {
            "id": game_id,
            "timestamp": datetime.now().isoformat(),
            "moves": game_data["moves"],
            "result": game_data["result"],
            "num_moves": len(game_data["moves"]),
            "positions": game_data["positions"]
        }
        training_state["game_history"].append(game_record)
        
        # Broadcast to connected clients
        asyncio.create_task(manager.broadcast({
            "type": "game_complete",
            "data": game_record
        }))
    
    def move_callback(move_data: Dict):
        """Called after each move in current game."""
        training_state["current_game"] = move_data
        
        # Broadcast to connected clients
        asyncio.create_task(manager.broadcast({
            "type": "move",
            "data": move_data
        }))
    
    def metrics_callback(metrics: Dict):
        """Called when metrics are updated."""
        training_state["metrics"].update(metrics)
        
        # Broadcast to connected clients
        asyncio.create_task(manager.broadcast({
            "type": "metrics",
            "data": metrics
        }))
    
    try:
        # Run training with callbacks
        # Note: This would require modifying TrainingOrchestrator to accept callbacks
        # For now, we'll simulate with a simple loop
        while training_state["is_training"] and orchestrator.total_games < config.target_games:
            # This is a placeholder - actual implementation would integrate with orchestrator
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"Training error: {e}")
        training_state["is_training"] = False


def position_to_dict(position: Position) -> Dict:
    """Convert Position to JSON-serializable dict."""
    board = position.board
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "legal_moves": [move.uci() for move in board.legal_moves],
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
