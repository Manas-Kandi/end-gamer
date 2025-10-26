"""FastAPI server with demo/mock data for visualization testing."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import random
from datetime import datetime
import chess

app = FastAPI(title="Chess Training Visualizer API (Demo Mode)")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4010", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo state
demo_state = {
    "is_training": False,
    "game_history": [],
    "metrics": {
        "total_games": 0,
        "iteration": 0,
        "win_rate": 0.0,
        "draw_rate": 0.0,
        "loss": 2.5,
        "curriculum_level": 0
    }
}

# WebSocket connections
active_connections = []


class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.get("/")
async def root():
    return {"status": "ok", "message": "Chess Training Visualizer API (Demo Mode)"}


@app.get("/api/status")
async def get_status():
    return {
        "is_training": demo_state["is_training"],
        "metrics": demo_state["metrics"],
        "config": {
            "target_games": 1000,
            "mcts_simulations": 400,
            "batch_size": 512
        }
    }


@app.post("/api/training/start")
async def start_training():
    if demo_state["is_training"]:
        return {"error": "Training already in progress"}, 400
    
    demo_state["is_training"] = True
    
    # Start demo training in background
    asyncio.create_task(run_demo_training())
    
    return {"status": "started", "config": "demo"}


@app.post("/api/training/stop")
async def stop_training():
    demo_state["is_training"] = False
    return {"status": "stopped"}


@app.get("/api/games")
async def get_games(limit: int = 50, offset: int = 0):
    games = demo_state["game_history"]
    total = len(games)
    games_slice = list(reversed(games))[offset:offset + limit]
    
    return {
        "games": games_slice,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/games/{game_id}")
async def get_game(game_id: int):
    if game_id < 0 or game_id >= len(demo_state["game_history"]):
        return {"error": "Game not found"}, 404
    
    return demo_state["game_history"][game_id]


@app.get("/api/metrics")
async def get_metrics():
    return demo_state["metrics"]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive with ping/pong
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def run_demo_training():
    """Simulate training with fake data."""
    print("🎮 Starting demo training...")
    
    while demo_state["is_training"]:
        # Simulate a game
        game_id = len(demo_state["game_history"])
        moves = generate_random_game()
        result = random.choice([1.0, 0.0, -1.0])
        
        game_data = {
            "id": game_id,
            "timestamp": datetime.now().isoformat(),
            "moves": moves,
            "result": result,
            "num_moves": len(moves),
            "positions": []
        }
        
        demo_state["game_history"].append(game_data)
        
        # Update metrics
        demo_state["metrics"]["total_games"] += 1
        demo_state["metrics"]["iteration"] = demo_state["metrics"]["total_games"] // 10
        demo_state["metrics"]["win_rate"] = min(0.95, demo_state["metrics"]["win_rate"] + random.uniform(0.001, 0.01))
        demo_state["metrics"]["draw_rate"] = min(0.95, demo_state["metrics"]["draw_rate"] + random.uniform(0.001, 0.01))
        demo_state["metrics"]["loss"] = max(0.1, demo_state["metrics"]["loss"] - random.uniform(0.01, 0.05))
        
        if demo_state["metrics"]["total_games"] % 25 == 0:
            demo_state["metrics"]["curriculum_level"] = min(2, demo_state["metrics"]["curriculum_level"] + 1)
        
        # Broadcast game completion
        await manager.broadcast({
            "type": "game_complete",
            "data": game_data
        })
        
        # Broadcast metrics update
        await manager.broadcast({
            "type": "metrics",
            "data": demo_state["metrics"]
        })
        
        # Simulate moves during the game
        board = chess.Board()
        for i, move_uci in enumerate(moves[:10]):  # Show first 10 moves
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                    
                    # Broadcast move
                    await manager.broadcast({
                        "type": "move",
                        "data": {
                            "fen": board.fen(),
                            "move_number": i + 1,
                            "move": move_uci,
                            "mcts_stats": {
                                "simulations": 400,
                                "nodes_explored": random.randint(800, 1500),
                                "search_time": random.uniform(0.3, 0.8)
                            },
                            "evaluation": {
                                "value": random.uniform(-0.5, 0.5),
                                "top_move_prob": random.uniform(0.2, 0.5)
                            }
                        }
                    })
                    
                    await asyncio.sleep(0.5)  # Slow down for visualization
            except:
                pass
        
        # Wait before next game
        await asyncio.sleep(2)
        
        print(f"✅ Demo game {game_id} completed ({len(moves)} moves, result: {result})")


def generate_random_game():
    """Generate a random chess game."""
    board = chess.Board()
    moves = []
    
    max_moves = random.randint(20, 60)
    
    for _ in range(max_moves):
        if board.is_game_over():
            break
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        move = random.choice(legal_moves)
        moves.append(move.uci())
        board.push(move)
    
    return moves


if __name__ == "__main__":
    import uvicorn
    print("🎮 Starting Chess Training Visualizer in DEMO MODE")
    print("=" * 60)
    print("This server generates fake training data for testing")
    print("Real training integration coming soon!")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
