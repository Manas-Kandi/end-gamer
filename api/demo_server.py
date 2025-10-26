"""FastAPI server with demo/mock data for visualization testing."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import random
from datetime import datetime
import chess
import json
from pathlib import Path

app = FastAPI(title="Chess Training Visualizer API (Demo Mode)")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4010", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage configuration
GAMES_DIR = Path("data/games")
GAMES_DIR.mkdir(parents=True, exist_ok=True)
GAMES_PER_FILE = 100

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


def load_all_games():
    """Load all games from disk on startup."""
    games = []
    game_files = sorted(GAMES_DIR.glob("games_*.json"))
    
    for game_file in game_files:
        try:
            with open(game_file, 'r') as f:
                batch = json.load(f)
                games.extend(batch)
        except Exception as e:
            print(f"Error loading {game_file}: {e}")
    
    return games


def save_game(game_data):
    """Save a game to disk, creating new file every 100 games."""
    demo_state["game_history"].append(game_data)
    
    # Determine which file this game belongs to
    game_id = game_data["id"]
    file_index = game_id // GAMES_PER_FILE
    file_path = GAMES_DIR / f"games_{file_index:04d}.json"
    
    # Load existing games in this batch
    if file_path.exists():
        with open(file_path, 'r') as f:
            batch = json.load(f)
    else:
        batch = []
    
    # Add new game
    batch.append(game_data)
    
    # Save batch
    with open(file_path, 'w') as f:
        json.dump(batch, f, indent=2)
    
    print(f"Saved game {game_id} to {file_path}")


def load_games_range(offset, limit):
    """Load games from disk with pagination."""
    all_games = load_all_games()
    total = len(all_games)
    
    # Return most recent games first
    games_reversed = list(reversed(all_games))
    games_slice = games_reversed[offset:offset + limit]
    
    return games_slice, total


# Load existing games on startup
print("Loading existing games from disk...")
demo_state["game_history"] = load_all_games()
demo_state["metrics"]["total_games"] = len(demo_state["game_history"])
print(f"Loaded {len(demo_state['game_history'])} games")

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
    # Load games from disk for pagination
    games_slice, total = load_games_range(offset, limit)
    
    return {
        "games": games_slice,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/games/{game_id}")
async def get_game(game_id: int):
    # Load all games to find specific one
    all_games = load_all_games()
    
    # Find game by ID
    for game in all_games:
        if game["id"] == game_id:
            return game
    
    return {"error": "Game not found"}, 404


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
    print("Starting demo training...")
    
    while demo_state["is_training"]:
        # Simulate a game
        game_id = demo_state["metrics"]["total_games"]
        moves, final_board = generate_random_game()
        
        # Determine actual result from final position
        if final_board.is_checkmate():
            # Winner is the side that just moved (opponent is checkmated)
            result = 1.0 if not final_board.turn else -1.0
            outcome = "checkmate"
            winner = "white" if result == 1.0 else "black"
        elif final_board.is_stalemate():
            result = 0.0
            outcome = "stalemate"
            winner = "draw"
        elif final_board.is_insufficient_material():
            result = 0.0
            outcome = "insufficient_material"
            winner = "draw"
        elif final_board.is_fifty_moves():
            result = 0.0
            outcome = "fifty_move_rule"
            winner = "draw"
        elif final_board.can_claim_threefold_repetition():
            result = 0.0
            outcome = "threefold_repetition"
            winner = "draw"
        else:
            result = 0.0
            outcome = "incomplete"
            winner = "draw"
        
        game_data = {
            "id": game_id,
            "timestamp": datetime.now().isoformat(),
            "moves": moves,
            "result": result,
            "num_moves": len(moves),
            "positions": [],
            "outcome": outcome,
            "winner": winner
        }
        
        # Save game to disk
        save_game(game_data)
        
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
        
        print(f"Game {game_id} completed ({len(moves)} moves, result: {result})")


def generate_random_game():
    """Generate a random chess game that plays until completion."""
    board = chess.Board()
    moves = []
    moves_since_capture = 0
    
    # Play until game is decisively over
    while True:
        # Check for terminal conditions
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            break
        
        # Check for draw by repetition or 50-move rule
        if board.is_fifty_moves() or board.can_claim_threefold_repetition():
            break
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Prefer captures and checks to make games more decisive
        captures = [m for m in legal_moves if board.is_capture(m)]
        checks = [m for m in legal_moves if board.gives_check(m)]
        
        # 70% chance to prefer captures/checks if available
        if (captures or checks) and random.random() < 0.7:
            preferred_moves = captures + checks
            move = random.choice(preferred_moves)
        else:
            move = random.choice(legal_moves)
        
        # Track if this was a capture
        was_capture = board.is_capture(move)
        
        moves.append(move.uci())
        board.push(move)
        
        # Reset counter on capture or pawn move
        if was_capture or board.piece_at(move.to_square).piece_type == chess.PAWN:
            moves_since_capture = 0
        else:
            moves_since_capture += 1
        
        # Safety limits
        if len(moves) > 300:  # Absolute maximum
            break
        if moves_since_capture > 50:  # Too many moves without progress
            break
    
    return moves, board


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Chess Training Visualizer - Demo Mode")
    print("=" * 60)
    print(f"Games directory: {GAMES_DIR.absolute()}")
    print(f"Loaded {len(demo_state['game_history'])} existing games")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
