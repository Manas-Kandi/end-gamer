#!/usr/bin/env python3
"""Command-line interface for playing against the trained chess engine."""

import argparse
import sys

import torch
import chess

from src.config.config import Config
from src.neural_net.chess_net import ChessNet
from src.mcts.mcts import MCTS
from src.chess_env.position import Position


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Play against trained chess engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play as white against the engine
  python play.py --model checkpoints/best_model.pt --color white
  
  # Play as black with faster engine
  python play.py --model checkpoints/best_model.pt --color black --simulations 200
  
  # Start from custom position
  python play.py --model checkpoints/best_model.pt --fen "8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--color',
        type=str,
        default='white',
        choices=['white', 'black'],
        help='Color to play as (default: white)'
    )
    
    parser.add_argument(
        '--fen',
        type=str,
        default=None,
        help='Starting position in FEN notation (default: king-pawn endgame)'
    )
    
    parser.add_argument(
        '--simulations',
        type=int,
        default=400,
        help='Number of MCTS simulations per move (default: 400)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use (default: cuda)'
    )
    
    parser.add_argument(
        '--show-analysis',
        action='store_true',
        help='Show move analysis and top alternatives'
    )
    
    return parser.parse_args()


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Get model architecture parameters
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_res_blocks = config.num_res_blocks
        num_filters = config.num_filters
    else:
        num_res_blocks = checkpoint.get('num_res_blocks', 3)
        num_filters = checkpoint.get('num_filters', 256)
    
    # Create and load model
    model = ChessNet(num_res_blocks=num_res_blocks, num_filters=num_filters)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model


def display_board(position):
    """Display the chess board."""
    print("\n" + "="*40)
    print(position.board)
    print("="*40)
    print(f"FEN: {position.board.fen()}")
    print()


def get_human_move(position):
    """Get move input from human player."""
    legal_moves = position.get_legal_moves()
    
    print("Legal moves:")
    move_list = [move.uci() for move in legal_moves]
    for i, move in enumerate(move_list):
        print(f"  {move}", end="  ")
        if (i + 1) % 6 == 0:
            print()
    print("\n")
    
    while True:
        move_str = input("Enter your move (e.g., e2e4) or 'quit' to exit: ").strip().lower()
        
        if move_str == 'quit':
            return None
        
        try:
            move = chess.Move.from_uci(move_str)
            if move in legal_moves:
                return move
            else:
                print("Illegal move. Please try again.")
        except ValueError:
            print("Invalid move format. Use UCI notation (e.g., e2e4).")


def get_engine_move(position, mcts, show_analysis=False):
    """Get move from chess engine."""
    print("Engine is thinking...")
    
    # Run MCTS search
    move_probs = mcts.get_move_probabilities(position)
    
    # Get best move
    best_move = max(move_probs.items(), key=lambda x: x[1])[0]
    
    if show_analysis:
        # Show top moves
        print("\nEngine analysis:")
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        for i, (move, prob) in enumerate(sorted_moves[:5]):
            print(f"  {i+1}. {move.uci()}: {prob:.1%}")
        print()
    
    print(f"Engine plays: {best_move.uci()}")
    return best_move


def check_game_over(position):
    """Check if game is over and return result message."""
    if not position.is_terminal():
        return None
    
    result = position.get_result()
    
    if result == 1.0:
        return "Checkmate! Current player wins."
    elif result == -1.0:
        return "Checkmate! Current player loses."
    else:
        return "Game drawn."


def main():
    """Main interactive play entry point."""
    args = parse_args()
    
    # Load model
    model = load_model(args.model, args.device)
    
    # Create MCTS
    mcts = MCTS(
        neural_net=model,
        num_simulations=args.simulations,
        c_puct=1.0,
        device=args.device
    )
    
    # Set up starting position
    if args.fen:
        try:
            board = chess.Board(args.fen)
            position = Position(board)
        except ValueError as e:
            print(f"Error: Invalid FEN string: {e}")
            sys.exit(1)
    else:
        # Default king-pawn endgame
        board = chess.Board("8/8/4k3/4P3/4K3/8/8/8 w - - 0 1")
        position = Position(board)
    
    # Determine who plays first
    human_color = chess.WHITE if args.color == 'white' else chess.BLACK
    
    print("\n" + "="*40)
    print("Chess Engine - Interactive Play")
    print("="*40)
    print(f"You are playing as: {args.color}")
    print(f"Engine simulations: {args.simulations}")
    print("="*40)
    
    # Game loop
    move_count = 0
    max_moves = 200
    
    while move_count < max_moves:
        # Display board
        display_board(position)
        
        # Check if game is over
        game_over_msg = check_game_over(position)
        if game_over_msg:
            print(game_over_msg)
            break
        
        # Determine whose turn it is
        current_player = position.board.turn
        
        if current_player == human_color:
            # Human's turn
            print("Your turn:")
            move = get_human_move(position)
            
            if move is None:
                print("Game quit by user.")
                break
        else:
            # Engine's turn
            print("Engine's turn:")
            move = get_engine_move(position, mcts, args.show_analysis)
        
        # Make the move
        position = position.make_move(move)
        move_count += 1
        print()
    
    if move_count >= max_moves:
        print(f"Game ended after {max_moves} moves (maximum reached).")
    
    # Display final position
    display_board(position)
    
    print("\nThanks for playing!")


if __name__ == '__main__':
    main()
