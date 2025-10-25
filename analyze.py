#!/usr/bin/env python3
"""Command-line interface for analyzing chess positions with the trained engine."""

import argparse
import sys

import torch
import chess
import numpy as np

from src.config.config import Config
from src.neural_net.chess_net import ChessNet
from src.mcts.mcts import MCTS
from src.chess_env.position import Position
from src.chess_env.move_encoder import MoveEncoder


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze chess positions with trained engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a position
  python analyze.py --model checkpoints/best_model.pt --fen "8/8/4k3/4P3/4K3/8/8/8 w - - 0 1"
  
  # Deep analysis with more simulations
  python analyze.py --model checkpoints/best_model.pt --fen "8/8/4k3/4P3/4K3/8/8/8 w - - 0 1" --simulations 800
  
  # Show top 10 moves
  python analyze.py --model checkpoints/best_model.pt --fen "8/8/4k3/4P3/4K3/8/8/8 w - - 0 1" --top-moves 10
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--fen',
        type=str,
        required=True,
        help='Position to analyze in FEN notation'
    )
    
    parser.add_argument(
        '--simulations',
        type=int,
        default=400,
        help='Number of MCTS simulations (default: 400)'
    )
    
    parser.add_argument(
        '--top-moves',
        type=int,
        default=5,
        help='Number of top moves to display (default: 5)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use (default: cuda)'
    )
    
    parser.add_argument(
        '--show-raw',
        action='store_true',
        help='Show raw neural network output before MCTS'
    )
    
    parser.add_argument(
        '--show-stats',
        action='store_true',
        help='Show detailed MCTS statistics'
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
    
    print("Model loaded successfully\n")
    return model


def display_position(position):
    """Display the chess position."""
    print("="*60)
    print("Position:")
    print("="*60)
    print(position.board)
    print()
    print(f"FEN: {position.board.fen()}")
    print(f"Turn: {'White' if position.board.turn == chess.WHITE else 'Black'}")
    print(f"Legal moves: {len(position.get_legal_moves())}")
    print("="*60)


def get_raw_evaluation(model, position, device):
    """Get raw neural network evaluation."""
    board_tensor = torch.from_numpy(position.to_tensor()).float()
    board_tensor = board_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    board_tensor = board_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
    
    policy_logits = policy_logits.cpu().numpy()[0]
    value = value.cpu().numpy()[0, 0]
    
    # Apply legal move mask
    move_mask = MoveEncoder.get_move_mask(position)
    masked_logits = policy_logits * move_mask + (1 - move_mask) * (-1e8)
    
    # Convert to probabilities
    exp_logits = np.exp(masked_logits - np.max(masked_logits))
    policy_probs = exp_logits / np.sum(exp_logits)
    
    return policy_probs, value


def display_raw_evaluation(position, policy_probs, value, top_n=5):
    """Display raw neural network evaluation."""
    print("\n" + "="*60)
    print("Raw Neural Network Evaluation (before MCTS)")
    print("="*60)
    print(f"Position value: {value:+.3f}")
    print(f"\nTop {top_n} moves by neural network prior:")
    
    legal_moves = position.get_legal_moves()
    move_probs = []
    
    for move in legal_moves:
        move_idx = MoveEncoder.encode_move(move)
        prob = policy_probs[move_idx]
        move_probs.append((move, prob))
    
    move_probs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (move, prob) in enumerate(move_probs[:top_n]):
        print(f"  {i+1}. {move.uci()}: {prob:.2%}")
    
    print("="*60)


def display_mcts_analysis(position, mcts, top_n=5, show_stats=False):
    """Display MCTS analysis."""
    print("\n" + "="*60)
    print("MCTS Analysis")
    print("="*60)
    
    # Run MCTS search
    print(f"Running {mcts.num_simulations} simulations...")
    move_probs = mcts.get_move_probabilities(position)
    
    # Sort moves by probability
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} moves after MCTS:")
    for i, (move, prob) in enumerate(sorted_moves[:top_n]):
        print(f"  {i+1}. {move.uci()}: {prob:.2%}")
    
    # Best move
    best_move = sorted_moves[0][0]
    print(f"\nRecommended move: {best_move.uci()}")
    
    # Show what happens after best move
    new_position = position.make_move(best_move)
    if not new_position.is_terminal():
        print(f"Position after {best_move.uci()}:")
        print(new_position.board)
    else:
        result = new_position.get_result()
        if result == 1.0:
            print(f"After {best_move.uci()}: Checkmate!")
        elif result == 0.0:
            print(f"After {best_move.uci()}: Draw")
        else:
            print(f"After {best_move.uci()}: Game over")
    
    if show_stats:
        print(f"\nMCTS Statistics:")
        print(f"  Simulations: {mcts.num_simulations}")
        print(f"  Exploration constant (c_puct): {mcts.c_puct}")
        print(f"  Unique positions explored: ~{mcts.num_simulations}")
    
    print("="*60)


def display_position_assessment(position):
    """Display assessment of the position."""
    print("\n" + "="*60)
    print("Position Assessment")
    print("="*60)
    
    # Check if terminal
    if position.is_terminal():
        result = position.get_result()
        if result == 1.0:
            print("Position: Checkmate (current player wins)")
        elif result == -1.0:
            print("Position: Checkmate (current player loses)")
        else:
            print("Position: Draw")
    else:
        print("Position: In progress")
        
        # Count material
        board = position.board
        white_pieces = len(board.pieces(chess.PAWN, chess.WHITE))
        black_pieces = len(board.pieces(chess.PAWN, chess.BLACK))
        
        print(f"Material: White {white_pieces} pawn(s), Black {black_pieces} pawn(s)")
        
        # Check for special positions
        if white_pieces == 1 and black_pieces == 0:
            print("Type: King and pawn vs King endgame")
        elif white_pieces == 0 and black_pieces == 1:
            print("Type: King vs King and pawn endgame")
        else:
            print("Type: Complex endgame")
    
    print("="*60)


def main():
    """Main analysis entry point."""
    args = parse_args()
    
    # Load model
    model = load_model(args.model, args.device)
    
    # Parse position
    try:
        board = chess.Board(args.fen)
        position = Position(board)
    except ValueError as e:
        print(f"Error: Invalid FEN string: {e}")
        sys.exit(1)
    
    # Display position
    display_position(position)
    
    # Display position assessment
    display_position_assessment(position)
    
    # Check if position is terminal
    if position.is_terminal():
        print("\nPosition is terminal. No analysis needed.")
        return
    
    # Get raw neural network evaluation if requested
    if args.show_raw:
        policy_probs, value = get_raw_evaluation(model, position, args.device)
        display_raw_evaluation(position, policy_probs, value, args.top_moves)
    
    # Create MCTS and run analysis
    mcts = MCTS(
        neural_net=model,
        num_simulations=args.simulations,
        c_puct=1.0,
        device=args.device
    )
    
    display_mcts_analysis(position, mcts, args.top_moves, args.show_stats)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
