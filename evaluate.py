#!/usr/bin/env python3
"""Command-line interface for evaluating trained chess engine models."""

import argparse
import sys
from pathlib import Path
import json

import torch

from src.config.config import Config
from src.neural_net.chess_net import ChessNet
from src.evaluation.evaluator import Evaluator
from src.evaluation.test_suite import TestSuite


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained chess engine model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with standard test suite
  python evaluate.py --model checkpoints/best_model.pt
  
  # Evaluate against specific opponent
  python evaluate.py --model checkpoints/best_model.pt --opponent stockfish
  
  # Quick evaluation with fewer games
  python evaluate.py --model checkpoints/best_model.pt --quick
  
  # Save results to file
  python evaluate.py --model checkpoints/best_model.pt --output results.json
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--suite',
        type=str,
        default='standard',
        choices=['standard', 'quick', 'comprehensive'],
        help='Test suite to use (default: standard)'
    )
    
    parser.add_argument(
        '--opponent',
        type=str,
        default=None,
        choices=['random', 'minimax_d3', 'minimax_d5', 'stockfish', 'all'],
        help='Specific opponent to evaluate against (default: all)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick evaluation with reduced number of games'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results JSON file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use for evaluation (default: cuda)'
    )
    
    parser.add_argument(
        '--mcts-simulations',
        type=int,
        default=400,
        help='Number of MCTS simulations per move (default: 400)'
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
        # Try to get from checkpoint directly
        num_res_blocks = checkpoint.get('num_res_blocks', 3)
        num_filters = checkpoint.get('num_filters', 256)
    
    # Create model
    model = ChessNet(num_res_blocks=num_res_blocks, num_filters=num_filters)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Display model info
    if 'total_games' in checkpoint:
        print(f"Model trained on: {checkpoint['total_games']:,} games")
    if 'iteration' in checkpoint:
        print(f"Training iteration: {checkpoint['iteration']}")
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    return model


def create_config(args):
    """Create configuration for evaluation."""
    config = Config()
    config.device = args.device
    config.mcts_simulations = args.mcts_simulations
    
    if args.quick:
        config.eval_games = 20
    
    return config


def display_results(results):
    """Display evaluation results in formatted output."""
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    # Position evaluation metrics
    if 'win_rate' in results:
        print(f"\nPosition Evaluation:")
        print(f"  Win rate (winning positions): {results['win_rate']:.1%}")
        print(f"  Draw rate (drawn positions): {results['draw_rate']:.1%}")
        print(f"  Move accuracy (vs tablebase): {results['move_accuracy']:.1%}")
    
    # Opponent benchmarks
    if 'opponent_results' in results:
        print(f"\nOpponent Benchmarks:")
        for opponent, score in results['opponent_results'].items():
            print(f"  vs {opponent}: {score:.1%}")
    
    # Elo rating
    if 'elo_estimate' in results:
        print(f"\nElo Rating:")
        print(f"  Estimated Elo: {results['elo_estimate']:.0f}")
    
    # Performance metrics
    if 'avg_move_time' in results:
        print(f"\nPerformance:")
        print(f"  Average move time: {results['avg_move_time']:.3f}s")
        print(f"  Nodes per second: {results['nodes_per_second']:.0f}")
    
    # Overall assessment
    print(f"\nOverall Assessment:")
    if results.get('win_rate', 0) >= 0.85 and results.get('draw_rate', 0) >= 0.90:
        print("  ✓ Model meets target performance criteria")
    else:
        print("  ✗ Model does not meet target performance criteria")
        if results.get('win_rate', 0) < 0.85:
            print(f"    - Win rate below target (85%): {results.get('win_rate', 0):.1%}")
        if results.get('draw_rate', 0) < 0.90:
            print(f"    - Draw rate below target (90%): {results.get('draw_rate', 0):.1%}")
    
    print("="*60)


def save_results(results, output_path):
    """Save evaluation results to JSON file."""
    print(f"\nSaving results to: {output_path}")
    
    # Convert any non-serializable types
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str, bool, list, dict)):
            serializable_results[key] = value
        else:
            serializable_results[key] = str(value)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print("Results saved successfully")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    """Main evaluation entry point."""
    args = parse_args()
    
    # Load model
    model = load_model(args.model, args.device)
    
    # Create configuration
    config = create_config(args)
    
    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = Evaluator(config)
    
    # Run evaluation
    print("\nRunning evaluation...")
    print(f"Test suite: {args.suite}")
    if args.opponent:
        print(f"Opponent: {args.opponent}")
    print(f"MCTS simulations: {args.mcts_simulations}")
    print()
    
    try:
        if args.opponent and args.opponent != 'all':
            # Evaluate against specific opponent
            results = evaluator.evaluate_against_opponent(model, args.opponent)
        else:
            # Run comprehensive evaluation
            results = evaluator.evaluate(model)
        
        # Display results
        display_results(results)
        
        # Save results if requested
        if args.output:
            save_results(results, args.output)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
