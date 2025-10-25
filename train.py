#!/usr/bin/env python3
"""Command-line interface for training the chess engine."""

import argparse
import signal
import sys
from pathlib import Path

from src.config.config import Config
from src.training.training_orchestrator import TrainingOrchestrator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train chess engine on king-pawn endgames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python train.py
  
  # Train with custom configuration
  python train.py --config configs/full_training.yaml
  
  # Resume training from checkpoint
  python train.py --resume checkpoints/checkpoint_50000.pt
  
  # Quick test run
  python train.py --config configs/quick_test.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration YAML file (default: configs/default.yaml)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use for training (overrides config)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel self-play workers (overrides config)'
    )
    
    parser.add_argument(
        '--target-games',
        type=int,
        default=None,
        help='Target number of games to train on (overrides config)'
    )
    
    return parser.parse_args()


def setup_signal_handlers(orchestrator):
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal. Saving checkpoint and shutting down...")
        try:
            checkpoint_path = orchestrator._save_checkpoint(suffix="_interrupted")
            print(f"Checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
        
        print("Cleaning up resources...")
        orchestrator.cleanup()
        
        print("Shutdown complete.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    try:
        config = Config.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Override config with command-line arguments
    if args.device is not None:
        config.device = args.device
        print(f"Overriding device: {args.device}")
    
    if args.num_workers is not None:
        config.num_workers = args.num_workers
        print(f"Overriding num_workers: {args.num_workers}")
    
    if args.target_games is not None:
        config.target_games = args.target_games
        print(f"Overriding target_games: {args.target_games}")
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}")
        sys.exit(1)
    
    # Display configuration summary
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Target games: {config.target_games:,}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"MCTS simulations: {config.mcts_simulations}")
    print(f"Parallel workers: {config.num_workers}")
    print(f"Checkpoint directory: {config.checkpoint_dir}")
    print(f"Log directory: {config.tensorboard_log_dir}")
    print("="*60 + "\n")
    
    # Create training orchestrator
    print("Initializing training orchestrator...")
    try:
        orchestrator = TrainingOrchestrator(config)
    except Exception as e:
        print(f"Error initializing orchestrator: {e}")
        sys.exit(1)
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers(orchestrator)
    
    # Start training
    try:
        print("\nStarting training...\n")
        orchestrator.train(
            resume_from_checkpoint=args.resume,
            progress_callback=display_progress
        )
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        
        # Display final statistics
        info = orchestrator.get_training_info()
        print(f"\nFinal Statistics:")
        print(f"  Total games: {info['total_games']:,}")
        print(f"  Total training steps: {info['total_training_steps']:,}")
        print(f"  Training time: {info['elapsed_time_hours']:.1f} hours")
        print(f"  Buffer utilization: {info['buffer_utilization']:.1%}")
        print(f"  Final learning rate: {info['current_lr']:.6f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        orchestrator.cleanup()


def display_progress(progress_info):
    """Display training progress callback."""
    iteration = progress_info['iteration']
    total_games = progress_info['total_games']
    target_games = progress_info['target_games']
    progress_pct = progress_info['progress_pct']
    iteration_time = progress_info['iteration_time']
    buffer_size = progress_info['buffer_size']
    curriculum_level = progress_info['curriculum_level']
    temperature = progress_info['temperature']
    
    # Calculate estimated time remaining
    if progress_pct > 0:
        total_time_estimate = iteration_time * (100 / progress_pct)
        time_remaining = total_time_estimate - (iteration_time * iteration)
        time_remaining_hours = time_remaining / 3600
    else:
        time_remaining_hours = 0
    
    print(f"\nIteration {iteration} Summary:")
    print(f"  Progress: {total_games:,}/{target_games:,} games ({progress_pct:.1f}%)")
    print(f"  Iteration time: {iteration_time:.1f}s")
    print(f"  Buffer size: {buffer_size:,}")
    print(f"  Curriculum level: {curriculum_level}")
    print(f"  Temperature: {temperature:.2f}")
    if time_remaining_hours > 0:
        print(f"  Estimated time remaining: {time_remaining_hours:.1f} hours")


if __name__ == '__main__':
    main()
