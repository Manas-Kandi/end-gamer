# Chess Engine: King-Pawn Endgames

A neural network-based chess engine specialized in king-pawn endgames using AlphaZero methodology. This project implements reinforcement learning through self-play to achieve expert-level performance (1800+ Elo) in these fundamental chess positions.

## Overview

King-pawn endgames, despite involving only three pieces, contain rich strategic concepts including opposition, key squares, zugzwang, and triangulation. This project uses reinforcement learning to discover these principles through self-play rather than relying on traditional endgame tablebases.

### Key Features

- **AlphaZero-style Architecture**: Combines deep neural networks with Monte Carlo Tree Search (MCTS)
- **Self-Play Training**: Learns optimal strategies without human-crafted knowledge
- **Specialized Focus**: Optimized specifically for king-pawn endgames
- **Real-Time Visualization**: Interactive web dashboard for monitoring training progress
- **Live Game Viewer**: Watch training games unfold in real-time with MCTS statistics
- **Game History Browser**: Browse and replay all training games with move-by-move analysis
- **Scalable Framework**: Designed for extension to other chess endgames
- **Comprehensive Evaluation**: Includes benchmarking against various opponents and tablebase verification
- **Optional Tablebase Integration**: Supports Syzygy tablebases for perfect endgame evaluation

## Project Structure

```
chess-engine-kp-endgame/
├── src/                    # Source code
│   ├── chess/             # Chess environment and logic
│   ├── neural/            # Neural network architecture
│   ├── mcts/              # Monte Carlo Tree Search
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Performance evaluation
│   └── scripts/           # Command-line interfaces
├── tests/                 # Unit and integration tests
├── configs/               # Configuration files
├── data/                  # Training data and positions
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs and metrics
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Package configuration
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended: RTX 3080 or better with 10GB+ VRAM)
- 16GB RAM minimum (32GB recommended)
- 100GB storage space

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/chess-engine-kp-endgame.git
   cd chess-engine-kp-endgame
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

### Alternative Installation with Optional Dependencies

For distributed training support:
```bash
pip install -e ".[distributed]"
```

For development tools:
```bash
pip install -e ".[dev]"
```

For all optional dependencies:
```bash
pip install -e ".[dev,distributed]"
```

### Optional: Syzygy Tablebase Setup

For enhanced evaluation with perfect endgame play:

1. Download Syzygy tablebases from http://tablebase.sesse.net/syzygy/
2. Set the tablebase path:
   ```bash
   export TABLEBASE_PATH=/path/to/tablebases
   ```
3. See `docs/tablebase_usage.md` for detailed setup and usage instructions

Note: Tablebases are optional. The engine works without them using heuristic evaluation.

## Hardware Requirements

### Minimum Requirements
- **GPU**: GTX 1060 or equivalent (6GB VRAM)
- **CPU**: 4-core with 8 threads
- **RAM**: 16GB
- **Storage**: 100GB

### Recommended Requirements
- **GPU**: RTX 3080 or better (10GB+ VRAM)
- **CPU**: 8-core with 16 threads
- **RAM**: 32GB
- **Storage**: 500GB NVMe SSD

### Training Time Estimates
- **Quick test run**: 2-4 hours (1,000 games)
- **Full training**: 50-100 hours (100,000 games) on RTX 3080

## Usage

### Real-Time Training Visualization

Launch the interactive web dashboard to monitor training in real-time:

```bash
# Quick start (starts both backend and frontend)
./start_visualization.sh
```

Or start services separately:

```bash
# Terminal 1: Start backend API
cd api
python server.py

# Terminal 2: Start frontend
cd frontend
npm install  # First time only
npm run dev
```

Then open your browser to `http://localhost:3000` to access:
- **Live Training Board**: Watch games being played in real-time
- **Game History**: Browse and replay all completed games
- **Metrics Dashboard**: View training progress with interactive charts
- **Control Panel**: Start/stop training and adjust configurations

See `docs/visualization_setup.md` for detailed setup instructions.

### Training a Model

Start training with default configuration:
```bash
python train.py --config configs/default.yaml
```

Resume training from checkpoint:
```bash
python train.py --config configs/default.yaml --resume checkpoints/checkpoint_50000.pt
```

Quick test run with reduced parameters:
```bash
python train.py --config configs/quick_test.yaml
```

Or start training from the web dashboard Control Panel.

### Evaluating a Model

Run comprehensive evaluation:
```bash
chess-evaluate --model checkpoints/best_model.pt --suite standard
```

Evaluate against specific opponents:
```bash
chess-evaluate --model checkpoints/best_model.pt --opponent stockfish
```

### Interactive Play

Play against the trained model:
```bash
chess-play --model checkpoints/best_model.pt --color white
```

### Position Analysis

Analyze a specific position:
```bash
chess-analyze --model checkpoints/best_model.pt --fen "8/8/8/8/3k4/8/3P4/3K4 w - - 0 1"
```

## Configuration

The system uses YAML configuration files located in the `configs/` directory:

- `default.yaml`: Standard training configuration
- `quick_test.yaml`: Fast testing with reduced parameters
- `full_training.yaml`: Complete 100K game training run

### Key Configuration Parameters

```yaml
# Neural Network
num_res_blocks: 3
num_filters: 256

# MCTS
mcts_simulations: 400
c_puct: 1.0

# Training
batch_size: 512
learning_rate: 0.001
weight_decay: 1e-4
target_games: 100000

# Hardware
device: "cuda"
num_workers: 8
```

## Performance Targets

The trained model aims to achieve:

- **Win Rate**: >85% in theoretically winning positions
- **Draw Rate**: >90% in theoretical draw positions
- **Move Accuracy**: >80% agreement with tablebase optimal moves
- **Elo Rating**: 1800+ against various opponents

## Development

### Running Tests

Run all tests:
```bash
pytest
```

Run specific test categories:
```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

### Code Quality

Format code:
```bash
black src/ tests/
```

Type checking:
```bash
mypy src/
```

Linting:
```bash
flake8 src/ tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir logs/
```

Key metrics to watch:
- Training loss convergence
- Win rate in evaluation positions
- Elo rating estimates
- MCTS efficiency (nodes per second)

## Troubleshooting

### Common Issues

**CUDA out of memory:**
- Reduce `batch_size` in configuration
- Reduce `mcts_simulations` for inference
- Use mixed precision training

**Slow training:**
- Increase `num_workers` for parallel self-play
- Use faster storage (NVMe SSD)
- Enable mixed precision training

**Poor performance:**
- Increase `target_games` for more training
- Adjust `c_puct` for MCTS exploration
- Verify position generation covers diverse scenarios

### Getting Help

- Check the [Issues](https://github.com/your-username/chess-engine-kp-endgame/issues) page
- Review the troubleshooting guide in the documentation
- Ensure your hardware meets the minimum requirements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{chess_engine_kp_endgame,
  title={Chess Engine: King-Pawn Endgames with AlphaZero},
  author={Chess Engine KP Team},
  year={2024},
  url={https://github.com/your-username/chess-engine-kp-endgame}
}
```

## Acknowledgments

- AlphaZero paper by DeepMind for the foundational methodology
- python-chess library for robust chess logic implementation
- PyTorch team for the deep learning framework
- Chess community for endgame theory and analysis

## Roadmap

- [ ] Complete initial implementation and training
- [ ] Extend to other pawn endgames (KPP, KPvKP)
- [ ] Add support for rook endgames
- [ ] Implement opening book integration
- [ ] Create web interface for interactive play
- [ ] Optimize for mobile deployment