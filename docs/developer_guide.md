# Developer Guide

## Architecture Overview

This document provides a comprehensive guide for developers who want to understand, modify, or extend the chess engine codebase.

## System Architecture

### High-Level Design

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                  Training Orchestrator                       │
│              (Coordinates entire pipeline)                   │
└────────────┬────────────────────────────────┬───────────────┘
             │                                │
             ▼                                ▼
┌──────────────────────┐              ┌──────────────────────┐
│   Self-Play Engine   │              │  Training Manager    │
│  (Game Generation)   │──────────────│  (Model Updates)     │
└──────────┬───────────┘              └──────────┬───────────┘
           │                                     │
           ▼                                     ▼
┌──────────────────────┐              ┌──────────────────────┐
│    MCTS Engine       │◄─────────────│   Neural Network     │
│  (Move Selection)    │              │  (Evaluation)        │
└──────────┬───────────┘              └──────────────────────┘
           │
           ▼
┌──────────────────────┐              ┌──────────────────────┐
│  Chess Environment   │              │  Evaluation Suite    │
│  (Game Rules)        │              │  (Performance Tests) │
└──────────────────────┘              └──────────────────────┘
```

### Module Dependencies

```
chess_env/          (No dependencies - foundation layer)
    ├── position.py
    ├── move_encoder.py
    └── position_generator.py

neural_net/         (Depends on: PyTorch)
    ├── chess_net.py
    ├── residual_block.py
    ├── policy_head.py
    └── value_head.py

mcts/               (Depends on: chess_env, neural_net)
    ├── mcts.py
    └── mcts_node.py

training/           (Depends on: all above)
    ├── training_orchestrator.py
    ├── trainer.py
    ├── self_play_worker.py
    ├── replay_buffer.py
    ├── training_example.py
    ├── parallel_self_play.py
    └── checkpoint_utils.py

evaluation/         (Depends on: all above)
    ├── evaluator.py
    ├── test_suite.py
    ├── benchmark_opponents.py
    └── tablebase.py

config/             (No dependencies)
    └── config.py

logging/            (Depends on: config)
    ├── metrics_logger.py
    └── resource_monitor.py

exceptions.py       (No dependencies)
```

## Core Components

### 1. Chess Environment (`src/chess_env/`)

**Purpose**: Encapsulate all chess-specific logic

#### Position Class

```python
class Position:
    """Immutable chess position representation"""
    
    def to_tensor(self) -> np.ndarray:
        """Convert to (12, 8, 8) neural network input
        
        Channels:
        0-5: White pieces (P, N, B, R, Q, K)
        6-11: Black pieces (p, n, b, r, q, k)
        """
```

**Design Decision**: Immutability ensures thread-safety for parallel self-play.

**Extension Point**: To add more piece types, increase channel count and update encoding.

#### MoveEncoder Class

```python
class MoveEncoder:
    """Stateless move encoding/decoding"""
    
    @staticmethod
    def encode_move(move: chess.Move) -> int:
        """Maps move to [0, 4095] using from_square * 64 + to_square"""
```

**Design Decision**: Simple encoding covers all possible moves. Promotions handled by chess.Move.

**Extension Point**: For larger boards, adjust encoding formula.

#### PositionGenerator Class

```python
class PositionGenerator:
    """Generate training positions with curriculum support"""
    
    def generate_position(self) -> Position:
        """Generate based on curriculum_level (0=simple, 1=medium, 2=complex)"""
```

**Design Decision**: Curriculum learning improves training efficiency.

**Extension Point**: Add new curriculum levels or position types.

### 2. Neural Network (`src/neural_net/`)

**Purpose**: Deep learning model for position evaluation

#### Architecture Decisions

1. **ResNet Backbone**: Proven effective for spatial data
2. **Dual Heads**: Separate policy and value prediction
3. **Batch Normalization**: Stabilizes training
4. **Small Model**: ~2M parameters for fast inference

#### ChessNet Class

```python
class ChessNet(nn.Module):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 12, 8, 8) board tensor
        Returns:
            policy: (batch, 4096) move logits
            value: (batch, 1) position evaluation [-1, 1]
        """
```

**Design Decision**: Return logits (not probabilities) to apply legal move masking.

**Extension Point**: 
- Increase `num_res_blocks` for more capacity
- Add attention mechanisms for long-range dependencies
- Implement squeeze-and-excitation blocks

### 3. MCTS Engine (`src/mcts/`)

**Purpose**: Tree search for move selection

#### MCTS Algorithm Flow

```
1. Selection: Traverse tree using UCB scores
2. Expansion: Add children for legal moves
3. Evaluation: Use neural network for leaf nodes
4. Backpropagation: Update visit counts and values
```

#### MCTSNode Class

```python
class MCTSNode:
    """Tree node with UCB-based selection"""
    
    def get_ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """UCB1 formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))"""
```

**Design Decision**: UCB1 balances exploration and exploitation.

**Extension Point**:
- Implement virtual loss for parallel MCTS
- Add Dirichlet noise for root exploration
- Implement progressive widening

#### MCTS Class

```python
class MCTS:
    def search(self, root_position: Position) -> np.ndarray:
        """Run simulations and return improved policy"""
```

**Design Decision**: Stateless search (no tree reuse) for simplicity.

**Extension Point**:
- Implement tree reuse between moves
- Add time-based search termination
- Batch neural network evaluations

### 4. Training Pipeline (`src/training/`)

**Purpose**: Orchestrate self-play and model updates

#### Training Loop

```
while games < target_games:
    1. Generate self-play games (parallel)
    2. Add examples to replay buffer
    3. Sample batches and train network
    4. Evaluate periodically
    5. Save checkpoints
```

#### TrainingOrchestrator Class

```python
class TrainingOrchestrator:
    """Main training coordinator"""
    
    def train(self) -> None:
        """Execute complete training pipeline"""
```

**Design Decision**: Separate self-play and training phases for clarity.

**Extension Point**:
- Implement asynchronous training (train while generating games)
- Add distributed training across multiple GPUs
- Implement prioritized experience replay

#### SelfPlayWorker Class

```python
class SelfPlayWorker:
    """Generate one self-play game"""
    
    def play_game(self) -> List[TrainingExample]:
        """Play game and return training examples"""
```

**Design Decision**: Each worker is independent for parallelization.

**Extension Point**:
- Add opening book for diverse starting positions
- Implement resignation threshold
- Add game augmentation (rotations, reflections)

#### ReplayBuffer Class

```python
class ReplayBuffer:
    """Store and sample training examples"""
    
    def sample_batch(self, batch_size: int) -> Tuple[...]:
        """Uniform random sampling"""
```

**Design Decision**: Simple FIFO buffer with uniform sampling.

**Extension Point**:
- Implement prioritized replay
- Add recency weighting
- Implement reservoir sampling for infinite streams

### 5. Evaluation Framework (`src/evaluation/`)

**Purpose**: Measure model performance

#### Evaluator Class

```python
class Evaluator:
    """Comprehensive performance evaluation"""
    
    def evaluate(self, neural_net: ChessNet) -> Dict[str, float]:
        """Return win_rate, draw_rate, move_accuracy, elo_estimate"""
```

**Design Decision**: Multiple metrics provide comprehensive view of performance.

**Extension Point**:
- Add perplexity metrics
- Implement skill rating (Glicko-2)
- Add position-specific analysis

## Design Patterns

### 1. Immutability Pattern

**Where**: Position class, TrainingExample

**Why**: Thread-safety for parallel processing

**Example**:
```python
def make_move(self, move: chess.Move) -> 'Position':
    """Return NEW position (doesn't modify self)"""
    new_board = deepcopy(self.board)
    new_board.push(move)
    return Position(new_board)
```

### 2. Strategy Pattern

**Where**: Benchmark opponents (RandomPlayer, MinimaxPlayer, etc.)

**Why**: Pluggable opponent implementations

**Example**:
```python
class Player(ABC):
    @abstractmethod
    def select_move(self, position: Position) -> chess.Move:
        pass

class RandomPlayer(Player):
    def select_move(self, position: Position) -> chess.Move:
        return random.choice(position.get_legal_moves())
```

### 3. Factory Pattern

**Where**: Config.from_yaml()

**Why**: Flexible configuration loading

**Example**:
```python
@classmethod
def from_yaml(cls, config_path: str) -> 'Config':
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return cls(**config_dict)
```

### 4. Observer Pattern

**Where**: MetricsLogger

**Why**: Decouple logging from training logic

**Example**:
```python
class MetricsLogger:
    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        self.log_file.write(f"{step},{tag},{value}\n")
```

## Extension Guide

### Adding New Endgames

To extend to other endgames (e.g., rook endgames):

1. **Update PositionGenerator**:
```python
class PositionGenerator:
    def __init__(self, endgame_type: str = 'king_pawn'):
        self.endgame_type = endgame_type
    
    def generate_position(self) -> Position:
        if self.endgame_type == 'king_pawn':
            return self._generate_king_pawn()
        elif self.endgame_type == 'rook':
            return self._generate_rook_endgame()
```

2. **Update Position.to_tensor()** if needed:
   - May need more channels for additional piece types
   - Update neural network input layer accordingly

3. **Update TestSuite**:
   - Add test positions for new endgame type
   - Update evaluation metrics

4. **Retrain Model**:
   - New endgame requires training from scratch
   - May need larger network for more complex endgames

### Adding New Neural Network Architectures

To experiment with different architectures:

1. **Create new network class**:
```python
class TransformerChessNet(nn.Module):
    """Transformer-based architecture"""
    def __init__(self, ...):
        # Implement transformer layers
        pass
```

2. **Update TrainingOrchestrator**:
```python
if config.network_type == 'resnet':
    self.neural_net = ChessNet(...)
elif config.network_type == 'transformer':
    self.neural_net = TransformerChessNet(...)
```

3. **Ensure interface compatibility**:
   - Must accept (batch, 12, 8, 8) input
   - Must return (policy_logits, value) tuple

### Adding New MCTS Variants

To implement MCTS improvements:

1. **Extend MCTS class**:
```python
class ParallelMCTS(MCTS):
    """MCTS with virtual loss for parallelization"""
    
    def search_parallel(self, root_position: Position, 
                       num_threads: int = 4) -> np.ndarray:
        # Implement parallel search with virtual loss
        pass
```

2. **Update SelfPlayWorker**:
```python
self.mcts = ParallelMCTS(neural_net, config.mcts_simulations, config.c_puct)
```

### Adding New Evaluation Metrics

To add custom evaluation metrics:

1. **Extend Evaluator class**:
```python
class Evaluator:
    def _evaluate_tactical_accuracy(self, neural_net: ChessNet) -> float:
        """Measure tactical pattern recognition"""
        # Implement custom metric
        pass
    
    def evaluate(self, neural_net: ChessNet) -> Dict[str, float]:
        metrics = super().evaluate(neural_net)
        metrics['tactical_accuracy'] = self._evaluate_tactical_accuracy(neural_net)
        return metrics
```

2. **Update MetricsLogger**:
```python
def log_evaluation(self, metrics: Dict[str, float], step: int):
    for name, value in metrics.items():
        self.log_scalar(f"eval/{name}", value, step)
```

## Testing Strategy

### Unit Tests

Each module has comprehensive unit tests:

```
tests/
├── test_chess_env/      # Position, MoveEncoder, PositionGenerator
├── test_neural_net/     # Network architecture
├── test_mcts/           # MCTS algorithm
├── test_training/       # Training pipeline
├── test_evaluation/     # Evaluation framework
├── test_config/         # Configuration management
└── test_logging/        # Logging and monitoring
```

### Integration Tests

Test component interactions:

```python
def test_complete_training_iteration():
    """Test full training iteration"""
    config = Config(target_games=100, games_per_iteration=10)
    orchestrator = TrainingOrchestrator(config)
    orchestrator.train()
    assert orchestrator.total_games == 100
```

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_mcts/

# With coverage
pytest --cov=src --cov-report=html

# Parallel execution
pytest -n auto
```

## Performance Optimization

### Profiling

Identify bottlenecks:

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run training iteration
orchestrator.train()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Common Bottlenecks

1. **MCTS Search**: 60-70% of time
   - Solution: Batch neural network evaluations
   - Solution: Implement parallel MCTS

2. **Neural Network Inference**: 20-30% of time
   - Solution: Use mixed precision (FP16)
   - Solution: Optimize batch size

3. **Data Loading**: 5-10% of time
   - Solution: Use DataLoader with multiple workers
   - Solution: Pin memory for faster GPU transfer

### Optimization Checklist

- [ ] Enable mixed precision training
- [ ] Batch neural network evaluations in MCTS
- [ ] Use DataLoader with num_workers > 0
- [ ] Pin memory for GPU tensors
- [ ] Compile model with torch.compile() (PyTorch 2.0+)
- [ ] Use cudnn.benchmark = True
- [ ] Profile and optimize hot paths

## Debugging Tips

### Common Issues

1. **NaN Losses**
   - Check learning rate (may be too high)
   - Verify gradient clipping is enabled
   - Check for invalid positions in training data

2. **Memory Leaks**
   - Ensure no references held to old game data
   - Use `torch.no_grad()` during inference
   - Clear MCTS tree after each game

3. **Slow Training**
   - Profile to identify bottleneck
   - Check GPU utilization (should be >80%)
   - Verify parallel workers are being used

### Debugging Tools

```python
# Enable PyTorch anomaly detection
torch.autograd.set_detect_anomaly(True)

# Log gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# Visualize MCTS tree
def print_tree(node: MCTSNode, depth: int = 0):
    print("  " * depth + f"Move: {node.move}, Visits: {node.visit_count}, Value: {node.get_value()}")
    for child in node.children.values():
        print_tree(child, depth + 1)
```

## Code Style and Conventions

### Python Style

- Follow PEP 8
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use docstrings for all public methods

### Naming Conventions

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Documentation

- Use Google-style docstrings
- Include parameter types and return types
- Provide usage examples for complex functions

Example:
```python
def search(self, root_position: Position) -> np.ndarray:
    """Run MCTS from root position.
    
    Args:
        root_position: Starting chess position
        
    Returns:
        Policy vector (4096,) based on visit counts
        
    Example:
        >>> mcts = MCTS(neural_net, num_simulations=400)
        >>> policy = mcts.search(position)
        >>> best_move_idx = np.argmax(policy)
    """
```

## Contributing

### Development Workflow

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes with tests
3. Run test suite: `pytest`
4. Check code style: `flake8 src/`
5. Update documentation
6. Submit pull request

### Pull Request Checklist

- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No performance regressions
- [ ] Changelog updated

## Resources

### Key Papers

1. **AlphaZero**: Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (2017)
2. **MCTS**: Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012)
3. **ResNets**: He et al., "Deep Residual Learning for Image Recognition" (2015)

### External Libraries

- **python-chess**: Chess move generation and validation
- **PyTorch**: Deep learning framework
- **TensorBoard**: Training visualization
- **python-chess-syzygy**: Tablebase access (optional)

### Useful Links

- [python-chess documentation](https://python-chess.readthedocs.io/)
- [PyTorch tutorials](https://pytorch.org/tutorials/)
- [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [MCTS survey](https://ieeexplore.ieee.org/document/6145622)
