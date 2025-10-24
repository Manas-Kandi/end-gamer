# Design Document

## Overview

This design document outlines the architecture and implementation strategy for a reinforcement learning-based chess engine specialized in king-pawn endgames. The system uses AlphaZero methodology, combining deep neural networks with Monte Carlo Tree Search (MCTS) to learn optimal play through self-play training.

### Core Design Principles

1. **Modularity**: Separate concerns between chess logic, neural network, MCTS, and training pipeline
2. **Scalability**: Design for parallel self-play and potential extension to other endgames
3. **Reproducibility**: Ensure deterministic behavior with proper seeding and configuration management
4. **Efficiency**: Optimize for training speed and inference performance
5. **Testability**: Enable comprehensive testing of individual components

### System Architecture

The system consists of five major subsystems:

1. **Chess Environment**: Position representation, move generation, and game rules
2. **Neural Network**: Policy and value prediction using deep learning
3. **MCTS Engine**: Tree search for move selection during self-play
4. **Training Pipeline**: Self-play generation, data collection, and model optimization
5. **Evaluation Framework**: Performance measurement and benchmarking

## Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Orchestrator                    │
│  - Configuration Management                                  │
│  - Training Loop Coordination                                │
│  - Checkpoint Management                                     │
└────────────┬────────────────────────────────┬───────────────┘
             │                                │
             ▼                                ▼
┌──────────────────────┐              ┌──────────────────────┐
│   Self-Play Engine   │              │  Training Manager    │
│  - Parallel Workers  │              │  - Data Batching     │
│  - Game Generation   │──────────────│  - Model Updates     │
│  - Data Collection   │              │  - Loss Computation  │
└──────────┬───────────┘              └──────────┬───────────┘
           │                                     │
           ▼                                     ▼
┌──────────────────────┐              ┌──────────────────────┐
│    MCTS Engine       │              │   Neural Network     │
│  - Tree Search       │◄─────────────│  - Policy Head       │
│  - UCB Selection     │              │  - Value Head        │
│  - NN Evaluation     │              │  - ResNet Backbone   │
└──────────┬───────────┘              └──────────────────────┘
           │
           ▼
┌──────────────────────┐              ┌──────────────────────┐
│  Chess Environment   │              │  Evaluation Suite    │
│  - Position Encoding │              │  - Benchmark Tests   │
│  - Move Generation   │              │  - Elo Calculation   │
│  - Game Rules        │              │  - Metrics Tracking  │
└──────────────────────┘              └──────────────────────┘
```

### Component Interaction Flow

**Training Iteration:**
1. Training Orchestrator initializes configuration and loads checkpoint
2. Self-Play Engine spawns parallel workers to generate games
3. Each worker uses MCTS Engine + Neural Network to play games
4. MCTS queries Chess Environment for legal moves and game state
5. Generated game data flows to Training Manager
6. Training Manager batches data and updates Neural Network
7. Evaluation Suite periodically tests new model
8. Training Orchestrator saves checkpoints and logs metrics

## Components and Interfaces

### 1. Chess Environment Module

**Purpose**: Encapsulate all chess-specific logic including position representation, move generation, and game rules.

**Key Classes:**

```python
class Position:
    """Represents a chess position in king-pawn endgame"""
    
    def __init__(self, board: chess.Board):
        self.board = board
        
    def to_tensor(self) -> np.ndarray:
        """Convert position to 8x8x12 neural network input"""
        # Returns: (8, 8, 12) numpy array
        
    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves in current position"""
        
    def make_move(self, move: chess.Move) -> 'Position':
        """Return new position after making move"""
        
    def is_terminal(self) -> bool:
        """Check if position is game-ending"""
        
    def get_result(self) -> float:
        """Get game result from current player perspective: 1.0 (win), 0.0 (draw), -1.0 (loss)"""
        
    def get_canonical_form(self) -> 'Position':
        """Return position from current player's perspective"""
```


```python
class MoveEncoder:
    """Encode/decode moves to neural network format"""
    
    @staticmethod
    def encode_move(move: chess.Move) -> int:
        """Convert chess.Move to integer index [0, 4095]"""
        # from_square * 64 + to_square
        
    @staticmethod
    def decode_move(move_idx: int) -> chess.Move:
        """Convert integer index to chess.Move"""
        
    @staticmethod
    def get_move_mask(position: Position) -> np.ndarray:
        """Get binary mask for legal moves (4096,)"""
```

```python
class PositionGenerator:
    """Generate random king-pawn endgame positions"""
    
    def __init__(self, curriculum_level: int = 0):
        self.curriculum_level = curriculum_level
        
    def generate_position(self) -> Position:
        """Generate random legal king-pawn position"""
        # Curriculum levels:
        # 0: Simple positions (pawn far advanced)
        # 1: Medium positions (requires king support)
        # 2: Complex positions (zugzwang, triangulation)
        
    def is_valid_kp_endgame(self, board: chess.Board) -> bool:
        """Verify position is valid king-pawn endgame"""
```

**Design Decisions:**

- Use `python-chess` library for move generation and validation (battle-tested, efficient)
- Canonical form ensures neural network always evaluates from current player's perspective
- Move encoding uses simple from-to representation (64×64 = 4096 possibilities)
- Position generator supports curriculum learning through difficulty levels

### 2. Neural Network Module

**Purpose**: Implement the deep learning model for position evaluation and move prediction.

**Architecture Details:**

```python
class ChessNet(nn.Module):
    """AlphaZero-style neural network for chess endgames"""
    
    def __init__(self, num_res_blocks: int = 3, num_filters: int = 256):
        super().__init__()
        
        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(12, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_head = PolicyHead(num_filters)
        
        # Value head
        self.value_head = ValueHead(num_filters)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 12, 8, 8) board tensor
        Returns:
            policy: (batch, 4096) move probabilities
            value: (batch, 1) position evaluation
        """
        x = self.input_conv(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
            
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
```


```python
class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
```

```python
class PolicyHead(nn.Module):
    """Policy head for move prediction"""
    
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 8 * 8, 4096)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # Logits, softmax applied with masking
```

```python
class ValueHead(nn.Module):
    """Value head for position evaluation"""
    
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x  # Range [-1, 1]
```

**Design Decisions:**

- ResNet architecture chosen for its proven effectiveness in image-like data
- 3 residual blocks balance model capacity with training speed
- 256 filters provide sufficient representational power for endgames
- Separate policy and value heads allow independent optimization
- Policy head outputs logits (softmax applied after legal move masking)
- Value head uses tanh activation to bound output to [-1, 1]

### 3. MCTS Engine Module

**Purpose**: Implement Monte Carlo Tree Search for move selection during self-play.

**Key Classes:**

```python
class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, position: Position, parent: Optional['MCTSNode'] = None, 
                 prior_prob: float = 0.0, move: Optional[chess.Move] = None):
        self.position = position
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior_prob = prior_prob  # P(s,a) from neural network
        
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.visit_count = 0
        self.total_value = 0.0
        
    def get_value(self) -> float:
        """Get mean action value Q(s,a)"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
        
    def get_ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """Calculate UCB score for node selection"""
        q_value = self.get_value()
        u_value = c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q_value + u_value
        
    def is_leaf(self) -> bool:
        """Check if node is a leaf (not expanded)"""
        return len(self.children) == 0
```


```python
class MCTS:
    """Monte Carlo Tree Search implementation"""
    
    def __init__(self, neural_net: ChessNet, num_simulations: int = 400, 
                 c_puct: float = 1.0, device: str = 'cuda'):
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        
    def search(self, root_position: Position) -> np.ndarray:
        """
        Run MCTS from root position
        Returns: policy vector (4096,) based on visit counts
        """
        root = MCTSNode(root_position)
        
        # Expand root
        self._expand_node(root)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree to leaf
            while not node.is_leaf() and not node.position.is_terminal():
                node = self._select_child(node)
                search_path.append(node)
            
            # Expansion and evaluation
            value = self._evaluate_node(node)
            
            # Backpropagation
            self._backpropagate(search_path, value)
        
        # Return policy based on visit counts
        return self._get_policy_from_visits(root)
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score"""
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            score = child.get_ucb_score(self.c_puct, node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def _expand_node(self, node: MCTSNode) -> None:
        """Expand node by adding children for all legal moves"""
        if node.position.is_terminal():
            return
            
        # Get neural network predictions
        policy_logits, _ = self._evaluate_position(node.position)
        
        # Create child nodes
        legal_moves = node.position.get_legal_moves()
        for move in legal_moves:
            move_idx = MoveEncoder.encode_move(move)
            prior_prob = policy_logits[move_idx]
            
            child_position = node.position.make_move(move)
            child_node = MCTSNode(child_position, parent=node, 
                                 prior_prob=prior_prob, move=move)
            node.children[move] = child_node
    
    def _evaluate_node(self, node: MCTSNode) -> float:
        """Evaluate node and expand if not terminal"""
        if node.position.is_terminal():
            return node.position.get_result()
        
        if node.is_leaf():
            self._expand_node(node)
        
        _, value = self._evaluate_position(node.position)
        return value
    
    def _evaluate_position(self, position: Position) -> Tuple[np.ndarray, float]:
        """Get neural network evaluation of position"""
        board_tensor = torch.from_numpy(position.to_tensor()).float()
        board_tensor = board_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.neural_net(board_tensor)
        
        # Apply legal move mask
        move_mask = position.get_move_mask()
        policy_logits = policy_logits.cpu().numpy()[0]
        policy_logits = policy_logits * move_mask + (1 - move_mask) * -1e8
        policy_probs = softmax(policy_logits)
        
        value = value.cpu().numpy()[0, 0]
        
        return policy_probs, value
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float) -> None:
        """Backpropagate value through search path"""
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += value
            value = -value  # Flip value for opponent
    
    def _get_policy_from_visits(self, root: MCTSNode) -> np.ndarray:
        """Convert visit counts to policy distribution"""
        policy = np.zeros(4096)
        
        for move, child in root.children.items():
            move_idx = MoveEncoder.encode_move(move)
            policy[move_idx] = child.visit_count
        
        # Normalize
        if policy.sum() > 0:
            policy = policy / policy.sum()
        
        return policy
```

**Design Decisions:**

- UCB1 formula balances exploration and exploitation
- Neural network provides both prior probabilities and leaf evaluation
- Visit count distribution used as improved policy target
- Value flipped during backpropagation (zero-sum game)
- Legal move masking ensures only valid moves considered


### 4. Training Pipeline Module

**Purpose**: Orchestrate self-play game generation, data collection, and neural network training.

**Key Classes:**

```python
class SelfPlayWorker:
    """Worker process for generating self-play games"""
    
    def __init__(self, neural_net: ChessNet, config: Config):
        self.mcts = MCTS(neural_net, config.mcts_simulations, config.c_puct)
        self.position_generator = PositionGenerator()
        self.config = config
        
    def play_game(self) -> List[TrainingExample]:
        """Play one self-play game and return training examples"""
        examples = []
        
        # Generate starting position
        position = self.position_generator.generate_position()
        
        move_count = 0
        while not position.is_terminal() and move_count < 200:
            # Run MCTS to get improved policy
            canonical_position = position.get_canonical_form()
            policy = self.mcts.search(canonical_position)
            
            # Store training example
            examples.append(TrainingExample(
                position=canonical_position.to_tensor(),
                policy=policy,
                value=None  # Filled in after game ends
            ))
            
            # Sample move from policy (with temperature)
            move = self._sample_move(policy, position, temperature=1.0)
            position = position.make_move(move)
            move_count += 1
        
        # Get game result
        result = position.get_result()
        
        # Fill in values (alternating for each player)
        for i, example in enumerate(examples):
            example.value = result if i % 2 == 0 else -result
        
        return examples
    
    def _sample_move(self, policy: np.ndarray, position: Position, 
                     temperature: float) -> chess.Move:
        """Sample move from policy with temperature"""
        legal_moves = position.get_legal_moves()
        move_probs = []
        
        for move in legal_moves:
            move_idx = MoveEncoder.encode_move(move)
            move_probs.append(policy[move_idx])
        
        # Apply temperature
        if temperature == 0:
            # Greedy
            best_idx = np.argmax(move_probs)
            return legal_moves[best_idx]
        else:
            move_probs = np.array(move_probs) ** (1.0 / temperature)
            move_probs = move_probs / move_probs.sum()
            move_idx = np.random.choice(len(legal_moves), p=move_probs)
            return legal_moves[move_idx]
```

```python
@dataclass
class TrainingExample:
    """Single training example from self-play"""
    position: np.ndarray  # (12, 8, 8)
    policy: np.ndarray    # (4096,)
    value: float          # {-1, 0, 1}
```

```python
class ReplayBuffer:
    """Store and sample training examples"""
    
    def __init__(self, max_size: int = 100000):
        self.buffer: Deque[TrainingExample] = deque(maxlen=max_size)
        
    def add_examples(self, examples: List[TrainingExample]) -> None:
        """Add training examples to buffer"""
        self.buffer.extend(examples)
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random batch for training"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        positions = []
        policies = []
        values = []
        
        for idx in indices:
            example = self.buffer[idx]
            positions.append(example.position)
            policies.append(example.policy)
            values.append(example.value)
        
        positions = torch.FloatTensor(np.array(positions))
        policies = torch.FloatTensor(np.array(policies))
        values = torch.FloatTensor(np.array(values)).unsqueeze(1)
        
        return positions, policies, values
    
    def __len__(self) -> int:
        return len(self.buffer)
```


```python
class Trainer:
    """Manages neural network training"""
    
    def __init__(self, neural_net: ChessNet, config: Config):
        self.neural_net = neural_net
        self.config = config
        self.optimizer = optim.Adam(
            neural_net.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=25000, gamma=0.1
        )
        
    def train_step(self, positions: torch.Tensor, target_policies: torch.Tensor,
                   target_values: torch.Tensor) -> Dict[str, float]:
        """Perform one training step"""
        self.neural_net.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        pred_policies, pred_values = self.neural_net(positions)
        
        # Compute losses
        policy_loss = F.cross_entropy(pred_policies, target_policies)
        value_loss = F.mse_loss(pred_values, target_values)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.neural_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
```

```python
class TrainingOrchestrator:
    """Coordinates the entire training process"""
    
    def __init__(self, config: Config):
        self.config = config
        self.neural_net = ChessNet(
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters
        ).to(config.device)
        
        self.trainer = Trainer(self.neural_net, config)
        self.replay_buffer = ReplayBuffer(max_size=config.buffer_size)
        self.evaluator = Evaluator(config)
        
        self.iteration = 0
        self.total_games = 0
        
    def train(self) -> None:
        """Main training loop"""
        while self.total_games < self.config.target_games:
            # Self-play phase
            print(f"Generating self-play games (iteration {self.iteration})...")
            new_examples = self._generate_self_play_games(
                num_games=self.config.games_per_iteration
            )
            self.replay_buffer.add_examples(new_examples)
            self.total_games += self.config.games_per_iteration
            
            # Training phase
            print(f"Training neural network...")
            self._train_network(num_steps=self.config.training_steps_per_iteration)
            
            # Evaluation phase
            if self.total_games % self.config.eval_frequency == 0:
                print(f"Evaluating model...")
                metrics = self.evaluator.evaluate(self.neural_net)
                self._log_metrics(metrics)
            
            # Checkpoint
            if self.total_games % self.config.checkpoint_frequency == 0:
                self._save_checkpoint()
            
            self.iteration += 1
    
    def _generate_self_play_games(self, num_games: int) -> List[TrainingExample]:
        """Generate self-play games in parallel"""
        # Use multiprocessing or Ray for parallel generation
        with mp.Pool(processes=self.config.num_workers) as pool:
            worker = SelfPlayWorker(self.neural_net, self.config)
            game_results = pool.starmap(worker.play_game, [()] * num_games)
        
        # Flatten list of lists
        all_examples = []
        for game_examples in game_results:
            all_examples.extend(game_examples)
        
        return all_examples
    
    def _train_network(self, num_steps: int) -> None:
        """Train network for specified number of steps"""
        for step in range(num_steps):
            # Sample batch
            positions, policies, values = self.replay_buffer.sample_batch(
                self.config.batch_size
            )
            
            # Move to device
            positions = positions.to(self.config.device)
            policies = policies.to(self.config.device)
            values = values.to(self.config.device)
            
            # Training step
            losses = self.trainer.train_step(positions, policies, values)
            
            # Log losses
            if step % 100 == 0:
                self._log_losses(losses)
    
    def _save_checkpoint(self) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'model_state_dict': self.neural_net.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'config': self.config
        }
        path = f"{self.config.checkpoint_dir}/checkpoint_{self.total_games}.pt"
        torch.save(checkpoint, path)
```

**Design Decisions:**

- Parallel self-play using multiprocessing for efficiency
- Replay buffer stores recent examples with maximum size limit
- Temperature sampling during self-play for exploration
- Gradient clipping prevents training instability
- Learning rate scheduling for fine-tuning in later stages
- Regular checkpointing enables recovery from failures


### 5. Evaluation Framework Module

**Purpose**: Measure model performance against benchmarks and track training progress.

**Key Classes:**

```python
class Evaluator:
    """Evaluate model performance"""
    
    def __init__(self, config: Config):
        self.config = config
        self.test_positions = self._load_test_positions()
        self.tablebase = self._load_tablebase()
        
    def evaluate(self, neural_net: ChessNet) -> Dict[str, float]:
        """Run comprehensive evaluation"""
        metrics = {}
        
        # Win rate in winning positions
        metrics['win_rate'] = self._evaluate_win_rate(neural_net)
        
        # Draw rate in drawn positions
        metrics['draw_rate'] = self._evaluate_draw_rate(neural_net)
        
        # Move quality vs tablebase
        metrics['move_accuracy'] = self._evaluate_move_accuracy(neural_net)
        
        # Elo estimate
        metrics['elo_estimate'] = self._estimate_elo(neural_net)
        
        return metrics
    
    def _evaluate_win_rate(self, neural_net: ChessNet) -> float:
        """Test win rate in theoretically winning positions"""
        mcts = MCTS(neural_net, num_simulations=400)
        wins = 0
        total = 0
        
        for position in self.test_positions['winning']:
            result = self._play_position_to_end(position, mcts)
            if result == 1.0:  # Win
                wins += 1
            total += 1
        
        return wins / total if total > 0 else 0.0
    
    def _evaluate_draw_rate(self, neural_net: ChessNet) -> float:
        """Test draw rate in theoretical draw positions"""
        mcts = MCTS(neural_net, num_simulations=400)
        draws = 0
        total = 0
        
        for position in self.test_positions['drawn']:
            result = self._play_position_to_end(position, mcts)
            if result == 0.0:  # Draw
                draws += 1
            total += 1
        
        return draws / total if total > 0 else 0.0
    
    def _evaluate_move_accuracy(self, neural_net: ChessNet) -> float:
        """Compare moves with tablebase optimal moves"""
        mcts = MCTS(neural_net, num_simulations=400)
        correct_moves = 0
        total = 0
        
        for position in self.test_positions['all']:
            # Get model's best move
            policy = mcts.search(position)
            model_move = self._get_best_move_from_policy(policy, position)
            
            # Get tablebase optimal move
            optimal_move = self.tablebase.get_best_move(position)
            
            if model_move == optimal_move:
                correct_moves += 1
            total += 1
        
        return correct_moves / total if total > 0 else 0.0
    
    def _estimate_elo(self, neural_net: ChessNet) -> float:
        """Estimate Elo rating through benchmark matches"""
        # Play against opponents of known strength
        results = {}
        
        opponents = [
            ('random', RandomPlayer(), 800),
            ('minimax_d3', MinimaxPlayer(depth=3), 1200),
            ('minimax_d5', MinimaxPlayer(depth=5), 1500),
            ('stockfish_limited', StockfishPlayer(depth=10), 1800)
        ]
        
        for name, opponent, opponent_elo in opponents:
            score = self._play_match(neural_net, opponent, num_games=50)
            results[name] = (score, opponent_elo)
        
        # Estimate Elo based on results
        estimated_elo = self._calculate_elo_from_results(results)
        return estimated_elo
    
    def _play_position_to_end(self, position: Position, mcts: MCTS, 
                             max_moves: int = 200) -> float:
        """Play position until terminal or max moves"""
        move_count = 0
        
        while not position.is_terminal() and move_count < max_moves:
            policy = mcts.search(position)
            move = self._get_best_move_from_policy(policy, position)
            position = position.make_move(move)
            move_count += 1
        
        if position.is_terminal():
            return position.get_result()
        else:
            # Timeout - check tablebase
            return self.tablebase.probe(position)
```

```python
class MetricsLogger:
    """Log training metrics to TensorBoard and files"""
    
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.log_file = open(f"{log_dir}/training.log", 'w')
        
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar metric"""
        self.writer.add_scalar(tag, value, step)
        
    def log_losses(self, losses: Dict[str, float], step: int) -> None:
        """Log training losses"""
        for name, value in losses.items():
            self.log_scalar(f"loss/{name}", value, step)
        
    def log_evaluation(self, metrics: Dict[str, float], step: int) -> None:
        """Log evaluation metrics"""
        for name, value in metrics.items():
            self.log_scalar(f"eval/{name}", value, step)
        
        # Also write to log file
        log_msg = f"Step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.log_file.write(log_msg + "\n")
        self.log_file.flush()
```

**Design Decisions:**

- Separate test sets for winning, drawn, and mixed positions
- Tablebase integration for ground truth verification
- Elo estimation through matches against calibrated opponents
- TensorBoard integration for real-time monitoring
- Comprehensive metrics covering multiple aspects of performance


## Data Models

### Configuration Model

```python
@dataclass
class Config:
    """Training configuration"""
    
    # Neural Network
    num_res_blocks: int = 3
    num_filters: int = 256
    
    # MCTS
    mcts_simulations: int = 400
    c_puct: float = 1.0
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 512
    buffer_size: int = 100000
    
    # Self-Play
    games_per_iteration: int = 100
    num_workers: int = 8
    temperature: float = 1.0
    
    # Training Schedule
    target_games: int = 100000
    training_steps_per_iteration: int = 100
    eval_frequency: int = 1000
    checkpoint_frequency: int = 5000
    
    # Curriculum Learning
    curriculum_schedule: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 0),      # 0-25K games: level 0 (simple)
        (25000, 1),  # 25K-75K games: level 1 (medium)
        (75000, 2)   # 75K-100K games: level 2 (complex)
    ])
    
    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    data_dir: str = './data'
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
```

### Training State Model

```python
@dataclass
class TrainingState:
    """Snapshot of training state for checkpointing"""
    
    iteration: int
    total_games: int
    model_state_dict: Dict
    optimizer_state_dict: Dict
    scheduler_state_dict: Dict
    replay_buffer: List[TrainingExample]
    config: Config
    metrics_history: List[Dict[str, float]]
    timestamp: str
    
    def save(self, path: str) -> None:
        """Save training state to disk"""
        torch.save(asdict(self), path)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingState':
        """Load training state from disk"""
        state_dict = torch.load(path)
        return cls(**state_dict)
```

### Test Suite Model

```python
@dataclass
class TestSuite:
    """Collection of test positions for evaluation"""
    
    winning_positions: List[Position]
    drawn_positions: List[Position]
    easy_positions: List[Position]
    medium_positions: List[Position]
    hard_positions: List[Position]
    
    @classmethod
    def from_pgn(cls, path: str) -> 'TestSuite':
        """Load test positions from PGN file"""
        # Parse PGN and categorize positions
        pass
    
    @classmethod
    def generate_standard_suite(cls) -> 'TestSuite':
        """Generate standard test suite with known positions"""
        # Include classic king-pawn endgame positions
        pass
```

## Error Handling

### Error Categories and Strategies

**1. Neural Network Errors**

```python
class NeuralNetworkError(Exception):
    """Base exception for neural network errors"""
    pass

class ModelLoadError(NeuralNetworkError):
    """Failed to load model checkpoint"""
    pass

class InferenceError(NeuralNetworkError):
    """Error during model inference"""
    pass
```

**Handling Strategy:**
- Validate checkpoint integrity before loading
- Implement fallback to previous checkpoint on load failure
- Catch CUDA out-of-memory errors and reduce batch size
- Log detailed error information for debugging

**2. MCTS Errors**

```python
class MCTSError(Exception):
    """Base exception for MCTS errors"""
    pass

class SearchTimeoutError(MCTSError):
    """MCTS search exceeded time limit"""
    pass

class InvalidNodeError(MCTSError):
    """Attempted operation on invalid node"""
    pass
```

**Handling Strategy:**
- Implement timeout mechanism for MCTS search
- Validate node state before operations
- Gracefully handle positions with no legal moves
- Return best available move if search incomplete

**3. Chess Logic Errors**

```python
class ChessError(Exception):
    """Base exception for chess logic errors"""
    pass

class IllegalMoveError(ChessError):
    """Attempted illegal move"""
    pass

class InvalidPositionError(ChessError):
    """Invalid chess position"""
    pass
```

**Handling Strategy:**
- Validate all moves before execution
- Verify position legality during generation
- Use python-chess validation as source of truth
- Log illegal move attempts for debugging

**4. Training Pipeline Errors**

```python
class TrainingError(Exception):
    """Base exception for training errors"""
    pass

class DataGenerationError(TrainingError):
    """Error during self-play data generation"""
    pass

class CheckpointError(TrainingError):
    """Error saving/loading checkpoint"""
    pass
```

**Handling Strategy:**
- Implement retry logic for failed self-play games
- Validate checkpoint integrity before overwriting
- Maintain multiple checkpoint versions
- Continue training with available data on partial failures

### Error Recovery Mechanisms

**Checkpoint Recovery:**
```python
def load_checkpoint_with_fallback(checkpoint_paths: List[str]) -> TrainingState:
    """Try loading checkpoints in order until success"""
    for path in checkpoint_paths:
        try:
            return TrainingState.load(path)
        except Exception as e:
            logging.warning(f"Failed to load {path}: {e}")
            continue
    raise CheckpointError("No valid checkpoint found")
```

**Graceful Degradation:**
```python
def safe_mcts_search(mcts: MCTS, position: Position, 
                     timeout: float = 10.0) -> np.ndarray:
    """Run MCTS with timeout and fallback"""
    try:
        with time_limit(timeout):
            return mcts.search(position)
    except SearchTimeoutError:
        logging.warning("MCTS timeout, using neural network policy")
        return mcts._evaluate_position(position)[0]
```


## Testing Strategy

### Unit Testing

**Chess Environment Tests:**
```python
class TestPosition(unittest.TestCase):
    def test_board_encoding(self):
        """Test position to tensor conversion"""
        # Verify correct channel encoding
        # Verify spatial relationships preserved
        
    def test_legal_moves(self):
        """Test legal move generation"""
        # Test various positions
        # Verify all legal moves found
        # Verify no illegal moves included
        
    def test_move_execution(self):
        """Test move application"""
        # Verify position updated correctly
        # Verify immutability of original position
        
    def test_terminal_detection(self):
        """Test game ending detection"""
        # Test checkmate detection
        # Test stalemate detection
        # Test 50-move rule
```

**Neural Network Tests:**
```python
class TestChessNet(unittest.TestCase):
    def test_forward_pass(self):
        """Test neural network forward pass"""
        # Verify output shapes
        # Verify value range [-1, 1]
        # Verify policy sums to 1 after softmax
        
    def test_gradient_flow(self):
        """Test backpropagation"""
        # Verify gradients computed
        # Verify no gradient explosion
        
    def test_model_save_load(self):
        """Test checkpoint save/load"""
        # Verify model state preserved
        # Verify predictions identical after reload
```

**MCTS Tests:**
```python
class TestMCTS(unittest.TestCase):
    def test_node_selection(self):
        """Test UCB selection"""
        # Verify highest UCB score selected
        # Verify exploration vs exploitation balance
        
    def test_backpropagation(self):
        """Test value backpropagation"""
        # Verify visit counts updated
        # Verify values flipped correctly
        
    def test_policy_extraction(self):
        """Test policy from visit counts"""
        # Verify normalization
        # Verify most visited move has highest probability
```

### Integration Testing

**Self-Play Integration:**
```python
class TestSelfPlay(unittest.TestCase):
    def test_complete_game(self):
        """Test full self-play game"""
        # Verify game completes
        # Verify training examples generated
        # Verify values assigned correctly
        
    def test_parallel_generation(self):
        """Test parallel self-play"""
        # Verify multiple workers function
        # Verify no data corruption
        # Verify performance improvement
```

**Training Pipeline Integration:**
```python
class TestTrainingPipeline(unittest.TestCase):
    def test_training_iteration(self):
        """Test complete training iteration"""
        # Verify self-play generation
        # Verify data added to buffer
        # Verify network updated
        # Verify metrics logged
        
    def test_checkpoint_recovery(self):
        """Test training resumption"""
        # Save checkpoint mid-training
        # Load and resume
        # Verify state preserved
```

### Performance Testing

**Benchmarks:**
```python
class TestPerformance(unittest.TestCase):
    def test_inference_speed(self):
        """Test neural network inference time"""
        # Target: <10ms per position
        
    def test_mcts_speed(self):
        """Test MCTS search time"""
        # Target: ~1 second for 400 simulations
        
    def test_self_play_throughput(self):
        """Test games generated per hour"""
        # Measure with different worker counts
        
    def test_memory_usage(self):
        """Test peak memory consumption"""
        # Verify within hardware limits
```

### Validation Testing

**Model Quality Tests:**
```python
class TestModelQuality(unittest.TestCase):
    def test_basic_positions(self):
        """Test performance on simple positions"""
        # King + pawn vs king (far advanced)
        # Should win >95%
        
    def test_opposition(self):
        """Test understanding of opposition"""
        # Positions requiring opposition
        # Compare with tablebase
        
    def test_zugzwang(self):
        """Test zugzwang recognition"""
        # Classic zugzwang positions
        # Verify correct evaluation
```

### Test Data

**Standard Test Positions:**
- Lucena position (winning technique)
- Philidor position (drawing technique)
- Key square positions
- Opposition positions
- Triangulation positions
- Edge cases (stalemate traps)

**Test Coverage Goals:**
- Unit tests: >80% code coverage
- Integration tests: All major workflows
- Performance tests: All critical paths
- Validation tests: Representative position set

## Deployment and Usage

### Command-Line Interface

```python
# Training
python train.py --config config.yaml

# Resume training
python train.py --config config.yaml --resume checkpoints/checkpoint_50000.pt

# Evaluation
python evaluate.py --model checkpoints/best_model.pt --test-suite data/test_positions.pgn

# Play against model
python play.py --model checkpoints/best_model.pt --human-color white

# Analyze position
python analyze.py --model checkpoints/best_model.pt --fen "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"
```

### Configuration Files

**config.yaml:**
```yaml
# Neural Network
num_res_blocks: 3
num_filters: 256

# MCTS
mcts_simulations: 400
c_puct: 1.0

# Training
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 512
buffer_size: 100000

# Self-Play
games_per_iteration: 100
num_workers: 8
temperature: 1.0

# Schedule
target_games: 100000
training_steps_per_iteration: 100
eval_frequency: 1000
checkpoint_frequency: 5000

# Paths
checkpoint_dir: ./checkpoints
log_dir: ./logs
data_dir: ./data
```

### Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config.yaml .

# Run training
CMD ["python", "src/train.py", "--config", "config.yaml"]
```

### API Interface

```python
class ChessEngineAPI:
    """Simple API for using trained model"""
    
    def __init__(self, model_path: str):
        self.neural_net = ChessNet()
        self.neural_net.load_state_dict(torch.load(model_path))
        self.mcts = MCTS(self.neural_net)
        
    def get_best_move(self, fen: str) -> str:
        """Get best move for position"""
        position = Position.from_fen(fen)
        policy = self.mcts.search(position)
        move = self._get_best_move_from_policy(policy, position)
        return move.uci()
    
    def evaluate_position(self, fen: str) -> float:
        """Get position evaluation"""
        position = Position.from_fen(fen)
        _, value = self.mcts._evaluate_position(position)
        return value
    
    def get_principal_variation(self, fen: str, depth: int = 5) -> List[str]:
        """Get principal variation"""
        position = Position.from_fen(fen)
        pv = []
        
        for _ in range(depth):
            if position.is_terminal():
                break
            policy = self.mcts.search(position)
            move = self._get_best_move_from_policy(policy, position)
            pv.append(move.uci())
            position = position.make_move(move)
        
        return pv
```

## Performance Optimization

### Training Optimizations

**1. Data Loading:**
- Use DataLoader with multiple workers
- Pin memory for faster GPU transfer
- Prefetch batches during training

**2. Mixed Precision Training:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    pred_policies, pred_values = neural_net(positions)
    loss = compute_loss(pred_policies, pred_values, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**3. Distributed Training:**
```python
# Use PyTorch DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model)

# Or use Ray for distributed self-play
@ray.remote
class SelfPlayWorker:
    def play_game(self):
        # Self-play logic
        pass
```

### Inference Optimizations

**1. Batch Inference:**
```python
def batch_evaluate_positions(positions: List[Position]) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate multiple positions in single forward pass"""
    batch_tensor = torch.stack([torch.from_numpy(p.to_tensor()) for p in positions])
    with torch.no_grad():
        policies, values = neural_net(batch_tensor)
    return policies.cpu().numpy(), values.cpu().numpy()
```

**2. Model Quantization:**
```python
# Post-training quantization for faster inference
quantized_model = torch.quantization.quantize_dynamic(
    neural_net, {nn.Linear}, dtype=torch.qint8
)
```

**3. ONNX Export:**
```python
# Export to ONNX for deployment
dummy_input = torch.randn(1, 12, 8, 8)
torch.onnx.export(neural_net, dummy_input, "model.onnx")
```

### Memory Optimizations

**1. Gradient Checkpointing:**
```python
# Trade compute for memory
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    x = self.input_conv(x)
    for res_block in self.res_blocks:
        x = checkpoint(res_block, x)
    return x
```

**2. Replay Buffer Management:**
```python
# Use memory-mapped files for large buffers
import numpy as np

class DiskBackedReplayBuffer:
    def __init__(self, max_size: int, data_dir: str):
        self.positions = np.memmap(
            f"{data_dir}/positions.dat",
            dtype='float32',
            mode='w+',
            shape=(max_size, 12, 8, 8)
        )
        # Similar for policies and values
```

## Monitoring and Debugging

### Logging Strategy

**Levels:**
- DEBUG: Detailed information for debugging
- INFO: General information about training progress
- WARNING: Unexpected behavior that doesn't stop training
- ERROR: Errors that require attention

**What to Log:**
- Training losses (every batch)
- Evaluation metrics (every N games)
- MCTS statistics (periodically)
- Resource usage (GPU memory, CPU)
- Checkpoint saves
- Errors and exceptions

### Visualization

**TensorBoard Dashboards:**
1. Training Dashboard: Losses over time
2. Evaluation Dashboard: Win rate, draw rate, Elo
3. Performance Dashboard: Games/hour, inference time
4. Resource Dashboard: GPU utilization, memory usage

**Custom Visualizations:**
```python
def visualize_policy(position: Position, policy: np.ndarray):
    """Visualize policy distribution on board"""
    # Create heatmap of move probabilities
    # Overlay on chess board
    pass

def plot_mcts_tree(root: MCTSNode, depth: int = 3):
    """Visualize MCTS search tree"""
    # Show visit counts and values
    # Highlight principal variation
    pass
```

### Debugging Tools

**Position Debugger:**
```python
def debug_position(position: Position, neural_net: ChessNet):
    """Detailed analysis of position"""
    print(f"FEN: {position.to_fen()}")
    print(f"Legal moves: {position.get_legal_moves()}")
    
    policy, value = neural_net.evaluate(position)
    print(f"Value: {value:.3f}")
    print(f"Top moves:")
    for move, prob in get_top_moves(policy, position, k=5):
        print(f"  {move}: {prob:.3f}")
```

**Training Debugger:**
```python
def check_training_health(trainer: Trainer):
    """Check for training issues"""
    # Check gradient norms
    # Check weight updates
    # Check loss trends
    # Detect overfitting
    pass
```

This comprehensive design provides a solid foundation for implementing the chess engine. The modular architecture allows for independent development and testing of components, while the detailed interfaces ensure smooth integration.
