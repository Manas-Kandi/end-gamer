# Training Guide

## Overview

This guide covers the complete training process for the king-pawn endgame chess engine, including hyperparameter tuning, training phases, monitoring, and troubleshooting.

## Training Process

### Quick Start

```bash
# Basic training with default configuration
python train.py --config configs/default.yaml

# Resume from checkpoint
python train.py --config configs/default.yaml --resume checkpoints/checkpoint_25000.pt

# Quick test run (reduced parameters)
python train.py --config configs/quick_test.yaml
```

### Training Phases

The training process consists of three phases with automatic curriculum progression:

#### Phase 1: Foundation (0-25K games)
- **Focus**: Basic endgame patterns and piece coordination
- **Curriculum Level**: 0 (simple positions - pawn far advanced)
- **Temperature**: 1.2 (high exploration)
- **Expected Duration**: ~8-12 hours on GPU
- **Key Metrics**: Win rate should reach 40-50% against random player

#### Phase 2: Refinement (25K-75K games)
- **Focus**: Opposition, king support, and tactical patterns
- **Curriculum Level**: 1 (medium positions - requires king support)
- **Temperature**: 1.0 (balanced exploration/exploitation)
- **Expected Duration**: ~24-36 hours on GPU
- **Key Metrics**: Win rate 60-70%, draw rate 70-80%

#### Phase 3: Mastery (75K-100K games)
- **Focus**: Zugzwang, triangulation, and optimal play
- **Curriculum Level**: 2 (complex positions)
- **Temperature**: 0.8 (low exploration, fine-tuning)
- **Expected Duration**: ~12-16 hours on GPU
- **Key Metrics**: Win rate >85%, draw rate >90%, Elo 1800+

### Total Training Time

- **GPU (RTX 3080 or better)**: 44-64 hours (~2-3 days)
- **GPU (GTX 1080 Ti)**: 60-90 hours (~3-4 days)
- **CPU only**: Not recommended (10-20x slower)

## Hyperparameters

### Neural Network Architecture

```yaml
num_res_blocks: 3        # Number of residual blocks
num_filters: 256         # Filters per convolutional layer
```

**Effects**:
- More blocks/filters → Better performance but slower training
- 3 blocks with 256 filters is optimal for king-pawn endgames
- Increasing to 5 blocks or 512 filters shows diminishing returns

### MCTS Configuration

```yaml
mcts_simulations: 400    # Simulations per move
c_puct: 1.0             # Exploration constant
```

**Effects**:
- **mcts_simulations**: More simulations → stronger play but slower self-play
  - 200: Fast but weaker training signal
  - 400: Balanced (recommended)
  - 800: Stronger but 2x slower
- **c_puct**: Controls exploration vs exploitation
  - 0.5: More exploitation (greedy)
  - 1.0: Balanced (recommended)
  - 2.0: More exploration (diverse games)

### Training Hyperparameters

```yaml
batch_size: 512          # Training batch size
learning_rate: 0.001     # Initial learning rate
weight_decay: 1e-4       # L2 regularization
```

**Effects**:
- **batch_size**: 
  - Larger → More stable gradients, better GPU utilization
  - 512 is optimal for most GPUs
  - Reduce to 256 if running out of memory
- **learning_rate**:
  - Too high (>0.01): Training instability, divergence
  - Too low (<0.0001): Slow convergence
  - 0.001 with decay is recommended
- **weight_decay**:
  - Prevents overfitting
  - 1e-4 is standard for chess networks

### Self-Play Configuration

```yaml
games_per_iteration: 100        # Games generated per iteration
training_steps_per_iteration: 100  # Training steps per iteration
num_workers: 4                  # Parallel self-play workers
```

**Effects**:
- **games_per_iteration**: More games → more diverse data
  - Balance with training_steps_per_iteration
  - Ratio of 1:1 works well
- **num_workers**: 
  - Set to number of CPU cores (up to 8)
  - More workers → faster self-play generation
  - Diminishing returns beyond 8 workers

### Learning Rate Schedule

```yaml
lr_scheduler: step       # Scheduler type
lr_step_size: 25000     # Games between LR reductions
lr_gamma: 0.1           # LR multiplication factor
```

The learning rate is reduced by 10x at 25K, 50K, and 75K games:
- 0-25K: lr = 0.001
- 25K-50K: lr = 0.0001
- 50K-75K: lr = 0.00001
- 75K+: lr = 0.000001

This schedule allows aggressive learning early and fine-tuning later.

### Curriculum Learning

```yaml
curriculum_schedule:
  0: 0        # 0-25K games: simple positions
  25000: 1    # 25K-75K games: medium positions
  75000: 2    # 75K+ games: complex positions
```

**Customization**:
- Adjust thresholds based on training progress
- If model struggles, extend simpler curriculum levels
- For faster training, advance curriculum earlier

## Monitoring Training

### TensorBoard

Launch TensorBoard to visualize training metrics:

```bash
tensorboard --logdir logs/tensorboard
```

**Key Metrics to Watch**:

1. **Loss Curves**
   - `loss/total_loss`: Should decrease steadily
   - `loss/policy_loss`: Measures move prediction accuracy
   - `loss/value_loss`: Measures position evaluation accuracy
   - Expect: Smooth decrease, stabilizing after 50K games

2. **Evaluation Metrics** (logged every 1000 games)
   - `eval/win_rate`: % of winning positions converted to wins
   - `eval/draw_rate`: % of drawn positions held to draws
   - `eval/move_accuracy`: % agreement with tablebase optimal moves
   - `eval/elo_estimate`: Estimated Elo rating

3. **Resource Utilization**
   - `resource/gpu_memory_mb`: GPU memory usage
   - `resource/cpu_percent`: CPU utilization
   - `resource/mcts_nodes_per_sec`: MCTS search speed

### Console Output

Training progress is printed to console:

```
Iteration 10 | Games: 1000/100000 | Buffer: 15234
Curriculum level: 0
Losses - Total: 2.456, Policy: 1.823, Value: 0.633
Evaluation - Win: 45.2%, Draw: 68.5%, Accuracy: 52.3%, Elo: 1245
Time: 1.2h elapsed, ~42.8h remaining
```

### Log Files

Detailed logs are saved to `logs/training.log`:
- Timestamp for each iteration
- Complete metrics history
- Error messages and warnings
- Checkpoint save confirmations

## Evaluation Metrics Interpretation

### Win Rate
- **Target**: >85% by end of training
- **Meaning**: % of theoretically winning positions converted to wins
- **Interpretation**:
  - <40%: Model not learning basic patterns
  - 40-60%: Learning piece coordination
  - 60-80%: Understanding opposition and king support
  - >85%: Strong endgame play

### Draw Rate
- **Target**: >90% by end of training
- **Meaning**: % of theoretical draws held without losing
- **Interpretation**:
  - <50%: Weak defensive play
  - 50-70%: Basic defensive understanding
  - 70-85%: Good defensive technique
  - >90%: Excellent draw-holding ability

### Move Accuracy
- **Target**: >75% by end of training
- **Meaning**: % agreement with Syzygy tablebase optimal moves
- **Interpretation**:
  - <40%: Random-level play
  - 40-60%: Understanding basic principles
  - 60-75%: Strong tactical play
  - >75%: Near-optimal play

### Elo Estimate
- **Target**: 1800+ by end of training
- **Meaning**: Estimated rating based on benchmark matches
- **Interpretation**:
  - <1000: Beginner level
  - 1000-1400: Intermediate level
  - 1400-1800: Advanced level
  - >1800: Expert level (target achieved)

## Optimization Tips

### Faster Training

1. **Reduce MCTS simulations** (400 → 200)
   - 2x faster self-play
   - Slightly weaker training signal
   - Good for initial experiments

2. **Increase parallel workers** (4 → 8)
   - Faster self-play generation
   - Requires more CPU cores
   - Diminishing returns beyond 8

3. **Enable mixed precision training**
   ```yaml
   mixed_precision: true
   ```
   - 1.5-2x faster training
   - Requires modern GPU (Volta or newer)
   - Minimal accuracy impact

4. **Reduce evaluation frequency** (1000 → 5000)
   - Less time spent on evaluation
   - Still get regular progress updates

### Better Performance

1. **Increase MCTS simulations** (400 → 800)
   - Stronger training signal
   - 2x slower self-play
   - Recommended for final training run

2. **Larger replay buffer** (100K → 200K)
   - More diverse training data
   - Requires more RAM
   - Better generalization

3. **More training steps per iteration** (100 → 200)
   - More thorough learning from each batch of games
   - Slower overall training
   - Better sample efficiency

4. **Extended training** (100K → 150K games)
   - Continued improvement beyond 100K
   - Diminishing returns after 120K
   - Recommended for maximum performance

### Memory Optimization

If running out of GPU memory:

1. **Reduce batch size** (512 → 256)
2. **Reduce num_filters** (256 → 128)
3. **Disable mixed precision** (if causing issues)
4. **Reduce num_data_workers** (4 → 2)

If running out of RAM:

1. **Reduce buffer_size** (100K → 50K)
2. **Reduce num_workers** (4 → 2)
3. **Reduce games_per_iteration** (100 → 50)

## Troubleshooting

### Training Not Converging

**Symptoms**: Loss not decreasing, metrics not improving

**Solutions**:
1. Check learning rate (may be too high or too low)
2. Verify data generation (inspect self-play games)
3. Ensure curriculum is progressing appropriately
4. Check for NaN losses (indicates instability)

### Overfitting

**Symptoms**: Training loss decreases but evaluation metrics plateau or worsen

**Solutions**:
1. Increase weight_decay (1e-4 → 1e-3)
2. Increase buffer_size for more diverse data
3. Add more self-play games per iteration
4. Reduce model capacity (fewer blocks/filters)

### Slow Training

**Symptoms**: Training taking much longer than expected

**Solutions**:
1. Check GPU utilization (should be >80%)
2. Increase batch_size for better GPU utilization
3. Increase num_workers for faster self-play
4. Enable mixed precision training
5. Reduce MCTS simulations if acceptable

### Out of Memory

**Symptoms**: CUDA out of memory or system RAM exhausted

**Solutions**:
1. Reduce batch_size
2. Reduce buffer_size
3. Reduce num_workers
4. Use gradient accumulation (modify trainer)
5. Close other applications

### Poor Evaluation Results

**Symptoms**: Metrics much worse than expected for training stage

**Solutions**:
1. Verify evaluation is using correct MCTS settings
2. Check if model is actually learning (inspect loss curves)
3. Ensure test positions are appropriate difficulty
4. Verify tablebase is loaded correctly (if using)

### Checkpoint Corruption

**Symptoms**: Cannot load checkpoint, training crashes on resume

**Solutions**:
1. Use `load_checkpoint_with_fallback()` to try older checkpoints
2. Check disk space (ensure enough space for checkpoints)
3. Verify checkpoint_frequency is not too low
4. Keep multiple checkpoint backups

## Advanced Configuration

### Custom Curriculum

Create custom curriculum for specific training goals:

```yaml
curriculum_schedule:
  0: 0        # Start with simple
  10000: 1    # Advance to medium earlier
  30000: 2    # Advance to complex earlier
  50000: 1    # Return to medium for reinforcement
  70000: 2    # Final complex phase
```

### Temperature Schedule

Customize exploration schedule:

```python
def get_exploration_temperature(self, games_played: int) -> float:
    if games_played < 20000:
        return 1.5  # Very high exploration
    elif games_played < 60000:
        return 1.0  # Balanced
    elif games_played < 90000:
        return 0.7  # Low exploration
    else:
        return 0.5  # Very low exploration (deterministic)
```

### Multi-Stage Training

Train in stages with different configurations:

```bash
# Stage 1: Fast exploration (0-30K)
python train.py --config configs/stage1_exploration.yaml

# Stage 2: Balanced training (30K-70K)
python train.py --config configs/stage2_balanced.yaml --resume checkpoints/checkpoint_30000.pt

# Stage 3: Fine-tuning (70K-100K)
python train.py --config configs/stage3_finetuning.yaml --resume checkpoints/checkpoint_70000.pt
```

## Best Practices

1. **Start with quick test run** to verify setup
2. **Monitor training closely** in first few hours
3. **Save checkpoints frequently** (every 5K games)
4. **Keep backup checkpoints** in case of corruption
5. **Use TensorBoard** for real-time monitoring
6. **Evaluate regularly** to track progress
7. **Document configuration changes** for reproducibility
8. **Use version control** for config files
9. **Test on validation set** before final evaluation
10. **Archive successful training runs** with full logs

## Expected Results Timeline

| Games | Win Rate | Draw Rate | Move Accuracy | Elo | Notes |
|-------|----------|-----------|---------------|-----|-------|
| 1K    | 20-30%   | 40-50%    | 30-40%        | 900 | Random-level play |
| 5K    | 35-45%   | 55-65%    | 40-50%        | 1100 | Basic patterns emerging |
| 10K   | 45-55%   | 65-75%    | 50-60%        | 1300 | Piece coordination |
| 25K   | 55-65%   | 70-80%    | 60-70%        | 1500 | Opposition understanding |
| 50K   | 70-80%   | 80-88%    | 68-75%        | 1650 | Strong tactical play |
| 75K   | 80-88%   | 88-93%    | 73-78%        | 1750 | Advanced techniques |
| 100K  | 85-92%   | 90-95%    | 75-82%        | 1800+ | Expert-level play |

These are approximate ranges - actual results may vary based on hardware, configuration, and random seed.
