# Tablebase Integration Usage Guide

This document explains how to use the Syzygy tablebase integration in the chess engine.

## Overview

The tablebase integration provides access to perfect endgame play through Syzygy tablebases. This is an optional feature that enhances evaluation accuracy but is not required for the engine to function.

## Installation

### 1. Install python-chess with tablebase support

The tablebase integration uses the `chess.syzygy` module from python-chess, which is included by default:

```bash
pip install python-chess
```

### 2. Download Syzygy Tablebases

Download the Syzygy tablebase files you need. For king-pawn endgames, you'll need 3-piece tablebases:

- Download from: http://tablebase.sesse.net/syzygy/
- For 3-piece endgames: Download the 3-men files (KPK, etc.)
- For more comprehensive coverage: Download 4-piece and 5-piece files

Example directory structure:
```
/path/to/tablebases/
├── KPK.rtbw
├── KPK.rtbz
├── KQK.rtbw
├── KQK.rtbz
└── ...
```

## Configuration

### Method 1: Environment Variable

Set the `TABLEBASE_PATH` environment variable:

```bash
export TABLEBASE_PATH=/path/to/tablebases
```

Then initialize the tablebase interface without arguments:

```python
from src.evaluation.tablebase import TablebaseInterface

tb = TablebaseInterface()
if tb.is_available():
    print(f"Tablebase loaded with {tb.get_max_pieces()} piece support")
```

### Method 2: Direct Path

Pass the path directly when initializing:

```python
from src.evaluation.tablebase import TablebaseInterface

tb = TablebaseInterface(tablebase_path="/path/to/tablebases")
if tb.is_available():
    print("Tablebase loaded successfully")
```

### Method 3: With Evaluator

Pass the tablebase path when creating an evaluator:

```python
from src.evaluation.evaluator import Evaluator
from src.config.config import Config

config = Config()
evaluator = Evaluator(config, tablebase_path="/path/to/tablebases")
```

## Usage Examples

### Basic Position Evaluation

```python
from src.evaluation.tablebase import TablebaseInterface
from src.chess_env.position import Position
import chess

# Initialize tablebase
tb = TablebaseInterface("/path/to/tablebases")

# Create a position
board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
position = Position(board)

# Probe for evaluation
result = tb.probe(position)
if result is not None:
    if result > 0:
        print("White is winning")
    elif result < 0:
        print("Black is winning")
    else:
        print("Position is drawn")
else:
    print("Position not in tablebase")
```

### Get Best Move

```python
# Get the optimal move from tablebase
best_move = tb.get_best_move(position)
if best_move:
    print(f"Best move: {best_move.uci()}")
else:
    print("No tablebase move available")
```

### Distance to Zeroing (DTZ)

```python
# Get distance to zeroing (moves until pawn move or capture)
dtz = tb.probe_dtz(position)
if dtz is not None:
    print(f"Distance to zeroing: {dtz} moves")
```

### Check Availability

```python
# Check if tablebase is available
if tb.is_available():
    print(f"Tablebase supports up to {tb.get_max_pieces()} pieces")
else:
    print("Tablebase not available - using heuristic evaluation")
```

## Integration with Evaluation

The evaluator automatically uses tablebase when available:

```python
from src.evaluation.evaluator import Evaluator
from src.config.config import Config
import torch

# Create config and neural network
config = Config()
neural_net = create_neural_network()  # Your network initialization

# Create evaluator with tablebase
evaluator = Evaluator(config, tablebase_path="/path/to/tablebases")

# Run evaluation - will use tablebase for move accuracy if available
metrics = evaluator.evaluate(neural_net)
print(f"Move accuracy: {metrics['move_accuracy']:.2%}")
```

## Graceful Fallback

The tablebase integration is designed to work gracefully when tablebases are not available:

```python
# Without tablebase path - still works
tb = TablebaseInterface()

# All methods return None gracefully
result = tb.probe(position)  # Returns None
best_move = tb.get_best_move(position)  # Returns None
dtz = tb.probe_dtz(position)  # Returns None

# System continues with heuristic evaluation
```

## Performance Considerations

### Tablebase Size

- 3-piece tablebases: ~1 MB (essential for KPK endgames)
- 4-piece tablebases: ~100 MB
- 5-piece tablebases: ~7 GB
- 6-piece tablebases: ~150 GB
- 7-piece tablebases: ~16 TB

For king-pawn endgames, 3-piece tablebases are sufficient.

### Lookup Speed

Tablebase lookups are very fast (microseconds), but:
- First access may be slower due to disk I/O
- Operating system will cache frequently accessed files
- Consider using SSD for better performance

### Memory Usage

Tablebases use memory-mapped files, so:
- They don't load entirely into RAM
- OS manages caching automatically
- Multiple processes can share the same tablebase files

## Troubleshooting

### Tablebase Not Loading

If the tablebase doesn't load:

1. Check the path exists:
   ```python
   import os
   print(os.path.exists("/path/to/tablebases"))
   ```

2. Check file permissions:
   ```bash
   ls -la /path/to/tablebases
   ```

3. Verify files are present:
   ```bash
   ls /path/to/tablebases/*.rtbw
   ```

### Position Not Found

If `probe()` returns `None`:

1. Check piece count:
   ```python
   piece_count = len(position.board.piece_map())
   max_pieces = tb.get_max_pieces()
   print(f"Position has {piece_count} pieces, tablebase supports {max_pieces}")
   ```

2. Verify position is legal:
   ```python
   print(position.board.is_valid())
   ```

### Performance Issues

If tablebase lookups are slow:

1. Use SSD storage for tablebase files
2. Ensure sufficient RAM for OS caching
3. Consider downloading only needed piece counts
4. Check disk I/O with system monitoring tools

## Example: Complete Evaluation with Tablebase

```python
from src.evaluation.evaluator import Evaluator
from src.evaluation.tablebase import TablebaseInterface
from src.config.config import Config
from src.neural_net.chess_net import ChessNet
import torch

# Initialize components
config = Config()
neural_net = ChessNet(
    num_res_blocks=config.num_res_blocks,
    num_filters=config.num_filters
).to(config.device)

# Load trained model
checkpoint = torch.load("checkpoints/best_model.pt")
neural_net.load_state_dict(checkpoint['model_state_dict'])

# Create evaluator with tablebase
evaluator = Evaluator(config, tablebase_path="/path/to/tablebases")

# Check tablebase status
if evaluator.tablebase.is_available():
    print(f"✓ Tablebase loaded ({evaluator.tablebase.get_max_pieces()} pieces)")
else:
    print("⚠ Tablebase not available - using heuristic evaluation")

# Run evaluation
print("Running evaluation...")
metrics = evaluator.evaluate(neural_net)

# Display results
print("\nEvaluation Results:")
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Draw Rate: {metrics['draw_rate']:.2%}")
print(f"Move Accuracy: {metrics['move_accuracy']:.2%}")
print(f"Estimated Elo: {metrics['elo_estimate']:.0f}")
```

## References

- Syzygy Tablebases: http://tablebase.sesse.net/syzygy/
- python-chess documentation: https://python-chess.readthedocs.io/
- Tablebase theory: https://en.wikipedia.org/wiki/Endgame_tablebase
