"""Custom exception classes for chess engine."""


class ChessEngineError(Exception):
    """Base exception for all chess engine errors."""
    pass


class NeuralNetworkError(ChessEngineError):
    """Base exception for neural network related errors."""
    pass


class ModelLoadError(NeuralNetworkError):
    """Exception raised when model loading fails."""
    
    def __init__(self, model_path: str, reason: str):
        self.model_path = model_path
        self.reason = reason
        super().__init__(f"Failed to load model from {model_path}: {reason}")


class ModelSaveError(NeuralNetworkError):
    """Exception raised when model saving fails."""
    
    def __init__(self, model_path: str, reason: str):
        self.model_path = model_path
        self.reason = reason
        super().__init__(f"Failed to save model to {model_path}: {reason}")


class InvalidModelArchitectureError(NeuralNetworkError):
    """Exception raised when model architecture is invalid or incompatible."""
    
    def __init__(self, expected: str, actual: str):
        self.expected = expected
        self.actual = actual
        super().__init__(f"Invalid model architecture. Expected: {expected}, Got: {actual}")


class MCTSError(ChessEngineError):
    """Base exception for MCTS related errors."""
    pass


class SearchTimeoutError(MCTSError):
    """Exception raised when MCTS search exceeds time limit."""
    
    def __init__(self, timeout_seconds: float, simulations_completed: int):
        self.timeout_seconds = timeout_seconds
        self.simulations_completed = simulations_completed
        super().__init__(
            f"MCTS search timed out after {timeout_seconds}s "
            f"({simulations_completed} simulations completed)"
        )


class InvalidSearchStateError(MCTSError):
    """Exception raised when MCTS encounters invalid search state."""
    
    def __init__(self, message: str):
        super().__init__(f"Invalid MCTS search state: {message}")


class ChessError(ChessEngineError):
    """Base exception for chess environment related errors."""
    pass


class InvalidPositionError(ChessError):
    """Exception raised when chess position is invalid."""
    
    def __init__(self, fen: str, reason: str):
        self.fen = fen
        self.reason = reason
        super().__init__(f"Invalid position '{fen}': {reason}")


class InvalidMoveError(ChessError):
    """Exception raised when move is invalid for position."""
    
    def __init__(self, move: str, position: str, reason: str = ""):
        self.move = move
        self.position = position
        self.reason = reason
        msg = f"Invalid move '{move}' for position '{position}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class PositionGenerationError(ChessError):
    """Exception raised when position generation fails."""
    
    def __init__(self, curriculum_level: int, attempts: int):
        self.curriculum_level = curriculum_level
        self.attempts = attempts
        super().__init__(
            f"Failed to generate valid position at curriculum level {curriculum_level} "
            f"after {attempts} attempts"
        )


class TrainingError(ChessEngineError):
    """Base exception for training related errors."""
    pass


class CheckpointError(TrainingError):
    """Exception raised for checkpoint related errors."""
    
    def __init__(self, checkpoint_path: str, reason: str):
        self.checkpoint_path = checkpoint_path
        self.reason = reason
        super().__init__(f"Checkpoint error at {checkpoint_path}: {reason}")


class CheckpointLoadError(CheckpointError):
    """Exception raised when checkpoint loading fails."""
    pass


class CheckpointSaveError(CheckpointError):
    """Exception raised when checkpoint saving fails."""
    pass


class CheckpointValidationError(CheckpointError):
    """Exception raised when checkpoint validation fails."""
    
    def __init__(self, checkpoint_path: str, validation_errors: list):
        self.validation_errors = validation_errors
        errors_str = ", ".join(validation_errors)
        super().__init__(checkpoint_path, f"Validation failed: {errors_str}")


class SelfPlayError(TrainingError):
    """Exception raised during self-play game generation."""
    
    def __init__(self, game_number: int, reason: str):
        self.game_number = game_number
        self.reason = reason
        super().__init__(f"Self-play game {game_number} failed: {reason}")


class DataError(TrainingError):
    """Exception raised for training data related errors."""
    pass


class ReplayBufferError(DataError):
    """Exception raised for replay buffer errors."""
    
    def __init__(self, message: str):
        super().__init__(f"Replay buffer error: {message}")


class InvalidTrainingExampleError(DataError):
    """Exception raised when training example is invalid."""
    
    def __init__(self, reason: str):
        super().__init__(f"Invalid training example: {reason}")


class ConfigurationError(ChessEngineError):
    """Exception raised for configuration related errors."""
    
    def __init__(self, parameter: str, value: any, reason: str):
        self.parameter = parameter
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid configuration for '{parameter}' = {value}: {reason}")


class EvaluationError(ChessEngineError):
    """Exception raised during model evaluation."""
    
    def __init__(self, evaluation_type: str, reason: str):
        self.evaluation_type = evaluation_type
        self.reason = reason
        super().__init__(f"Evaluation '{evaluation_type}' failed: {reason}")


class TablebaseError(ChessEngineError):
    """Exception raised for tablebase related errors."""
    
    def __init__(self, message: str):
        super().__init__(f"Tablebase error: {message}")


class TablebaseNotAvailableError(TablebaseError):
    """Exception raised when tablebase is not available."""
    
    def __init__(self, tablebase_path: str = None):
        if tablebase_path:
            super().__init__(f"Tablebase not available at {tablebase_path}")
        else:
            super().__init__("Tablebase not available")


class ResourceError(ChessEngineError):
    """Exception raised for resource related errors (memory, GPU, etc.)."""
    
    def __init__(self, resource_type: str, reason: str):
        self.resource_type = resource_type
        self.reason = reason
        super().__init__(f"Resource error ({resource_type}): {reason}")


class OutOfMemoryError(ResourceError):
    """Exception raised when system runs out of memory."""
    
    def __init__(self, required_mb: float, available_mb: float):
        self.required_mb = required_mb
        self.available_mb = available_mb
        super().__init__(
            "memory",
            f"Insufficient memory. Required: {required_mb:.1f}MB, Available: {available_mb:.1f}MB"
        )


class GPUError(ResourceError):
    """Exception raised for GPU related errors."""
    
    def __init__(self, reason: str):
        super().__init__("GPU", reason)
