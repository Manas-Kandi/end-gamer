"""Unit tests for custom exception classes."""

import pytest
from src.exceptions import (
    ChessEngineError,
    NeuralNetworkError,
    ModelLoadError,
    ModelSaveError,
    InvalidModelArchitectureError,
    MCTSError,
    SearchTimeoutError,
    InvalidSearchStateError,
    ChessError,
    InvalidPositionError,
    InvalidMoveError,
    PositionGenerationError,
    TrainingError,
    CheckpointError,
    CheckpointLoadError,
    CheckpointSaveError,
    CheckpointValidationError,
    SelfPlayError,
    DataError,
    ReplayBufferError,
    InvalidTrainingExampleError,
    ConfigurationError,
    EvaluationError,
    TablebaseError,
    TablebaseNotAvailableError,
    ResourceError,
    OutOfMemoryError,
    GPUError,
)


class TestExceptionHierarchy:
    """Test exception class hierarchy."""
    
    def test_base_exception(self):
        """Test base ChessEngineError."""
        error = ChessEngineError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_neural_network_error_hierarchy(self):
        """Test NeuralNetworkError is subclass of ChessEngineError."""
        error = NeuralNetworkError("Test")
        assert isinstance(error, ChessEngineError)
        assert isinstance(error, Exception)
    
    def test_mcts_error_hierarchy(self):
        """Test MCTSError is subclass of ChessEngineError."""
        error = MCTSError("Test")
        assert isinstance(error, ChessEngineError)
    
    def test_chess_error_hierarchy(self):
        """Test ChessError is subclass of ChessEngineError."""
        error = ChessError("Test")
        assert isinstance(error, ChessEngineError)
    
    def test_training_error_hierarchy(self):
        """Test TrainingError is subclass of ChessEngineError."""
        error = TrainingError("Test")
        assert isinstance(error, ChessEngineError)


class TestModelErrors:
    """Test model-related exception classes."""
    
    def test_model_load_error(self):
        """Test ModelLoadError with path and reason."""
        error = ModelLoadError("/path/to/model.pt", "File not found")
        
        assert isinstance(error, NeuralNetworkError)
        assert error.model_path == "/path/to/model.pt"
        assert error.reason == "File not found"
        assert "Failed to load model" in str(error)
        assert "/path/to/model.pt" in str(error)
        assert "File not found" in str(error)
    
    def test_model_save_error(self):
        """Test ModelSaveError with path and reason."""
        error = ModelSaveError("/path/to/model.pt", "Permission denied")
        
        assert isinstance(error, NeuralNetworkError)
        assert error.model_path == "/path/to/model.pt"
        assert error.reason == "Permission denied"
        assert "Failed to save model" in str(error)
    
    def test_invalid_model_architecture_error(self):
        """Test InvalidModelArchitectureError."""
        error = InvalidModelArchitectureError("ResNet-3", "ResNet-5")
        
        assert isinstance(error, NeuralNetworkError)
        assert error.expected == "ResNet-3"
        assert error.actual == "ResNet-5"
        assert "Invalid model architecture" in str(error)
        assert "Expected: ResNet-3" in str(error)
        assert "Got: ResNet-5" in str(error)


class TestMCTSErrors:
    """Test MCTS-related exception classes."""
    
    def test_search_timeout_error(self):
        """Test SearchTimeoutError with timeout and simulations."""
        error = SearchTimeoutError(5.0, 150)
        
        assert isinstance(error, MCTSError)
        assert error.timeout_seconds == 5.0
        assert error.simulations_completed == 150
        assert "timed out" in str(error)
        assert "5.0s" in str(error)
        assert "150 simulations" in str(error)
    
    def test_invalid_search_state_error(self):
        """Test InvalidSearchStateError."""
        error = InvalidSearchStateError("No legal moves available")
        
        assert isinstance(error, MCTSError)
        assert "Invalid MCTS search state" in str(error)
        assert "No legal moves available" in str(error)


class TestChessErrors:
    """Test chess environment exception classes."""
    
    def test_invalid_position_error(self):
        """Test InvalidPositionError with FEN and reason."""
        fen = "8/8/8/8/8/8/8/8 w - - 0 1"
        error = InvalidPositionError(fen, "Missing kings")
        
        assert isinstance(error, ChessError)
        assert error.fen == fen
        assert error.reason == "Missing kings"
        assert "Invalid position" in str(error)
        assert fen in str(error)
        assert "Missing kings" in str(error)
    
    def test_invalid_move_error(self):
        """Test InvalidMoveError with move and position."""
        error = InvalidMoveError("e2e5", "starting position", "Pawn cannot move 3 squares")
        
        assert isinstance(error, ChessError)
        assert error.move == "e2e5"
        assert error.position == "starting position"
        assert error.reason == "Pawn cannot move 3 squares"
        assert "Invalid move" in str(error)
        assert "e2e5" in str(error)
    
    def test_invalid_move_error_without_reason(self):
        """Test InvalidMoveError without reason."""
        error = InvalidMoveError("e2e5", "starting position")
        
        assert error.reason == ""
        assert "Invalid move" in str(error)
    
    def test_position_generation_error(self):
        """Test PositionGenerationError."""
        error = PositionGenerationError(curriculum_level=2, attempts=1000)
        
        assert isinstance(error, ChessError)
        assert error.curriculum_level == 2
        assert error.attempts == 1000
        assert "Failed to generate valid position" in str(error)
        assert "level 2" in str(error)
        assert "1000 attempts" in str(error)


class TestTrainingErrors:
    """Test training-related exception classes."""
    
    def test_checkpoint_error(self):
        """Test base CheckpointError."""
        error = CheckpointError("/path/to/checkpoint.pt", "Corrupted file")
        
        assert isinstance(error, TrainingError)
        assert error.checkpoint_path == "/path/to/checkpoint.pt"
        assert error.reason == "Corrupted file"
        assert "Checkpoint error" in str(error)
    
    def test_checkpoint_load_error(self):
        """Test CheckpointLoadError."""
        error = CheckpointLoadError("/path/to/checkpoint.pt", "File not found")
        
        assert isinstance(error, CheckpointError)
        assert isinstance(error, TrainingError)
    
    def test_checkpoint_save_error(self):
        """Test CheckpointSaveError."""
        error = CheckpointSaveError("/path/to/checkpoint.pt", "Disk full")
        
        assert isinstance(error, CheckpointError)
        assert isinstance(error, TrainingError)
    
    def test_checkpoint_validation_error(self):
        """Test CheckpointValidationError with validation errors."""
        validation_errors = ["Missing model_state_dict", "Invalid config"]
        error = CheckpointValidationError("/path/to/checkpoint.pt", validation_errors)
        
        assert isinstance(error, CheckpointError)
        assert error.validation_errors == validation_errors
        assert "Validation failed" in str(error)
        assert "Missing model_state_dict" in str(error)
        assert "Invalid config" in str(error)
    
    def test_self_play_error(self):
        """Test SelfPlayError."""
        error = SelfPlayError(game_number=42, reason="MCTS search failed")
        
        assert isinstance(error, TrainingError)
        assert error.game_number == 42
        assert error.reason == "MCTS search failed"
        assert "Self-play game 42 failed" in str(error)
        assert "MCTS search failed" in str(error)


class TestDataErrors:
    """Test data-related exception classes."""
    
    def test_data_error_hierarchy(self):
        """Test DataError is subclass of TrainingError."""
        error = DataError("Test")
        assert isinstance(error, TrainingError)
    
    def test_replay_buffer_error(self):
        """Test ReplayBufferError."""
        error = ReplayBufferError("Buffer is full")
        
        assert isinstance(error, DataError)
        assert "Replay buffer error" in str(error)
        assert "Buffer is full" in str(error)
    
    def test_invalid_training_example_error(self):
        """Test InvalidTrainingExampleError."""
        error = InvalidTrainingExampleError("Policy does not sum to 1.0")
        
        assert isinstance(error, DataError)
        assert "Invalid training example" in str(error)
        assert "Policy does not sum to 1.0" in str(error)


class TestConfigurationError:
    """Test configuration exception class."""
    
    def test_configuration_error(self):
        """Test ConfigurationError with parameter, value, and reason."""
        error = ConfigurationError("batch_size", -1, "Must be positive")
        
        assert isinstance(error, ChessEngineError)
        assert error.parameter == "batch_size"
        assert error.value == -1
        assert error.reason == "Must be positive"
        assert "Invalid configuration" in str(error)
        assert "batch_size" in str(error)
        assert "-1" in str(error)
        assert "Must be positive" in str(error)


class TestEvaluationError:
    """Test evaluation exception class."""
    
    def test_evaluation_error(self):
        """Test EvaluationError."""
        error = EvaluationError("win_rate", "No test positions available")
        
        assert isinstance(error, ChessEngineError)
        assert error.evaluation_type == "win_rate"
        assert error.reason == "No test positions available"
        assert "Evaluation 'win_rate' failed" in str(error)
        assert "No test positions available" in str(error)


class TestTablebaseErrors:
    """Test tablebase exception classes."""
    
    def test_tablebase_error(self):
        """Test base TablebaseError."""
        error = TablebaseError("Probe failed")
        
        assert isinstance(error, ChessEngineError)
        assert "Tablebase error" in str(error)
        assert "Probe failed" in str(error)
    
    def test_tablebase_not_available_error_with_path(self):
        """Test TablebaseNotAvailableError with path."""
        error = TablebaseNotAvailableError("/path/to/tablebase")
        
        assert isinstance(error, TablebaseError)
        assert "not available" in str(error)
        assert "/path/to/tablebase" in str(error)
    
    def test_tablebase_not_available_error_without_path(self):
        """Test TablebaseNotAvailableError without path."""
        error = TablebaseNotAvailableError()
        
        assert isinstance(error, TablebaseError)
        assert "Tablebase not available" in str(error)


class TestResourceErrors:
    """Test resource-related exception classes."""
    
    def test_resource_error(self):
        """Test base ResourceError."""
        error = ResourceError("CPU", "High utilization")
        
        assert isinstance(error, ChessEngineError)
        assert error.resource_type == "CPU"
        assert error.reason == "High utilization"
        assert "Resource error (CPU)" in str(error)
        assert "High utilization" in str(error)
    
    def test_out_of_memory_error(self):
        """Test OutOfMemoryError."""
        error = OutOfMemoryError(required_mb=16000.0, available_mb=8000.0)
        
        assert isinstance(error, ResourceError)
        assert error.required_mb == 16000.0
        assert error.available_mb == 8000.0
        assert "Insufficient memory" in str(error)
        assert "16000.0MB" in str(error)
        assert "8000.0MB" in str(error)
    
    def test_gpu_error(self):
        """Test GPUError."""
        error = GPUError("CUDA out of memory")
        
        assert isinstance(error, ResourceError)
        assert error.resource_type == "GPU"
        assert "Resource error (GPU)" in str(error)
        assert "CUDA out of memory" in str(error)


class TestExceptionRaising:
    """Test that exceptions can be raised and caught properly."""
    
    def test_raise_and_catch_model_load_error(self):
        """Test raising and catching ModelLoadError."""
        with pytest.raises(ModelLoadError) as exc_info:
            raise ModelLoadError("/path/to/model.pt", "File not found")
        
        assert exc_info.value.model_path == "/path/to/model.pt"
        assert exc_info.value.reason == "File not found"
    
    def test_catch_as_base_exception(self):
        """Test catching specific exception as base ChessEngineError."""
        with pytest.raises(ChessEngineError):
            raise ModelLoadError("/path/to/model.pt", "File not found")
    
    def test_catch_multiple_exception_types(self):
        """Test catching multiple exception types."""
        def raise_various_errors(error_type):
            if error_type == "model":
                raise ModelLoadError("/path", "error")
            elif error_type == "mcts":
                raise SearchTimeoutError(5.0, 100)
            elif error_type == "chess":
                raise InvalidPositionError("fen", "error")
        
        # All should be catchable as ChessEngineError
        for error_type in ["model", "mcts", "chess"]:
            with pytest.raises(ChessEngineError):
                raise_various_errors(error_type)
    
    def test_exception_attributes_accessible(self):
        """Test that exception attributes are accessible after catching."""
        try:
            raise SearchTimeoutError(10.0, 250)
        except SearchTimeoutError as e:
            assert e.timeout_seconds == 10.0
            assert e.simulations_completed == 250
        except Exception:
            pytest.fail("Should have caught SearchTimeoutError")


class TestExceptionMessages:
    """Test exception message formatting."""
    
    def test_all_exceptions_have_meaningful_messages(self):
        """Test that all exceptions produce meaningful error messages."""
        exceptions_to_test = [
            (ModelLoadError("/path", "reason"), ["Failed to load", "/path", "reason"]),
            (SearchTimeoutError(5.0, 100), ["timed out", "5.0", "100"]),
            (InvalidPositionError("fen", "reason"), ["Invalid position", "fen", "reason"]),
            (CheckpointValidationError("/path", ["error1", "error2"]), ["Validation failed", "error1", "error2"]),
            (OutOfMemoryError(1000.0, 500.0), ["Insufficient memory", "1000.0", "500.0"]),
        ]
        
        for exception, expected_substrings in exceptions_to_test:
            error_message = str(exception)
            for substring in expected_substrings:
                assert substring in error_message, \
                    f"Expected '{substring}' in error message: {error_message}"
