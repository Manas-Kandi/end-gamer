"""Unit tests for ResourceMonitor class."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from threading import Event

from src.logging.resource_monitor import ResourceMonitor, ResourceSnapshot, MCTSStats


class TestResourceMonitor:
    """Test cases for ResourceMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = ResourceMonitor(monitoring_interval=0.1)  # Short interval for testing
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.monitor._is_monitoring:
            self.monitor.stop_monitoring()
    
    @patch('src.logging.resource_monitor.psutil')
    @patch('src.logging.resource_monitor.TORCH_AVAILABLE', False)
    def test_get_current_resources_without_gpu(self, mock_psutil):
        """Test getting current resources without GPU."""
        # Mock psutil calls
        mock_psutil.cpu_percent.return_value = 75.5
        mock_memory = Mock()
        mock_memory.used = 8192 * 1024 * 1024  # 8GB in bytes
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 1024  # 1GB in bytes
        mock_psutil.Process.return_value = mock_process
        
        monitor = ResourceMonitor()
        snapshot = monitor.get_current_resources()
        
        assert isinstance(snapshot, ResourceSnapshot)
        assert snapshot.cpu_percent == 75.5
        assert snapshot.memory_mb == 8192.0
        assert snapshot.process_memory_mb == 1024.0
        assert snapshot.gpu_memory_mb is None
        assert snapshot.gpu_utilization is None
        assert snapshot.timestamp > 0
    
    @patch('src.logging.resource_monitor.psutil')
    @patch('src.logging.resource_monitor.TORCH_AVAILABLE', True)
    @patch('src.logging.resource_monitor.torch')
    def test_get_current_resources_with_gpu(self, mock_torch, mock_psutil):
        """Test getting current resources with GPU."""
        # Mock psutil calls
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.used = 4096 * 1024 * 1024  # 4GB in bytes
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 512 * 1024 * 1024  # 512MB in bytes
        mock_psutil.Process.return_value = mock_process
        
        # Mock torch calls
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2048 * 1024 * 1024  # 2GB in bytes
        
        monitor = ResourceMonitor()
        snapshot = monitor.get_current_resources()
        
        assert snapshot.cpu_percent == 50.0
        assert snapshot.memory_mb == 4096.0
        assert snapshot.process_memory_mb == 512.0
        assert snapshot.gpu_memory_mb == 2048.0
    
    @patch('src.logging.resource_monitor.psutil')
    def test_monitoring_lifecycle(self, mock_psutil):
        """Test starting and stopping monitoring."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.used = 4096 * 1024 * 1024
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 512 * 1024 * 1024
        mock_psutil.Process.return_value = mock_process
        
        monitor = ResourceMonitor(monitoring_interval=0.05)
        
        # Initially not monitoring
        assert not monitor._is_monitoring
        assert len(monitor._resource_history) == 0
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor._is_monitoring
        assert monitor._monitoring_thread is not None
        
        # Wait for some data collection
        time.sleep(0.2)
        
        # Should have collected some data
        assert len(monitor._resource_history) > 0
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor._is_monitoring
        
        # History should be preserved
        history_size = len(monitor._resource_history)
        assert history_size > 0
        
        # Wait a bit more - no new data should be collected
        time.sleep(0.1)
        assert len(monitor._resource_history) == history_size
    
    def test_monitoring_double_start_stop(self):
        """Test that double start/stop doesn't cause issues."""
        monitor = ResourceMonitor()
        
        # Double start should not cause issues
        monitor.start_monitoring()
        monitor.start_monitoring()  # Should log warning but not crash
        
        assert monitor._is_monitoring
        
        # Double stop should not cause issues
        monitor.stop_monitoring()
        monitor.stop_monitoring()  # Should not crash
        
        assert not monitor._is_monitoring
    
    @patch('src.logging.resource_monitor.psutil')
    def test_resource_history_management(self, mock_psutil):
        """Test resource history size management."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.used = 4096 * 1024 * 1024
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 512 * 1024 * 1024
        mock_psutil.Process.return_value = mock_process
        
        monitor = ResourceMonitor()
        monitor._max_history_size = 5  # Small size for testing
        
        # Add more snapshots than max size
        for i in range(10):
            snapshot = monitor.get_current_resources()
            monitor._add_to_history(snapshot)
        
        # Should maintain max size
        assert len(monitor._resource_history) == 5
        
        # Test get_resource_history
        all_history = monitor.get_resource_history()
        assert len(all_history) == 5
        
        last_3 = monitor.get_resource_history(last_n=3)
        assert len(last_3) == 3
        
        # Should be the most recent ones
        assert last_3 == all_history[-3:]
    
    @patch('src.logging.resource_monitor.psutil')
    def test_resource_summary(self, mock_psutil):
        """Test resource usage summary calculation."""
        # Mock psutil with varying values
        cpu_values = [10.0, 20.0, 30.0, 40.0, 50.0]
        memory_values = [1000, 2000, 3000, 4000, 5000]  # MB
        
        mock_memory = Mock()
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        monitor = ResourceMonitor()
        
        # Add snapshots with different values
        for i, (cpu, mem) in enumerate(zip(cpu_values, memory_values)):
            mock_psutil.cpu_percent.return_value = cpu
            mock_memory.used = mem * 1024 * 1024
            mock_psutil.virtual_memory.return_value = mock_memory
            mock_process.memory_info.return_value.rss = (mem // 2) * 1024 * 1024
            
            snapshot = monitor.get_current_resources()
            monitor._add_to_history(snapshot)
        
        summary = monitor.get_resource_summary()
        
        # Check CPU stats
        assert summary['cpu_percent']['min'] == 10.0
        assert summary['cpu_percent']['max'] == 50.0
        assert summary['cpu_percent']['avg'] == 30.0
        
        # Check memory stats
        assert summary['memory_mb']['min'] == 1000.0
        assert summary['memory_mb']['max'] == 5000.0
        assert summary['memory_mb']['avg'] == 3000.0
        
        # Check process memory stats
        assert summary['process_memory_mb']['min'] == 500.0
        assert summary['process_memory_mb']['max'] == 2500.0
        assert summary['process_memory_mb']['avg'] == 1500.0
    
    def test_resource_summary_empty_history(self):
        """Test resource summary with empty history."""
        monitor = ResourceMonitor()
        summary = monitor.get_resource_summary()
        assert summary == {}
    
    def test_mcts_timing(self):
        """Test MCTS timing functionality."""
        monitor = ResourceMonitor()
        
        # Initially no stats
        stats = monitor.get_mcts_stats(400)
        assert stats is None
        
        # Start timing
        monitor.start_mcts_timing()
        
        # Simulate some searches
        time.sleep(0.01)  # Small delay to ensure time passes
        monitor.record_mcts_search(nodes_evaluated=100, search_depth=8.5)
        monitor.record_mcts_search(nodes_evaluated=150, search_depth=9.5)
        monitor.record_mcts_search(nodes_evaluated=120, search_depth=7.0)
        
        time.sleep(0.01)  # Ensure some time has passed
        
        # Get stats
        stats = monitor.get_mcts_stats(simulations=400)
        assert stats is not None
        assert isinstance(stats, MCTSStats)
        
        # Check calculations
        assert stats.total_nodes == 370  # 100 + 150 + 120
        assert stats.avg_depth == (8.5 + 9.5 + 7.0) / 3  # 8.33...
        assert stats.simulations == 400
        assert stats.nodes_per_second > 0
        assert stats.search_time_ms > 0
        
        # Reset stats
        monitor.reset_mcts_stats()
        stats = monitor.get_mcts_stats(400)
        assert stats is None
    
    def test_mcts_timing_edge_cases(self):
        """Test MCTS timing edge cases."""
        monitor = ResourceMonitor()
        
        # Start timing but no searches recorded
        monitor.start_mcts_timing()
        stats = monitor.get_mcts_stats(400)
        assert stats is None
        
        # Record search immediately after start (zero elapsed time)
        monitor.start_mcts_timing()
        monitor.record_mcts_search(100, 8.0)
        
        # Might get None if elapsed time is too small
        stats = monitor.get_mcts_stats(400)
        # This could be None or valid stats depending on timing precision
    
    def test_log_to_metrics_logger(self):
        """Test logging to MetricsLogger."""
        monitor = ResourceMonitor()
        mock_logger = Mock()
        
        # Test resource logging
        with patch.object(monitor, 'get_current_resources') as mock_get_resources:
            mock_snapshot = ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=75.0,
                memory_mb=8192.0,
                gpu_memory_mb=4096.0,
                process_memory_mb=1024.0
            )
            mock_get_resources.return_value = mock_snapshot
            
            monitor.log_resources_to_metrics_logger(mock_logger)
            
            mock_logger.log_resource_usage.assert_called_once_with(
                cpu_percent=75.0,
                memory_mb=8192.0,
                gpu_memory_mb=4096.0
            )
        
        # Test MCTS stats logging
        mock_logger.reset_mock()
        
        with patch.object(monitor, 'get_mcts_stats') as mock_get_stats:
            mock_stats = MCTSStats(
                nodes_per_second=1500.0,
                avg_depth=8.5,
                simulations=400,
                total_nodes=6000,
                search_time_ms=266.7
            )
            mock_get_stats.return_value = mock_stats
            
            monitor.log_mcts_stats_to_metrics_logger(mock_logger, 400)
            
            mock_logger.log_mcts_statistics.assert_called_once_with(
                nodes_per_second=1500.0,
                avg_depth=8.5,
                simulations=400
            )
        
        # Test when no MCTS stats available
        mock_logger.reset_mock()
        
        with patch.object(monitor, 'get_mcts_stats') as mock_get_stats:
            mock_get_stats.return_value = None
            
            monitor.log_mcts_stats_to_metrics_logger(mock_logger, 400)
            
            mock_logger.log_mcts_statistics.assert_not_called()
    
    @patch('src.logging.resource_monitor.psutil')
    @patch('src.logging.resource_monitor.TORCH_AVAILABLE', True)
    @patch('src.logging.resource_monitor.torch')
    def test_get_system_info_with_gpu(self, mock_torch, mock_psutil):
        """Test getting system information with GPU."""
        # Mock psutil
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.cpu_count.side_effect = lambda logical=False: 16 if logical else 8
        mock_memory = Mock()
        mock_memory.total = 32 * 1024**3  # 32GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_process = Mock()
        mock_process.pid = 12345
        mock_psutil.Process.return_value = mock_process
        
        # Mock torch
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 3080"
        
        mock_properties = Mock()
        mock_properties.total_memory = 10 * 1024**3  # 10GB
        mock_torch.cuda.get_device_properties.return_value = mock_properties
        
        monitor = ResourceMonitor()
        info = monitor.get_system_info()
        
        assert info['cpu_count'] == 8
        assert info['cpu_count_logical'] == 16
        assert info['memory_total_gb'] == 32.0
        assert info['python_process_id'] == 12345
        assert info['gpu_available'] is True
        assert info['gpu_count'] == 2
        assert info['gpu_name'] == "NVIDIA RTX 3080"
        assert info['gpu_memory_total_gb'] == 10.0
    
    @patch('src.logging.resource_monitor.psutil')
    @patch('src.logging.resource_monitor.TORCH_AVAILABLE', False)
    def test_get_system_info_without_gpu(self, mock_psutil):
        """Test getting system information without GPU."""
        # Mock psutil
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.cpu_count.side_effect = lambda logical=False: 8 if logical else 4
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_process = Mock()
        mock_process.pid = 54321
        mock_psutil.Process.return_value = mock_process
        
        monitor = ResourceMonitor()
        info = monitor.get_system_info()
        
        assert info['cpu_count'] == 4
        assert info['cpu_count_logical'] == 8
        assert info['memory_total_gb'] == 16.0
        assert info['python_process_id'] == 54321
        assert info['gpu_available'] is False
        assert 'gpu_count' not in info
        assert 'gpu_name' not in info
    
    def test_context_manager(self):
        """Test using ResourceMonitor as context manager."""
        with patch('src.logging.resource_monitor.psutil'):
            with ResourceMonitor(monitoring_interval=0.05) as monitor:
                assert monitor._is_monitoring
                time.sleep(0.1)  # Let it collect some data
            
            # Should have stopped monitoring
            assert not monitor._is_monitoring
    
    @patch('src.logging.resource_monitor.psutil')
    def test_monitoring_with_exception(self, mock_psutil):
        """Test monitoring continues despite exceptions."""
        # Mock psutil to raise exception sometimes
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second call
                raise Exception("Test exception")
            return 50.0
        
        mock_psutil.cpu_percent.side_effect = side_effect
        mock_memory = Mock()
        mock_memory.used = 4096 * 1024 * 1024
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 512 * 1024 * 1024
        mock_psutil.Process.return_value = mock_process
        
        monitor = ResourceMonitor(monitoring_interval=0.05)
        monitor.start_monitoring()
        
        # Wait for multiple monitoring cycles
        time.sleep(0.2)
        
        monitor.stop_monitoring()
        
        # Should have collected some data despite the exception
        # (at least from the successful calls)
        assert len(monitor._resource_history) >= 1
    
    def test_gpu_availability_check(self):
        """Test GPU availability checking."""
        # Test without torch
        with patch('src.logging.resource_monitor.TORCH_AVAILABLE', False):
            monitor = ResourceMonitor()
            assert not monitor._gpu_available
        
        # Test with torch but no CUDA
        with patch('src.logging.resource_monitor.TORCH_AVAILABLE', True):
            with patch('src.logging.resource_monitor.torch') as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                monitor = ResourceMonitor()
                assert not monitor._gpu_available
        
        # Test with torch and CUDA
        with patch('src.logging.resource_monitor.TORCH_AVAILABLE', True):
            with patch('src.logging.resource_monitor.torch') as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                monitor = ResourceMonitor()
                assert monitor._gpu_available
        
        # Test with torch but exception
        with patch('src.logging.resource_monitor.TORCH_AVAILABLE', True):
            with patch('src.logging.resource_monitor.torch') as mock_torch:
                mock_torch.cuda.is_available.side_effect = Exception("CUDA error")
                monitor = ResourceMonitor()
                assert not monitor._gpu_available