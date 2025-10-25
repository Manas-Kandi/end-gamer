"""Resource monitoring for system performance tracking."""

import time
import psutil
from typing import Dict, Optional, Any
from dataclasses import dataclass
from threading import Thread, Event
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    process_memory_mb: Optional[float] = None


@dataclass
class MCTSStats:
    """MCTS performance statistics."""
    nodes_per_second: float
    avg_depth: float
    simulations: int
    total_nodes: int
    search_time_ms: float


class ResourceMonitor:
    """Monitor system resources and performance metrics.
    
    Tracks CPU usage, memory consumption, GPU metrics, and provides
    utilities for measuring MCTS performance.
    """
    
    def __init__(self, monitoring_interval: float = 5.0):
        """Initialize resource monitor.
        
        Args:
            monitoring_interval: Seconds between resource measurements
        """
        self.monitoring_interval = monitoring_interval
        self.process = psutil.Process()
        
        # Threading for continuous monitoring
        self._monitoring_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._is_monitoring = False
        
        # Resource history
        self._resource_history = []
        self._max_history_size = 1000
        
        # MCTS timing
        self._mcts_start_time: Optional[float] = None
        self._mcts_node_count = 0
        self._mcts_search_count = 0
        self._mcts_total_depth = 0.0
        
        # GPU availability check
        self._gpu_available = self._check_gpu_availability()
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        if not TORCH_AVAILABLE:
            return False
        
        try:
            return torch.cuda.is_available()
        except Exception:
            return False
    
    def get_current_resources(self) -> ResourceSnapshot:
        """Get current resource usage snapshot.
        
        Returns:
            ResourceSnapshot with current resource usage
        """
        timestamp = time.time()
        
        # CPU and system memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.used / (1024 * 1024)
        
        # Process memory
        process_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # GPU metrics
        gpu_memory_mb = None
        gpu_utilization = None
        
        if self._gpu_available:
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                # GPU utilization requires nvidia-ml-py, so we'll skip it for now
                # gpu_utilization = self._get_gpu_utilization()
            except Exception as e:
                self.logger.warning(f"Failed to get GPU metrics: {e}")
        
        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
            process_memory_mb=process_memory_mb
        )
    
    def start_monitoring(self) -> None:
        """Start continuous resource monitoring in background thread."""
        if self._is_monitoring:
            self.logger.warning("Resource monitoring already started")
            return
        
        self._stop_event.clear()
        self._monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self._is_monitoring = True
        self.logger.info(f"Started resource monitoring (interval: {self.monitoring_interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop continuous resource monitoring."""
        if not self._is_monitoring:
            return
        
        self._stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
        
        self._is_monitoring = False
        self.logger.info("Stopped resource monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                snapshot = self.get_current_resources()
                self._add_to_history(snapshot)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next interval or stop event
            self._stop_event.wait(self.monitoring_interval)
    
    def _add_to_history(self, snapshot: ResourceSnapshot) -> None:
        """Add snapshot to history with size limit."""
        self._resource_history.append(snapshot)
        
        # Maintain history size limit
        if len(self._resource_history) > self._max_history_size:
            self._resource_history.pop(0)
    
    def get_resource_history(self, last_n: Optional[int] = None) -> list[ResourceSnapshot]:
        """Get resource usage history.
        
        Args:
            last_n: Number of recent snapshots to return (all if None)
            
        Returns:
            List of ResourceSnapshot objects
        """
        if last_n is None:
            return self._resource_history.copy()
        else:
            return self._resource_history[-last_n:]
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary statistics of resource usage.
        
        Returns:
            Dictionary with min, max, avg resource usage
        """
        if not self._resource_history:
            return {}
        
        cpu_values = [s.cpu_percent for s in self._resource_history]
        memory_values = [s.memory_mb for s in self._resource_history]
        process_memory_values = [s.process_memory_mb for s in self._resource_history if s.process_memory_mb]
        
        summary = {
            'cpu_percent': {
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': sum(cpu_values) / len(cpu_values)
            },
            'memory_mb': {
                'min': min(memory_values),
                'max': max(memory_values),
                'avg': sum(memory_values) / len(memory_values)
            }
        }
        
        if process_memory_values:
            summary['process_memory_mb'] = {
                'min': min(process_memory_values),
                'max': max(process_memory_values),
                'avg': sum(process_memory_values) / len(process_memory_values)
            }
        
        # GPU metrics if available
        gpu_memory_values = [s.gpu_memory_mb for s in self._resource_history if s.gpu_memory_mb is not None]
        if gpu_memory_values:
            summary['gpu_memory_mb'] = {
                'min': min(gpu_memory_values),
                'max': max(gpu_memory_values),
                'avg': sum(gpu_memory_values) / len(gpu_memory_values)
            }
        
        return summary
    
    def start_mcts_timing(self) -> None:
        """Start timing MCTS search performance."""
        self._mcts_start_time = time.time()
        self._mcts_node_count = 0
        self._mcts_search_count = 0
        self._mcts_total_depth = 0.0
    
    def record_mcts_search(self, nodes_evaluated: int, search_depth: float) -> None:
        """Record MCTS search statistics.
        
        Args:
            nodes_evaluated: Number of nodes evaluated in this search
            search_depth: Average depth of this search
        """
        self._mcts_node_count += nodes_evaluated
        self._mcts_search_count += 1
        self._mcts_total_depth += search_depth
    
    def get_mcts_stats(self, simulations: int) -> Optional[MCTSStats]:
        """Get current MCTS performance statistics.
        
        Args:
            simulations: Number of simulations per search
            
        Returns:
            MCTSStats object or None if timing not started
        """
        if self._mcts_start_time is None or self._mcts_search_count == 0:
            return None
        
        elapsed_time = time.time() - self._mcts_start_time
        
        if elapsed_time <= 0:
            return None
        
        nodes_per_second = self._mcts_node_count / elapsed_time
        avg_depth = self._mcts_total_depth / self._mcts_search_count
        search_time_ms = (elapsed_time / self._mcts_search_count) * 1000
        
        return MCTSStats(
            nodes_per_second=nodes_per_second,
            avg_depth=avg_depth,
            simulations=simulations,
            total_nodes=self._mcts_node_count,
            search_time_ms=search_time_ms
        )
    
    def reset_mcts_stats(self) -> None:
        """Reset MCTS timing statistics."""
        self._mcts_start_time = None
        self._mcts_node_count = 0
        self._mcts_search_count = 0
        self._mcts_total_depth = 0.0
    
    def log_resources_to_metrics_logger(self, metrics_logger) -> None:
        """Log current resources to MetricsLogger.
        
        Args:
            metrics_logger: MetricsLogger instance
        """
        snapshot = self.get_current_resources()
        
        metrics_logger.log_resource_usage(
            cpu_percent=snapshot.cpu_percent,
            memory_mb=snapshot.memory_mb,
            gpu_memory_mb=snapshot.gpu_memory_mb
        )
    
    def log_mcts_stats_to_metrics_logger(self, metrics_logger, simulations: int) -> None:
        """Log MCTS statistics to MetricsLogger.
        
        Args:
            metrics_logger: MetricsLogger instance
            simulations: Number of simulations per search
        """
        stats = self.get_mcts_stats(simulations)
        if stats:
            metrics_logger.log_mcts_statistics(
                nodes_per_second=stats.nodes_per_second,
                avg_depth=stats.avg_depth,
                simulations=stats.simulations
            )
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for logging.
        
        Returns:
            Dictionary with system information
        """
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_process_id': self.process.pid
        }
        
        if self._gpu_available:
            try:
                info['gpu_available'] = True
                info['gpu_count'] = torch.cuda.device_count()
                info['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
                info['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else None
            except Exception as e:
                info['gpu_available'] = False
                info['gpu_error'] = str(e)
        else:
            info['gpu_available'] = False
        
        return info
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()