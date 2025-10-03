"""
Performance monitoring and benchmarking utilities for RAMwich.
"""
import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    operations_per_second: float = 0.0
    energy_efficiency: float = 0.0  # Operations per joule
    throughput_gops: float = 0.0  # Giga operations per second
    latency_ms: float = 0.0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Performance monitoring and profiling for RAMwich simulations.
    
    Provides comprehensive performance tracking including:
    - Execution time measurement
    - Memory usage monitoring
    - CPU utilization tracking
    - Custom metric collection
    """
    
    def __init__(self, enable_detailed_monitoring: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            enable_detailed_monitoring: Enable detailed system monitoring
        """
        self.enable_detailed_monitoring = enable_detailed_monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_metrics: Optional[PerformanceMetrics] = None
        self._start_time: Optional[float] = None
        self._start_memory: Optional[float] = None
        self._process = psutil.Process() if enable_detailed_monitoring else None
    
    @contextmanager
    def monitor_execution(self, operation_name: str = "simulation"):
        """
        Context manager for monitoring execution performance.
        
        Args:
            operation_name: Name of the operation being monitored
            
        Yields:
            PerformanceMetrics: Current metrics object for real-time updates
            
        Example:
            >>> monitor = PerformanceMonitor()
            >>> with monitor.monitor_execution("simulation") as metrics:
            ...     # Run simulation
            ...     simulator.run()
            >>> print(f"Execution time: {metrics.execution_time:.2f}s")
        """
        self.start_monitoring()
        metrics = self.current_metrics
        
        try:
            yield metrics
        finally:
            self.stop_monitoring()
            if not self.current_metrics:
                logger.warning(f"No metrics collected for {operation_name}")
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.current_metrics = PerformanceMetrics()
        self._start_time = time.time()
        
        if self.enable_detailed_monitoring and self._process:
            try:
                memory_info = self._process.memory_info()
                self._start_memory = memory_info.rss / 1024 / 1024  # MB
                self.current_metrics.memory_usage_mb = self._start_memory
            except Exception as e:
                logger.warning(f"Failed to get initial memory info: {e}")
                self._start_memory = 0.0
    
    def stop_monitoring(self) -> PerformanceMetrics:
        """
        Stop monitoring and calculate final metrics.
        
        Returns:
            PerformanceMetrics: Final performance metrics
        """
        if not self.current_metrics or not self._start_time:
            logger.warning("Monitoring was not started")
            return PerformanceMetrics()
        
        # Calculate execution time
        end_time = time.time()
        self.current_metrics.execution_time = end_time - self._start_time
        
        # Get final system metrics
        if self.enable_detailed_monitoring and self._process:
            try:
                # Memory usage
                memory_info = self._process.memory_info()
                current_memory = memory_info.rss / 1024 / 1024  # MB
                self.current_metrics.memory_usage_mb = current_memory
                self.current_metrics.peak_memory_mb = max(
                    self.current_metrics.peak_memory_mb, current_memory
                )
                
                # CPU usage
                self.current_metrics.cpu_usage_percent = self._process.cpu_percent()
                
            except Exception as e:
                logger.warning(f"Failed to get final system metrics: {e}")
        
        # Store metrics in history
        self.metrics_history.append(self.current_metrics)
        
        final_metrics = self.current_metrics
        self.current_metrics = None
        self._start_time = None
        self._start_memory = None
        
        return final_metrics
    
    def update_custom_metric(self, name: str, value: Any) -> None:
        """
        Update a custom metric during monitoring.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if self.current_metrics:
            self.current_metrics.additional_metrics[name] = value
    
    def calculate_throughput(self, total_operations: int) -> None:
        """
        Calculate throughput metrics.
        
        Args:
            total_operations: Total number of operations performed
        """
        if not self.current_metrics:
            return
            
        if self.current_metrics.execution_time > 0:
            self.current_metrics.operations_per_second = total_operations / self.current_metrics.execution_time
            self.current_metrics.throughput_gops = self.current_metrics.operations_per_second / 1e9
    
    def calculate_energy_efficiency(self, total_energy_j: float, total_operations: int) -> None:
        """
        Calculate energy efficiency metrics.
        
        Args:
            total_energy_j: Total energy consumption in joules
            total_operations: Total number of operations
        """
        if not self.current_metrics or total_energy_j <= 0:
            return
            
        self.current_metrics.energy_efficiency = total_operations / total_energy_j
    
    def calculate_latency(self, total_cycles: int, clock_frequency_hz: float) -> None:
        """
        Calculate latency metrics.
        
        Args:
            total_cycles: Total number of clock cycles
            clock_frequency_hz: Clock frequency in Hz
        """
        if not self.current_metrics or clock_frequency_hz <= 0:
            return
            
        self.current_metrics.latency_ms = (total_cycles / clock_frequency_hz) * 1000
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns:
            Dict containing performance summary
        """
        if not self.metrics_history:
            return {"status": "No metrics available"}
        
        latest = self.metrics_history[-1]
        
        summary = {
            "execution_time_s": latest.execution_time,
            "memory_usage_mb": latest.memory_usage_mb,
            "peak_memory_mb": latest.peak_memory_mb,
            "cpu_usage_percent": latest.cpu_usage_percent,
            "operations_per_second": latest.operations_per_second,
            "throughput_gops": latest.throughput_gops,
            "energy_efficiency_ops_per_j": latest.energy_efficiency,
            "latency_ms": latest.latency_ms,
            "total_measurements": len(self.metrics_history),
        }
        
        # Add custom metrics
        if latest.additional_metrics:
            summary["custom_metrics"] = latest.additional_metrics
        
        return summary
    
    def get_historical_data(self) -> List[Dict[str, Any]]:
        """
        Get historical performance data.
        
        Returns:
            List of performance measurements
        """
        return [
            {
                "timestamp": i,
                "execution_time": m.execution_time,
                "memory_usage_mb": m.memory_usage_mb,
                "cpu_usage_percent": m.cpu_usage_percent,
                "operations_per_second": m.operations_per_second,
                "throughput_gops": m.throughput_gops,
                **m.additional_metrics
            }
            for i, m in enumerate(self.metrics_history)
        ]


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for RAMwich performance evaluation.
    """
    
    def __init__(self, monitor: Optional[PerformanceMonitor] = None):
        """
        Initialize benchmark suite.
        
        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor or PerformanceMonitor()
        self.benchmark_results: Dict[str, PerformanceMetrics] = {}
    
    def run_benchmark(self, name: str, benchmark_func, *args, **kwargs) -> PerformanceMetrics:
        """
        Run a single benchmark.
        
        Args:
            name: Benchmark name
            benchmark_func: Function to benchmark
            *args: Arguments for benchmark function
            **kwargs: Keyword arguments for benchmark function
            
        Returns:
            PerformanceMetrics: Benchmark results
        """
        logger.info(f"Running benchmark: {name}")
        
        with self.monitor.monitor_execution(name) as metrics:
            try:
                result = benchmark_func(*args, **kwargs)
                metrics.additional_metrics["benchmark_result"] = result
                metrics.additional_metrics["status"] = "success"
            except Exception as e:
                logger.error(f"Benchmark {name} failed: {e}")
                metrics.additional_metrics["error"] = str(e)
                metrics.additional_metrics["status"] = "failed"
        
        final_metrics = self.monitor.metrics_history[-1]
        self.benchmark_results[name] = final_metrics
        
        logger.info(f"Benchmark {name} completed in {final_metrics.execution_time:.2f}s")
        return final_metrics
    
    def run_scaling_benchmark(self, name: str, benchmark_func, scale_params: List[Dict[str, Any]]) -> Dict[str, PerformanceMetrics]:
        """
        Run scaling benchmarks with different parameters.
        
        Args:
            name: Base benchmark name
            benchmark_func: Function to benchmark
            scale_params: List of parameter dictionaries for scaling
            
        Returns:
            Dict mapping parameter sets to performance metrics
        """
        scaling_results = {}
        
        for i, params in enumerate(scale_params):
            benchmark_name = f"{name}_scale_{i}"
            logger.info(f"Running scaling benchmark {benchmark_name} with params: {params}")
            
            metrics = self.run_benchmark(benchmark_name, benchmark_func, **params)
            scaling_results[str(params)] = metrics
        
        return scaling_results
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark report.
        
        Returns:
            Dict containing benchmark report
        """
        if not self.benchmark_results:
            return {"status": "No benchmarks run"}
        
        report = {
            "summary": {
                "total_benchmarks": len(self.benchmark_results),
                "successful_benchmarks": sum(
                    1 for m in self.benchmark_results.values()
                    if m.additional_metrics.get("status") == "success"
                ),
                "failed_benchmarks": sum(
                    1 for m in self.benchmark_results.values()
                    if m.additional_metrics.get("status") == "failed"
                ),
            },
            "results": {},
            "performance_analysis": {}
        }
        
        # Individual benchmark results
        for name, metrics in self.benchmark_results.items():
            report["results"][name] = {
                "execution_time_s": metrics.execution_time,
                "memory_usage_mb": metrics.memory_usage_mb,
                "throughput_gops": metrics.throughput_gops,
                "energy_efficiency": metrics.energy_efficiency,
                "status": metrics.additional_metrics.get("status", "unknown"),
                "custom_metrics": {
                    k: v for k, v in metrics.additional_metrics.items()
                    if k not in ["status", "benchmark_result", "error"]
                }
            }
        
        # Performance analysis
        successful_metrics = [
            m for m in self.benchmark_results.values()
            if m.additional_metrics.get("status") == "success"
        ]
        
        if successful_metrics:
            execution_times = [m.execution_time for m in successful_metrics]
            memory_usages = [m.memory_usage_mb for m in successful_metrics]
            
            report["performance_analysis"] = {
                "avg_execution_time_s": sum(execution_times) / len(execution_times),
                "min_execution_time_s": min(execution_times),
                "max_execution_time_s": max(execution_times),
                "avg_memory_usage_mb": sum(memory_usages) / len(memory_usages),
                "peak_memory_usage_mb": max(memory_usages),
            }
        
        return report