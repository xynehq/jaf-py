"""
Performance monitoring and metrics collection for JAF framework.

This module provides comprehensive performance tracking capabilities including
execution timing, memory usage, token counting, and cache hit rate monitoring.
"""

import time
import psutil
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from contextlib import asynccontextmanager

from .types import TraceEvent, RunId, TraceId


@dataclass(frozen=True)
class PerformanceMetrics:
    """Comprehensive performance metrics for agent execution."""
    execution_time_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    token_count: int
    cache_hit_rate: float
    llm_call_count: int
    tool_call_count: int
    error_count: int
    retry_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'execution_time_ms': self.execution_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'token_count': self.token_count,
            'cache_hit_rate': self.cache_hit_rate,
            'llm_call_count': self.llm_call_count,
            'tool_call_count': self.tool_call_count,
            'error_count': self.error_count,
            'retry_count': self.retry_count
        }


@dataclass(frozen=True)
class PerformanceEvent:
    """Performance-related trace event."""
    type: str = 'performance_metrics'
    data: PerformanceMetrics = field(default_factory=lambda: PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0))
    timestamp: float = field(default_factory=time.time)
    run_id: Optional[RunId] = None
    trace_id: Optional[TraceId] = None


class PerformanceMonitor:
    """
    Performance monitoring system for JAF agents.
    
    Tracks execution metrics, memory usage, and provides performance insights.
    """
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory: float = 0
        self.token_count: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.llm_calls: int = 0
        self.tool_calls: int = 0
        self.errors: int = 0
        self.retries: int = 0
        self.process = psutil.Process()
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        self.peak_memory = self.start_memory
    
    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and return collected metrics."""
        if self.start_time is None:
            raise ValueError("Monitoring not started")
        
        execution_time = (time.time() - self.start_time) * 1000  # Convert to ms
        current_memory = self._get_memory_usage()
        cache_hit_rate = self._calculate_cache_hit_rate()
        
        return PerformanceMetrics(
            execution_time_ms=execution_time,
            memory_usage_mb=current_memory,
            peak_memory_mb=self.peak_memory,
            token_count=self.token_count,
            cache_hit_rate=cache_hit_rate,
            llm_call_count=self.llm_calls,
            tool_call_count=self.tool_calls,
            error_count=self.errors,
            retry_count=self.retries
        )
    
    def record_llm_call(self, token_count: int = 0) -> None:
        """Record an LLM call with optional token count."""
        self.llm_calls += 1
        self.token_count += token_count
        self._update_peak_memory()
    
    def record_tool_call(self) -> None:
        """Record a tool call."""
        self.tool_calls += 1
        self._update_peak_memory()
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1
    
    def record_error(self) -> None:
        """Record an error occurrence."""
        self.errors += 1
    
    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.retries += 1
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert bytes to MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def _update_peak_memory(self) -> None:
        """Update peak memory usage."""
        current_memory = self._get_memory_usage()
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        total_cache_operations = self.cache_hits + self.cache_misses
        if total_cache_operations == 0:
            return 0.0
        return (self.cache_hits / total_cache_operations) * 100


@asynccontextmanager
async def monitor_performance(
    run_id: Optional[RunId] = None,
    trace_id: Optional[TraceId] = None,
    on_complete: Optional[Callable[[PerformanceMetrics], None]] = None
) -> AsyncIterator[PerformanceMonitor]:
    """
    Context manager for performance monitoring.
    
    Usage:
        async with monitor_performance() as monitor:
            monitor.record_llm_call(150)  # 150 tokens
            # ... agent execution
    """
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        yield monitor
    finally:
        metrics = monitor.stop_monitoring()
        
        if on_complete:
            on_complete(metrics)
        
        # Emit performance event
        event = PerformanceEvent(
            data=metrics,
            run_id=run_id,
            trace_id=trace_id
        )


class PerformanceCollector:
    """
    Collects and aggregates performance metrics across multiple runs.
    """
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.run_metrics: Dict[str, PerformanceMetrics] = {}
    
    def collect_metrics(self, metrics: PerformanceMetrics, run_id: Optional[str] = None) -> None:
        """Collect performance metrics from a run."""
        self.metrics_history.append(metrics)
        if run_id:
            self.run_metrics[run_id] = metrics
    
    def get_average_metrics(self, last_n: Optional[int] = None) -> Optional[PerformanceMetrics]:
        """Get average metrics across runs."""
        if not self.metrics_history:
            return None
        
        metrics_to_analyze = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        if not metrics_to_analyze:
            return None
        
        count = len(metrics_to_analyze)
        
        return PerformanceMetrics(
            execution_time_ms=sum(m.execution_time_ms for m in metrics_to_analyze) / count,
            memory_usage_mb=sum(m.memory_usage_mb for m in metrics_to_analyze) / count,
            peak_memory_mb=max(m.peak_memory_mb for m in metrics_to_analyze),
            token_count=sum(m.token_count for m in metrics_to_analyze) / count,
            cache_hit_rate=sum(m.cache_hit_rate for m in metrics_to_analyze) / count,
            llm_call_count=sum(m.llm_call_count for m in metrics_to_analyze) / count,
            tool_call_count=sum(m.tool_call_count for m in metrics_to_analyze) / count,
            error_count=sum(m.error_count for m in metrics_to_analyze) / count,
            retry_count=sum(m.retry_count for m in metrics_to_analyze) / count
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        recent_metrics = self.get_average_metrics(last_n=10)
        all_time_metrics = self.get_average_metrics()
        
        return {
            'total_runs': len(self.metrics_history),
            'recent_average': recent_metrics.to_dict() if recent_metrics else None,
            'all_time_average': all_time_metrics.to_dict() if all_time_metrics else None,
            'performance_trends': self._analyze_trends()
        }
    
    def _analyze_trends(self) -> Dict[str, str]:
        """Analyze performance trends."""
        if len(self.metrics_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent = self.metrics_history[-5:] if len(self.metrics_history) >= 5 else self.metrics_history
        older = self.metrics_history[:-5] if len(self.metrics_history) >= 10 else []
        
        if not older:
            return {'status': 'insufficient_historical_data'}
        
        recent_avg_time = sum(m.execution_time_ms for m in recent) / len(recent)
        older_avg_time = sum(m.execution_time_ms for m in older) / len(older)
        
        time_trend = 'improving' if recent_avg_time < older_avg_time else 'degrading'
        
        recent_avg_memory = sum(m.memory_usage_mb for m in recent) / len(recent)
        older_avg_memory = sum(m.memory_usage_mb for m in older) / len(older)
        
        memory_trend = 'improving' if recent_avg_memory < older_avg_memory else 'degrading'
        
        return {
            'execution_time_trend': time_trend,
            'memory_usage_trend': memory_trend,
            'recent_avg_time_ms': recent_avg_time,
            'older_avg_time_ms': older_avg_time,
            'recent_avg_memory_mb': recent_avg_memory,
            'older_avg_memory_mb': older_avg_memory
        }


# Global performance collector instance
global_performance_collector = PerformanceCollector()


def get_performance_summary() -> Dict[str, Any]:
    """Get global performance summary."""
    return global_performance_collector.get_performance_summary()
