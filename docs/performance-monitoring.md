# Performance Monitoring

JAF's performance monitoring system provides comprehensive insights into agent execution performance, resource utilization, and system metrics. This system helps track and optimize performance in production environments.

## Overview

The performance monitoring system offers:

- **Execution Metrics**: Track execution time, memory usage, and resource consumption
- **LLM and Tool Tracking**: Monitor LLM calls, tool executions, and token usage
- **Cache Performance**: Track cache hit rates and efficiency
- **Error and Retry Monitoring**: Monitor error rates and retry patterns
- **Historical Analysis**: Collect and analyze performance trends over time

## Core Components

### PerformanceMonitor

The central monitoring component that tracks execution metrics:

```python
from jaf.core.performance import PerformanceMonitor, monitor_performance

# Create and use performance monitor
monitor = PerformanceMonitor()

# Start monitoring
monitor.start_monitoring()

# Record various events during execution
monitor.record_llm_call(token_count=150)
monitor.record_tool_call()
monitor.record_cache_hit()
monitor.record_error()

# Stop monitoring and get metrics
metrics = monitor.stop_monitoring()

print(f"Execution Time: {metrics.execution_time_ms}ms")
print(f"Memory Usage: {metrics.memory_usage_mb}MB")
print(f"Peak Memory: {metrics.peak_memory_mb}MB")
print(f"Token Count: {metrics.token_count}")
print(f"Cache Hit Rate: {metrics.cache_hit_rate}%")
print(f"LLM Calls: {metrics.llm_call_count}")
print(f"Tool Calls: {metrics.tool_call_count}")
print(f"Errors: {metrics.error_count}")
print(f"Retries: {metrics.retry_count}")
```

### PerformanceMetrics

Comprehensive metrics data structure:

```python
from jaf.core.performance import PerformanceMetrics

# PerformanceMetrics contains:
# - execution_time_ms: Total execution time in milliseconds
# - memory_usage_mb: Current memory usage in MB
# - peak_memory_mb: Peak memory usage during execution
# - token_count: Total tokens processed
# - cache_hit_rate: Cache hit rate percentage
# - llm_call_count: Number of LLM calls made
# - tool_call_count: Number of tool calls made
# - error_count: Number of errors encountered
# - retry_count: Number of retry attempts

# Convert metrics to dictionary for serialization
metrics_dict = metrics.to_dict()
print(f"Metrics: {metrics_dict}")
```

## Advanced Features

### Context Manager for Performance Monitoring

Use the context manager for automatic performance tracking:

```python
from jaf.core.performance import monitor_performance
import asyncio

# Use context manager for automatic monitoring
async def run_with_monitoring():
    async with monitor_performance() as monitor:
        # Simulate agent execution
        monitor.record_llm_call(token_count=150)
        await asyncio.sleep(0.1)  # Simulate processing
        
        monitor.record_tool_call()
        await asyncio.sleep(0.05)  # Simulate tool execution
        
        monitor.record_cache_hit()
        
        # Monitor automatically stops and returns metrics
        # when exiting the context
    
    print("Performance monitoring completed automatically")

# Run with callback
async def run_with_callback():
    def on_complete(metrics):
        print(f"Execution completed in {metrics.execution_time_ms}ms")
        print(f"Peak memory: {metrics.peak_memory_mb}MB")
    
    async with monitor_performance(on_complete=on_complete) as monitor:
        monitor.record_llm_call(token_count=200)
        # ... execution logic
```

### Performance Collection and Analysis

Collect and analyze performance across multiple runs:

```python
from jaf.core.performance import PerformanceCollector, get_performance_summary

# Create collector for aggregating metrics
collector = PerformanceCollector()

# Simulate multiple runs
async def simulate_multiple_runs():
    for i in range(10):
        async with monitor_performance() as monitor:
            # Simulate varying workloads
            monitor.record_llm_call(token_count=100 + i * 10)
            monitor.record_tool_call()
            
            if i % 3 == 0:
                monitor.record_error()
            else:
                monitor.record_cache_hit()
            
            await asyncio.sleep(0.1 + i * 0.01)  # Varying execution time
        
        # Collect metrics from this run
        metrics = monitor.stop_monitoring()
        collector.collect_metrics(metrics, run_id=f"run_{i}")

# Analyze collected performance data
def analyze_performance():
    # Get average metrics
    avg_metrics = collector.get_average_metrics()
    if avg_metrics:
        print(f"Average execution time: {avg_metrics.execution_time_ms:.2f}ms")
        print(f"Average memory usage: {avg_metrics.memory_usage_mb:.2f}MB")
        print(f"Average cache hit rate: {avg_metrics.cache_hit_rate:.1f}%")
    
    # Get recent performance (last 5 runs)
    recent_avg = collector.get_average_metrics(last_n=5)
    if recent_avg:
        print(f"Recent average execution time: {recent_avg.execution_time_ms:.2f}ms")
    
    # Get comprehensive summary
    summary = collector.get_performance_summary()
    print(f"Performance summary: {summary}")
    
    # Get global performance summary
    global_summary = get_performance_summary()
    print(f"Global performance: {global_summary}")

# Usage
asyncio.run(simulate_multiple_runs())
analyze_performance()
```

## Advanced Monitoring Features

### Resource Profiling

Deep dive into resource usage patterns:

```python
from jaf.core.performance import ResourceProfiler, ProfilerConfig

class DetailedResourceMonitor:
    def __init__(self):
        self.profiler = ResourceProfiler()
        self.profiling_sessions = {}
    
    async def profile_agent_execution(self, agent_name: str, execution_func):
        """Profile a complete agent execution."""
        
        # Start profiling session
        session = self.profiler.start_session(
            session_id=f"{agent_name}_{int(time.time())}",
            config=ProfilerConfig(
                track_memory_allocations=True,
                track_cpu_usage=True,
                track_io_operations=True,
                sample_rate_ms=100
            )
        )
        
        try:
            # Execute with profiling
            result = await execution_func()
            
            # Get detailed profile
            profile = session.get_profile()
            
            # Analyze performance patterns
            analysis = self._analyze_profile(profile)
            
            return {
                'result': result,
                'performance_profile': profile,
                'analysis': analysis,
                'recommendations': self._generate_recommendations(analysis)
            }
            
        finally:
            session.end()
    
    def _analyze_profile(self, profile):
        """Analyze performance profile for insights."""
        return {
            'peak_memory_usage': profile.peak_memory_mb,
            'avg_cpu_usage': profile.avg_cpu_percent,
            'io_bottlenecks': profile.io_wait_time_ms,
            'gc_pressure': profile.garbage_collection_time_ms,
            'hot_spots': profile.cpu_hot_spots,
            'memory_leaks': profile.potential_memory_leaks
        }
    
    def _generate_recommendations(self, analysis):
        """Generate optimization recommendations."""
        recommendations = []
        
        if analysis['peak_memory_usage'] > 1000:  # MB
            recommendations.append({
                'type': 'memory_optimization',
                'description': 'Consider reducing batch sizes or implementing streaming',
                'priority': 'high'
            })
        
        if analysis['avg_cpu_usage'] > 80:  # Percent
            recommendations.append({
                'type': 'cpu_optimization',
                'description': 'Consider async processing or load balancing',
                'priority': 'medium'
            })
        
        if analysis['io_bottlenecks'] > 1000:  # ms
            recommendations.append({
                'type': 'io_optimization',
                'description': 'Consider connection pooling or caching',
                'priority': 'high'
            })
        
        return recommendations
```

### Predictive Performance Analytics

Forecast performance trends and capacity needs:

```python
from jaf.core.performance import PredictiveAnalyzer, TrendAnalysis

class PerformancePredictor:
    def __init__(self):
        self.analyzer = PredictiveAnalyzer()
        self.historical_data = []
    
    def analyze_trends(self, time_range: str = '7d'):
        """Analyze performance trends over time."""
        
        # Get historical performance data
        historical_metrics = self._get_historical_metrics(time_range)
        
        # Perform trend analysis
        trend_analysis = self.analyzer.analyze_trends(historical_metrics)
        
        return TrendAnalysis(
            cpu_trend=trend_analysis.cpu_trend,
            memory_trend=trend_analysis.memory_trend,
            response_time_trend=trend_analysis.response_time_trend,
            throughput_trend=trend_analysis.throughput_trend,
            error_rate_trend=trend_analysis.error_rate_trend,
            predictions=self._generate_predictions(trend_analysis)
        )
    
    def _generate_predictions(self, trend_analysis):
        """Generate performance predictions."""
        predictions = {}
        
        # Predict resource needs
        if trend_analysis.memory_trend.slope > 0.1:  # Growing memory usage
            days_to_limit = self._calculate_days_to_memory_limit(trend_analysis.memory_trend)
            predictions['memory_capacity'] = {
                'warning': f"Memory usage trending up, may reach limits in {days_to_limit} days",
                'recommendation': 'Consider scaling up or optimizing memory usage'
            }
        
        # Predict performance degradation
        if trend_analysis.response_time_trend.slope > 0.05:  # Increasing response times
            predictions['performance_degradation'] = {
                'warning': 'Response times trending upward',
                'recommendation': 'Investigate performance bottlenecks'
            }
        
        # Predict capacity needs
        if trend_analysis.throughput_trend.slope > 0.2:  # Increasing load
            predicted_load = self._predict_future_load(trend_analysis.throughput_trend)
            predictions['capacity_planning'] = {
                'predicted_load': predicted_load,
                'recommendation': f'Plan for {predicted_load:.1f}x current capacity in 30 days'
            }
        
        return predictions
    
    def get_optimization_opportunities(self):
        """Identify optimization opportunities."""
        current_metrics = self._get_current_metrics()
        
        opportunities = []
        
        # Check for underutilized resources
        if current_metrics.cpu_usage_percent < 30:
            opportunities.append({
                'type': 'resource_optimization',
                'description': 'CPU underutilized, consider consolidating workloads',
                'potential_savings': '20-30% cost reduction'
            })
        
        # Check for cache optimization opportunities
        if current_metrics.cache_hit_rate < 0.8:
            opportunities.append({
                'type': 'cache_optimization',
                'description': 'Low cache hit rate, consider cache tuning',
                'potential_improvement': '15-25% response time improvement'
            })
        
        # Check for batch processing opportunities
        if current_metrics.small_request_ratio > 0.7:
            opportunities.append({
                'type': 'batching_optimization',
                'description': 'Many small requests, consider request batching',
                'potential_improvement': '30-40% throughput improvement'
            })
        
        return opportunities
```

## Real-time Monitoring and Alerting

### Performance Alerting System

Set up intelligent performance alerts:

```python
from jaf.core.performance import PerformanceAlertManager, AlertRule, AlertSeverity

class IntelligentPerformanceAlerting:
    def __init__(self):
        self.alert_manager = PerformanceAlertManager()
        self._setup_alert_rules()
    
    def _setup_alert_rules(self):
        """Configure comprehensive performance alert rules."""
        
        # Critical performance alerts
        self.alert_manager.add_rule(AlertRule(
            name='critical_response_time',
            condition=lambda metrics: metrics.avg_response_time_ms > 5000,
            severity=AlertSeverity.CRITICAL,
            action=self._handle_critical_performance,
            cooldown_minutes=5,
            description='Response time exceeds 5 seconds'
        ))
        
        self.alert_manager.add_rule(AlertRule(
            name='memory_pressure',
            condition=lambda metrics: metrics.memory_usage_percent > 90,
            severity=AlertSeverity.CRITICAL,
            action=self._handle_memory_pressure,
            cooldown_minutes=2,
            description='Memory usage above 90%'
        ))
        
        # Warning level alerts
        self.alert_manager.add_rule(AlertRule(
            name='degraded_performance',
            condition=lambda metrics: (
                metrics.avg_response_time_ms > 2000 and 
                metrics.error_rate > 0.05
            ),
            severity=AlertSeverity.WARNING,
            action=self._handle_performance_degradation,
            cooldown_minutes=10,
            description='Performance degradation detected'
        ))
        
        self.alert_manager.add_rule(AlertRule(
            name='resource_inefficiency',
            condition=lambda metrics: (
                metrics.cpu_usage_percent < 20 and 
                metrics.memory_usage_percent < 30
            ),
            severity=AlertSeverity.INFO,
            action=self._handle_resource_underutilization,
            cooldown_minutes=60,
            description='Resources underutilized'
        ))
    
    def _handle_critical_performance(self, metrics):
        """Handle critical performance issues."""
        logger.critical(f"Critical performance issue: {metrics.avg_response_time_ms}ms response time")
        
        # Immediate actions
        self._enable_emergency_mode()
        self._scale_up_resources()
        self._notify_on_call_team()
        
        # Diagnostic actions
        self._start_detailed_profiling()
        self._capture_performance_snapshot()
    
    def _handle_memory_pressure(self, metrics):
        """Handle memory pressure situations."""
        logger.critical(f"Memory pressure: {metrics.memory_usage_percent}% usage")
        
        # Immediate relief actions
        self._trigger_garbage_collection()
        self._reduce_cache_sizes()
        self._limit_concurrent_requests()
        
        # Scaling actions
        self._request_additional_memory()
        self._consider_horizontal_scaling()
    
    def _handle_performance_degradation(self, metrics):
        """Handle gradual performance degradation."""
        logger.warning(f"Performance degradation: {metrics.avg_response_time_ms}ms, {metrics.error_rate} error rate")
        
        # Analysis actions
        self._analyze_performance_trends()
        self._check_resource_bottlenecks()
        self._review_recent_changes()
        
        # Mitigation actions
        self._optimize_current_workload()
        self._adjust_performance_parameters()
```

### Real-time Dashboard Integration

Create real-time performance dashboards:

```python
from jaf.core.performance import PerformanceDashboard, DashboardConfig

class RealTimePerformanceDashboard:
    def __init__(self):
        self.dashboard = PerformanceDashboard()
        self.metrics_buffer = []
        self.update_interval_seconds = 5
    
    async def start_dashboard(self, port: int = 8080):
        """Start real-time performance dashboard."""
        
        # Configure dashboard
        config = DashboardConfig(
            update_interval_ms=1000,
            max_data_points=1000,
            enable_real_time_charts=True,
            enable_alerts_panel=True,
            enable_predictions_panel=True
        )
        
        # Start dashboard server
        await self.dashboard.start_server(port=port, config=config)
        
        # Start metrics collection
        asyncio.create_task(self._collect_metrics_loop())
        
        print(f"Performance dashboard available at http://localhost:{port}")
    
    async def _collect_metrics_loop(self):
        """Continuously collect and update metrics."""
        while True:
            try:
                # Collect current metrics
                metrics = self._collect_current_metrics()
                
                # Update dashboard
                await self.dashboard.update_metrics(metrics)
                
                # Store for trend analysis
                self.metrics_buffer.append(metrics)
                if len(self.metrics_buffer) > 1000:
                    self.metrics_buffer.pop(0)  # Keep last 1000 points
                
                await asyncio.sleep(self.update_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.update_interval_seconds)
    
    def get_dashboard_data(self):
        """Get current dashboard data for API endpoints."""
        if not self.metrics_buffer:
            return {'error': 'No metrics available'}
        
        current_metrics = self.metrics_buffer[-1]
        
        return {
            'current_metrics': {
                'cpu_usage': current_metrics.cpu_usage_percent,
                'memory_usage': current_metrics.memory_usage_mb,
                'response_time': current_metrics.avg_response_time_ms,
                'throughput': current_metrics.requests_per_minute,
                'error_rate': current_metrics.error_rate
            },
            'trends': self._calculate_trends(),
            'alerts': self._get_active_alerts(),
            'predictions': self._get_performance_predictions(),
            'recommendations': self._get_optimization_recommendations()
        }
    
    def _calculate_trends(self):
        """Calculate performance trends from recent data."""
        if len(self.metrics_buffer) < 10:
            return {}
        
        recent_metrics = self.metrics_buffer[-10:]
        
        return {
            'cpu_trend': self._calculate_trend([m.cpu_usage_percent for m in recent_metrics]),
            'memory_trend': self._calculate_trend([m.memory_usage_mb for m in recent_metrics]),
            'response_time_trend': self._calculate_trend([m.avg_response_time_ms for m in recent_metrics]),
            'throughput_trend': self._calculate_trend([m.requests_per_minute for m in recent_metrics])
        }
```

## Integration with External Monitoring

### Prometheus Integration

Export metrics to Prometheus:

```python
from jaf.core.performance import PrometheusExporter
from prometheus_client import Counter, Histogram, Gauge

class PrometheusPerformanceExporter:
    def __init__(self):
        self.exporter = PrometheusExporter()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up Prometheus metrics."""
        
        # Counters
        self.request_count = Counter(
            'jaf_requests_total',
            'Total number of requests',
            ['agent_name', 'status']
        )
        
        self.tool_calls_count = Counter(
            'jaf_tool_calls_total',
            'Total number of tool calls',
            ['tool_name', 'status']
        )
        
        # Histograms
        self.response_time_histogram = Histogram(
            'jaf_response_time_seconds',
            'Response time distribution',
            ['agent_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.tool_execution_histogram = Histogram(
            'jaf_tool_execution_seconds',
            'Tool execution time distribution',
            ['tool_name'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        # Gauges
        self.active_agents = Gauge(
            'jaf_active_agents',
            'Number of active agents'
        )
        
        self.memory_usage = Gauge(
            'jaf_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'jaf_cpu_usage_percent',
            'CPU usage percentage'
        )
    
    def record_request(self, agent_name: str, response_time_seconds: float, status: str):
        """Record request metrics."""
        self.request_count.labels(agent_name=agent_name, status=status).inc()
        self.response_time_histogram.labels(agent_name=agent_name).observe(response_time_seconds)
    
    def record_tool_call(self, tool_name: str, execution_time_seconds: float, status: str):
        """Record tool call metrics."""
        self.tool_calls_count.labels(tool_name=tool_name, status=status).inc()
        self.tool_execution_histogram.labels(tool_name=tool_name).observe(execution_time_seconds)
    
    def update_system_metrics(self, metrics):
        """Update system-level metrics."""
        self.active_agents.set(metrics.active_agents)
        self.memory_usage.set(metrics.memory_usage_mb * 1024 * 1024)  # Convert to bytes
        self.cpu_usage.set(metrics.cpu_usage_percent)
```

### Grafana Dashboard Configuration

Example Grafana dashboard configuration:

```json
{
  "dashboard": {
    "title": "JAF Performance Dashboard",
    "panels": [
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(jaf_response_time_seconds_sum[5m]) / rate(jaf_response_time_seconds_count[5m])",
            "legendFormat": "Avg Response Time"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(jaf_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(jaf_requests_total{status=\"error\"}[5m]) / rate(jaf_requests_total[5m]) * 100",
            "legendFormat": "Error Rate %"
          }
        ]
      },
      {
        "title": "Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "jaf_cpu_usage_percent",
            "legendFormat": "CPU %"
          },
          {
            "expr": "jaf_memory_usage_bytes / 1024 / 1024",
            "legendFormat": "Memory MB"
          }
        ]
      }
    ]
  }
}
```

## Best Practices

### 1. Balanced Monitoring

Monitor what matters without overwhelming the system:

```python
# Good: Focused monitoring
monitoring_config = {
    'essential_metrics': ['response_time', 'error_rate', 'throughput'],
    'detailed_metrics': ['memory_usage', 'cpu_usage'],
    'debug_metrics': ['gc_time', 'thread_count'],  # Only in debug mode
    'sampling_rate': 0.1  # Sample 10% for detailed metrics
}
```

### 2. Proactive Alerting

Set up alerts that prevent issues rather than just reporting them:

```python
# Good: Predictive alerting
def setup_proactive_alerts():
    return [
        AlertRule(
            name='trending_response_time',
            condition=lambda metrics: metrics.response_time_trend.slope > 0.1,
            description='Response time trending upward'
        ),
        AlertRule(
            name='capacity_warning',
            condition=lambda metrics: metrics.predicted_capacity_days < 7,
            description='Capacity limit predicted within 7 days'
        )
    ]
```

### 3. Performance Budgets

Set and enforce performance budgets:

```python
# Good: Performance budget enforcement
class PerformanceBudget:
    def __init__(self):
        self.budgets = {
            'max_response_time_ms': 2000,
            'max_memory_usage_mb': 1024,
            'min_cache_hit_rate': 0.8,
            'max_error_rate': 0.01
        }
    
    def check_budget_compliance(self, metrics):
        violations = []
        
        if metrics.avg_response_time_ms > self.budgets['max_response_time_ms']:
            violations.append('response_time_exceeded')
        
        if metrics.memory_usage_mb > self.budgets['max_memory_usage_mb']:
            violations.append('memory_budget_exceeded')
        
        return violations
```

## Example: Complete Performance Monitoring Setup

Here's a comprehensive example showing a complete performance monitoring implementation:

```python
import asyncio
import time
from jaf.core.performance import (
    PerformanceMonitor, PerformanceAlertManager, PrometheusExporter,
    PerformanceDashboard, AutoOptimizer
)

class ComprehensivePerformanceSystem:
    """Complete performance monitoring and optimization system."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.alert_manager = PerformanceAlertManager()
        self.prometheus_exporter = PrometheusExporter()
        self.dashboard = PerformanceDashboard()
        self.optimizer = AutoOptimizer()
        
        self.performance_history = []
        self.optimization_history = []
    
    async def initialize(self):
        """Initialize the complete performance system."""
        
        # Set up monitoring
        await self.monitor.start()
        
        # Configure alerts
        self._setup_comprehensive_alerts()
        
        # Start Prometheus exporter
        await self.prometheus_exporter.start(port=9090)
        
        # Start dashboard
        await self.dashboard.start(port=8080)
        
        # Start optimization loop
        asyncio.create_task(self._optimization_loop())
        
        print("🚀 Comprehensive performance system initialized")
        print("📊 Dashboard: http://localhost:8080")
        print("📈 Metrics: http://localhost:9090/metrics")
    
    def _setup_comprehensive_alerts(self):
        """Set up comprehensive alerting rules."""
        
        # Performance alerts
        self.alert_manager.add_rule({
            'name': 'response_time_sla_breach',
            'condition': lambda m: m.p95_response_time_ms > 3000,
            'severity': 'critical',
            'action': self._handle_sla_breach
        })
        
        # Resource alerts
        self.alert_manager.add_rule({
            'name': 'memory_leak_detection',
            'condition': lambda m: self._detect_memory_leak(m),
            'severity': 'warning',
            'action': self._handle_memory_leak
        })
        
        # Capacity alerts
        self.alert_manager.add_rule({
            'name': 'capacity_planning',
            'condition': lambda m: self._predict_capacity_exhaustion(m) < 7,
            'severity': 'info',
            'action': self._handle_capacity_planning
        })
    
    async def _optimization_loop(self):
        """Continuous optimization loop."""
        while True:
            try:
                # Collect current metrics
                metrics = await self.monitor.get_comprehensive_metrics()
                
                # Store for trend analysis
                self.performance_history.append(metrics)
                if len(self.performance_history) > 1000:
                    self.performance_history.pop(0)
                
                # Check for optimization opportunities
                if self.optimizer.should_optimize(metrics):
                    optimizations = await self.optimizer.generate_optimizations(
                        current_metrics=metrics,
                        history=self.performance_history[-50:]
                    )
                    
                    # Apply optimizations
                    for optimization in optimizations:
                        success = await self._apply_optimization(optimization)
                        self.optimization_history.append({
                            'optimization': optimization,
                            'success': success,
                            'timestamp': time.time()
                        })
                
                # Update external systems
                await self._update_external_systems(metrics)
                
                # Wait for next cycle
                await asyncio.sleep(60)  # Optimize every minute
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)
    
    async def _apply_optimization(self, optimization):
        """Apply a specific optimization."""
        try:
            if optimization.type == 'cache_tuning':
                await self._tune_cache_parameters(optimization.parameters)
            elif optimization.type == 'resource_scaling':
                await self._scale_resources(optimization.parameters)
            elif optimization.type == 'load_balancing':
                await self._adjust_load_balancing(optimization.parameters)
            
            logger.info(f"Applied optimization: {optimization.type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply optimization {optimization.type}: {e}")
            return False
    
    async def _update_external_systems(self, metrics):
        """Update external monitoring systems."""
        
        # Update Prometheus metrics
        self.prometheus_exporter.update_metrics(metrics)
        
        # Update dashboard
        await self.dashboard.update_real_time_data(metrics)
        
        # Send to external APM (if configured)
        if hasattr(self, 'apm_client'):
            await self.apm_client.send_metrics(metrics)
    
    def get_performance_report(self, time_range: str = '24h'):
        """Generate comprehensive performance report."""
        
        # Get metrics for time range
        metrics = self._get_metrics_for_range(time_range)
        
        # Calculate statistics
        stats = self._calculate_performance_statistics(metrics)
        
        # Generate insights
        insights = self._generate_performance_insights(stats)
        
        # Create recommendations
        recommendations = self._generate_recommendations(insights)
        
        return {
            'time_range': time_range,
            'summary': stats,
            'insights': insights,
            'recommendations': recommendations,
            'optimization_history': self.optimization_history[-10:],
            'trend_analysis': self._analyze_trends(metrics)
        }

# Usage example
async def main():
    """Demonstrate comprehensive performance monitoring."""
    
    # Initialize performance system
    perf_
