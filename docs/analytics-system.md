# Analytics System

JAF provides a comprehensive analytics system that enables conversation insights, agent performance tracking, and system monitoring. This system helps understand agent behavior and optimize performance in production environments.

## Overview

The analytics system consists of three main components:

- **ConversationAnalytics**: Analyzes individual conversations for sentiment, engagement, and resolution patterns
- **AgentAnalytics**: Tracks agent performance, tool usage, and execution patterns  
- **SystemAnalytics**: Monitors overall system health, resource usage, and operational metrics

## Core Components

### ConversationAnalytics

Provides insights into conversation quality and user engagement:

```python
from jaf.core.analytics import AnalyticsEngine, analyze_conversation_quality
from jaf.core.types import Message, ContentRole
import time

# Create analytics engine
analytics = AnalyticsEngine()

# Create messages for analysis
messages = [
    Message(role=ContentRole.USER, content='I need help with my order'),
    Message(role=ContentRole.ASSISTANT, content='I\'d be happy to help you with your order. Can you provide your order number?'),
    Message(role=ContentRole.USER, content='Yes, it\'s #12345'),
    Message(role=ContentRole.ASSISTANT, content='Thank you! I found your order. It was shipped yesterday and should arrive tomorrow.')
]

# Analyze conversation with start and end times
start_time = time.time() - 210  # 3.5 minutes ago
end_time = time.time()

conversation_analytics = analytics.analyze_conversation(messages, start_time, end_time)

print(f"Total Messages: {conversation_analytics.total_messages}")
print(f"User Messages: {conversation_analytics.user_messages}")
print(f"Assistant Messages: {conversation_analytics.assistant_messages}")
print(f"Average Message Length: {conversation_analytics.average_message_length}")
print(f"Duration: {conversation_analytics.conversation_duration_minutes} minutes")
print(f"Topic Keywords: {conversation_analytics.topic_keywords}")
print(f"Sentiment Score: {conversation_analytics.sentiment_score}")
print(f"Engagement Score: {conversation_analytics.engagement_score}")
print(f"Resolution Status: {conversation_analytics.resolution_status}")
```

**Available Metrics:**
- **total_messages**: Total number of messages in conversation
- **user_messages**: Number of user messages
- **assistant_messages**: Number of assistant messages  
- **tool_messages**: Number of tool messages
- **average_message_length**: Average length of messages
- **conversation_duration_minutes**: Duration in minutes
- **topic_keywords**: Extracted keywords and topics
- **sentiment_score**: Sentiment analysis score
- **engagement_score**: User engagement level (0-100)
- **resolution_status**: 'resolved', 'ongoing', or 'escalated'

### AgentAnalytics

Tracks agent performance and behavior patterns:

```python
from jaf.core.analytics import AnalyticsEngine

# Create analytics engine
analytics = AnalyticsEngine()

# Record agent performance data
analytics.record_agent_performance(
    agent_name='CustomerSupportAgent',
    success=True,
    response_time_ms=1250,
    tool_name='search_order',
    error_type=None
)

# Get agent analytics
agent_analytics = analytics.agent_analyzer.get_agent_analytics('CustomerSupportAgent')

if agent_analytics:
    print(f"Agent: {agent_analytics.agent_name}")
    print(f"Total Invocations: {agent_analytics.total_invocations}")
    print(f"Success Rate: {agent_analytics.success_rate}%")
    print(f"Average Response Time: {agent_analytics.average_response_time_ms}ms")
    print(f"Tool Usage: {agent_analytics.tool_usage_frequency}")
    print(f"Handoff Patterns: {agent_analytics.handoff_patterns}")
    print(f"Error Patterns: {agent_analytics.error_patterns}")
    print(f"Satisfaction Score: {agent_analytics.user_satisfaction_score}")
    print(f"Specializations: {agent_analytics.specialization_areas}")
```

**Tracked Metrics:**
- **total_invocations**: Number of times agent was invoked
- **success_rate**: Percentage of successful executions
- **average_response_time_ms**: Average response time in milliseconds
- **tool_usage_frequency**: Dictionary of tool usage counts
- **handoff_patterns**: Agent-to-agent handoff patterns
- **error_patterns**: Types and frequency of errors
- **user_satisfaction_score**: Average user satisfaction
- **specialization_areas**: Identified specialization areas

### SystemAnalytics

Monitors overall system health and performance:

```python
from jaf.core.analytics import AnalyticsEngine
from jaf.core.performance import PerformanceMetrics

# Create analytics engine
analytics = AnalyticsEngine()

# Record system metrics
metrics = PerformanceMetrics(
    execution_time_ms=1500,
    memory_usage_mb=128,
    token_count=450,
    error_count=0,
    retry_count=0
)

analytics.record_system_metrics(metrics, 'CustomerSupportAgent')

# Get system analytics
system_analytics = analytics.system_analyzer.get_system_analytics()

print(f"Total Conversations: {system_analytics.total_conversations}")
print(f"Active Agents: {system_analytics.active_agents}")
print(f"Peak Concurrent Sessions: {system_analytics.peak_concurrent_sessions}")
print(f"Resource Utilization: {system_analytics.resource_utilization}")
print(f"Performance Trends: {system_analytics.performance_trends}")
print(f"Bottlenecks: {system_analytics.bottlenecks}")
print(f"Recommendations: {system_analytics.optimization_recommendations}")
```

**System Metrics:**
- **total_conversations**: Total number of conversations processed
- **active_agents**: Number of active agents
- **peak_concurrent_sessions**: Peak concurrent session count
- **resource_utilization**: Memory, CPU, and other resource usage
- **performance_trends**: Historical performance data
- **bottlenecks**: Identified system bottlenecks
- **optimization_recommendations**: System optimization suggestions

## Advanced Usage

### Comprehensive Analytics Report

Get a complete analytics report across all dimensions:

```python
from jaf.core.analytics import AnalyticsEngine, get_analytics_report

# Create analytics engine
analytics = AnalyticsEngine()

# Record some sample data
analytics.record_agent_performance('SupportAgent', True, 1200, 'search_tool')
analytics.record_agent_performance('SalesAgent', True, 800, 'crm_tool')

# Get comprehensive analytics report
report = analytics.get_comprehensive_analytics()

print(f"Report Timestamp: {report['timestamp']}")
print(f"System Analytics: {report['system']}")
print(f"Agent Analytics: {report['agents']}")
print(f"Summary: {report['summary']}")

# Use global analytics function
global_report = get_analytics_report()
print(f"Global Analytics: {global_report}")
```

### Recording Agent Interactions

Track detailed agent interactions:

```python
from jaf.core.analytics import AnalyticsEngine

analytics = AnalyticsEngine()

# Record tool usage
analytics.agent_analyzer.record_tool_usage('CustomerAgent', 'search_orders')
analytics.agent_analyzer.record_tool_usage('CustomerAgent', 'update_status')

# Record agent handoffs
analytics.agent_analyzer.record_handoff('CustomerAgent', 'TechnicalAgent')

# Record errors
analytics.agent_analyzer.record_error('CustomerAgent', 'timeout_error')

# Record satisfaction scores
analytics.agent_analyzer.record_satisfaction('CustomerAgent', 4.5)

# Get detailed agent analytics
agent_data = analytics.agent_analyzer.get_agent_analytics('CustomerAgent')
print(f"Agent Performance: {agent_data}")
```

### System Monitoring

Monitor system-wide performance:

```python
from jaf.core.analytics import AnalyticsEngine
from jaf.core.performance import PerformanceMetrics

analytics = AnalyticsEngine()

# Start conversation tracking
analytics.system_analyzer.record_conversation_start('SupportAgent')

# Record performance metrics
metrics = PerformanceMetrics(
    execution_time_ms=1500,
    memory_usage_mb=256,
    token_count=500,
    error_count=0,
    retry_count=1
)
analytics.system_analyzer.record_performance_metrics(metrics)

# End conversation
analytics.system_analyzer.record_conversation_end()

# Get system insights
system_data = analytics.system_analyzer.get_system_analytics()
print(f"System Performance: {system_data}")
```

## Best Practices

### 1. Efficient Data Collection

Focus on meaningful metrics:

```python
# Good: Collect actionable metrics
analytics.record_agent_performance(
    agent_name='SupportAgent',
    success=True,
    response_time_ms=1200,
    tool_name='search_orders'
)

# Analyze conversations periodically
if conversation_complete:
    conversation_analytics = analytics.analyze_conversation(
        messages, start_time, end_time
    )
```

### 2. Performance Monitoring

Regular system health checks:

```python
# Monitor system performance
system_analytics = analytics.system_analyzer.get_system_analytics()

# Check for bottlenecks
if system_analytics.bottlenecks:
    print(f"Bottlenecks detected: {system_analytics.bottlenecks}")
    
# Review recommendations
for rec in system_analytics.optimization_recommendations:
    print(f"Recommendation: {rec}")
```

### 3. Data-Driven Optimization

Use analytics to improve performance:

```python
# Get comprehensive report
report = analytics.get_comprehensive_analytics()

# Identify top performing agents
top_agents = report['summary']['top_performing_agents']
for agent in top_agents:
    print(f"Top Agent: {agent['name']} - Score: {agent['combined_score']}")

# Review key insights
for insight in report['summary']['key_insights']:
    print(f"Insight: {insight}")
```

## Example: Production Analytics Setup

Here's a complete example for production use:

```python
import time
from jaf.core.analytics import AnalyticsEngine, analyze_conversation_quality
from jaf.core.types import Message, ContentRole
from jaf.core.performance import PerformanceMetrics

def setup_production_analytics():
    """Set up analytics for production environment."""
    
    # Create analytics engine
    analytics = AnalyticsEngine()
    
    # Example: Process a customer support conversation
    messages = [
        Message(role=ContentRole.USER, content='I have an issue with my order'),
        Message(role=ContentRole.ASSISTANT, content='I can help you with that. What\'s your order number?'),
        Message(role=ContentRole.USER, content='Order #12345'),
        Message(role=ContentRole.ASSISTANT, content='I found your order. It will be delivered tomorrow.'),
        Message(role=ContentRole.USER, content='Perfect, thank you!')
    ]
    
    # Record conversation timing
    start_time = time.time() - 300  # 5 minutes ago
    end_time = time.time()
    
    # Analyze conversation
    conversation_analytics = analytics.analyze_conversation(messages, start_time, end_time)
    print(f"Conversation Quality: {conversation_analytics}")
    
    # Record agent performance
    analytics.record_agent_performance(
        agent_name='SupportAgent',
        success=True,
        response_time_ms=1200,
        tool_name='order_lookup'
    )
    
    # Record system metrics
    metrics = PerformanceMetrics(
        execution_time_ms=1200,
        memory_usage_mb=128,
        token_count=350,
        error_count=0,
        retry_count=0
    )
    analytics.record_system_metrics(metrics, 'SupportAgent')
    
    # Get comprehensive report
    report = analytics.get_comprehensive_analytics()
    
    print("\n=== Analytics Report ===")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Total Conversations: {report['system'].total_conversations}")
    print(f"Active Agents: {report['system'].active_agents}")
    
    if report['agents']:
        for agent_name, agent_data in report['agents'].items():
            print(f"\nAgent: {agent_name}")
            print(f"  Success Rate: {agent_data.success_rate:.1f}%")
            print(f"  Avg Response Time: {agent_data.average_response_time_ms:.0f}ms")
            print(f"  Specializations: {agent_data.specialization_areas}")
    
    print(f"\nKey Insights:")
    for insight in report['summary']['key_insights']:
        print(f"  - {insight}")
    
    return analytics

if __name__ == "__main__":
    analytics = setup_production_analytics()
```

The analytics system provides essential insights for monitoring and optimizing your JAF deployment in production environments.

## Next Steps

- Learn about [Performance Monitoring](performance-monitoring.md) for system optimization
- Explore [Workflow Orchestration](workflow-orchestration.md) for complex automation
- Check [Streaming Responses](streaming-responses.md) for real-time interactions
- Review [Plugin System](plugin-system.md) for extensibility
