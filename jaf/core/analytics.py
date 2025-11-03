"""
Advanced analytics and insights for JAF framework.

This module provides sophisticated analytics capabilities including
conversation analysis, agent performance insights, and usage patterns.
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, Counter
from datetime import datetime, timedelta

from .types import Message, ContentRole, RunState, TraceEvent, get_text_content
from .performance import PerformanceMetrics


@dataclass(frozen=True)
class ConversationAnalytics:
    """Analytics for conversation patterns and quality."""

    total_messages: int
    user_messages: int
    assistant_messages: int
    tool_messages: int
    average_message_length: float
    conversation_duration_minutes: float
    topic_keywords: List[str]
    sentiment_score: float
    engagement_score: float
    resolution_status: str  # 'resolved', 'ongoing', 'escalated'


@dataclass(frozen=True)
class AgentAnalytics:
    """Analytics for individual agent performance."""

    agent_name: str
    total_invocations: int
    success_rate: float
    average_response_time_ms: float
    tool_usage_frequency: Dict[str, int]
    handoff_patterns: Dict[str, int]
    error_patterns: Dict[str, int]
    user_satisfaction_score: float
    specialization_areas: List[str]


@dataclass(frozen=True)
class SystemAnalytics:
    """System-wide analytics and insights."""

    total_conversations: int
    active_agents: int
    peak_concurrent_sessions: int
    resource_utilization: Dict[str, float]
    performance_trends: Dict[str, List[float]]
    bottlenecks: List[str]
    optimization_recommendations: List[str]


class ConversationAnalyzer:
    """Analyzes conversation patterns and extracts insights."""

    def __init__(self):
        self.keyword_extractors = {
            "technical": ["error", "bug", "issue", "problem", "fix", "debug"],
            "business": ["revenue", "cost", "profit", "customer", "sales", "market"],
            "support": ["help", "assistance", "question", "how to", "tutorial"],
            "creative": ["design", "create", "generate", "brainstorm", "idea"],
        }

    def analyze_conversation(
        self, messages: List[Message], start_time: float, end_time: float
    ) -> ConversationAnalytics:
        """Analyze a complete conversation."""
        user_msgs = [m for m in messages if m.role == ContentRole.USER]
        assistant_msgs = [m for m in messages if m.role == ContentRole.ASSISTANT]
        tool_msgs = [m for m in messages if m.role == ContentRole.TOOL]

        # Calculate basic metrics
        total_length = sum(len(m.content or "") for m in messages)
        avg_length = total_length / len(messages) if messages else 0
        duration = (end_time - start_time) / 60  # Convert to minutes

        # Extract topics and keywords
        all_text = " ".join(m.content or "" for m in messages)
        keywords = self._extract_keywords(all_text)

        # Calculate sentiment and engagement
        sentiment = self._calculate_sentiment(user_msgs + assistant_msgs)
        engagement = self._calculate_engagement(messages)

        # Determine resolution status
        resolution = self._determine_resolution_status(messages)

        return ConversationAnalytics(
            total_messages=len(messages),
            user_messages=len(user_msgs),
            assistant_messages=len(assistant_msgs),
            tool_messages=len(tool_msgs),
            average_message_length=avg_length,
            conversation_duration_minutes=duration,
            topic_keywords=keywords,
            sentiment_score=sentiment,
            engagement_score=engagement,
            resolution_status=resolution,
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from conversation text."""
        text_lower = text.lower()
        found_keywords = []

        for category, keywords in self.keyword_extractors.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(f"{category}:{keyword}")

        # Add simple word frequency analysis
        words = text_lower.split()
        word_freq = Counter(word for word in words if len(word) > 4)
        top_words = [word for word, count in word_freq.most_common(5) if count > 1]

        return found_keywords + top_words

    def _calculate_sentiment(self, messages: List[Message]) -> float:
        """Calculate sentiment score (simplified implementation)."""
        positive_words = ["good", "great", "excellent", "perfect", "amazing", "helpful", "thanks"]
        negative_words = ["bad", "terrible", "awful", "wrong", "error", "problem", "issue"]

        total_score = 0
        total_words = 0

        for message in messages:
            content_text = get_text_content(message.content)
            if not content_text:
                continue

            words = content_text.lower().split()
            total_words += len(words)

            for word in words:
                if word in positive_words:
                    total_score += 1
                elif word in negative_words:
                    total_score -= 1

        return total_score / max(total_words, 1) * 100  # Normalize to percentage

    def _calculate_engagement(self, messages: List[Message]) -> float:
        """Calculate engagement score based on conversation patterns."""
        if len(messages) < 2:
            return 0.0

        # Factors that indicate engagement
        user_messages = [m for m in messages if m.role == ContentRole.USER]
        assistant_messages = [m for m in messages if m.role == ContentRole.ASSISTANT]

        # Message frequency
        message_score = min(len(messages) / 10, 1.0) * 30

        # Response balance
        if user_messages and assistant_messages:
            balance = min(len(user_messages), len(assistant_messages)) / max(
                len(user_messages), len(assistant_messages)
            )
            balance_score = balance * 40
        else:
            balance_score = 0

        # Message length variety (indicates thoughtful responses)
        lengths = [len(m.content or "") for m in messages]
        if lengths:
            length_variety = (max(lengths) - min(lengths)) / max(max(lengths), 1)
            variety_score = min(length_variety, 1.0) * 30
        else:
            variety_score = 0

        return message_score + balance_score + variety_score

    def _determine_resolution_status(self, messages: List[Message]) -> str:
        """Determine if the conversation was resolved."""
        if not messages:
            return "ongoing"

        last_messages = messages[-3:] if len(messages) >= 3 else messages
        last_text = " ".join(m.content or "" for m in last_messages).lower()

        resolution_indicators = ["thank you", "thanks", "solved", "resolved", "perfect", "exactly"]
        escalation_indicators = ["escalate", "manager", "supervisor", "complaint"]

        if any(indicator in last_text for indicator in resolution_indicators):
            return "resolved"
        elif any(indicator in last_text for indicator in escalation_indicators):
            return "escalated"
        else:
            return "ongoing"


class AgentPerformanceAnalyzer:
    """Analyzes individual agent performance and behavior."""

    def __init__(self):
        self.agent_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "invocations": 0,
                "successes": 0,
                "response_times": [],
                "tool_usage": defaultdict(int),
                "handoffs": defaultdict(int),
                "errors": defaultdict(int),
                "satisfaction_scores": [],
            }
        )

    def record_agent_invocation(self, agent_name: str, success: bool, response_time_ms: float):
        """Record an agent invocation."""
        metrics = self.agent_metrics[agent_name]
        metrics["invocations"] += 1
        if success:
            metrics["successes"] += 1
        metrics["response_times"].append(response_time_ms)

    def record_tool_usage(self, agent_name: str, tool_name: str):
        """Record tool usage by an agent."""
        self.agent_metrics[agent_name]["tool_usage"][tool_name] += 1

    def record_handoff(self, from_agent: str, to_agent: str):
        """Record an agent handoff."""
        self.agent_metrics[from_agent]["handoffs"][to_agent] += 1

    def record_error(self, agent_name: str, error_type: str):
        """Record an error for an agent."""
        self.agent_metrics[agent_name]["errors"][error_type] += 1

    def record_satisfaction(self, agent_name: str, score: float):
        """Record user satisfaction score for an agent."""
        self.agent_metrics[agent_name]["satisfaction_scores"].append(score)

    def get_agent_analytics(self, agent_name: str) -> Optional[AgentAnalytics]:
        """Get comprehensive analytics for a specific agent."""
        if agent_name not in self.agent_metrics:
            return None

        metrics = self.agent_metrics[agent_name]

        # Calculate success rate
        success_rate = (metrics["successes"] / max(metrics["invocations"], 1)) * 100

        # Calculate average response time
        avg_response_time = sum(metrics["response_times"]) / max(len(metrics["response_times"]), 1)

        # Calculate satisfaction score
        satisfaction_scores = metrics["satisfaction_scores"]
        avg_satisfaction = (
            sum(satisfaction_scores) / max(len(satisfaction_scores), 1)
            if satisfaction_scores
            else 0
        )

        # Determine specialization areas
        specializations = self._determine_specializations(metrics["tool_usage"])

        return AgentAnalytics(
            agent_name=agent_name,
            total_invocations=metrics["invocations"],
            success_rate=success_rate,
            average_response_time_ms=avg_response_time,
            tool_usage_frequency=dict(metrics["tool_usage"]),
            handoff_patterns=dict(metrics["handoffs"]),
            error_patterns=dict(metrics["errors"]),
            user_satisfaction_score=avg_satisfaction,
            specialization_areas=specializations,
        )

    def _determine_specializations(self, tool_usage: Dict[str, int]) -> List[str]:
        """Determine agent specialization areas based on tool usage."""
        if not tool_usage:
            return []

        # Define tool categories
        tool_categories = {
            "math": ["calculate", "compute", "math", "formula"],
            "search": ["search", "find", "lookup", "query"],
            "file": ["read", "write", "file", "document"],
            "communication": ["email", "message", "notify", "send"],
            "data": ["analyze", "process", "transform", "export"],
        }

        category_scores = defaultdict(int)

        for tool, count in tool_usage.items():
            tool_lower = tool.lower()
            for category, keywords in tool_categories.items():
                if any(keyword in tool_lower for keyword in keywords):
                    category_scores[category] += count

        # Return top specializations
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        return [category for category, score in sorted_categories[:3] if score > 0]


class SystemAnalyzer:
    """Analyzes system-wide performance and provides optimization insights."""

    def __init__(self):
        self.conversation_count = 0
        self.active_agents: Set[str] = set()
        self.concurrent_sessions = 0
        self.peak_concurrent = 0
        self.performance_history: List[PerformanceMetrics] = []
        self.resource_usage: Dict[str, List[float]] = defaultdict(list)

    def record_conversation_start(self, agent_name: str):
        """Record the start of a new conversation."""
        self.conversation_count += 1
        self.active_agents.add(agent_name)
        self.concurrent_sessions += 1
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent_sessions)

    def record_conversation_end(self):
        """Record the end of a conversation."""
        self.concurrent_sessions = max(0, self.concurrent_sessions - 1)

    def record_performance_metrics(self, metrics: PerformanceMetrics):
        """Record system performance metrics."""
        self.performance_history.append(metrics)

        # Track resource usage trends
        self.resource_usage["memory"].append(metrics.memory_usage_mb)
        self.resource_usage["execution_time"].append(metrics.execution_time_ms)
        self.resource_usage["token_usage"].append(metrics.token_count)

    def get_system_analytics(self) -> SystemAnalytics:
        """Get comprehensive system analytics."""
        # Calculate resource utilization
        resource_util = {}
        for resource, values in self.resource_usage.items():
            if values:
                recent_values = values[-10:]  # Last 10 measurements
                resource_util[resource] = {
                    "current": recent_values[-1] if recent_values else 0,
                    "average": sum(recent_values) / len(recent_values),
                    "peak": max(values),
                    "trend": self._calculate_trend(recent_values),
                }

        # Calculate performance trends
        trends = {}
        if len(self.performance_history) >= 5:
            recent_metrics = self.performance_history[-5:]
            trends = {
                "response_time": [m.execution_time_ms for m in recent_metrics],
                "memory_usage": [m.memory_usage_mb for m in recent_metrics],
                "error_rate": [m.error_count for m in recent_metrics],
            }

        # Identify bottlenecks and recommendations
        bottlenecks = self._identify_bottlenecks()
        recommendations = self._generate_recommendations(bottlenecks, resource_util)

        return SystemAnalytics(
            total_conversations=self.conversation_count,
            active_agents=len(self.active_agents),
            peak_concurrent_sessions=self.peak_concurrent,
            resource_utilization=resource_util,
            performance_trends=trends,
            bottlenecks=bottlenecks,
            optimization_recommendations=recommendations,
        )

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"

        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def _identify_bottlenecks(self) -> List[str]:
        """Identify system bottlenecks."""
        bottlenecks = []

        if not self.performance_history:
            return bottlenecks

        recent_metrics = (
            self.performance_history[-5:]
            if len(self.performance_history) >= 5
            else self.performance_history
        )

        # Check for high response times
        avg_response_time = sum(m.execution_time_ms for m in recent_metrics) / len(recent_metrics)
        if avg_response_time > 5000:  # 5 seconds
            bottlenecks.append("high_response_time")

        # Check for high memory usage
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        if avg_memory > 500:  # 500 MB
            bottlenecks.append("high_memory_usage")

        # Check for high error rates
        avg_errors = sum(m.error_count for m in recent_metrics) / len(recent_metrics)
        if avg_errors > 0.1:  # More than 10% error rate
            bottlenecks.append("high_error_rate")

        # Check for excessive retries
        avg_retries = sum(m.retry_count for m in recent_metrics) / len(recent_metrics)
        if avg_retries > 1:
            bottlenecks.append("excessive_retries")

        return bottlenecks

    def _generate_recommendations(
        self, bottlenecks: List[str], resource_util: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if "high_response_time" in bottlenecks:
            recommendations.append("Consider implementing response caching or optimizing LLM calls")

        if "high_memory_usage" in bottlenecks:
            recommendations.append("Implement conversation compression or increase memory limits")

        if "high_error_rate" in bottlenecks:
            recommendations.append(
                "Review error handling strategies and implement circuit breakers"
            )

        if "excessive_retries" in bottlenecks:
            recommendations.append("Optimize retry policies and implement exponential backoff")

        # Check resource trends
        for resource, data in resource_util.items():
            if isinstance(data, dict) and data.get("trend") == "increasing":
                recommendations.append(f"Monitor {resource} usage - showing increasing trend")

        if not recommendations:
            recommendations.append(
                "System performance is optimal - no immediate optimizations needed"
            )

        return recommendations


class AnalyticsEngine:
    """Main analytics engine that coordinates all analysis components."""

    def __init__(self):
        self.conversation_analyzer = ConversationAnalyzer()
        self.agent_analyzer = AgentPerformanceAnalyzer()
        self.system_analyzer = SystemAnalyzer()
        self.analytics_history: List[Dict[str, Any]] = []

    def analyze_conversation(
        self, messages: List[Message], start_time: float, end_time: float
    ) -> ConversationAnalytics:
        """Analyze a conversation and return insights."""
        return self.conversation_analyzer.analyze_conversation(messages, start_time, end_time)

    def record_agent_performance(
        self,
        agent_name: str,
        success: bool,
        response_time_ms: float,
        tool_name: Optional[str] = None,
        error_type: Optional[str] = None,
    ):
        """Record agent performance data."""
        self.agent_analyzer.record_agent_invocation(agent_name, success, response_time_ms)

        if tool_name:
            self.agent_analyzer.record_tool_usage(agent_name, tool_name)

        if error_type:
            self.agent_analyzer.record_error(agent_name, error_type)

    def record_system_metrics(self, metrics: PerformanceMetrics, agent_name: str):
        """Record system-wide metrics."""
        self.system_analyzer.record_performance_metrics(metrics)
        self.system_analyzer.record_conversation_start(agent_name)

    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics across all dimensions."""
        system_analytics = self.system_analyzer.get_system_analytics()

        # Get analytics for all agents
        agent_analytics = {}
        for agent_name in self.system_analyzer.active_agents:
            agent_data = self.agent_analyzer.get_agent_analytics(agent_name)
            if agent_data:
                agent_analytics[agent_name] = agent_data

        analytics_report = {
            "timestamp": datetime.now().isoformat(),
            "system": system_analytics,
            "agents": agent_analytics,
            "summary": {
                "total_conversations": system_analytics.total_conversations,
                "active_agents": system_analytics.active_agents,
                "top_performing_agents": self._get_top_performing_agents(agent_analytics),
                "key_insights": self._generate_key_insights(system_analytics, agent_analytics),
            },
        }

        self.analytics_history.append(analytics_report)
        return analytics_report

    def _get_top_performing_agents(
        self, agent_analytics: Dict[str, AgentAnalytics]
    ) -> List[Dict[str, Any]]:
        """Get top performing agents by success rate and satisfaction."""
        if not agent_analytics:
            return []

        agents_with_scores = []
        for name, analytics in agent_analytics.items():
            # Combined score: success rate + satisfaction
            combined_score = (analytics.success_rate + analytics.user_satisfaction_score) / 2
            agents_with_scores.append(
                {
                    "name": name,
                    "success_rate": analytics.success_rate,
                    "satisfaction_score": analytics.user_satisfaction_score,
                    "combined_score": combined_score,
                }
            )

        # Sort by combined score and return top 3
        agents_with_scores.sort(key=lambda x: x["combined_score"], reverse=True)
        return agents_with_scores[:3]

    def _generate_key_insights(
        self, system_analytics: SystemAnalytics, agent_analytics: Dict[str, AgentAnalytics]
    ) -> List[str]:
        """Generate key insights from the analytics data."""
        insights = []

        # System insights
        if system_analytics.peak_concurrent_sessions > 10:
            insights.append(
                f"High concurrent usage detected: {system_analytics.peak_concurrent_sessions} peak sessions"
            )

        if system_analytics.bottlenecks:
            insights.append(
                f"Performance bottlenecks identified: {', '.join(system_analytics.bottlenecks)}"
            )

        # Agent insights
        if agent_analytics:
            avg_success_rate = sum(a.success_rate for a in agent_analytics.values()) / len(
                agent_analytics
            )
            if avg_success_rate > 90:
                insights.append(
                    f"Excellent agent performance: {avg_success_rate:.1f}% average success rate"
                )
            elif avg_success_rate < 70:
                insights.append(
                    f"Agent performance needs attention: {avg_success_rate:.1f}% average success rate"
                )

        # Usage patterns
        total_conversations = system_analytics.total_conversations
        if total_conversations > 100:
            insights.append(
                f"High system usage: {total_conversations} total conversations processed"
            )

        if not insights:
            insights.append("System operating normally with good performance metrics")

        return insights


# Global analytics engine instance
global_analytics_engine = AnalyticsEngine()


def get_analytics_report() -> Dict[str, Any]:
    """Get comprehensive analytics report."""
    return global_analytics_engine.get_comprehensive_analytics()


def analyze_conversation_quality(
    messages: List[Message], start_time: float, end_time: float
) -> ConversationAnalytics:
    """Analyze conversation quality and patterns."""
    return global_analytics_engine.analyze_conversation(messages, start_time, end_time)
