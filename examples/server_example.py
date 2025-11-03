#!/usr/bin/env python3
"""
JAF RAG Example - Using LiteLLM with Gemini.

This example demonstrates how to create a RAG (Retrieval-Augmented Generation)
agent using LiteLLM proxy with Gemini models.
"""

import asyncio
import os
from typing import Any, Dict, List

from pydantic import BaseModel

from jaf import (
    Agent,
    RunConfig,
    RunState,
    ToolSchema,
    generate_run_id,
    generate_trace_id,
    run,
)
from jaf.core.tool_results import ToolErrorCodes, ToolResponse, ToolResult
from jaf.core.types import Message
from jaf.providers.model import make_litellm_provider


class RAGQueryArgs(BaseModel):
    """Arguments for RAG query tool."""

    query: str
    max_results: int = 5


class LiteLLMRAGTool:
    """
    A tool that performs Retrieval-Augmented Generation using LiteLLM.
    """

    def __init__(self):
        self.knowledge_base = self._create_mock_knowledge_base()

    def _create_mock_knowledge_base(self) -> List[Dict[str, Any]]:
        """Create a mock knowledge base for demonstration."""
        return [
            {
                "id": "doc1",
                "title": "Python Programming Basics",
                "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python's syntax is clean and expressive, making it an excellent choice for beginners and experienced developers alike.",
                "metadata": {"category": "programming", "level": "beginner"},
            },
            {
                "id": "doc2",
                "title": "Machine Learning with Python",
                "content": "Python is the leading language for machine learning due to its rich ecosystem of libraries. Scikit-learn provides simple and efficient tools for data mining and analysis. TensorFlow and PyTorch enable deep learning and neural network development. NumPy and Pandas handle numerical computations and data manipulation.",
                "metadata": {"category": "ml", "level": "intermediate"},
            },
            {
                "id": "doc3",
                "title": "Web Development with FastAPI",
                "content": "FastAPI is a modern, fast web framework for building APIs with Python. It provides automatic API documentation with Swagger UI, data validation using Pydantic, async support for high performance, and type hints for better developer experience. FastAPI is built on Starlette and uses standard Python type declarations.",
                "metadata": {"category": "web", "level": "intermediate"},
            },
            {
                "id": "doc4",
                "title": "Data Science Workflow",
                "content": "A typical data science workflow includes data collection from various sources, data cleaning and preprocessing, exploratory data analysis, feature engineering, model selection and training, model evaluation and validation, and finally model deployment. Python provides excellent tools for each step including Pandas, Matplotlib, Seaborn, and Jupyter notebooks.",
                "metadata": {"category": "data-science", "level": "advanced"},
            },
            {
                "id": "doc5",
                "title": "Agent Frameworks and AI",
                "content": "Agent frameworks like JAF (Juspay Agent Framework) enable building sophisticated AI applications with tool integration, conversation management, state handling, and model orchestration capabilities. These frameworks provide abstractions for creating autonomous agents that can interact with external systems and maintain context across conversations.",
                "metadata": {"category": "ai", "level": "advanced"},
            },
            {
                "id": "doc6",
                "title": "Google AI and Gemini",
                "content": "Google AI's Gemini models are multimodal large language models that can understand and generate text, code, images, and more. Gemini Pro is optimized for a wide range of reasoning tasks, while Gemini Ultra provides the highest capability for complex tasks. The models support long context windows and advanced reasoning capabilities.",
                "metadata": {"category": "ai", "level": "advanced"},
            },
            {
                "id": "doc7",
                "title": "LiteLLM Proxy Server",
                "content": "LiteLLM is a proxy server that provides a unified interface for 100+ language models including OpenAI, Anthropic, Google AI, AWS Bedrock, and more. It handles API key management, rate limiting, caching, and provides OpenAI-compatible endpoints. This makes it easy to switch between different model providers without changing your application code.",
                "metadata": {"category": "ai", "level": "intermediate"},
            },
        ]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="litellm_rag",
            description="Search knowledge base and retrieve relevant information",
            parameters=RAGQueryArgs,
        )

    async def execute(self, args: RAGQueryArgs, context: Any) -> ToolResult:
        """Execute RAG query."""
        try:
            # Step 1: Retrieve relevant documents
            relevant_docs = self._semantic_search(args.query, args.max_results)

            if not relevant_docs:
                return ToolResponse.success(
                    "I couldn't find any relevant information in the knowledge base for your query.",
                    {"query": args.query, "results_count": 0},
                )

            # Step 2: Format the retrieved information
            formatted_response = self._format_retrieved_docs(relevant_docs, args.query)

            return ToolResponse.success(
                formatted_response,
                {
                    "query": args.query,
                    "results_count": len(relevant_docs),
                    "sources": [
                        {"title": doc["title"], "category": doc["metadata"]["category"]}
                        for doc in relevant_docs
                    ],
                },
            )

        except Exception as e:
            return ToolResponse.error(
                ToolErrorCodes.EXECUTION_FAILED,
                f"RAG query failed: {e!s}",
                {"error": str(e), "query": args.query},
            )

    def _semantic_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform semantic search on the knowledge base."""
        query_lower = query.lower()
        scored_docs = []

        for doc in self.knowledge_base:
            score = 0
            # Enhanced scoring based on keyword matches
            title_matches = sum(1 for word in query_lower.split() if word in doc["title"].lower())
            content_matches = sum(
                1 for word in query_lower.split() if word in doc["content"].lower()
            )

            # Category relevance
            category_keywords = {
                "programming": ["python", "code", "programming", "language", "syntax"],
                "ml": ["machine learning", "ai", "model", "training", "neural"],
                "web": ["web", "api", "fastapi", "server", "http"],
                "data-science": ["data", "analysis", "visualization", "pandas"],
                "ai": [
                    "ai",
                    "agent",
                    "framework",
                    "intelligent",
                    "autonomous",
                    "litellm",
                    "gemini",
                ],
            }

            category = doc["metadata"]["category"]
            if category in category_keywords:
                category_matches = sum(
                    1 for keyword in category_keywords[category] if keyword in query_lower
                )
                score += category_matches * 1.5

            score += title_matches * 3 + content_matches

            if score > 0:
                scored_docs.append((score, doc))

        # Sort by score and return top results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:max_results]]

    def _format_retrieved_docs(self, docs: List[Dict[str, Any]], query: str) -> str:
        """Format retrieved documents for the model."""
        if not docs:
            return "No relevant information found."

        formatted_parts = [f"Retrieved information for query: '{query}'\n"]

        for i, doc in enumerate(docs, 1):
            formatted_parts.append(f"[Source {i}] {doc['title']}")
            formatted_parts.append(
                f"Category: {doc['metadata']['category']} | Level: {doc['metadata']['level']}"
            )
            formatted_parts.append(f"Content: {doc['content']}")
            formatted_parts.append("")  # Add spacing

        return "\n".join(formatted_parts)


def create_litellm_rag_agent() -> Agent:
    """Create a RAG-enabled agent using LiteLLM."""

    def rag_instructions(state: RunState) -> str:
        return """You are a knowledgeable AI assistant with access to a specialized knowledge base through the LiteLLM proxy.

When users ask questions, you should:
1. Use the litellm_rag tool to search for relevant information in the knowledge base
2. Provide comprehensive answers based on the retrieved information
3. Always cite your sources when providing information from the knowledge base
4. Be specific and detailed in your responses
5. If the knowledge base doesn't contain relevant information, be honest about the limitations

You have access to information about programming, machine learning, web development, data science, AI frameworks, and LiteLLM proxy configuration.

Always use the search tool when users ask questions that could be answered with information from the knowledge base."""

    # Create the RAG tool
    rag_tool = LiteLLMRAGTool()

    return Agent(
        name="litellm_rag_assistant",
        instructions=rag_instructions,
        tools=[rag_tool],
        output_codec=None,
        handoffs=None,
        model_config=None,
    )


async def demo_litellm_rag():
    """Demonstrate a conversation with the LiteLLM RAG agent."""
    print("🤖 JAF LiteLLM RAG Demo Starting...")

    # Get configuration from environment
    litellm_url = os.getenv("LITELLM_URL", "http://localhost:4000")
    litellm_api_key = os.getenv("LITELLM_API_KEY", "your_api_key_here")
    litellm_model = os.getenv("LITELLM_MODEL", "gemini-2.5-pro")

    print(f"📡 LiteLLM URL: {litellm_url}")
    print(f"🔑 API Key: {'Set' if litellm_api_key != 'your_api_key_here' else 'Using default'}")
    print(f"🧠 Model: {litellm_model}")

    # Create the RAG agent
    rag_agent = create_litellm_rag_agent()

    # Create LiteLLM model provider
    model_provider = make_litellm_provider(litellm_url, litellm_api_key)

    # Create run configuration
    run_config = RunConfig(
        agent_registry={"litellm_rag_assistant": rag_agent},
        model_provider=model_provider,
        model_override=litellm_model,
        max_turns=10,
    )

    # Demo questions
    demo_questions = [
        "What is Python and why is it popular for programming?",
        "How can I use Python for machine learning?",
        "Tell me about FastAPI for web development",
        "What is JAF and how does it help with AI applications?",
        "Explain the data science workflow using Python",
        "What is LiteLLM and how does it work?",
        "How do Gemini models compare to other AI models?",
    ]

    print(f"📚 Knowledge base contains {len(rag_agent.tools[0].knowledge_base)} documents")
    print("💬 Running demo conversations...\n")

    for i, question in enumerate(demo_questions, 1):
        print(f"🔍 Demo Question {i}: {question}")
        print("-" * 60)

        # Create initial state
        initial_state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[Message(role="user", content=question)],
            current_agent_name="litellm_rag_assistant",
            context={},
            turn_count=0,
        )

        # Run the conversation
        try:
            result = await run(initial_state, run_config)

            # Display result
            if hasattr(result.outcome, "output") and result.outcome.output:
                print(f"🤖 Assistant: {result.outcome.output}")
            else:
                print(f"❌ Error: {result.outcome}")

        except Exception as e:
            print(f"❌ Error running conversation: {e}")

        print("\n" + "=" * 80 + "\n")


async def interactive_litellm_rag():
    """Run an interactive LiteLLM RAG demo."""
    print("🤖 Interactive JAF LiteLLM RAG Demo")
    print("Type your questions and get answers from the knowledge base!")
    print("Type 'quit' or 'exit' to stop.\n")

    # Get configuration from environment
    litellm_url = os.getenv("LITELLM_URL", "http://localhost:4000")
    litellm_api_key = os.getenv("LITELLM_API_KEY", "your_api_key_here")
    litellm_model = os.getenv("LITELLM_MODEL", "gemini-2.5-pro")

    print(f"📡 LiteLLM URL: {litellm_url}")
    print(f"🔑 API Key: {'Set' if litellm_api_key != 'your_api_key_here' else 'Using default'}")
    print(f"🧠 Model: {litellm_model}")

    # Create the RAG agent
    rag_agent = create_litellm_rag_agent()

    # Create LiteLLM model provider
    model_provider = make_litellm_provider(litellm_url, litellm_api_key)

    # Create run configuration
    run_config = RunConfig(
        agent_registry={"litellm_rag_assistant": rag_agent},
        model_provider=model_provider,
        model_override=litellm_model,
        max_turns=10,
    )

    print(f"📚 Knowledge base loaded with {len(rag_agent.tools[0].knowledge_base)} documents")
    print("🔧 Configuration ready\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            print("🔍 Searching knowledge base and generating response...")

            # Create initial state
            initial_state = RunState(
                run_id=generate_run_id(),
                trace_id=generate_trace_id(),
                messages=[Message(role="user", content=user_input)],
                current_agent_name="litellm_rag_assistant",
                context={},
                turn_count=0,
            )

            # Run the conversation
            result = await run(initial_state, run_config)

            # Display result
            if hasattr(result.outcome, "output") and result.outcome.output:
                print(f"🤖 Assistant: {result.outcome.output}\n")
            else:
                print(f"❌ Error: {result.outcome}\n")

        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


async def main():
    """Main entry point."""
    print("🚀 JAF LiteLLM RAG Example")
    print("=========================\n")

    # Load environment variables
    try:
        from dotenv import load_dotenv

        load_dotenv()
        print("📄 Loaded environment variables from .env file")
    except ImportError:
        print("💡 Install python-dotenv to use .env files: pip install python-dotenv")

    # Check configuration
    litellm_url = os.getenv("LITELLM_URL", "http://localhost:4000")
    litellm_api_key = os.getenv("LITELLM_API_KEY", "your_api_key_here")
    litellm_model = os.getenv("LITELLM_MODEL", "gemini-2.5-pro")
    port = os.getenv("PORT", "3003")
    host = os.getenv("HOST", "127.0.0.1")

    print("⚙️  Configuration:")
    print(f"   - LiteLLM URL: {litellm_url}")
    print(
        f"   - LiteLLM API Key: {'✅ Set' if litellm_api_key != 'your_api_key_here' else '❌ Default (update needed)'}"
    )
    print(f"   - LiteLLM Model: {litellm_model}")
    print(f"   - Server Port: {port}")
    print(f"   - Server Host: {host}")
    print()

    if litellm_api_key == "your_api_key_here":
        print("⚠️  Warning: Using default API key. Update LITELLM_API_KEY in your .env file")
        print("   This may work for local testing but won't work with actual model providers")
        print()

    # Choose demo mode
    try:
        mode = input(
            "Choose demo mode:\n1. Automated demo with sample questions\n2. Interactive chat\nEnter 1 or 2: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        mode = "1"

    if mode == "1":
        await demo_litellm_rag()
    elif mode == "2":
        await interactive_litellm_rag()
    else:
        print("Invalid choice. Running automated demo...")
        await demo_litellm_rag()


if __name__ == "__main__":
    asyncio.run(main())
