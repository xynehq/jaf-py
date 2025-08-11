#!/usr/bin/env python3
"""
JAF RAG Example - Python equivalent of the TypeScript rag-demo.

This example demonstrates how to create a RAG (Retrieval-Augmented Generation) 
agent using Google's Generative AI with Vertex AI.
"""

import asyncio
import os
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from jaf import Agent, Tool, ToolSchema, RunState, RunConfig, run, generate_trace_id, generate_run_id
from jaf.core.types import Message, create_trace_id, create_run_id
from jaf.providers.model import make_litellm_provider
from jaf.core.tool_results import ToolResult, ToolResultStatus


# Google Vertex AI imports (optional - install with: pip install google-generativeai google-auth)
try:
    import google.generativeai as genai
    from google.auth import default
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    print("⚠️  Google AI libraries not installed. Install with: pip install google-generativeai google-auth")


class RAGQueryArgs(BaseModel):
    """Arguments for RAG query tool."""
    query: str
    max_results: int = 5


class VertexAIRAGTool:
    """
    A tool that performs Retrieval-Augmented Generation using Google Vertex AI.
    
    This is a simplified example. In production, you would:
    1. Connect to a real vector database (e.g., Pinecone, Weaviate, etc.)
    2. Use proper embeddings for semantic search
    3. Implement chunking and indexing strategies
    4. Add relevance scoring and filtering
    """
    
    def __init__(self, project_id: Optional[str] = None, location: str = "us-central1"):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.knowledge_base = self._create_mock_knowledge_base()
    
    def _create_mock_knowledge_base(self) -> List[Dict[str, Any]]:
        """Create a mock knowledge base for demonstration."""
        return [
            {
                "id": "doc1",
                "title": "Python Programming Basics",
                "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                "metadata": {"category": "programming", "level": "beginner"}
            },
            {
                "id": "doc2", 
                "title": "Machine Learning with Python",
                "content": "Python is widely used in machine learning due to libraries like scikit-learn, TensorFlow, and PyTorch. These libraries provide tools for data preprocessing, model training, and evaluation.",
                "metadata": {"category": "ml", "level": "intermediate"}
            },
            {
                "id": "doc3",
                "title": "Web Development with FastAPI",
                "content": "FastAPI is a modern, fast web framework for building APIs with Python. It provides automatic API documentation, data validation, and async support out of the box.",
                "metadata": {"category": "web", "level": "intermediate"}
            },
            {
                "id": "doc4",
                "title": "Data Science Workflow",
                "content": "A typical data science workflow includes data collection, cleaning, exploration, modeling, and deployment. Python provides excellent tools for each step of this process.",
                "metadata": {"category": "data-science", "level": "advanced"}
            },
            {
                "id": "doc5",
                "title": "Agent Frameworks and AI",
                "content": "Agent frameworks like JAF (Juspay Agent Framework) enable building AI applications with tool integration, conversation management, and model orchestration capabilities.",
                "metadata": {"category": "ai", "level": "advanced"}
            }
        ]
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="vertex_ai_rag",
            description="Search knowledge base and generate answers using Vertex AI RAG",
            parameters=RAGQueryArgs
        )
    
    async def execute(self, args: RAGQueryArgs, context: Any) -> ToolResult:
        """Execute RAG query with mock retrieval and generation."""
        try:
            # Step 1: Retrieve relevant documents (mock semantic search)
            relevant_docs = self._semantic_search(args.query, args.max_results)
            
            if not relevant_docs:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    data="I couldn't find any relevant information in the knowledge base for your query.",
                    metadata={"query": args.query, "results_count": 0}
                )
            
            # Step 2: Prepare context for generation
            context_text = self._prepare_context(relevant_docs)
            
            # Step 3: Generate response (using mock generation)
            # In a real implementation, this would use Vertex AI
            generated_response = await self._generate_response(args.query, context_text)
            
            # Step 4: Format the response
            response_data = {
                "answer": generated_response,
                "sources": [
                    {
                        "title": doc["title"],
                        "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                        "metadata": doc["metadata"]
                    }
                    for doc in relevant_docs
                ]
            }
            
            formatted_response = self._format_response(generated_response, relevant_docs)
            
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                data=formatted_response,
                metadata={
                    "query": args.query,
                    "results_count": len(relevant_docs),
                    "sources": response_data["sources"]
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error_message=f"RAG query failed: {str(e)}",
                data={"error": str(e), "query": args.query}
            )
    
    def _semantic_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform semantic search on the knowledge base (mock implementation)."""
        # In a real implementation, this would:
        # 1. Convert query to embeddings
        # 2. Search vector database
        # 3. Return top-k results based on similarity
        
        # For demo, use simple keyword matching
        query_lower = query.lower()
        scored_docs = []
        
        for doc in self.knowledge_base:
            score = 0
            # Simple scoring based on keyword matches
            title_matches = sum(1 for word in query_lower.split() if word in doc["title"].lower())
            content_matches = sum(1 for word in query_lower.split() if word in doc["content"].lower())
            
            score = title_matches * 2 + content_matches  # Title matches weighted higher
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:max_results]]
    
    def _prepare_context(self, docs: List[Dict[str, Any]]) -> str:
        """Prepare context text from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Source {i}] {doc['title']}: {doc['content']}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_response(self, query: str, context: str) -> str:
        """Generate response using the retrieved context (mock implementation)."""
        # In a real implementation, this would call Vertex AI
        # For demo, create a simple template-based response
        
        if GOOGLE_AI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            try:
                return await self._generate_with_google_ai(query, context)
            except Exception as e:
                print(f"Google AI generation failed, using fallback: {e}")
        
        # Fallback: Template-based response
        return (
            f"Based on the available information, here's what I found about '{query}':\n\n"
            f"The knowledge base contains relevant information from multiple sources. "
            f"The key points include information about the topic from various perspectives. "
            f"For more detailed information, please refer to the sources provided below."
        )
    
    async def _generate_with_google_ai(self, query: str, context: str) -> str:
        """Generate response using Google Generative AI."""
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-pro')
            
            prompt = f"""
Based on the following context, please answer the user's question accurately and comprehensively.

Context:
{context}

Question: {query}

Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please indicate what information is available and what might be missing.
"""
            
            response = await model.generate_content_async(prompt)
            return response.text
            
        except Exception as e:
            print(f"Google AI API error: {e}")
            raise
    
    def _format_response(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """Format the final response with sources."""
        response_parts = [answer]
        
        if sources:
            response_parts.append("\n\n**Sources:**")
            for i, doc in enumerate(sources, 1):
                source_line = f"{i}. **{doc['title']}** - {doc['metadata']['category']} ({doc['metadata']['level']})"
                response_parts.append(source_line)
        
        return "\n".join(response_parts)


def create_rag_agent() -> Agent:
    """Create a RAG-enabled agent."""
    
    def rag_instructions(state: RunState) -> str:
        return """You are a knowledgeable assistant with access to a specialized knowledge base through RAG (Retrieval-Augmented Generation).

When users ask questions, you should:
1. Use the vertex_ai_rag tool to search for relevant information
2. Provide comprehensive answers based on the retrieved information
3. Always cite your sources when providing information
4. If the knowledge base doesn't contain relevant information, be honest about the limitations

You have access to information about programming, machine learning, web development, and AI frameworks.
Feel free to search the knowledge base for any technical questions."""
    
    # Create the RAG tool
    rag_tool = VertexAIRAGTool(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    )
    
    return Agent(
        name="rag_assistant",
        instructions=rag_instructions,
        tools=[rag_tool],
        output_codec=None,
        handoffs=None,
        model_config=None
    )


async def demo_rag_conversation():
    """Demonstrate a conversation with the RAG agent."""
    print("🤖 JAF RAG Demo Starting...")
    
    # Create the RAG agent
    rag_agent = create_rag_agent()
    
    # Create model provider
    model_provider = make_litellm_provider(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here")
    )
    
    # Create run configuration
    run_config = RunConfig(
        agent_registry={"rag_assistant": rag_agent},
        model_provider=model_provider,
        max_turns=10,
        model_override=os.getenv("MODEL_NAME", "gpt-4o")
    )
    
    # Demo questions
    demo_questions = [
        "What is Python and why is it popular for programming?",
        "How can I use Python for machine learning?",
        "Tell me about FastAPI for web development",
        "What is JAF and how does it help with AI applications?"
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
            messages=[Message(role='user', content=question)],
            current_agent_name="rag_assistant",
            context={},
            turn_count=0
        )
        
        # Run the conversation
        result = await run(initial_state, run_config)
        
        # Display result
        if hasattr(result.outcome, 'output') and result.outcome.output:
            print(f"🤖 Assistant: {result.outcome.output}")
        else:
            print(f"❌ Error: {result.outcome}")
        
        print("\n" + "=" * 80 + "\n")


async def interactive_rag_demo():
    """Run an interactive RAG demo."""
    print("🤖 Interactive JAF RAG Demo")
    print("Type your questions and get answers from the knowledge base!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Create the RAG agent
    rag_agent = create_rag_agent()
    
    # Create model provider
    model_provider = make_litellm_provider(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here")
    )
    
    # Create run configuration
    run_config = RunConfig(
        agent_registry={"rag_assistant": rag_agent},
        model_provider=model_provider,
        max_turns=10,
        model_override=os.getenv("MODEL_NAME", "gpt-4o")
    )
    
    print(f"📚 Knowledge base loaded with {len(rag_agent.tools[0].knowledge_base)} documents")
    print("🔧 Configuration ready\n")
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("🔍 Searching knowledge base and generating response...")
            
            # Create initial state
            initial_state = RunState(
                run_id=generate_run_id(),
                trace_id=generate_trace_id(),
                messages=[Message(role='user', content=user_input)],
                current_agent_name="rag_assistant",
                context={},
                turn_count=0
            )
            
            # Run the conversation
            result = await run(initial_state, run_config)
            
            # Display result
            if hasattr(result.outcome, 'output') and result.outcome.output:
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
    print("🚀 JAF RAG Example")
    print("==================\n")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("📄 Loaded environment variables from .env file")
    except ImportError:
        print("💡 Install python-dotenv to use .env files: pip install python-dotenv")
    
    # Check configuration
    print("⚙️  Configuration:")
    print(f"   - OpenAI API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")
    print(f"   - Google API Key: {'✅ Set' if os.getenv('GOOGLE_API_KEY') else '❌ Not set (will use fallback)'}")
    print(f"   - Google AI Available: {'✅ Yes' if GOOGLE_AI_AVAILABLE else '❌ No (install google-generativeai)'}")
    print()
    
    # Choose demo mode
    try:
        mode = input("Choose demo mode:\n1. Automated demo with sample questions\n2. Interactive chat\nEnter 1 or 2: ").strip()
    except (EOFError, KeyboardInterrupt):
        # Default to automated demo when running in non-interactive environments
        mode = "1"
    
    if mode == "1":
        await demo_rag_conversation()
    elif mode == "2":
        await interactive_rag_demo()
    else:
        print("Invalid choice. Running automated demo...")
        await demo_rag_conversation()


if __name__ == "__main__":
    asyncio.run(main())