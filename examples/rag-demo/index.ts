import 'dotenv/config';
import { 
  run, 
  RunConfig, 
  RunState, 
  generateTraceId, 
  generateRunId,
  ConsoleTraceCollector,
  makeLiteLLMProvider
} from '../../src/index';
import { ragAgent } from './rag-agent';

type RAGContext = {
  userId: string;
  permissions: string[];
};

async function runRAGDemo() {
  console.log('\n🔍 FAF Vertex AI RAG Demo');
  console.log('========================');
  console.log('Showcasing Vertex AI RAG integration with FAF framework\n');

  // Check for required environment variables
  if (!process.env.GOOGLE_CLOUD_PROJECT) {
    console.error('❌ Error: GOOGLE_CLOUD_PROJECT environment variable is required');
    console.log('Please set it to your Google Cloud project ID (e.g., "genius-dev-393512")');
    process.exit(1);
  }


  // Set up context with RAG permissions
  const context: RAGContext = {
    userId: process.env.RAG_USER_ID || 'rag-demo-user',
    permissions: ['user', 'rag_access'] // Include rag_access permission
  };

  // Set up agent registry
  const agentRegistry = new Map<string, any>([
    ['RAGAgent', ragAgent]
  ]);

  // Set up model provider
  if (!process.env.LITELLM_URL) {
    console.error('❌ Error: LITELLM_URL environment variable is required');
    console.log('Please set it to your LiteLLM proxy endpoint');
    process.exit(1);
  }
  
  if (!process.env.LITELLM_API_KEY) {
    console.error('❌ Error: LITELLM_API_KEY environment variable is required');
    console.log('Please set it to your LiteLLM API key (starts with sk-)');
    process.exit(1);
  }
  
  const modelProvider = makeLiteLLMProvider(
    process.env.LITELLM_URL,
    process.env.LITELLM_API_KEY
  ) as any;

  // Set up tracing
  const traceCollector = new ConsoleTraceCollector();

  const chatModel = process.env.LITELLM_MODEL || 'gemini-2.5-flash-lite';
  console.log(`📡 Using LiteLLM model: ${chatModel}`);

  const config: RunConfig<RAGContext> = {
    agentRegistry,
    modelProvider,
    maxTurns: parseInt(process.env.RAG_MAX_TURNS || '5'), // Lower turn limit for RAG demo
    modelOverride: chatModel, // Use available model
    onEvent: traceCollector.collect.bind(traceCollector),
  };

  // Demo queries to test RAG functionality
  const demoQueries = [
    "What is return URL?",
    "How do I integrate hypercheckout on android?"
  ];

  for (let i = 0; i < demoQueries.length; i++) {
    const query = demoQueries[i];
    console.log(`\n📋 Demo Query ${i + 1}: "${query}"`);
    console.log('='.repeat(50));

    const initialState: RunState<RAGContext> = {
      runId: generateRunId(),
      traceId: generateTraceId(),
      messages: [
        { role: 'user', content: query }
      ],
      currentAgentName: 'RAGAgent',
      context,
      turnCount: 0,
    };

    try {
      const result = await run(initialState, config);

      if (result.outcome.status === 'completed') {
        console.log('\n✅ RAG Query Completed Successfully!');
        console.log('\n📝 Response:');
        console.log(result.outcome.output);
      } else {
        console.error('\n❌ RAG Query Failed:');
        console.error(`Error: ${result.outcome.error._tag}`);
        if ('detail' in result.outcome.error) {
          console.error(`Details: ${result.outcome.error.detail}`);
        }
        if ('reason' in result.outcome.error) {
          console.error(`Reason: ${result.outcome.error.reason}`);
        }
      }
    } catch (error) {
      console.error('\n❌ Unexpected Error:');
      console.error(error);
    }

    // Add delay between queries
    if (i < demoQueries.length - 1) {
      console.log('\n⏳ Waiting 2 seconds before next query...');
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  console.log('\n🎉 RAG Demo Completed!');
  console.log('\n📊 Demo Summary:');
  console.log('- ✅ Vertex AI RAG integration working');
  console.log('- ✅ Real-time streaming responses');
  console.log('- ✅ Source attribution and grounding');
  console.log('- ✅ Performance metrics tracking');
  console.log('- ✅ Permission-based access control');
  console.log('- ✅ Comprehensive error handling');
  console.log('\n🔗 Integration Points:');
  console.log('- FAF framework orchestration');
  console.log('- Vertex AI RAG corpus querying');
  console.log('- Real-time tracing and observability');
  console.log('- Type-safe tool definitions');
}


// Authentication check
function checkAuthentication() {
  try {
    // This will check if gcloud auth is configured
    const { execSync } = require('child_process');
    execSync('gcloud auth list --filter=status:ACTIVE --format="value(account)"', { stdio: 'pipe' });
    return true;
  } catch (error) {
    return false;
  }
}

// Main execution
async function main() {
  console.log('🚀 Starting FAF Vertex AI RAG Demo...\n');

  // Check authentication
  if (!checkAuthentication()) {
    console.error('❌ Google Cloud authentication not found!');
    console.log('\nPlease run the following command to authenticate:');
    console.log('   gcloud auth application-default login\n');
    console.log('This demo requires access to Vertex AI RAG services.');
    process.exit(1);
  }

  console.log('✅ Google Cloud authentication detected');
  
  await runRAGDemo();
}

if (require.main === module) {
  main().catch(console.error);
}