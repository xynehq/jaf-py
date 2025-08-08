import { z } from 'zod';
import {
  RunState,
  RunConfig,
  RunResult,
  FAFError,
  Message,
  TraceEvent,
  Agent,
  Tool,
} from './types.js';

export async function run<Ctx, Out>(
  initialState: RunState<Ctx>,
  config: RunConfig<Ctx>
): Promise<RunResult<Out>> {
  try {
    config.onEvent?.({
      type: 'run_start',
      data: { runId: initialState.runId, traceId: initialState.traceId }
    });

    const result = await runInternal<Ctx, Out>(initialState, config);
    
    config.onEvent?.({
      type: 'run_end',
      data: { outcome: result.outcome }
    });

    return result;
  } catch (error) {
    const errorResult: RunResult<Out> = {
      finalState: initialState,
      outcome: {
        status: 'error',
        error: {
          _tag: 'ModelBehaviorError',
          detail: error instanceof Error ? error.message : String(error)
        }
      }
    } as RunResult<Out>;

    config.onEvent?.({
      type: 'run_end',
      data: { outcome: errorResult.outcome }
    });

    return errorResult;
  }
}

async function runInternal<Ctx, Out>(
  state: RunState<Ctx>,
  config: RunConfig<Ctx>
): Promise<RunResult<Out>> {
  if (state.turnCount === 0) {
    const firstUserMessage = state.messages.find(m => m.role === 'user');
    if (firstUserMessage && config.initialInputGuardrails) {
      for (const guardrail of config.initialInputGuardrails) {
        const result = await guardrail(firstUserMessage.content);
        if (!result.isValid) {
          return {
            finalState: state,
            outcome: {
              status: 'error',
              error: {
                _tag: 'InputGuardrailTripwire',
                reason: result.errorMessage
              }
            }
          };
        }
      }
    }
  }

  const maxTurns = config.maxTurns ?? 50;
  if (state.turnCount >= maxTurns) {
    return {
      finalState: state,
      outcome: {
        status: 'error',
        error: {
          _tag: 'MaxTurnsExceeded',
          turns: state.turnCount
        }
      }
    };
  }

  const currentAgent = config.agentRegistry.get(state.currentAgentName);
  if (!currentAgent) {
    return {
      finalState: state,
      outcome: {
        status: 'error',
        error: {
          _tag: 'AgentNotFound',
          agentName: state.currentAgentName
        }
      }
    };
  }

  console.log(`[FAF:ENGINE] Using agent: ${currentAgent.name}`);
  console.log(`[FAF:ENGINE] Agent has ${currentAgent.tools?.length || 0} tools available`);
  if (currentAgent.tools) {
    console.log(`[FAF:ENGINE] Available tools:`, currentAgent.tools.map(t => t.schema.name));
  }

  const model = config.modelOverride ?? currentAgent.modelConfig?.name ?? "gpt-4o";
  
  config.onEvent?.({
    type: 'llm_call_start',
    data: { agentName: currentAgent.name, model }
  });

  const llmResponse = await config.modelProvider.getCompletion(state, currentAgent, config);
  
  config.onEvent?.({
    type: 'llm_call_end',
    data: { choice: llmResponse }
  });

  if (!llmResponse.message) {
    return {
      finalState: state,
      outcome: {
        status: 'error',
        error: {
          _tag: 'ModelBehaviorError',
          detail: 'No message in model response'
        }
      }
    };
  }

  const assistantMessage: Message = {
    role: 'assistant',
    content: llmResponse.message.content || '',
    tool_calls: llmResponse.message.tool_calls
  };

  const newMessages = [...state.messages, assistantMessage];

  if (llmResponse.message.tool_calls && llmResponse.message.tool_calls.length > 0) {
    console.log(`[FAF:ENGINE] Processing ${llmResponse.message.tool_calls.length} tool calls`);
    console.log(`[FAF:ENGINE] Tool calls:`, llmResponse.message.tool_calls);
    
    const toolResults = await executeToolCalls(
      llmResponse.message.tool_calls,
      currentAgent,
      state,
      config
    );
    
    console.log(`[FAF:ENGINE] Tool execution completed. Results count:`, toolResults.length);

    if (toolResults.some(r => r.isHandoff)) {
      const handoffResult = toolResults.find(r => r.isHandoff);
      if (handoffResult) {
        const targetAgent = handoffResult.targetAgent!;
        
        if (!currentAgent.handoffs?.includes(targetAgent)) {
          return {
            finalState: { ...state, messages: newMessages },
            outcome: {
              status: 'error',
              error: {
                _tag: 'HandoffError',
                detail: `Agent ${currentAgent.name} cannot handoff to ${targetAgent}`
              }
            }
          };
        }

        config.onEvent?.({
          type: 'handoff',
          data: { from: currentAgent.name, to: targetAgent }
        });

        const nextState: RunState<Ctx> = {
          ...state,
          messages: [...newMessages, ...toolResults.map(r => r.message)],
          currentAgentName: targetAgent,
          turnCount: state.turnCount + 1
        };

        return runInternal(nextState, config);
      }
    }

    const nextState: RunState<Ctx> = {
      ...state,
      messages: [...newMessages, ...toolResults.map(r => r.message)],
      turnCount: state.turnCount + 1
    };

    return runInternal(nextState, config);
  }

  if (llmResponse.message.content) {
    if (currentAgent.outputCodec) {
      const parseResult = currentAgent.outputCodec.safeParse(
        tryParseJSON(llmResponse.message.content)
      );
      
      if (!parseResult.success) {
        return {
          finalState: { ...state, messages: newMessages },
          outcome: {
            status: 'error',
            error: {
              _tag: 'DecodeError',
              errors: parseResult.error.issues
            }
          }
        };
      }

      if (config.finalOutputGuardrails) {
        for (const guardrail of config.finalOutputGuardrails) {
          const result = await guardrail(parseResult.data);
          if (!result.isValid) {
            return {
              finalState: { ...state, messages: newMessages },
              outcome: {
                status: 'error',
                error: {
                  _tag: 'OutputGuardrailTripwire',
                  reason: result.errorMessage
                }
              }
            };
          }
        }
      }

      return {
        finalState: { ...state, messages: newMessages },
        outcome: {
          status: 'completed',
          output: parseResult.data as Out
        }
      };
    } else {
      if (config.finalOutputGuardrails) {
        for (const guardrail of config.finalOutputGuardrails) {
          const result = await guardrail(llmResponse.message.content);
          if (!result.isValid) {
            return {
              finalState: { ...state, messages: newMessages },
              outcome: {
                status: 'error',
                error: {
                  _tag: 'OutputGuardrailTripwire',
                  reason: result.errorMessage
                }
              }
            };
          }
        }
      }

      return {
        finalState: { ...state, messages: newMessages },
        outcome: {
          status: 'completed',
          output: llmResponse.message.content as Out
        }
      };
    }
  }

  return {
    finalState: { ...state, messages: newMessages },
    outcome: {
      status: 'error',
      error: {
        _tag: 'ModelBehaviorError',
        detail: 'Model produced neither content nor tool calls'
      }
    }
  };
}

type ToolCallResult = {
  message: Message;
  isHandoff?: boolean;
  targetAgent?: string;
};

async function executeToolCalls<Ctx>(
  toolCalls: Array<{
    id: string;
    type: 'function';
    function: {
      name: string;
      arguments: string;
    };
  }>,
  agent: Agent<Ctx, any>,
  state: RunState<Ctx>,
  config: RunConfig<Ctx>
): Promise<ToolCallResult[]> {
  const results = await Promise.all(
    toolCalls.map(async (toolCall): Promise<ToolCallResult> => {
      config.onEvent?.({
        type: 'tool_call_start',
        data: {
          toolName: toolCall.function.name,
          args: tryParseJSON(toolCall.function.arguments)
        }
      });

      try {
        const tool = agent.tools?.find(t => t.schema.name === toolCall.function.name);
        
        if (!tool) {
          const errorResult = JSON.stringify({
            error: "tool_not_found",
            message: `Tool ${toolCall.function.name} not found`,
            tool_name: toolCall.function.name,
          });

          config.onEvent?.({
            type: 'tool_call_end',
            data: { toolName: toolCall.function.name, result: errorResult }
          });

          return {
            message: {
              role: 'tool',
              content: errorResult,
              tool_call_id: toolCall.id
            }
          };
        }

        const rawArgs = tryParseJSON(toolCall.function.arguments);
        const parseResult = tool.schema.parameters.safeParse(rawArgs);

        if (!parseResult.success) {
          const errorResult = JSON.stringify({
            error: "validation_error",
            message: `Invalid arguments for ${toolCall.function.name}: ${parseResult.error.message}`,
            tool_name: toolCall.function.name,
            validation_errors: parseResult.error.issues
          });

          config.onEvent?.({
            type: 'tool_call_end',
            data: { toolName: toolCall.function.name, result: errorResult }
          });

          return {
            message: {
              role: 'tool',
              content: errorResult,
              tool_call_id: toolCall.id
            }
          };
        }

        console.log(`[FAF:ENGINE] About to execute tool: ${toolCall.function.name}`);
        console.log(`[FAF:ENGINE] Tool args:`, parseResult.data);
        console.log(`[FAF:ENGINE] Tool context:`, state.context);
        
        const toolResult = await tool.execute(parseResult.data, state.context);
        
        // Handle both string and ToolResult formats
        let resultString: string;
        let toolResultObj: any = null;
        
        if (typeof toolResult === 'string') {
          resultString = toolResult;
          console.log(`[FAF:ENGINE] Tool ${toolCall.function.name} returned string:`, resultString);
        } else {
          // It's a ToolResult object
          toolResultObj = toolResult;
          const { toolResultToString } = await import('./tool-results');
          resultString = toolResultToString(toolResult);
          console.log(`[FAF:ENGINE] Tool ${toolCall.function.name} returned ToolResult:`, toolResult);
          console.log(`[FAF:ENGINE] Converted to string:`, resultString);
        }

        config.onEvent?.({
          type: 'tool_call_end',
          data: { 
            toolName: toolCall.function.name, 
            result: resultString,
            toolResult: toolResultObj,
            status: toolResultObj?.status || 'success'
          }
        });

        const handoffCheck = tryParseJSON(resultString);
        if (handoffCheck && typeof handoffCheck === 'object' && 'handoff_to' in handoffCheck) {
          return {
            message: {
              role: 'tool',
              content: resultString,
              tool_call_id: toolCall.id
            },
            isHandoff: true,
            targetAgent: handoffCheck.handoff_to as string
          };
        }

        return {
          message: {
            role: 'tool',
            content: resultString,
            tool_call_id: toolCall.id
          }
        };

      } catch (error) {
        const errorResult = JSON.stringify({
          error: "execution_error",
          message: error instanceof Error ? error.message : String(error),
          tool_name: toolCall.function.name,
        });

        config.onEvent?.({
          type: 'tool_call_end',
          data: { toolName: toolCall.function.name, result: errorResult }
        });

        return {
          message: {
            role: 'tool',
            content: errorResult,
            tool_call_id: toolCall.id
          }
        };
      }
    })
  );

  return results;
}

function tryParseJSON(str: string): any {
  try {
    return JSON.parse(str);
  } catch {
    return str;
  }
}