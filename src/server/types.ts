import { z } from 'zod';
import { Agent, RunConfig } from '../core/types';

export interface ServerConfig<Ctx> {
  port?: number;
  host?: string;
  cors?: boolean;
  runConfig: RunConfig<Ctx>;
  agentRegistry: Map<string, Agent<Ctx, any>>;
}

// Request/Response schemas
export const httpMessageSchema = z.object({
  role: z.enum(['user', 'assistant', 'system']),
  content: z.string()
});

export const chatRequestSchema = z.object({
  messages: z.array(httpMessageSchema),
  agentName: z.string(),
  context: z.any().optional(),
  maxTurns: z.number().optional(),
  stream: z.boolean().default(false)
});

export type ChatRequest = z.infer<typeof chatRequestSchema>;
export type HttpMessage = z.infer<typeof httpMessageSchema>;

// Extended message schema that includes tool calls and responses
export const fullMessageSchema = z.union([
  httpMessageSchema,
  z.object({
    role: z.literal('assistant'),
    content: z.string(),
    tool_calls: z.array(z.object({
      id: z.string(),
      type: z.literal('function'),
      function: z.object({
        name: z.string(),
        arguments: z.string()
      })
    })).optional()
  }),
  z.object({
    role: z.literal('tool'),
    content: z.string(),
    tool_call_id: z.string().optional()
  })
]);

export const chatResponseSchema = z.object({
  success: z.boolean(),
  data: z.object({
    runId: z.string(),
    traceId: z.string(),
    messages: z.array(fullMessageSchema),
    outcome: z.object({
      status: z.enum(['completed', 'error', 'max_turns']),
      output: z.string().optional(),
      error: z.any().optional()
    }),
    turnCount: z.number(),
    executionTimeMs: z.number()
  }).optional(),
  error: z.string().optional()
});

export type ChatResponse = z.infer<typeof chatResponseSchema>;

export const agentListResponseSchema = z.object({
  success: z.boolean(),
  data: z.object({
    agents: z.array(z.object({
      name: z.string(),
      description: z.string(),
      tools: z.array(z.string())
    }))
  }).optional(),
  error: z.string().optional()
});

export type AgentListResponse = z.infer<typeof agentListResponseSchema>;

export const healthResponseSchema = z.object({
  status: z.literal('healthy'),
  timestamp: z.string(),
  version: z.string(),
  uptime: z.number()
});

export type HealthResponse = z.infer<typeof healthResponseSchema>;