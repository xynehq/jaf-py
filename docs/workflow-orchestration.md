# Workflow Orchestration

JAF's workflow orchestration system enables the creation of complex, multi-step automation processes that can coordinate multiple agents, tools, and conditional logic. This system provides sophisticated workflow capabilities for enterprise scenarios.

## Overview

The workflow system provides:

- **Step-based Execution**: Define workflows as sequences of executable steps
- **Conditional Logic**: Branch workflow execution based on runtime conditions
- **Parallel Processing**: Execute multiple steps simultaneously for improved performance
- **Error Handling**: Robust error recovery and retry mechanisms
- **State Management**: Maintain workflow state and context across execution steps
- **Agent Integration**: Seamless integration with JAF agents and tools

## Core Components

### Workflow Definition

Workflows are created using the Workflow class and WorkflowBuilder:

```python
from jaf.core.workflows import Workflow, WorkflowBuilder, WorkflowContext
from jaf.core.workflows import AgentStep, ToolStep, ConditionalStep, ParallelStep
from jaf import Agent, Tool

# Create a simple workflow
workflow = Workflow(
    workflow_id="customer_onboarding",
    name="Customer Onboarding Process",
    description="Complete customer onboarding workflow"
)

# Add steps to workflow
agent_step = AgentStep("welcome_step", welcome_agent, "Welcome new customer")
workflow.add_step(agent_step)

# Or use the builder pattern
workflow = WorkflowBuilder("customer_onboarding", "Customer Onboarding") \
    .add_agent_step("welcome", welcome_agent, "Welcome new customer") \
    .add_tool_step("create_account", account_tool, {"type": "standard"}) \
    .build()
```

### Workflow Execution

Execute workflows with comprehensive monitoring and control:

```python
from jaf.core.workflows import Workflow, WorkflowContext

# Create execution context
context = WorkflowContext(
    workflow_id="onboarding_001",
    user_context={"customer_id": "cust_12345"},
    variables={
        "customer_email": "john@example.com",
        "verification_status": "pending"
    },
    metadata={
        "started_by": "system",
        "priority": "high"
    }
)

# Execute workflow
result = await workflow.execute(context)

print(f"Workflow Status: {result.status}")
print(f"Execution Time: {result.total_execution_time_ms}ms")
print(f"Steps Completed: {len(result.steps)}")
print(f"Success Rate: {result.success_rate}%")
```

## Step Types

### AgentStep

Execute JAF agents within workflows:

```python
from jaf.core.workflows import AgentStep
from jaf import Agent

# Create an agent
def instructions(state):
    return "You are a helpful customer service agent."

customer_agent = Agent(
    name="CustomerServiceAgent",
    instructions=instructions,
    tools=[]
)

# Create agent step
agent_step = AgentStep(
    step_id="customer_service",
    agent=customer_agent,
    message="Handle customer inquiry about billing"
)

# Configure step options
agent_step.with_timeout(60).with_retry(3)

# Add execution conditions
agent_step.add_condition(lambda context: context.variables.get("priority") == "high")
```

### ToolStep

Execute tools with parameter mapping:

```python
from jaf.core.workflows import ToolStep
from jaf import Tool

# Create a tool
email_tool = Tool(
    name="send_email",
    description="Send email to customer",
    # ... tool implementation
)

# Create tool step
tool_step = ToolStep(
    step_id="send_welcome_email",
    tool=email_tool,
    args={
        "to": "customer@example.com",
        "subject": "Welcome to our service",
        "template": "welcome_email"
    }
)

# Configure step options
tool_step.with_timeout(30).with_retry(2)
```

### ConditionalStep

Branch execution based on runtime conditions:

```python
from jaf.core.workflows import ConditionalStep, AgentStep, ToolStep

# Create conditional step
conditional_step = ConditionalStep(
    step_id="payment_check",
    condition=lambda context: context.variables.get("payment_amount", 0) > 1000,
    true_step=AgentStep("approval", high_value_agent, "Review high-value transaction"),
    false_step=ToolStep("auto_approve", auto_approve_tool, {"auto_approve": True})
)
```

### ParallelStep

Execute multiple steps simultaneously:

```python
from jaf.core.workflows import ParallelStep, ToolStep, AgentStep

# Create steps for parallel execution
create_record_step = ToolStep("create_record", database_tool, {"table": "customers"})
send_email_step = ToolStep("send_email", email_tool, {"template": "welcome"})
assign_manager_step = AgentStep("assign_manager", manager_agent, "Assign account manager")

# Create parallel step
parallel_step = ParallelStep(
    step_id="customer_setup",
    steps=[create_record_step, send_email_step, assign_manager_step],
    wait_for_all=True  # Wait for all steps to complete
)

# Configure timeout
parallel_step.with_timeout(120)
```

### LoopStep

Iterate over data or repeat until conditions are met:

```python
from jaf.core.workflows import LoopStep, ToolStep

# Create step to execute in loop
process_step = ToolStep("process_item", processing_tool, {})

# Create loop step with condition
loop_step = LoopStep(
    step_id="process_orders",
    step=process_step,
    condition=lambda context, iteration: iteration < len(context.variables.get("orders", [])),
    max_iterations=10
)

# Configure timeout
loop_step.with_timeout(300)
```

## Advanced Features

### Error Handling and Recovery

Workflows support built-in error handling and retry mechanisms:

```python
from jaf.core.workflows import Workflow, AgentStep, ToolStep

# Create workflow with error handling
workflow = Workflow("payment_processing", "Payment Processing")

# Add error handler for specific step
def payment_error_handler(step_result, context):
    if "timeout" in step_result.error:
        # Return a retry step
        return ToolStep("retry_payment", payment_tool, {"retry": True})
    else:
        # Return a fallback step
        return AgentStep("manual_review", review_agent, "Manual payment review required")

workflow.add_error_handler("process_payment", payment_error_handler)

# Configure step-level retry
payment_step = ToolStep("process_payment", payment_tool, {"amount": 100})
payment_step.with_retry(max_retries=3).with_timeout(30)
workflow.add_step(payment_step)
```

### State Management

Workflow context maintains state across execution steps:

```python
from jaf.core.workflows import WorkflowContext

# Create context with initial state
context = WorkflowContext(
    workflow_id="customer_onboarding",
    user_context={"customer_id": "12345"},
    variables={
        "customer_tier": "premium",
        "verification_status": "pending"
    }
)

# Context is automatically updated with step results
# Access updated context after execution
result = await workflow.execute(context)
final_context = result.context

print(f"Final variables: {final_context.variables}")
print(f"Step results: {[step.output for step in result.steps]}")
```

### Streaming Execution

Monitor workflow execution in real-time:

```python
from jaf.core.workflows import execute_workflow_stream

# Stream workflow execution
async for step_result in execute_workflow_stream(workflow, context):
    print(f"Step {step_result.step_id}: {step_result.status}")
    if step_result.is_success:
        print(f"  Output: {step_result.output}")
    elif step_result.is_failure:
        print(f"  Error: {step_result.error}")
    print(f"  Execution time: {step_result.execution_time_ms}ms")
```

## Best Practices

### 1. Design for Idempotency

Ensure workflow steps can be safely retried:

```python
# Good: Idempotent step
idempotent_step = ToolStep("create_user_account") \
    .with_params_function(
        lambda state: {
            "user_id": state["user_id"],
            "email": state["email"],
            "upsert": True  # Create or update
        }
    )

# Good: Check before action
safe_step = ConditionalStep("create_if_not_exists") \
    .condition(lambda state: not user_exists(state["user_id"])) \
    .if_true(ToolStep("create_user_account"))
```

### 2. Handle Partial Failures

Design workflows to handle partial failures gracefully:

```python
# Parallel step with failure handling
robust_parallel = ParallelStep("multi_system_update") \
    .add_branch(ToolStep("update_crm").with_retry(max_attempts=3)) \
    .add_branch(ToolStep("update_billing").with_retry(max_attempts=3)) \
    .add_branch(ToolStep("update_analytics").with_retry(max_attempts=2)) \
    .with_failure_policy("continue_on_partial_failure") \
    .with_minimum_success_count(2)  # Require at least 2 successes
```

### 3. Use Timeouts Appropriately

Set realistic timeouts for all steps:

```python
# Different timeouts for different step types
workflow = WorkflowBuilder("data_processing") \
    .add_step(
        ToolStep("quick_validation")
        .with_timeout(10)  # Fast operation
    ) \
    .add_step(
        ToolStep("heavy_computation")
        .with_timeout(300)  # Allow 5 minutes
    ) \
    .add_step(
        AgentStep("human_review")
        .with_timeout(3600)  # Allow 1 hour
    ) \
    .build()
```

### 4. Implement Proper Logging

Add comprehensive logging for debugging:

```python
from jaf.core.workflows import WorkflowLogger

# Custom logger
class DetailedWorkflowLogger(WorkflowLogger):
    def log_step_start(self, step_name, state):
        logger.info(f"Starting step: {step_name}", extra={
            "workflow_id": state.workflow_id,
            "step_name": step_name,
            "state_keys": list(state.data.keys())
        })
    
    def log_step_complete(self, step_name, result, duration_ms):
        logger.info(f"Completed step: {step_name} in {duration_ms}ms", extra={
            "step_name": step_name,
            "duration_ms": duration_ms,
            "success": result.success
        })

# Use custom logger
workflow = WorkflowBuilder("logged_workflow") \
    .with_logger(DetailedWorkflowLogger()) \
    .build()
```

## Example: Complete E-commerce Order Processing

Here's a comprehensive example showing a complete e-commerce order processing workflow:

```python
import asyncio
from jaf.core.workflows import WorkflowBuilder, WorkflowEngine, WorkflowContext
from jaf.core.workflows import AgentStep, ToolStep, ConditionalStep, ParallelStep

async def create_order_processing_workflow():
    """Create a comprehensive order processing workflow."""
    
    return WorkflowBuilder("ecommerce_order_processing") \
        .description("Complete e-commerce order processing pipeline") \
        .add_step(
            # Step 1: Validate order
            ToolStep("validate_order")
            .with_params_function(
                lambda state: {"order_id": state["order_id"]}
            )
            .with_timeout(30)
            .with_retry(max_attempts=3)
        ) \
        .add_step(
            # Step 2: Check inventory
            ConditionalStep("inventory_check")
            .condition(lambda state: state.get("order_valid", False))
            .if_true(
                ToolStep("check_inventory")
                .with_params_function(
                    lambda state: {"items": state["order_items"]}
                )
            )
            .if_false(
                AgentStep("order_validation_agent")
                .with_input("Handle invalid order")
            )
        ) \
        .add_step(
            # Step 3: Process payment
            ConditionalStep("payment_processing")
            .condition(lambda state: state.get("inventory_available", False))
            .if_true(
                ToolStep("process_payment")
                .with_params_function(
                    lambda state: {
                        "amount": state["order_total"],
                        "payment_method": state["payment_method"]
                    }
                )
                .with_timeout(60)
                .with_retry(max_attempts=2)
            )
            .if_false(
                AgentStep("inventory_agent")
                .with_input("Handle out of stock items")
            )
        ) \
        .add_step(
            # Step 4: Parallel fulfillment
            ConditionalStep("fulfillment_check")
            .condition(lambda state: state.get("payment_successful", False))
            .if_true(
                ParallelStep("order_fulfillment")
                .add_branch(
                    # Update inventory
                    ToolStep("update_inventory")
                    .with_params_function(
                        lambda state: {"items": state["order_items"]}
                    )
                )
                .add_branch(
                    # Generate shipping label
                    ToolStep("create_shipping_label")
                    .with_params_function(
                        lambda state: {
                            "address": state["shipping_address"],
                            "items": state["order_items"]
                        }
                    )
                )
                .add_branch(
                    # Send confirmation email
                    ToolStep("send_confirmation_email")
                    .with_params_function(
                        lambda state: {
                            "customer_email": state["customer_email"],
                            "order_id": state["order_id"]
                        }
                    )
                )
                .add_branch(
                    # Update CRM
                    ToolStep("update_crm")
                    .with_params_function(
                        lambda state: {
                            "customer_id": state["customer_id"],
                            "order_value": state["order_total"]
                        }
                    )
                )
                .with_timeout(120)
                .with_failure_policy("continue_on_partial_failure")
                .with_minimum_success_count(3)
            )
        ) \
        .add_step(
            # Step 5: Final notification
            AgentStep("fulfillment_agent")
            .with_input_function(
                lambda state: f"Complete order fulfillment for order {state['order_id']}"
            )
            .with_context_function(
                lambda state: {
                    "order_status": state.get("fulfillment_status", "unknown"),
                    "customer_tier": state.get("customer_tier", "standard")
                }
            )
        ) \
        .with_error_handler(
            lambda error, step, state: {
                "action": "retry" if "network" in str(error).lower() else "fail",
                "escalate": True,
                "notify_team": True
            }
        ) \
        .build()

async def main():
    """Demonstrate the complete order processing workflow."""
    
    # Create workflow
    workflow = await create_order_processing_workflow()
    
    # Create engine
    engine = WorkflowEngine()
    
    # Create execution context
    context = WorkflowContext(
        workflow_id="order_12345",
        initial_data={
            "order_id": "ORD-12345",
            "customer_id": "CUST-67890",
            "customer_email": "customer@example.com",
            "order_items": [
                {"sku": "ITEM-001", "quantity": 2, "price": 29.99},
                {"sku": "ITEM-002", "quantity": 1, "price": 49.99}
            ],
            "order_total": 109.97,
            "payment_method": "credit_card",
            "shipping_address": {
                "street": "123 Main St",
                "city": "Anytown",
                "state": "CA",
                "zip": "12345"
            },
            "customer_tier": "premium"
        },
        metadata={
            "started_by": "order_service",
            "priority": "normal",
            "source": "web"
        }
    )
    
    # Execute workflow
    print("🚀 Starting order processing workflow...")
    result = await engine.execute_workflow(workflow, context)
    
    # Display results
    print(f"✅ Workflow completed with status: {result.status}")
    print(f"⏱️ Total execution time: {result.execution_time_ms}ms")
    print(f"📊 Steps completed: {result.steps_completed}/{result.total_steps}")
    
    if result.status == "failed":
        print(f"❌ Error: {result.error}")
    else:
        print(f"🎉 Order {context.initial_data['order_id']} processed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
```

The workflow orchestration system provides the foundation for building sophisticated, enterprise-grade automation that can handle complex business processes with reliability and observability.

## Next Steps

- Learn about [Analytics System](analytics-system.md) for workflow monitoring
- Explore [Performance Monitoring](performance-monitoring.md) for optimization
- Check [Streaming Responses](streaming-responses.md) for real-time updates
- Review [Plugin System](plugin-system.md) for extensibility
