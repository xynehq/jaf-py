"""
Advanced guardrails implementation for JAF framework.

This module provides LLM-based guardrails with caching, circuit breaking,
and execution strategies for input validation and output filtering.
"""

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .types import (
    Agent,
    RunConfig,
    RunState,
    ValidationResult,
    ValidValidationResult,
    InvalidValidationResult,
    Guardrail,
    AdvancedGuardrailsConfig,
    validate_guardrails_config,
    json_parse_llm_output,
    get_text_content,
    Message,
    ContentRole,
    create_run_id,
    create_trace_id,
    GuardrailEvent,
    GuardrailEventData,
    GuardrailViolationEvent,
    GuardrailViolationEventData,
)

# Constants for content length limits
SHORT_TIMEOUT_MAX_CONTENT = 10000
LONG_TIMEOUT_MAX_CONTENT = 50000
CIRCUIT_BREAKER_CLEANUP_MAX_AGE = 10 * 60 * 1000  # 10 minutes

# Constants for timeout values
DEFAULT_FAST_MODEL_TIMEOUT_MS = 10000
DEFAULT_TIMEOUT_MS = 5000
GUARDRAIL_TIMEOUT_MS = 10000
OUTPUT_GUARDRAIL_TIMEOUT_MS = 15000


class GuardrailCircuitBreaker:
    """Circuit breaker for guardrail execution to handle repeated failures."""

    def __init__(self, max_failures: int = 5, reset_time_ms: int = 60000):
        self.failures = 0
        self.last_failure_time = 0
        self.max_failures = max_failures
        self.reset_time_ms = reset_time_ms

    def is_open(self) -> bool:
        """Check if circuit breaker is open (blocking requests)."""
        if self.failures < self.max_failures:
            return False

        time_since_last_failure = (time.time() * 1000) - self.last_failure_time
        if time_since_last_failure > self.reset_time_ms:
            self.failures = 0
            return False

        return True

    def record_failure(self) -> None:
        """Record a failure."""
        self.failures += 1
        self.last_failure_time = time.time() * 1000

    def record_success(self) -> None:
        """Record a success, resetting the failure count."""
        self.failures = 0

    def should_be_cleaned_up(self, max_age: int) -> bool:
        """Check if this circuit breaker should be cleaned up."""
        now = time.time() * 1000
        return (
            self.last_failure_time > 0
            and (now - self.last_failure_time) > max_age
            and not self.is_open()
        )


@dataclass
class CacheEntry:
    """Cache entry for guardrail results."""

    result: ValidationResult
    timestamp: float
    hit_count: int = 1


class GuardrailCache:
    """LRU cache for guardrail results."""

    def __init__(self, max_size: int = 1000, ttl_ms: int = 300000):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.ttl_ms = ttl_ms

    def _create_key(self, stage: str, rule_prompt: str, content: str, model_name: str) -> str:
        """Create a cache key."""
        content_hash = self._hash_string(content[:1000])
        rule_hash = self._hash_string(rule_prompt)
        return f"guardrail_{stage}_{model_name}_{rule_hash}_{content_hash}_{len(content)}"

    def _hash_string(self, s: str) -> str:
        """Simple hash function for strings."""
        hash_val = 0
        for char in s:
            hash_val = ((hash_val << 5) - hash_val) + ord(char)
            hash_val = hash_val & 0xFFFFFFFF  # Keep it 32-bit
        return str(abs(hash_val))

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        return (time.time() * 1000) - entry.timestamp > self.ttl_ms

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if len(self.cache) < self.max_size:
            return

        lru_key: Optional[str] = None
        lru_score = float("inf")
        now = time.time() * 1000

        for key, entry in self.cache.items():
            age_hours = (now - entry.timestamp) / (1000 * 60 * 60)
            score = entry.hit_count / (1 + age_hours)
            if score < lru_score:
                lru_score = score
                lru_key = key

        if lru_key:
            del self.cache[lru_key]

    def get(
        self, stage: str, rule_prompt: str, content: str, model_name: str
    ) -> Optional[ValidationResult]:
        """Get cached result."""
        key = self._create_key(stage, rule_prompt, content, model_name)
        entry = self.cache.get(key)

        if not entry or self._is_expired(entry):
            if entry:
                del self.cache[key]
            return None

        entry.hit_count += 1
        entry.timestamp = time.time() * 1000

        return entry.result

    def set(
        self, stage: str, rule_prompt: str, content: str, model_name: str, result: ValidationResult
    ) -> None:
        """Cache a result."""
        key = self._create_key(stage, rule_prompt, content, model_name)

        self._evict_lru()

        self.cache[key] = CacheEntry(result=result, timestamp=time.time() * 1000, hit_count=1)

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {"size": len(self.cache), "max_size": self.max_size}


# Global instances
_guardrail_cache = GuardrailCache()
_circuit_breakers: Dict[str, GuardrailCircuitBreaker] = {}


def _get_circuit_breaker(stage: str, model_name: str) -> GuardrailCircuitBreaker:
    """Get or create a circuit breaker for a stage/model combination."""
    key = f"{stage}-{model_name}"
    if key not in _circuit_breakers:
        _circuit_breakers[key] = GuardrailCircuitBreaker()
    return _circuit_breakers[key]


async def _with_timeout(awaitable, timeout_ms: int, error_message: str):
    """Run an awaitable with a timeout."""
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_ms / 1000)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Timeout: {error_message}")


async def _create_llm_guardrail(
    config: RunConfig,
    stage: str,
    rule_prompt: str,
    fast_model: Optional[str] = None,
    fail_safe: str = "allow",
    timeout_ms: int = 30000,
) -> Guardrail:
    """Create an LLM-based guardrail function."""

    async def guardrail_func(content: Any) -> ValidationResult:
        content_str = str(content) if not isinstance(content, str) else content

        model_to_use = fast_model or config.default_fast_model
        if not model_to_use:
            print(
                f"[JAF:GUARDRAILS] No fast model available for LLM guardrail evaluation, using failSafe: {fail_safe}"
            )
            return (
                ValidValidationResult()
                if fail_safe == "allow"
                else InvalidValidationResult(
                    error_message="No model available for guardrail evaluation"
                )
            )

        # Check cache first
        cached_result = _guardrail_cache.get(stage, rule_prompt, content_str, model_to_use)
        if cached_result:
            print(f"[JAF:GUARDRAILS] Cache hit for {stage} guardrail")
            return cached_result

        # Check circuit breaker
        circuit_breaker = _get_circuit_breaker(stage, model_to_use)
        if circuit_breaker.is_open():
            print(
                f"[JAF:GUARDRAILS] Circuit breaker open for {stage} guardrail on model {model_to_use}, using failSafe: {fail_safe}"
            )
            return (
                ValidValidationResult()
                if fail_safe == "allow"
                else InvalidValidationResult(
                    error_message="Circuit breaker open - too many recent failures"
                )
            )

        # Validate content
        if not content_str:
            print(f"[JAF:GUARDRAILS] Invalid content provided to {stage} guardrail")
            return (
                ValidValidationResult()
                if fail_safe == "allow"
                else InvalidValidationResult(error_message="Invalid content provided to guardrail")
            )

        # Check content length
        max_content_length = (
            SHORT_TIMEOUT_MAX_CONTENT if timeout_ms < 10000 else LONG_TIMEOUT_MAX_CONTENT
        )
        if len(content_str) > max_content_length:
            print(
                f"[JAF:GUARDRAILS] Content too large for {stage} guardrail ({len(content_str)} chars, max: {max_content_length})"
            )
            return (
                ValidValidationResult()
                if fail_safe == "allow"
                else InvalidValidationResult(
                    error_message=f"Content too large for guardrail evaluation ({len(content_str)} > {max_content_length} chars)"
                )
            )

        # Prepare content for evaluation
        sanitized_content = content_str.replace('"""', "[TRIPLE_QUOTE]").replace("\n", " ")[:2000]

        eval_prompt = f"""You are a guardrail validator for {stage}.
Rules:
{rule_prompt}

Decide if the {stage} {"user message" if stage == "input" else "assistant output"} complies with the rules.
Return a JSON object with keys: {{"allowed": boolean, "reason": string}}. Do not include extra text.
{stage.capitalize()} {"user message" if stage == "input" else "assistant output"}:
\"\"\"
{sanitized_content}
\"\"\""""

        try:
            # Create temporary state for guardrail evaluation
            temp_state = RunState(
                run_id=create_run_id("guardrail-eval"),
                trace_id=create_trace_id("guardrail-eval"),
                messages=[Message(role=ContentRole.USER, content=eval_prompt)],
                current_agent_name="guardrail-evaluator",
                context={},
                turn_count=0,
            )

            # Create evaluation agent
            def eval_instructions(state: RunState) -> str:
                return "You are a guardrail validator. Return only valid JSON."

            eval_agent = Agent(
                name="guardrail-evaluator",
                instructions=eval_instructions,
                model_config={"name": model_to_use} if hasattr(config, "ModelConfig") else None,
            )

            # Create guardrail config (no guardrails to avoid recursion)
            guardrail_config = RunConfig(
                agent_registry=config.agent_registry,
                model_provider=config.model_provider,
                max_turns=1,
                default_fast_model=config.default_fast_model,
                model_override=model_to_use,
                initial_input_guardrails=None,
                final_output_guardrails=None,
                on_event=None,
                prefer_streaming=config.prefer_streaming,
            )

            # Execute with timeout
            completion_promise = config.model_provider.get_completion(
                temp_state, eval_agent, guardrail_config
            )
            response = await _with_timeout(
                completion_promise,
                timeout_ms,
                f"{stage} guardrail evaluation timed out after {timeout_ms}ms",
            )

            # Handle different response formats
            response_content = None
            if hasattr(response, "message") and response.message:
                if hasattr(response.message, "content"):
                    response_content = response.message.content
            elif isinstance(response, dict):
                if "message" in response and response["message"]:
                    if isinstance(response["message"], dict) and "content" in response["message"]:
                        response_content = response["message"]["content"]
                    elif hasattr(response["message"], "content"):
                        response_content = response["message"].content

            if not response_content:
                circuit_breaker.record_success()
                result = ValidValidationResult()
                _guardrail_cache.set(stage, rule_prompt, content_str, model_to_use, result)
                return result

            # Parse response
            parsed = json_parse_llm_output(response_content)
            allowed = bool(parsed.get("allowed", True) if parsed else True)
            reason = str(
                parsed.get("reason", "Guardrail violation") if parsed else "Guardrail violation"
            )

            circuit_breaker.record_success()

            result = (
                ValidValidationResult()
                if allowed
                else InvalidValidationResult(error_message=reason)
            )

            _guardrail_cache.set(stage, rule_prompt, content_str, model_to_use, result)
            return result

        except Exception as e:
            circuit_breaker.record_failure()

            error_message = str(e)
            is_timeout = "Timeout" in error_message

            log_message = f"[JAF:GUARDRAILS] {stage} guardrail evaluation failed"
            if is_timeout:
                print(f"{log_message} due to timeout ({timeout_ms}ms), using failSafe: {fail_safe}")
            else:
                print(f"{log_message}, using failSafe: {fail_safe} - {error_message}")

            return (
                ValidValidationResult()
                if fail_safe == "allow"
                else InvalidValidationResult(
                    error_message=f"Guardrail evaluation failed: {error_message}"
                )
            )

    return guardrail_func


async def build_effective_guardrails(
    current_agent: Agent, config: RunConfig
) -> Tuple[List[Guardrail], List[Guardrail]]:
    """Build effective input and output guardrails for an agent."""
    effective_input_guardrails: List[Guardrail] = []
    effective_output_guardrails: List[Guardrail] = []

    try:
        raw_guardrails_cfg = (
            current_agent.advanced_config.guardrails if current_agent.advanced_config else None
        )
        guardrails_cfg = validate_guardrails_config(raw_guardrails_cfg)

        fast_model = guardrails_cfg.fast_model or config.default_fast_model
        if not fast_model and (guardrails_cfg.input_prompt or guardrails_cfg.output_prompt):
            print(
                "[JAF:GUARDRAILS] No fast model available for LLM guardrails - skipping LLM-based validation"
            )

        print(
            "[JAF:GUARDRAILS] Configuration:",
            {
                "hasInputPrompt": bool(guardrails_cfg.input_prompt),
                "hasOutputPrompt": bool(guardrails_cfg.output_prompt),
                "requireCitations": guardrails_cfg.require_citations,
                "executionMode": guardrails_cfg.execution_mode,
                "failSafe": guardrails_cfg.fail_safe,
                "timeoutMs": guardrails_cfg.timeout_ms,
                "fastModel": fast_model or "none",
            },
        )

        # Start with global guardrails
        effective_input_guardrails = list(config.initial_input_guardrails or [])
        effective_output_guardrails = list(config.final_output_guardrails or [])

        # Add input prompt guardrail
        if guardrails_cfg.input_prompt and guardrails_cfg.input_prompt.strip():
            input_guardrail = await _create_llm_guardrail(
                config,
                "input",
                guardrails_cfg.input_prompt,
                fast_model,
                guardrails_cfg.fail_safe,
                guardrails_cfg.timeout_ms,
            )
            effective_input_guardrails.append(input_guardrail)

        # Add citation requirement guardrail
        if guardrails_cfg.require_citations:

            def citation_guardrail(output: Any) -> ValidationResult:
                def find_text(val: Any) -> str:
                    if isinstance(val, str):
                        return val
                    elif isinstance(val, list):
                        return " ".join(find_text(item) for item in val)
                    elif isinstance(val, dict):
                        return " ".join(find_text(v) for v in val.values())
                    else:
                        return str(val)

                text = find_text(output)
                has_citation = bool(re.search(r"\[(\d+)\]", text))
                return (
                    ValidValidationResult()
                    if has_citation
                    else InvalidValidationResult(
                        error_message="Missing required [n] citation in output"
                    )
                )

            effective_output_guardrails.append(citation_guardrail)

        # Add output prompt guardrail
        if guardrails_cfg.output_prompt and guardrails_cfg.output_prompt.strip():
            output_guardrail = await _create_llm_guardrail(
                config,
                "output",
                guardrails_cfg.output_prompt,
                fast_model,
                guardrails_cfg.fail_safe,
                guardrails_cfg.timeout_ms,
            )
            effective_output_guardrails.append(output_guardrail)

    except Exception as e:
        print(f"[JAF:GUARDRAILS] Failed to configure advanced guardrails: {e}")
        # Fall back to global guardrails only
        effective_input_guardrails = list(config.initial_input_guardrails or [])
        effective_output_guardrails = list(config.final_output_guardrails or [])

    return effective_input_guardrails, effective_output_guardrails


async def execute_input_guardrails_sequential(
    input_guardrails: List[Guardrail], first_user_message: Message, config: RunConfig
) -> ValidationResult:
    """Execute input guardrails sequentially."""
    if not input_guardrails:
        return ValidValidationResult()

    print(f"[JAF:GUARDRAILS] Starting {len(input_guardrails)} input guardrails (sequential)")

    content = get_text_content(first_user_message.content)

    for i, guardrail in enumerate(input_guardrails):
        guardrail_name = f"input-guardrail-{i + 1}"

        try:
            print(f"[JAF:GUARDRAILS] Starting {guardrail_name}")

            timeout_ms = GUARDRAIL_TIMEOUT_MS
            result = await _with_timeout(
                guardrail(content)
                if asyncio.iscoroutinefunction(guardrail)
                else guardrail(content),
                timeout_ms,
                f"{guardrail_name} execution timed out after {timeout_ms}ms",
            )

            print(f"[JAF:GUARDRAILS] {guardrail_name} completed: {result}")

            if not result.is_valid:
                error_message = getattr(result, "error_message", "Guardrail violation")
                print(f"🚨 {guardrail_name} violation: {error_message}")
                if config.on_event:
                    config.on_event(
                        GuardrailViolationEvent(
                            data=GuardrailViolationEventData(stage="input", reason=error_message)
                        )
                    )
                return result

        except Exception as error:
            error_message = str(error)
            print(f"[JAF:GUARDRAILS] {guardrail_name} failed: {error_message}")

            is_system_error = "Timeout" in error_message or "Circuit breaker" in error_message

            if is_system_error:
                print(
                    f"[JAF:GUARDRAILS] {guardrail_name} system error, continuing: {error_message}"
                )
                continue
            else:
                if config.on_event:
                    config.on_event(
                        GuardrailViolationEvent(
                            data=GuardrailViolationEventData(stage="input", reason=error_message)
                        )
                    )
                return InvalidValidationResult(error_message=error_message)

    print("✅ All input guardrails passed (sequential).")
    return ValidValidationResult()


async def execute_input_guardrails_parallel(
    input_guardrails: List[Guardrail], first_user_message: Message, config: RunConfig
) -> ValidationResult:
    """Execute input guardrails in parallel."""
    if not input_guardrails:
        return ValidValidationResult()

    print(f"[JAF:GUARDRAILS] Starting {len(input_guardrails)} input guardrails")

    content = get_text_content(first_user_message.content)

    async def run_guardrail(guardrail: Guardrail, index: int):
        guardrail_name = f"input-guardrail-{index + 1}"

        try:
            print(f"[JAF:GUARDRAILS] Starting {guardrail_name}")

            timeout_ms = (
                DEFAULT_FAST_MODEL_TIMEOUT_MS if config.default_fast_model else DEFAULT_TIMEOUT_MS
            )

            if asyncio.iscoroutinefunction(guardrail):
                result = await _with_timeout(
                    guardrail(content),
                    timeout_ms,
                    f"{guardrail_name} execution timed out after {timeout_ms}ms",
                )
            else:
                result = guardrail(content)

            print(f"[JAF:GUARDRAILS] {guardrail_name} completed: {result}")
            return {"result": result, "guardrail_index": index}

        except Exception as error:
            error_message = str(error)
            print(f"[JAF:GUARDRAILS] {guardrail_name} failed: {error_message}")

            return {
                "result": ValidValidationResult(),
                "guardrail_index": index,
                "warning": f"Guardrail {index + 1} failed but was skipped: {error_message}",
            }

    try:
        # Run all guardrails in parallel
        tasks = [run_guardrail(guardrail, i) for i, guardrail in enumerate(input_guardrails)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        print("[JAF:GUARDRAILS] Input guardrails completed. Checking results...")

        warnings = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_message = str(result)
                print(f"[JAF:GUARDRAILS] Input guardrail {i + 1} promise rejected: {error_message}")
                warnings.append(f"Guardrail {i + 1} failed: {error_message}")
                continue

            if "warning" in result:
                warnings.append(result["warning"])

            validation_result = result["result"]
            if not validation_result.is_valid:
                error_message = getattr(validation_result, "error_message", "Guardrail violation")
                print(
                    f"🚨 Input guardrail {result['guardrail_index'] + 1} violation: {error_message}"
                )
                if config.on_event:
                    config.on_event(
                        GuardrailViolationEvent(
                            data=GuardrailViolationEventData(stage="input", reason=error_message)
                        )
                    )
                return validation_result

        if warnings:
            print(f"[JAF:GUARDRAILS] {len(warnings)} guardrail warnings: {warnings}")

        print("✅ All input guardrails passed.")
        return ValidValidationResult()

    except Exception as error:
        print(f"[JAF:GUARDRAILS] Catastrophic failure in input guardrail execution: {error}")
        return ValidValidationResult()  # Fail gracefully


async def execute_output_guardrails(
    output_guardrails: List[Guardrail], output: Any, config: RunConfig
) -> ValidationResult:
    """Execute output guardrails sequentially."""
    if not output_guardrails:
        return ValidValidationResult()

    print(f"[JAF:GUARDRAILS] Checking {len(output_guardrails)} output guardrails")

    for i, guardrail in enumerate(output_guardrails):
        guardrail_name = f"output-guardrail-{i + 1}"

        try:
            timeout_ms = OUTPUT_GUARDRAIL_TIMEOUT_MS

            if asyncio.iscoroutinefunction(guardrail):
                result = await _with_timeout(
                    guardrail(output),
                    timeout_ms,
                    f"{guardrail_name} execution timed out after {timeout_ms}ms",
                )
            else:
                result = guardrail(output)

            if not result.is_valid:
                error_message = getattr(result, "error_message", "Guardrail violation")
                print(f"🚨 {guardrail_name} violation: {error_message}")
                if config.on_event:
                    config.on_event(
                        GuardrailViolationEvent(
                            data=GuardrailViolationEventData(stage="output", reason=error_message)
                        )
                    )
                return result

            print(f"✅ {guardrail_name} passed")

        except Exception as error:
            error_message = str(error)
            print(f"[JAF:GUARDRAILS] {guardrail_name} failed: {error_message}")

            is_system_error = "Timeout" in error_message or "Circuit breaker" in error_message

            if is_system_error:
                print(
                    f"[JAF:GUARDRAILS] {guardrail_name} system error, allowing output: {error_message}"
                )
                continue
            else:
                if config.on_event:
                    config.on_event(
                        GuardrailViolationEvent(
                            data=GuardrailViolationEventData(stage="output", reason=error_message)
                        )
                    )
                return InvalidValidationResult(error_message=error_message)

    print("✅ All output guardrails passed")
    return ValidValidationResult()


def cleanup_circuit_breakers() -> None:
    """Clean up old circuit breakers."""
    to_remove = []
    for key, breaker in _circuit_breakers.items():
        if breaker.should_be_cleaned_up(CIRCUIT_BREAKER_CLEANUP_MAX_AGE):
            to_remove.append(key)

    for key in to_remove:
        del _circuit_breakers[key]


class GuardrailCacheManager:
    """Manager for guardrail cache operations."""

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """Get cache statistics."""
        return _guardrail_cache.get_stats()

    @staticmethod
    def clear() -> None:
        """Clear cache."""
        _guardrail_cache.clear()

    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        """Get cache metrics."""
        stats = _guardrail_cache.get_stats()
        return {
            **stats,
            "utilization_percent": (stats["size"] / stats["max_size"]) * 100,
            "circuit_breakers_count": len(_circuit_breakers),
        }

    @staticmethod
    def log_stats() -> None:
        """Log cache statistics."""
        metrics = GuardrailCacheManager.get_metrics()
        print("[JAF:GUARDRAILS] Cache stats:", metrics)

    @staticmethod
    def cleanup() -> None:
        """Cleanup old entries."""
        cleanup_circuit_breakers()


# Export the cache manager
guardrail_cache_manager = GuardrailCacheManager()
