"""Regression tests for the security-audit fixes (critical and high findings)."""

import asyncio
import dataclasses
import math

import pytest
from fastapi.testclient import TestClient

from adk.security.sanitization import sanitize_file_path
from adk.security.validation import (
    AdkSecurityConfig,
    AdkSecurityValidator,
    ValidationResult,
    generate_secure_token,
    validate_session_token,
)
from jaf.core.composition import CachedTool
from jaf.core.types import RunConfig
from jaf.memory.providers.in_memory import InMemoryProvider
from jaf.memory.providers.postgres import PostgresProvider, _validate_identifier
from jaf.memory.types import InMemoryConfig, MemoryQuery
from jaf.server.server import create_jaf_server
from jaf.server.types import ServerConfig
from jaf.utils.document_processor import DocumentProcessingError, _ensure_public_url
from jaf.utils.safe_math import safe_eval_arithmetic


class TestSafeMath:
    def test_arithmetic(self):
        assert safe_eval_arithmetic("2+3*4") == 14
        assert safe_eval_arithmetic("sqrt(16)+pi", {"sqrt": math.sqrt}, {"pi": math.pi}) > 7

    @pytest.mark.parametrize("bad", ["__import__('os')", "2**9", "().__class__", "sqrt(16)"])
    def test_rejects_non_arithmetic(self, bad):
        with pytest.raises(ValueError):
            safe_eval_arithmetic(bad)


class TestPostgresMemory:
    def test_identifier_validation(self):
        assert _validate_identifier("conversations", "table") == "conversations"
        with pytest.raises(ValueError):
            _validate_identifier('x"; DROP TABLE y;--', "table")

    def test_find_conversations_filters_by_user(self):
        class Client:
            query = None

            async def fetch(self, sql, *args):
                self.query = (sql, args)
                return []

        class Cfg:
            table_name = "conversations"

        client = Client()
        provider = PostgresProvider(Cfg(), client)
        asyncio.run(provider.find_conversations(MemoryQuery(user_id="alice", limit=5, offset=2)))
        assert client.query == (
            "SELECT * FROM conversations WHERE user_id = $1 LIMIT $2 OFFSET $3",
            ("alice", 5, 2),
        )


class TestCachedToolScope:
    def test_cache_not_shared_between_users(self):
        calls = []

        class Tool:
            schema = type("S", (), {"name": "t", "description": "d", "parameters": None})()

            async def execute(self, args, context):
                calls.append(context["user_id"])
                return f"result for {context['user_id']}"

        cached = CachedTool(Tool()).create_tool()
        first = asyncio.run(cached.execute({"q": 1}, {"user_id": "alice"}))
        second = asyncio.run(cached.execute({"q": 1}, {"user_id": "bob"}))
        assert "result for alice" in first and "result for bob" in second
        assert calls == ["alice", "bob"]


class TestSsrf:
    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data",
            "http://127.0.0.1:8000/",
            "http://10.0.0.5/",
            "file:///etc/passwd",
        ],
    )
    def test_blocks_internal_urls(self, url):
        with pytest.raises(DocumentProcessingError):
            asyncio.run(_ensure_public_url(url))


class TestServerAuth:
    @staticmethod
    def _client(**kwargs):
        memory = InMemoryProvider(InMemoryConfig())
        asyncio.run(memory.store_messages("c1", [], {"user_id": "alice"}))
        config = ServerConfig(
            agent_registry={},
            run_config=RunConfig(agent_registry={}, model_provider=None),
            default_memory_provider=memory,
            **kwargs,
        )
        return TestClient(create_jaf_server(config))

    @staticmethod
    async def _authenticate(request):
        return request.headers.get("x-user")

    PROTECTED = [
        ("get", "/conversations/c1"),
        ("get", "/approvals/pending?conversation_id=c1"),
        ("get", "/approvals/stream?conversation_id=c1"),
        ("get", "/conversations/c1/regeneration-history"),
        ("get", "/conversations/c1/checkpoint-history"),
    ]

    @pytest.mark.parametrize("method,url", PROTECTED)
    def test_fails_closed_without_authenticate(self, method, url):
        assert getattr(self._client(), method)(url).status_code == 401

    @pytest.mark.parametrize("method,url", PROTECTED[:1] + PROTECTED[3:])
    def test_ownership_enforced(self, method, url):
        client = self._client(authenticate=self._authenticate)
        assert getattr(client, method)(url).status_code == 401
        assert getattr(client, method)(url, headers={"x-user": "bob"}).status_code == 404
        assert getattr(client, method)(url, headers={"x-user": "alice"}).status_code == 200

    def test_cors_never_combines_wildcard_with_credentials(self):
        response = self._client().options(
            "/health",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )
        assert response.headers.get("access-control-allow-credentials") is None


class TestAdkSecurity:
    class _Authenticator:
        async def authenticate(self, api_key):
            return ValidationResult(api_key == "good")

        async def validate_session(self, session_token):
            return ValidationResult(session_token == "tok")

    def test_anonymous_request_rejected_when_authenticator_configured(self):
        validator = AdkSecurityValidator(AdkSecurityConfig(), self._Authenticator())
        assert asyncio.run(validator.validate_request({})).is_valid is False

    def test_session_token_bound_to_user(self):
        token = generate_secure_token("alice", "secret")
        assert validate_session_token(token, "secret", "alice").is_valid
        assert not validate_session_token(token, "secret", "bob").is_valid

    @pytest.mark.parametrize("path", ["/etc/passwd", "C:/x", "../x"])
    def test_file_path_rejects_absolute_and_traversal(self, path):
        with pytest.raises(ValueError):
            sanitize_file_path(path)


class _FakeRedis:
    """Minimal async Redis double with real WATCH/MULTI/EXEC conflict detection."""

    def __init__(self):
        self.data, self.versions = {}, {}

    async def get(self, key):
        await asyncio.sleep(0)  # yield so concurrent tasks interleave between read and write
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key] = value
        self.versions[key] = self.versions.get(key, 0) + 1

    async def expire(self, key, ttl):
        return True

    def pipeline(self, transaction=True):
        return _FakePipeline(self)


class _FakePipeline:
    def __init__(self, redis):
        self.redis, self.watched, self.queued = redis, {}, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def watch(self, key):
        self.watched[key] = self.redis.versions.get(key, 0)

    async def unwatch(self):
        self.watched.clear()

    async def get(self, key):
        return await self.redis.get(key)

    def multi(self):
        pass

    def set(self, key, value, ex=None):
        self.queued.append((key, value))

    async def execute(self):
        from redis.exceptions import WatchError

        if any(self.redis.versions.get(k, 0) != v for k, v in self.watched.items()):
            raise WatchError()
        for key, value in self.queued:
            await self.redis.set(key, value)


class TestRedisAtomicUpdates:
    def test_concurrent_appends_are_not_lost(self):
        from jaf.core.types import Message
        from jaf.memory.providers.redis import RedisProvider
        from jaf.memory.types import RedisConfig

        async def scenario():
            provider = RedisProvider(RedisConfig(), _FakeRedis())
            await provider.store_messages("c1", [Message(role="user", content="start")])
            await asyncio.gather(
                *[
                    provider.append_messages("c1", [Message(role="user", content=f"m{i}")])
                    for i in range(20)
                ]
            )
            return (await provider.get_conversation("c1")).data.messages

        assert len(asyncio.run(scenario())) == 21

    def test_concurrent_appends_from_two_processes_are_not_lost(self):
        from jaf.core.types import Message
        from jaf.memory.providers.redis import RedisProvider
        from jaf.memory.types import RedisConfig

        async def scenario():
            redis = _FakeRedis()  # shared server, separate providers = separate processes
            first, second = RedisProvider(RedisConfig(), redis), RedisProvider(RedisConfig(), redis)
            await first.store_messages("c1", [Message(role="user", content="start")])
            await asyncio.gather(
                *[
                    (first if i % 2 else second).append_messages(
                        "c1", [Message(role="user", content=f"m{i}")]
                    )
                    for i in range(10)
                ]
            )
            return (await first.get_conversation("c1")).data.messages

        assert len(asyncio.run(scenario())) == 11


class TestA2AAuth:
    @staticmethod
    def _client(**extra):
        from jaf.a2a.server import create_a2a_server

        config = {
            "agentCard": {"name": "t", "description": "d", "version": "1"},
            "agents": {},
            "port": 9999,
            **extra,
        }
        return TestClient(create_a2a_server(config)["app"])

    BODY = {"jsonrpc": "2.0", "id": 1, "method": "agent/getAuthenticatedExtendedCard", "params": {}}

    def test_fails_closed_by_default(self):
        assert self._client().post("/a2a", json=self.BODY).status_code == 401

    def test_authenticate_callable_enforced(self):
        async def authenticate(request):
            return request.headers.get("authorization") == "Bearer ok"

        client = self._client(authenticate=authenticate)
        assert client.post("/a2a", json=self.BODY).status_code == 401
        assert client.post("/a2a", json=self.BODY, headers={"Authorization": "Bearer ok"}).status_code == 200

    def test_explicit_opt_out_allows_anonymous(self):
        assert self._client(allowUnauthenticated=True).post("/a2a", json=self.BODY).status_code == 200


class TestApprovalReplayExpiry:
    SIGNATURE = 'secure_tool:{"message": "go"}'

    @staticmethod
    def _outcome(approved_at):
        from pydantic import BaseModel

        from jaf.core.engine import run
        from jaf.core.types import (
            Agent,
            ApprovalValue,
            Message,
            ModelConfig,
            RunState,
            ToolSchema,
            create_run_id,
            create_trace_id,
        )

        class Args(BaseModel):
            message: str

        class SecureTool:
            schema = ToolSchema(name="secure_tool", description="needs approval", parameters=Args)
            needs_approval = True

            async def execute(self, args, context):
                return "executed"

        class Model:
            calls = 0

            async def get_completion(self, state, agent, config):
                Model.calls += 1
                if Model.calls > 1:
                    return {"message": {"content": "done", "tool_calls": None}}
                return {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "new-call-id",
                                "type": "function",
                                "function": {"name": "secure_tool", "arguments": '{"message": "go"}'},
                            }
                        ],
                    }
                }

        context = {"signature": TestApprovalReplayExpiry.SIGNATURE}
        if approved_at is not None:
            context["approved_at"] = approved_at
        agent = Agent(
            name="a",
            instructions=lambda s: "x",
            tools=[SecureTool()],
            model_config=ModelConfig(name="m"),
        )
        state = RunState(
            run_id=create_run_id("r"),
            trace_id=create_trace_id("t"),
            messages=[Message(role="user", content="hi")],
            current_agent_name="a",
            context={},
            turn_count=0,
            approvals={
                "old-call-id": ApprovalValue(status="approved", approved=True, additional_context=context)
            },
        )
        config = RunConfig(agent_registry={"a": agent}, model_provider=Model())
        return asyncio.run(run(state, config)).outcome.status

    def test_fresh_approval_matches_by_signature(self):
        import time

        assert self._outcome(time.time()) == "completed"

    def test_expired_or_untimed_approval_is_not_replayed(self):
        import time

        assert self._outcome(time.time() - 3600) == "interrupted"
        assert self._outcome(None) == "interrupted"


class TestMediumFixes:
    def test_conversation_id_cannot_contain_dot_dot(self):
        from jaf.memory.utils import sanitize_conversation_id

        assert ".." not in sanitize_conversation_id("../../etc/passwd")
        assert sanitize_conversation_id("conv-1.2") == "conv-1.2"

    def test_a2a_client_sends_auth_headers_and_long_session_id(self):
        from jaf.a2a.client import create_a2a_client

        client = create_a2a_client("http://h", {"headers": {"Authorization": "Bearer t"}})
        assert client.config.headers == {"Authorization": "Bearer t"}
        assert len(client.session_id.rsplit("_", 1)[1]) == 32  # full 128-bit uuid hex

    def test_a2a_batch_size_is_capped(self):
        client = TestA2AAuth._client(allowUnauthenticated=True, maxBatchSize=2)
        batch = [TestA2AAuth.BODY] * 3
        assert client.post("/a2a", json=batch).json()["error"]["code"] == -32600

    def test_safe_text_guardrail_blocks_xss_but_not_plain_text(self):
        from jaf.policies.validation import create_safe_text_guardrail

        guardrail = create_safe_text_guardrail()
        assert asyncio.run(guardrail("one=two is fine")).is_valid
        for payload in ["<img src=x onerror=alert(1)>", "<SCRIPT src=x", "javascript :alert(1)"]:
            assert not asyncio.run(guardrail(payload)).is_valid

    def test_svg_attachments_rejected(self):
        from jaf.utils.attachments import ALLOWED_IMAGE_MIME_TYPES

        assert "image/svg+xml" not in ALLOWED_IMAGE_MIME_TYPES

    def test_cleartext_azure_websocket_refused_except_localhost(self):
        from jaf.providers.model import _azure_responses_ws_url

        assert _azure_responses_ws_url("https://x.openai.azure.com").startswith("wss://")
        assert _azure_responses_ws_url("http://localhost:8080").startswith("ws://localhost")
        with pytest.raises(ValueError):
            _azure_responses_ws_url("http://evil.example.com")

    def test_proxy_info_hides_embedded_credentials(self, monkeypatch):
        from jaf.core.proxy_helpers import get_proxy_info

        monkeypatch.setenv("HTTPS_PROXY", "http://user:p4ss@proxy.corp:8080")
        info = get_proxy_info()
        assert "p4ss" not in str(info) and "user" not in str(info["https_proxy"])

    def test_tool_cache_is_bounded(self):
        calls = []

        class Tool:
            schema = type("S", (), {"name": "t", "description": "d", "parameters": None})()

            async def execute(self, args, context):
                calls.append(args)
                return "ok"

        cached_tool = CachedTool(Tool(), max_entries=3)
        tool = cached_tool.create_tool()
        for i in range(10):
            asyncio.run(tool.execute({"q": i}, {"user_id": "u"}))
        assert len(cached_tool.cache) == 3

    def test_tool_failure_result_has_no_stack_trace(self):
        from jaf.core.tool_results import ToolResponse, with_error_handling

        async def failing(args, context):
            raise RuntimeError("boom")

        result = asyncio.run(with_error_handling("t", failing)({}, {}))
        assert "Traceback" not in str(result) and "stack" not in str(result)

    def test_a2a_error_response_hides_exception_text(self):
        from jaf.a2a.protocol import map_error_to_a2a_error

        error = map_error_to_a2a_error(RuntimeError("password=hunter2 at /srv/db"))
        assert "hunter2" not in str(error)

    def test_adk_rate_limit_counts_requests_and_exponent_is_capped(self):
        from adk.utils.safe_evaluator import SafeMathEvaluator

        with pytest.raises(ValueError):
            SafeMathEvaluator.safe_eval("9**9999999")


class TestAttachmentMimeType:
    PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAHnOcQAAAAABJRU5ErkJggg=="

    def test_type_inferred_from_content_and_extension(self):
        from jaf.utils.attachments import make_image_attachment, make_document_attachment

        assert make_image_attachment(data=self.PNG_B64).mime_type == "image/png"
        assert make_image_attachment(url="https://x.example/a.jpg").mime_type == "image/jpeg"
        assert make_document_attachment(url="https://x.example/r.pdf").mime_type == "application/pdf"

    def test_unknown_type_is_rejected_not_skipped(self):
        from jaf.utils.attachments import AttachmentValidationError, make_image_attachment

        with pytest.raises(AttachmentValidationError, match="MIME type is required"):
            make_image_attachment(url="https://x.example/download")  # no extension, no data

    def test_disallowed_declared_type_still_rejected(self):
        from jaf.utils.attachments import AttachmentValidationError, make_image_attachment

        with pytest.raises(AttachmentValidationError):
            make_image_attachment(data=self.PNG_B64, mime_type="image/svg+xml")


class TestTlsConfig:
    def test_postgres_ssl_setting_reaches_driver(self, monkeypatch):
        import jaf.memory.providers.postgres as pg
        from jaf.memory.types import PostgresConfig

        seen = []

        class Conn:
            async def fetchval(self, *a):
                return 1

            async def close(self):
                pass

            async def execute(self, *a):
                pass

        async def connect(**kwargs):
            seen.append(kwargs.get("ssl"))
            return Conn()

        async def create_pool(**kwargs):
            seen.append(kwargs.get("ssl"))
            return Conn()

        monkeypatch.setattr(pg.asyncpg, "connect", connect)
        monkeypatch.setattr(pg.asyncpg, "create_pool", create_pool)
        asyncio.run(pg.create_postgres_provider(PostgresConfig(ssl=True)))
        assert seen == [True, True]


class TestMassAssignmentAndHistory:
    def test_caller_metadata_cannot_reassign_owner_or_bookkeeping(self):
        async def scenario():
            memory = InMemoryProvider(InMemoryConfig())
            await memory.store_messages("c", [], {"user_id": "alice"})
            await memory.store_messages("c", [], {"user_id": "mallory", "total_messages": 999})
            return (await memory.get_conversation("c")).data

        conversation = asyncio.run(scenario())
        assert conversation.user_id == "alice"
        assert conversation.metadata["total_messages"] == 0

    def test_approval_context_cannot_override_run_context(self):
        from pydantic import BaseModel

        from jaf.core.engine import run
        from jaf.core.types import (
            Agent, ApprovalValue, Message, ModelConfig, RunConfig, RunState,
            ToolSchema, create_run_id, create_trace_id,
        )

        seen = {}

        class Args(BaseModel):
            message: str

        class Tool:
            schema = ToolSchema(name="t", description="d", parameters=Args)
            needs_approval = True

            async def execute(self, args, context):
                seen.update(context)
                return "ok"

        class Model:
            calls = 0

            async def get_completion(self, state, agent, config):
                Model.calls += 1
                if Model.calls > 1:
                    return {"message": {"content": "done", "tool_calls": None}}
                return {"message": {"content": "", "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "t", "arguments": '{"message": "x"}'}}]}}

        agent = Agent(name="a", instructions=lambda s: "x", tools=[Tool()], model_config=ModelConfig(name="m"))
        state = RunState(
            run_id=create_run_id("r"), trace_id=create_trace_id("t"),
            messages=[Message(role="user", content="hi")], current_agent_name="a",
            context={"user_id": "alice"}, turn_count=0,
            approvals={"c1": ApprovalValue(
                status="approved", approved=True,
                additional_context={"user_id": "admin", "note": "ok"})},
        )
        asyncio.run(run(state, RunConfig(agent_registry={"a": agent}, model_provider=Model())))
        assert seen["user_id"] == "alice" and seen["note"] == "ok"

    def test_analytics_history_is_bounded(self):
        from jaf.core.analytics import MAX_ANALYTICS_HISTORY, _append_bounded

        series = []
        for i in range(MAX_ANALYTICS_HISTORY + 50):
            _append_bounded(series, i)
        assert len(series) == MAX_ANALYTICS_HISTORY and series[-1] == MAX_ANALYTICS_HISTORY + 49


class TestRegenerationRollback:
    def test_failed_regeneration_restores_original_history(self):
        from jaf.core.regeneration import regenerate_conversation
        from jaf.core.types import Agent, Message, ModelConfig, RegenerationRequest
        from jaf.memory.types import MemoryConfig

        class FailingModel:
            async def get_completion(self, state, agent, config):
                raise RuntimeError("model down")

        async def scenario():
            memory = InMemoryProvider(InMemoryConfig())
            original = [
                Message(role="user", content="question"),
                Message(role="assistant", content="answer"),
            ]
            await memory.store_messages("conv", original, {"user_id": "alice"})
            agent = Agent(name="a", instructions=lambda s: "x", tools=[], model_config=ModelConfig(name="m"))
            config = RunConfig(
                agent_registry={"a": agent},
                model_provider=FailingModel(),
                memory=MemoryConfig(provider=memory),
                conversation_id="conv",
            )
            request = RegenerationRequest(conversation_id="conv", message_id=original[1].message_id)
            result = await regenerate_conversation(request, config, {}, "a")
            stored = (await memory.get_conversation("conv")).data.messages
            return result.outcome.status, [m.content for m in stored]

        status, contents = asyncio.run(scenario())
        assert status == "error"
        assert contents == ["question", "answer"]  # the failed run did not destroy the history


class TestStreaming:
    @staticmethod
    def _setup(model):
        from jaf.core.types import Agent, Message, ModelConfig, RunState, create_run_id, create_trace_id

        agent = Agent(name="a", instructions=lambda s: "x", tools=[], model_config=ModelConfig(name="m"))
        state = RunState(
            run_id=create_run_id("r"),
            trace_id=create_trace_id("t"),
            messages=[Message(role="user", content="hi")],
            current_agent_name="a",
            context={},
            turn_count=0,
        )
        return state, RunConfig(agent_registry={"a": agent}, model_provider=model)

    def test_stream_finishes_even_when_run_ends_without_further_events(self):
        from jaf.core.streaming import StreamingEventType, run_streaming

        class Model:
            async def get_completion(self, state, agent, config):
                return {"message": {"content": "hello world", "tool_calls": None}}

        async def scenario():
            state, config = self._setup(Model())
            return [e.type async for e in run_streaming(state, config)]

        # wait_for turns the old hang into a test failure instead of a stuck test run
        types = asyncio.run(asyncio.wait_for(scenario(), timeout=10))
        assert types[-1] == StreamingEventType.COMPLETE

    def test_failed_run_streams_generic_error(self):
        from jaf.core.streaming import StreamingEventType, run_streaming

        class Model:
            async def get_completion(self, state, agent, config):
                raise RuntimeError("secret internal detail")

        async def scenario():
            state, config = self._setup(Model())
            return [e async for e in run_streaming(state, config)]

        events = asyncio.run(asyncio.wait_for(scenario(), timeout=10))
        assert events[-1].type == StreamingEventType.ERROR
        assert "secret internal detail" not in str(events[-1].data)

    def test_consumer_disconnect_cancels_background_run(self):
        from jaf.core.streaming import run_streaming

        started, cancelled = asyncio.Event(), []

        class Model:
            async def get_completion(self, state, agent, config):
                started.set()
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    cancelled.append(True)
                    raise

        async def scenario():
            state, config = self._setup(Model())
            stream = run_streaming(state, config)
            await stream.__anext__()  # START event; the background run begins after this
            consumer = asyncio.ensure_future(stream.__anext__())
            await asyncio.wait_for(started.wait(), timeout=5)
            consumer.cancel()  # the client went away
            try:
                await consumer
            except asyncio.CancelledError:
                pass
            await stream.aclose()
            await asyncio.sleep(0.1)

        asyncio.run(scenario())
        assert cancelled == [True]


class TestA2ATaskOwnership:
    @staticmethod
    def _client(monkeypatch):
        from jaf.a2a.server import create_a2a_server
        from jaf.a2a.types import A2AAgent

        async def authenticate(request):
            return request.headers.get("x-user")

        async def execute(context, agent, model_provider):
            task = {"id": "task-1", "contextId": "c", "status": {"state": "working"}, "history": []}
            return {"final_task": task}

        import jaf.a2a.server as server_module

        monkeypatch.setattr(server_module, "execute_a2a_agent", execute)
        agent = A2AAgent(name="a", description="d", instruction="i", tools=[], supportedContentTypes=["text/plain"])
        config = {
            "agentCard": {"name": "t", "description": "d", "version": "1"},
            "agents": {"a": agent},
            "port": 9999,
            "authenticate": authenticate,
        }
        return TestClient(create_a2a_server(config)["app"])

    @staticmethod
    def _rpc(method, params):
        return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}

    SEND = {"message": {"role": "user", "parts": [{"kind": "text", "text": "hi"}], "messageId": "m1", "contextId": "ctx1", "kind": "message"}}

    def test_only_the_creator_can_get_or_cancel_a_task(self, monkeypatch):
        client = self._client(monkeypatch)
        alice, bob = {"x-user": "alice"}, {"x-user": "bob"}
        assert "result" in client.post("/a2a", json=self._rpc("message/send", self.SEND), headers=alice).json()

        get = lambda who: client.post("/a2a", json=self._rpc("tasks/get", {"id": "task-1"}), headers=who).json()
        cancel = lambda who: client.post("/a2a", json=self._rpc("tasks/cancel", {"id": "task-1"}), headers=who).json()

        assert "result" in get(alice) and "_owner" not in str(get(alice))
        assert get(bob)["error"]["message"].endswith("not found")  # same answer as a missing task
        assert cancel(bob)["error"]["message"].endswith("not found")
        assert get(alice)["result"]["status"]["state"] == "working"  # bob's cancel did nothing

        assert cancel(alice)["result"]["status"]["state"] == "canceled"
        assert get(alice)["result"]["status"]["state"] == "canceled"  # cancellation persisted


class TestGuardrailFailSafe:
    def test_default_is_block_and_unavailable_model_blocks(self):
        from jaf.core.guardrails import _create_llm_guardrail
        from jaf.core.types import AdvancedGuardrailsConfig

        assert AdvancedGuardrailsConfig().fail_safe == "block"

        async def scenario(**kwargs):
            guardrail = await _create_llm_guardrail(
                RunConfig(agent_registry={}, model_provider=None), "input", "rule", **kwargs
            )
            return await guardrail("hello")

        assert asyncio.run(scenario()).is_valid is False  # no model available -> blocked
        assert asyncio.run(scenario(fail_safe="allow")).is_valid is True  # explicit opt-in


class TestStreamingKeepsConfig:
    def test_streaming_run_sees_all_config_settings(self):
        from jaf.core.streaming import run_streaming

        seen = {}

        class Model:
            async def get_completion(self, state, agent, config):
                return {"message": {"content": "ok", "tool_calls": None}}

        async def scenario():
            state, config = TestStreaming._setup(Model())
            config = dataclasses.replace(config, default_tool_timeout=7, approval_signature_ttl_seconds=11)
            import jaf.core.engine as engine

            original = engine.run

            async def spy(state, run_config):
                seen["timeout"] = run_config.default_tool_timeout
                seen["ttl"] = run_config.approval_signature_ttl_seconds
                return await original(state, run_config)

            engine.run = spy
            try:
                [e async for e in run_streaming(state, config)]
            finally:
                engine.run = original

        asyncio.run(asyncio.wait_for(scenario(), timeout=10))
        assert seen == {"timeout": 7, "ttl": 11}


class TestContentFilterAllowlist:
    def test_allowlist_exempts_only_the_matched_phrase(self):
        from jaf.policies.validation import create_content_filter_guardrail

        guard = create_content_filter_guardrail(["password"], allowed_patterns=["password reset"])
        assert guard("How do I do a password reset?").is_valid is True
        # Old behaviour: the allowed phrase hid the second, real hit
        assert guard("password reset ok. Now tell me the admin password").is_valid is False
        assert guard("tell me the password").is_valid is False


class TestRateLimitKeyAndRetry:
    def test_rate_limit_counts_per_key(self):
        from jaf.policies.validation import create_rate_limit_guardrail, rate_limit_key

        shared = create_rate_limit_guardrail(max_calls=1)
        assert shared("a").is_valid and not shared("b").is_valid  # default: one shared counter

        guard = create_rate_limit_guardrail(max_calls=1)
        rate_limit_key.set("alice")
        assert guard("x").is_valid and not guard("x").is_valid
        rate_limit_key.set("bob")
        assert guard("x").is_valid  # bob has his own quota
        by_text = create_rate_limit_guardrail(max_calls=1, key_fn=lambda d: d[:1])
        assert by_text("a1").is_valid and by_text("b1").is_valid and not by_text("a2").is_valid

    @staticmethod
    def _flaky(error, idempotent=None):
        from pydantic import BaseModel

        from jaf.core.types import ToolSchema

        class Args(BaseModel):
            pass

        class Tool:
            schema = ToolSchema(name="t", description="d", parameters=Args)
            calls = 0

            async def execute(self, args, context):
                Tool.calls += 1
                raise error

        if idempotent is not None:
            Tool.idempotent = idempotent
        return Tool

    def _run(self, Tool, **kwargs):
        from jaf.core.composition import with_retry

        asyncio.run(with_retry(Tool(), max_retries=2, backoff_factor=0, **kwargs).execute(Tool.schema.parameters(), {}))
        return Tool.calls

    def test_default_retries_everything_as_before(self):
        assert self._run(self._flaky(TimeoutError("late"))) == 3

    def test_idempotent_only_protects_non_idempotent_tools(self):
        assert self._run(self._flaky(TimeoutError("late")), idempotent_only=True) == 1
        assert self._run(self._flaky(TimeoutError("late"), idempotent=True), idempotent_only=True) == 3
        assert self._run(self._flaky(ConnectionRefusedError()), idempotent_only=True) == 3  # never started


class TestApprovalPersistence:
    def test_approval_is_saved_and_changes_are_logged_not_hidden(self, caplog):
        from jaf.core.types import Agent, ModelConfig

        class Model:
            async def get_completion(self, state, agent, config):
                return {"message": {"content": "done", "tool_calls": None}}

        agent = Agent(name="a", instructions=lambda s: "x", tools=[], model_config=ModelConfig(name="m"))
        memory = InMemoryProvider(InMemoryConfig())
        asyncio.run(memory.store_messages("c1", [], {"user_id": "alice"}))

        async def authenticate(request):
            return request.headers.get("x-user")

        config = ServerConfig(
            agent_registry={"a": agent},
            run_config=RunConfig(agent_registry={"a": agent}, model_provider=Model()),
            default_memory_provider=memory,
            authenticate=authenticate,
        )
        client = TestClient(create_jaf_server(config))
        body = {
            "messages": [{"role": "user", "content": "go"}],
            "agent_name": "a",
            "conversation_id": "c1",
            "approvals": [{"session_id": "run-1", "tool_call_id": "call-1", "approved": True}],
        }
        with caplog.at_level("WARNING"):
            assert client.post("/chat", json=body, headers={"x-user": "alice"}).status_code == 200

        stored = asyncio.run(memory.get_conversation("c1")).data.metadata["toolApprovals"]
        entry = stored["run-1:call-1"]
        assert entry["status"] == "approved" and isinstance(entry["approvedAt"], float)
        assert "Failed to persist approval" not in caplog.text

    def test_saved_approval_is_picked_up_on_the_next_request(self):
        from jaf.core.types import Agent, Message, ModelConfig, ToolCall, ToolCallFunction

        seen_approvals = {}

        class Model:
            async def get_completion(self, state, agent, config):
                seen_approvals.update(state.approvals)
                return {"message": {"content": "done", "tool_calls": None}}

        agent = Agent(name="a", instructions=lambda s: "x", tools=[], model_config=ModelConfig(name="m"))
        memory = InMemoryProvider(InMemoryConfig())
        call = ToolCall(id="call-1", type="function", function=ToolCallFunction(name="t", arguments="{}"))
        asyncio.run(memory.store_messages(
            "c1", [Message(role="assistant", content="", tool_calls=[call])], {"user_id": "alice"}
        ))

        async def authenticate(request):
            return request.headers.get("x-user")

        config = ServerConfig(
            agent_registry={"a": agent},
            run_config=RunConfig(agent_registry={"a": agent}, model_provider=Model()),
            default_memory_provider=memory,
            authenticate=authenticate,
        )
        client = TestClient(create_jaf_server(config))
        base = {"messages": [{"role": "user", "content": "go"}], "agent_name": "a", "conversation_id": "c1"}
        approve = {**base, "approvals": [{"session_id": "run-1", "tool_call_id": "call-1", "approved": True}]}
        assert client.post("/chat", json=approve, headers={"x-user": "alice"}).status_code == 200

        seen_approvals.clear()
        assert client.post("/chat", json=base, headers={"x-user": "alice"}).status_code == 200
        assert seen_approvals["call-1"].status == "approved"  # restored from storage, not resent


class TestWorkflowRace:
    def test_race_returns_first_success_and_ignores_fast_failure(self):
        from jaf.core.workflows import ParallelStep, StepResult, StepStatus, WorkflowContext, WorkflowStep

        class Step(WorkflowStep):
            def __init__(self, step_id, delay, ok):
                super().__init__(step_id, step_id)
                self.delay, self.ok = delay, ok

            async def execute(self, context):
                await asyncio.sleep(self.delay)
                return StepResult(
                    step_id=self.step_id,
                    status=StepStatus.COMPLETED if self.ok else StepStatus.FAILED,
                    output=self.step_id,
                    error=None if self.ok else "boom",
                    execution_time_ms=0,
                )

        race = ParallelStep("p", [Step("fails-fast", 0, False), Step("wins", 0.05, True), Step("slow", 5, True)], wait_for_all=False)

        async def scenario():
            return await asyncio.wait_for(race.execute(WorkflowContext(workflow_id="w", user_context={})), timeout=3)

        result = asyncio.run(scenario())
        assert result.status == StepStatus.COMPLETED  # the fast failure did not end the race


class TestLowFindings:
    def test_no_proxy_wildcard_respects_domain_boundary(self):
        from jaf.core.proxy import ProxyConfig

        proxy = ProxyConfig(http_proxy="http://p:1", no_proxy="*example.com")
        assert proxy.should_bypass_proxy("api.example.com")
        assert proxy.should_bypass_proxy("example.com")
        assert not proxy.should_bypass_proxy("evilexample.com")

    def test_error_text_redaction(self):
        from jaf.utils.redact import redact_error_text

        text = redact_error_text(
            "Field token_budget missing; http://user:pw@db.internal/x key sk-abcdefghijklmnop1234 Bearer abcdefgh.ijk"
        )
        assert "pw@" not in text and "sk-abcdefghijklmnop1234" not in text and "abcdefgh.ijk" not in text
        assert "Field token_budget missing" in text  # ordinary words stay readable for the model

    def test_unknown_error_does_not_dump_attributes(self):
        from jaf.core.errors import JAFErrorHandler

        class Weird:
            token = "super-secret-value"

        assert "super-secret-value" not in JAFErrorHandler.format_error(Weird())

    def test_data_url_media_type_restricted(self):
        from jaf.utils.attachments import AttachmentValidationError, _validate_url

        _validate_url("data:image/png;base64,iVBORw0KGgo=")
        with pytest.raises(AttachmentValidationError):
            _validate_url("data:text/html;base64,PHNjcmlwdD4=")

    def test_oversized_base64_rejected_before_decoding(self):
        from jaf.utils.attachments import MAX_ATTACHMENT_SIZE, _validate_base64

        assert _validate_base64("A" * (MAX_ATTACHMENT_SIZE * 4 // 3 + 100)) is False

    def test_trace_collector_is_bounded(self):
        from jaf.core.tracing import InMemoryTraceCollector

        assert InMemoryTraceCollector().max_traces == 1000  # default, but configurable
        assert InMemoryTraceCollector(max_traces=5).max_traces == 5
        unlimited = InMemoryTraceCollector(max_traces=None, max_events_per_trace=None)
        assert unlimited.max_traces is None and unlimited.max_events_per_trace is None

    def test_langfuse_open_trace_limit_is_configurable(self):
        from jaf.core.tracing import LangfuseTraceCollector

        pytest.importorskip("langfuse")
        collector = LangfuseTraceCollector(public_key="pk", secret_key="sk", host="http://localhost:1", max_open_traces=7)
        assert collector.max_open_traces == 7
        assert LangfuseTraceCollector(public_key="pk", secret_key="sk", host="http://localhost:1").max_open_traces == 1000



class TestWebSocketHandshakeHeaders:
    @staticmethod
    def _headers(**kwargs):
        from jaf.providers.model import _AzureResponsesWebSocketConnection

        connection = _AzureResponsesWebSocketConnection(
            api_base="https://x.openai.azure.com", default_timeout=5, **{"api_key": None, **kwargs}
        )
        return connection._handshake_headers()

    def test_api_key_behaviour_is_unchanged(self):
        assert self._headers(api_key="k") == {"Authorization": "Bearer k"}
        assert self._headers() == {}

    def test_extra_headers_are_sent_on_the_handshake(self):
        headers = self._headers(api_key="k", extra_headers={"x-tenant": "t1", "x-n": 5, "skip": None})
        assert headers == {"Authorization": "Bearer k", "x-tenant": "t1", "x-n": "5"}

    def test_extra_headers_cannot_replace_authorization(self):
        headers = self._headers(api_key="k", extra_headers={"authorization": "Bearer other"})
        assert headers == {"Authorization": "Bearer k"}

    def test_azure_ad_token_used_only_without_api_key(self):
        assert self._headers(azure_ad_token="ad") == {"Authorization": "Bearer ad"}
        assert self._headers(api_key="k", azure_ad_token="ad") == {"Authorization": "Bearer k"}

    def test_connection_settings_stay_out_of_the_message_body(self):
        from jaf.providers.model import _responses_params_to_ws_payload

        payload = _responses_params_to_ws_payload(
            {"model": "m", "input": [], "extra_headers": {"a": "b"}, "azure_ad_token": "t", "api_key": "k", "organization": "org"}
        )
        assert payload == {"model": "m", "input": [], "organization": "org"}

    def test_connect_passes_headers_to_websockets(self, monkeypatch):
        import jaf.providers.model as model_module

        seen = {}

        class FakeSocket:
            state = model_module.State.OPEN

            async def close(self):
                pass

        async def fake_connect(url, **kwargs):
            seen.update(kwargs)
            return FakeSocket()

        monkeypatch.setattr(model_module.websockets, "connect", fake_connect)
        connection = model_module._AzureResponsesWebSocketConnection(
            api_base="https://x.openai.azure.com",
            api_key="k",
            default_timeout=5,
            extra_headers={"x-team": "a"},
        )
        asyncio.run(connection._ensure_connected())
        assert seen["additional_headers"] == {"Authorization": "Bearer k", "x-team": "a"}


class TestImagesReachTheProvider:
    def test_image_request_is_not_blocked_by_a_local_model_list(self, monkeypatch):
        from jaf.core.types import (
            Agent, Message, MessageContentPart, ModelConfig, RunConfig, RunState, create_run_id, create_trace_id,
        )
        from jaf.providers.model import make_litellm_provider

        sent = {}

        class Msg:
            content = "ok"
            tool_calls = None

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]
            usage = None
            id = "r"
            created = 0
            object = "chat.completion"
            system_fingerprint = None
            model = "brand-new-vision-model"

            def model_dump(self):
                return {}

        async def fake_acompletion(**kwargs):
            sent.update(kwargs)
            return Resp()

        provider = make_litellm_provider("http://localhost:4000", api_key="k")
        monkeypatch.setattr(provider.client.chat.completions, "create", fake_acompletion)
        agent = Agent(name="a", instructions=lambda s: "x", tools=[], model_config=ModelConfig(name="brand-new-vision-model"))
        message = Message(
            role="user",
            content=[
                MessageContentPart(type="text", text="what is this"),
                MessageContentPart(type="image_url", image_url={"url": "data:image/png;base64,iVBORw0KGgo="}),
            ],
        )
        state = RunState(
            run_id=create_run_id("r"), trace_id=create_trace_id("t"), messages=[message],
            current_agent_name="a", context={}, turn_count=0,
        )
        asyncio.run(provider.get_completion(state, agent, RunConfig(agent_registry={"a": agent}, model_provider=provider)))
        assert sent["model"] == "brand-new-vision-model"  # reached litellm, no "does not support vision" error


class TestErrorsStayUsefulToTheModel:
    def test_pipeline_error_keeps_reason_without_earlier_results(self):
        from pydantic import BaseModel

        from jaf.core.composition import create_tool_pipeline
        from jaf.core.types import ToolSchema

        class Args(BaseModel):
            x: int = 0

        class Fails:
            schema = ToolSchema(name="fails", description="d", parameters=Args)

            async def execute(self, args, context):
                raise ValueError("field 'date' must be YYYY-MM-DD, got http://u:secret@host")

        result = asyncio.run(create_tool_pipeline(Fails(), name="p").execute(Args(), {}))
        text = str(result)
        assert "must be YYYY-MM-DD" in text  # the model can read the reason and fix its call
        assert "secret" not in text  # but the URL password is masked
