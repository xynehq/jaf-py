"""Transport resilience of the Azure Responses WebSocket connection.

Covers the keepalive configuration and the retry-once-before-first-event rule in
`_AzureResponsesWebSocketConnection.create_response`.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.frames import Close
from websockets.protocol import State

from jaf.providers.model import (
    AzureResponsesWebSocketConnection,
)


def _closed() -> ConnectionClosed:
    """The error websockets raises when its keepalive kills the socket."""
    return ConnectionClosed(None, None)


class _FakeSocket:
    """Scripted socket. `recv_script` entries are either dicts (sent as JSON
    events) or exceptions (raised). `send_error` fails the initial send."""

    def __init__(self, recv_script=None, send_error=None):
        self._recv_script = list(recv_script or [])
        self._send_error = send_error
        self.closed = False
        self.state = State.OPEN
        self.sent = []

    async def send(self, _payload):
        self.sent.append(_payload)
        if self._send_error is not None:
            raise self._send_error

    async def recv(self):
        if not self._recv_script:
            raise AssertionError("recv() called more times than scripted")
        item = self._recv_script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return json.dumps(item)

    async def close(self):
        self.closed = True
        self.state = State.CLOSED


def _connection_with(sockets, **kwargs):
    """An AzureResponsesWebSocketConnection whose Nth connect yields sockets[N]."""
    connection = AzureResponsesWebSocketConnection(
        api_base="https://example.openai.azure.com",
        api_key="k",
        default_timeout=None,
        **kwargs,
    )
    connect = AsyncMock(side_effect=list(sockets))
    return connection, connect


async def _drain(connection, payload=None):
    return [event async for event in connection.create_response(payload or {"model": "m"})]


@pytest.mark.asyncio
async def test_keepalive_defaults_allow_more_slack_than_the_library():
    """A busy event loop must not trip the 1011 keepalive close. The library
    default ping_timeout is 20s; ours is deliberately larger."""
    socket = _FakeSocket(recv_script=[{"type": "response.completed"}])
    connection, connect = _connection_with([socket])

    with patch("jaf.providers.model.websockets.connect", connect):
        await _drain(connection)

    kwargs = connect.await_args.kwargs
    assert kwargs["ping_interval"] == 20
    assert kwargs["ping_timeout"] == 60


@pytest.mark.asyncio
@pytest.mark.parametrize("interval, timeout", [(5, 123), (None, 60), (20, None)])
async def test_keepalive_overrides_reach_each_new_socket(interval, timeout):
    dead = _FakeSocket(send_error=_closed())
    healthy = _FakeSocket(recv_script=[{"type": "response.completed"}])
    connection, connect = _connection_with(
        [dead, healthy], ping_interval=interval, ping_timeout=timeout
    )

    with patch("jaf.providers.model.websockets.connect", connect):
        await _drain(connection)

    assert connect.await_count == 2
    for call in connect.await_args_list:
        assert call.kwargs["ping_interval"] == interval
        assert call.kwargs["ping_timeout"] == timeout


@pytest.mark.asyncio
async def test_retries_on_a_fresh_connection_when_send_dies_before_any_event():
    dead = _FakeSocket(send_error=_closed())
    healthy = _FakeSocket(recv_script=[{"type": "response.completed"}])
    connection, connect = _connection_with([dead, healthy])

    with patch("jaf.providers.model.websockets.connect", connect):
        events = await _drain(connection)

    assert [e["type"] for e in events] == ["response.completed"]
    assert connect.await_count == 2, "should have opened a fresh socket"
    assert dead.closed, "the dead socket must be closed, not returned to the pool"


@pytest.mark.asyncio
async def test_retries_when_the_socket_dies_on_the_first_recv():
    dead = _FakeSocket(recv_script=[_closed()])
    healthy = _FakeSocket(recv_script=[{"type": "response.completed"}])
    connection, connect = _connection_with([dead, healthy])

    with patch("jaf.providers.model.websockets.connect", connect):
        events = await _drain(connection)

    assert [e["type"] for e in events] == ["response.completed"]
    assert connect.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["response.created", "response.output_text.delta"])
async def test_does_not_retry_once_an_event_reached_the_caller(event_type):
    """Mid-stream death must surface: the caller already holds partial state,
    so silently restarting the response would duplicate it."""
    socket = _FakeSocket(
        recv_script=[{"type": event_type}, _closed()]
    )
    connection, connect = _connection_with([socket])

    with (
        patch("jaf.providers.model.websockets.connect", connect),
        pytest.raises(ConnectionClosed),
    ):
        await _drain(connection)

    assert connect.await_count == 1, "must not reconnect after yielding"


@pytest.mark.asyncio
async def test_retries_at_most_once():
    first = _FakeSocket(recv_script=[_closed()])
    second = _FakeSocket(recv_script=[_closed()])
    connection, connect = _connection_with([first, second])

    with (
        patch("jaf.providers.model.websockets.connect", connect),
        pytest.raises(ConnectionClosed),
    ):
        await _drain(connection)

    assert connect.await_count == 2, "exactly one retry, then give up"


@pytest.mark.asyncio
async def test_server_side_error_event_is_not_retried():
    """response.failed is a semantic error from Azure, not a transport blip."""
    socket = _FakeSocket(recv_script=[{"type": "response.failed", "error": {"code": "bad"}}])
    connection, connect = _connection_with([socket])

    with (
        patch("jaf.providers.model.websockets.connect", connect),
        pytest.raises(RuntimeError, match="Azure Responses WebSocket error"),
    ):
        await _drain(connection)

    assert connect.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [State.CLOSING, State.CLOSED])
async def test_reconnects_before_sending_on_an_idle_closed_socket(state):
    old = _FakeSocket(recv_script=[{"type": "response.completed"}])
    healthy = _FakeSocket(recv_script=[{"type": "response.completed"}])
    connection, connect = _connection_with([old, healthy])

    with patch("jaf.providers.model.websockets.connect", connect):
        await _drain(connection)
        old.state = state
        await _drain(connection)

    assert len(old.sent) == 1, "must not attempt another send on the closed socket"
    assert len(healthy.sent) == 1
    assert old.closed
    assert connect.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["send", "receive"])
async def test_idle_disconnect_race_retries_exact_payload_and_logs_codes(phase, caplog):
    old = _FakeSocket(recv_script=[{"type": "response.completed"}])
    healthy = _FakeSocket(recv_script=[{"type": "response.completed"}])
    connection, connect = _connection_with([old, healthy])
    payload = {
        "model": "m",
        "previous_response_id": "resp_previous",
        "input": [{"role": "user", "content": "private prompt"}],
    }
    error = ConnectionClosed(None, Close(1011, "private close reason"))

    with patch("jaf.providers.model.websockets.connect", connect):
        await _drain(connection)
        # Still OPEN at the preflight check: failure races with send/receive.
        if phase == "send":
            old._send_error = error
        else:
            old._recv_script = [error]
        events = await _drain(connection, payload)

    assert events == [{"type": "response.completed"}]
    assert old.sent[-1] == healthy.sent[0]
    assert json.loads(healthy.sent[0]) == {"type": "response.create", **payload}
    assert connect.await_count == 2
    assert old.closed
    assert connection._dirty is False
    assert "closed before first event; retrying once" in caplog.text
    assert "sent_close_code=1011" in caplog.text
    assert "private prompt" not in caplog.text
    assert "private close reason" not in caplog.text


@pytest.mark.asyncio
async def test_successful_response_leaves_the_connection_reusable():
    """A completed response must not mark the socket dirty, or the next call
    throws away a perfectly good connection."""
    socket = _FakeSocket(
        recv_script=[{"type": "response.output_text.delta"}, {"type": "response.completed"}]
    )
    connection, connect = _connection_with([socket])

    with patch("jaf.providers.model.websockets.connect", connect):
        await _drain(connection)
        assert connection._dirty is False
        # a second call on the same (healthy) socket must not reconnect
        socket._recv_script = [{"type": "response.completed"}]
        await _drain(connection)

    assert connect.await_count == 1
