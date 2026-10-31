"""Unit-тесты AssistAIClient. HTTP только через respx, без живых API."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from respx import MockRouter

from Assist_ai import (
    AssistAIClient,
    AssistAIError,
    BaseTool,
    ImageModelScope,
    _ToolCallBuilder,
)
from bot import _extract_generated_images

CHAT_URL = "https://api.proxyapi.ru/v1/chat/completions"


class _RecordingTool(BaseTool):
    """Инструмент-заглушка: запоминает аргументы и не ходит в сеть."""

    def __init__(self, tool_name: str) -> None:
        self._name = tool_name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Тестовый инструмент."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return json.dumps({"ok": True, "echo": kwargs}, ensure_ascii=False)


def _completion(message: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": message,
            }
        ]
    }


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return _completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        },
        "tool_calls",
    )


def _sse(events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for event in events:
        lines.append("data: " + json.dumps(event, ensure_ascii=False))
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines)


def _stream_delta(
    delta: dict[str, Any], finish_reason: str | None = None
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ]
    }


def test_tool_arguments_parsing_robustness() -> None:
    """Строка JSON и уже готовый dict принимаются, список аргументов отвергается."""
    parsed = AssistAIClient._parse_tool_arguments(
        '{"from_currency": "USD", "amount": 100}'
    )
    assert parsed == {"from_currency": "USD", "amount": 100}

    ready = {"to_currency": "RUB", "amount": 1}
    assert AssistAIClient._parse_tool_arguments(ready) is ready
    assert AssistAIClient._parse_tool_arguments(None) == {}
    assert AssistAIClient._parse_tool_arguments("") == {}

    with pytest.raises(ValueError, match="JSON-объектом"):
        AssistAIClient._parse_tool_arguments('["USD", "RUB"]')
    with pytest.raises(ValueError, match="Неподдерживаемый тип"):
        AssistAIClient._parse_tool_arguments(["USD", "RUB"])


@pytest.mark.asyncio
async def test_chat_completion_tool_call_execution(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Вызов convert_currency дописывает role=tool и второй запрос возвращает текст."""
    arguments = {"from_currency": "USD", "to_currency": "RUB", "amount": 100}
    tool_body = json.dumps(
        {
            "from_currency": "USD",
            "to_currency": "RUB",
            "amount": 100,
            "rate": 90.0,
            "result": 9000.0,
        },
        ensure_ascii=False,
    )
    recorded = _RecordingTool("convert_currency")

    async def execute(**kwargs: Any) -> str:
        recorded.calls.append(kwargs)
        return tool_body

    assist_client._tools_registry["convert_currency"].execute = execute  # type: ignore[method-assign]

    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json=_tool_call("call_fx_1", "convert_currency", json.dumps(arguments)),
            ),
            httpx.Response(
                200,
                json=_completion(
                    {"role": "assistant", "content": "100 долларов — это 9000 рублей."},
                    "stop",
                ),
            ),
        ]
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Сколько 100 долларов в рублях?"}
    ]

    reply = await assist_client.chat_completion(messages)

    assert reply == "100 долларов — это 9000 рублей."
    assert route.call_count == 2
    assert recorded.calls == [arguments]
    tool_messages = [item for item in messages if item.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_fx_1"
    assert json.loads(tool_messages[0]["content"])["result"] == 9000.0
    second_messages = json.loads(route.calls[1].request.content)["messages"]
    assert any(item.get("role") == "tool" for item in second_messages)
    assert [item["role"] for item in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_stream_chat_tool_call_stitching(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Фрагменты delta.tool_calls склеиваются в get_weather_by_city до вызова инструмента."""
    fragments = [
        {"index": 0, "id": "call_w_1", "function": {"name": "get_we"}},
        {
            "index": 0,
            "function": {
                "name": "ather_by_city",
                "arguments": '{"city_name": "Москва"}',
            },
        },
    ]
    builder = _ToolCallBuilder()
    for fragment in fragments:
        builder.add(fragment)
    assert builder.calls()[0]["function"]["name"] == "get_weather_by_city"

    stub = _RecordingTool("get_weather_by_city")
    assist_client.register_tool(stub)
    first_stream = _sse(
        [
            _stream_delta({"tool_calls": [fragments[0]]}),
            _stream_delta({"tool_calls": [fragments[1]]}),
            _stream_delta({}, "tool_calls"),
        ]
    )
    second_stream = _sse(
        [
            _stream_delta({"content": "В Москве "}),
            _stream_delta({"content": "ясно."}, "stop"),
        ]
    )
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, text=first_stream),
            httpx.Response(200, text=second_stream),
        ]
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Какая погода в Москве?"}
    ]

    deltas = [delta async for delta in assist_client.stream_chat(messages)]

    assert route.call_count == 2
    assert stub.calls == [{"city_name": "Москва"}]
    tool_deltas = [delta for delta in deltas if delta.kind == "tool"]
    assert tool_deltas
    assert tool_deltas[0].text.startswith("get_weather_by_city")
    assert (
        "".join(delta.text for delta in deltas if delta.kind == "content")
        == "В Москве ясно."
    )
    tool_message = next(item for item in messages if item.get("tool_calls"))
    assert tool_message["tool_calls"][0]["function"]["name"] == "get_weather_by_city"


@pytest.mark.asyncio
async def test_http_timeout_handling(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Таймаут httpx не выходит наружу: _post_json поднимает AssistAIError."""
    respx_mock.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(AssistAIError) as captured:
        await assist_client._post_json(
            "chat/completions", {"model": "openai/gpt-4o", "messages": []}
        )

    error = captured.value
    assert isinstance(error.__cause__, httpx.TimeoutException)
    assert "Превышено время ожидания" in error.message
    assert "chat/completions" in error.message


@pytest.mark.asyncio
async def test_max_tool_iterations_limit(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Шестой ответ с tool_calls не запрашивается: на пятом круге цикл прерывается."""
    assert assist_client.config.max_tool_iterations == 5
    stub = _RecordingTool("convert_currency")
    assist_client.register_tool(stub)
    responses = [
        httpx.Response(
            200,
            json=_tool_call(
                f"call_{index}",
                "convert_currency",
                '{"from_currency": "USD", "to_currency": "RUB", "amount": 1}',
            ),
        )
        for index in range(6)
    ]
    route = respx_mock.post(CHAT_URL).mock(side_effect=responses)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Курс доллара"}]

    with pytest.raises(AssistAIError, match="Превышен лимит вызовов инструментов: 5"):
        await assist_client.chat_completion(messages)

    assert route.call_count == 5
    assert len(stub.calls) == 5
    assert sum(1 for item in messages if item.get("role") == "tool") == 5


@pytest.mark.asyncio
async def test_generate_image_tool_packs_url_and_base64(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Инструмент не бросает исключения и помечает url отдельно от base64."""
    respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"url": "https://cdn.example/cat.png"},
                    {"b64_json": "aGVsbG8="},
                ]
            },
        )
    )
    tool = assist_client._tools_registry["generate_image"]

    with ImageModelScope(None) as scope:
        payload = json.loads(await tool.execute(prompt="кот в шляпе", size="1024x1024"))

    assert payload["status"] == "success"
    assert payload["prompt"] == "кот в шляпе"
    assert payload["image_count"] == 2
    assert payload["images"] == [{"type": "url"}, {"type": "base64"}]
    assert "aGVsbG8=" not in json.dumps(payload)
    assert scope.images == [
        {"type": "url", "value": "https://cdn.example/cat.png"},
        {"type": "base64", "value": "aGVsbG8="},
    ]


@pytest.mark.asyncio
async def test_generate_image_tool_returns_error_json(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Сбой ProxyAPI остаётся JSON-ошибкой для модели, а не исключением."""
    respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(500, text="boom")
    )
    tool = assist_client._tools_registry["generate_image"]

    payload = json.loads(await tool.execute(prompt="схема"))

    assert "error" in payload
    assert "images" not in payload


@pytest.mark.asyncio
async def test_image_model_scope_used_when_tool_omits_model(
    assist_client: AssistAIClient,
) -> None:
    """Модель персоны подставляется, если инструмент вызван без model."""
    seen: dict[str, str | None] = {}

    async def _fake_generate(
        prompt: str,
        model: str | None = None,
        size: str = "1024x1024",
    ) -> list[str]:
        seen["model"] = model
        seen["prompt"] = prompt
        return ["https://cdn.example/cat.png"]

    assist_client.generate_image = _fake_generate  # type: ignore[method-assign]
    tool = assist_client._tools_registry["generate_image"]
    with ImageModelScope("openai/gpt-image-1.5"):
        payload = json.loads(await tool.execute(prompt="кот"))

    assert seen["model"] == "openai/gpt-image-1.5"
    assert payload["status"] == "success"


@pytest.mark.asyncio
async def test_image_size_scope_used_when_tool_omits_size(
    assist_client: AssistAIClient,
) -> None:
    """Размер персоны подставляется, если инструмент вызван без size."""
    seen: dict[str, str | None] = {}
    sizes: list[str] = []

    async def _fake_generate(
        prompt: str,
        model: str | None = None,
        size: str = "1024x1024",
    ) -> list[str]:
        sizes.append(size)
        seen["model"] = model
        return ["https://cdn.example/cat.png"]

    assist_client.generate_image = _fake_generate  # type: ignore[method-assign]
    tool = assist_client._tools_registry["generate_image"]
    with ImageModelScope("black-forest-labs/flux-3-image", "2K"):
        explicit = json.loads(await tool.execute(prompt="кот", size="1K"))
        scoped = json.loads(await tool.execute(prompt="кот"))

    assert explicit["status"] == "success"
    assert scoped["status"] == "success"
    assert sizes == ["1K", "2K"]
    assert seen["model"] == "black-forest-labs/flux-3-image"


@pytest.mark.asyncio
async def test_flux_pixel_size_becomes_resolution_tier(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """FLUX не принимает 1024x1024: в запрос уходит ярус 1K."""
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/cat.png"}]}
        )
    )

    images = await assist_client.generate_image(
        "кот",
        model="black-forest-labs/flux-3-image",
        size="1024x1024",
    )

    body = json.loads(route.calls.last.request.content)
    assert body["size"] == "1K"
    assert "aspect_ratio" not in body
    assert images == ["https://cdn.example/cat.png"]


@pytest.mark.asyncio
async def test_openai_image_keeps_pixel_size(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Пиксельный size остаётся у моделей, которым он нужен."""
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/cat.png"}]}
        )
    )

    await assist_client.generate_image(
        "кот",
        model="openai/gpt-image-1-mini",
        size="1024x1024",
    )

    body = json.loads(route.calls.last.request.content)
    assert body["size"] == "1024x1024"
    assert "aspect_ratio" not in body


def test_extract_generated_images_matches_tool_call_id() -> None:
    """Бот берёт только картинки вызова generate_image, не чужие tool-сообщения."""
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_img",
                    "type": "function",
                    "function": {"name": "generate_image", "arguments": "{}"},
                },
                {
                    "id": "call_fx",
                    "type": "function",
                    "function": {"name": "convert_currency", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_fx",
            "content": json.dumps(
                {"images": [{"type": "url", "value": "https://nope"}]}
            ),
        },
        {
            "role": "tool",
            "tool_call_id": "call_img",
            "content": json.dumps(
                {
                    "status": "success",
                    "images": [{"type": "url", "value": "https://cdn.example/cat.png"}],
                }
            ),
        },
        {"role": "tool", "tool_call_id": "call_img", "content": "не json"},
    ]

    assert _extract_generated_images(messages) == [
        {"type": "url", "value": "https://cdn.example/cat.png"}
    ]
