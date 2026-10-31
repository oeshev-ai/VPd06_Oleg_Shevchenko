"""История темы после generate_image: tool-раунд без base64 и факт успеха."""

from __future__ import annotations

import json

import pytest

from bot import (
    _compact_tool_content,
    _image_completion_text,
    _remember_model_turn,
    _sanitize_history_message,
)
from topic_history import InMemoryTopicStorage

AID = "standard"


def test_compact_tool_content_strips_image_values() -> None:
    raw = json.dumps(
        {
            "status": "success",
            "prompt": "кот",
            "images": [{"type": "base64", "value": "a" * 100}],
        },
        ensure_ascii=False,
    )

    compacted = json.loads(_compact_tool_content(raw))

    assert compacted["status"] == "success"
    assert compacted["prompt"] == "кот"
    assert compacted["images"] == [{"type": "base64"}]
    assert "value" not in compacted["images"][0]
    assert "note" in compacted


def test_sanitize_drops_reasoning_and_keeps_tool_calls() -> None:
    message = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "думаю",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "arguments": '{"prompt":"кот"}',
                },
            }
        ],
    }

    stored = _sanitize_history_message(message)

    assert stored is not None
    assert "reasoning_content" not in stored
    assert stored["tool_calls"][0]["function"]["name"] == "generate_image"


def test_sanitize_ignores_user_messages() -> None:
    assert _sanitize_history_message({"role": "user", "content": "hi"}) is None


def test_image_completion_text_uses_prompt() -> None:
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps(
                {
                    "status": "success",
                    "prompt": "рыжий кот",
                    "images": [{"type": "url"}],
                },
                ensure_ascii=False,
            ),
        }
    ]

    text = _image_completion_text(messages, 1)

    assert "рыжий кот" in text
    assert "generate_image" in text


@pytest.mark.asyncio
async def test_remember_model_turn_stores_tool_round_without_base64() -> None:
    storage = InMemoryTopicStorage()
    topic_id = await storage.create_topic(1, AID, "Рисование")
    await storage.append_message(
        1, AID, topic_id, {"role": "user", "content": "нарисуй кота"}, 50
    )
    new_messages = [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "секрет",
            "tool_calls": [
                {
                    "id": "call_img",
                    "type": "function",
                    "function": {
                        "name": "generate_image",
                        "arguments": '{"prompt":"кот"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_img",
            "content": json.dumps(
                {
                    "status": "success",
                    "prompt": "кот",
                    "images": [{"type": "base64", "value": "AAAA"}],
                },
                ensure_ascii=False,
            ),
        },
        {"role": "assistant", "content": ""},
    ]

    await _remember_model_turn(
        1,
        AID,
        topic_id,
        storage,
        new_messages,
        answer="",
        images=[{"type": "base64", "value": "AAAA"}],
    )

    history = await storage.get_history(1, AID, topic_id)
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["tool_calls"][0]["function"]["name"] == "generate_image"
    assert "reasoning_content" not in history[1]
    tool_payload = json.loads(history[2]["content"])
    assert tool_payload["images"] == [{"type": "base64"}]
    assert "value" not in tool_payload["images"][0]
    assert history[3]["role"] == "assistant"
    assert "кот" in history[3]["content"]
    assert "generate_image" in history[3]["content"]
    await storage.close()
