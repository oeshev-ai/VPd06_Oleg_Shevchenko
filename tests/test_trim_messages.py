"""Обрезка истории не должна оставлять сирот role=tool."""

from __future__ import annotations

from history_storage import trim_messages


def test_trim_does_not_start_with_orphan_tool() -> None:
    messages = [
        {"role": "user", "content": "нарисуй"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "generate_image", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "готово"},
    ]

    out = trim_messages(messages, 3)

    assert out == [] or out[0].get("role") == "user"
    assert not out or out[0].get("role") != "tool"


def test_trim_keeps_latest_user_turn() -> None:
    messages = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": "b"},
    ]

    out = trim_messages(messages, 3)

    assert [(item["role"], item["content"]) for item in out] == [
        ("user", "2"),
        ("assistant", "b"),
    ]


def test_trim_keeps_system_and_aligns_to_user() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "draw"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "generate_image", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "ok"},
    ]

    out = trim_messages(messages, 4)

    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    assert out[1]["content"] == "next"
    assert not any(item.get("role") == "tool" for item in out)
