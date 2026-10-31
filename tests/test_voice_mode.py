"""Решение об озвучке ответа по voice_mode и типу ввода."""

from __future__ import annotations

from assistant_storage import DEFAULT_SYSTEM_PROMPT, AssistantProfile
from bot import _should_send_voice_reply


def _profile(mode: str) -> AssistantProfile:
    return AssistantProfile(
        name="Тест",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        voice_mode=mode,  # type: ignore[arg-type]
    )


def test_voice_mode_off_ignores_voice() -> None:
    assert _should_send_voice_reply(_profile("off"), "voice") is False


def test_voice_mode_auto_on_voice() -> None:
    assert _should_send_voice_reply(_profile("auto"), "voice") is True


def test_voice_mode_auto_on_text() -> None:
    assert _should_send_voice_reply(_profile("auto"), "text") is False


def test_voice_mode_always_on_text() -> None:
    assert _should_send_voice_reply(_profile("always"), "text") is True


def test_voice_mode_always_on_voice() -> None:
    assert _should_send_voice_reply(_profile("always"), "voice") is True
