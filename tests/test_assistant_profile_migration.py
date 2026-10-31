"""Старые профили без голосовых полей читаются с дефолтами."""

from __future__ import annotations

from assistant_storage import DEFAULT_SYSTEM_PROMPT, AssistantProfile


def test_old_profile_gets_voice_defaults() -> None:
    profile = AssistantProfile.model_validate(
        {
            "id": "abc12345",
            "name": "Старый",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "text_model": "openai/gpt-4o-mini",
            "image_model": None,
            "image_size": None,
        }
    )

    assert profile.voice_mode == "auto"
    assert profile.tts_model is None
    assert profile.tts_voice is None
    assert profile.tts_format is None
    assert profile.stt_language is None
