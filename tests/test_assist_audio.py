"""STT и TTS AssistAIClient. HTTP только через respx."""

from __future__ import annotations

import httpx
import pytest
from respx import MockRouter

from Assist_ai import AssistAIClient, AssistAIError, SpeechResult

TRANSCRIBE_URL = "https://api.proxyapi.ru/v1/audio/transcriptions"
SPEECH_URL = "https://api.proxyapi.ru/v1/audio/speech"


@pytest.mark.asyncio
async def test_transcribe_audio_success(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    respx_mock.post(TRANSCRIBE_URL).mock(
        return_value=httpx.Response(200, json={"text": "привет"})
    )

    text = await assist_client.transcribe_audio(b"fake-ogg", filename="voice.ogg")

    assert text == "привет"


@pytest.mark.asyncio
async def test_transcribe_audio_empty_text(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    respx_mock.post(TRANSCRIBE_URL).mock(
        return_value=httpx.Response(200, json={"text": ""})
    )

    text = await assist_client.transcribe_audio(b"fake-ogg", filename="voice.ogg")

    assert text == ""


@pytest.mark.asyncio
async def test_transcribe_audio_timeout(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    respx_mock.post(TRANSCRIBE_URL).mock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(AssistAIError) as captured:
        await assist_client.transcribe_audio(b"fake-ogg", filename="voice.ogg")

    assert "Превышено время ожидания" in captured.value.message
    assert "audio/transcriptions" in captured.value.message


@pytest.mark.asyncio
async def test_synthesize_speech_success(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    payload = b"ogg-bytes"
    respx_mock.post(SPEECH_URL).mock(return_value=httpx.Response(200, content=payload))

    speech = await assist_client.synthesize_speech("Привет")

    assert isinstance(speech, SpeechResult)
    assert speech.audio == payload
    assert speech.filename == "answer.ogg"
    assert speech.mime_type == "audio/ogg"


@pytest.mark.asyncio
async def test_synthesize_speech_empty_audio(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    respx_mock.post(SPEECH_URL).mock(return_value=httpx.Response(200, content=b""))

    with pytest.raises(AssistAIError, match="пустой аудиоответ"):
        await assist_client.synthesize_speech("Привет")
