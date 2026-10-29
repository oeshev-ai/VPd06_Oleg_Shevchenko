"""Общие фикстуры тестов AssistAIClient.

Ключи тестовые. Файл ``.env`` не читается, внешние API не вызываются.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio

from Assist_ai import AssistAIClient, AssistAIConfig


@pytest_asyncio.fixture
async def assist_client() -> AsyncIterator[AssistAIClient]:
    """Клиент ProxyAPI с фейковыми ключами и лимитом в 5 вызовов инструментов."""
    client = AssistAIClient(
        AssistAIConfig(
            proxyapi_key="test-proxy-key",
            openweather_api_key="",
            exchangerate_api_key="test-fx-key",
            base_url="https://api.proxyapi.ru/v1",
            default_text_model="openai/gpt-4o",
            timeout=5.0,
            max_tool_iterations=5,
        )
    )
    try:
        yield client
    finally:
        await client.close()
