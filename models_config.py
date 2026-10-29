"""Списки моделей ProxyAPI для кнопок бота.

Идентификаторы взяты из каталога https://proxyapi.ru/models/browse.
Каталог не запрашивается при каждом нажатии. Пустой выбор в профиле
означает модель по умолчанию из ``AssistAIConfig``.
"""

from __future__ import annotations

AVAILABLE_TEXT_MODELS: tuple[str, ...] = (
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-3.8-flash",
    "anthropic/claude-sonnet-5-5",
    "deepseek/deepseek-v4.1-flash",
    "deepseek/deepseek-chat",
)

# Картинки: только модели генерации из каталога. Дефолт клиента — gpt-image-1-mini.
AVAILABLE_IMAGE_MODELS: tuple[str, ...] = (
    "openai/gpt-image-1-mini",
    "openai/gpt-image-1.5",
    "openai/gpt-image-2",
    "black-forest-labs/flux-3-image",
)
