"""
Production-ready Telegram-бот с интеграцией AssistAIClient.

Клиент LLM создаётся один раз в startup и живёт в ``workflow_data``.
Цикл Function Calling уже внутри ``stream_chat`` и здесь не повторяется.
Ответ языковой модели приходит потоком и сразу правится в сообщении Telegram.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import os
import shlex
import sys
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Literal

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import SendMessage
from aiogram.methods.base import TelegramMethod
from aiogram.types import (
    BotCommand,
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    MenuButtonCommands,
    Message,
    ReplyKeyboardMarkup,
    TelegramObject,
)
from dotenv import load_dotenv
from pydantic import ValidationError

from Assist_ai import AssistAIClient, AssistAIConfig, AssistAIError, ImageModelScope
from assistant_storage import (
    DEFAULT_ASSISTANT_ID,
    DEFAULT_ASSISTANT_NAME,
    AssistantProfile,
    AssistantStorage,
    AssistantStorageError,
    create_assistant_storage,
)
from history_storage import RedisConfig
from models_config import AVAILABLE_IMAGE_MODELS, AVAILABLE_TEXT_MODELS
from topic_history import (
    TopicConfig,
    TopicHistoryStorage,
    TopicStorageError,
    create_topic_storage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("TelegramBot")

TELEGRAM_TEXT_LIMIT = 4096
STREAM_EDIT_INTERVAL = 0.5
PAGE_SIZE = 10
WELCOME = (
    "Привет. Я ассистент: отвечаю на обычные вопросы и умею получать актуальные данные.\n\n"
    "Погода: текущая, прогноз, качество воздуха, сравнение городов.\n"
    "Валюты: конвертация суммы из одной валюты в другую.\n"
    "Изображения: генерация по текстовому описанию.\n\n"
    "🎤 Голосовые сообщения:\n"
    "Отправьте голосовое — я распознаю речь и отвечу.\n"
    "Если у персоны включён голосовой режим, ответ будет также озвучен.\n\n"
    "🎙 Настройки голоса:\n"
    "Откройте настройки персоны и выберите режим: off / auto / always.\n"
    "Команда /voice открывает те же настройки.\n\n"
    "Команды:\n"
    "/assistants — персоны и кнопка «Использовать»\n"
    '/create_assistant Имя "промпт" — новая персона\n'
    "/delete_assistant <id> — удалить персону\n"
    "/model — текстовая модель активного ассистента\n"
    "/image_model — модель картинок\n"
    "/voice — режим и голос TTS активной персоны\n"
    "/new_topic <название> — новая тема, контекст с нуля\n"
    "/topics — список тем\n"
    "/switch <id или название> — вернуться к теме\n"
    "/clear_topic — очистить только текущую тему\n"
    "/image <описание> — картинка\n"
    "/cancel — отменить ввод\n"
    "/help — эта справка\n\n"
    "Внизу кнопки: ассистенты, темы и настройки."
)
InputKind = Literal["text", "voice"]
MAX_VOICE_DURATION_SECONDS = 600
MAX_VOICE_SIZE_MB = 20
SHOW_RECOGNIZED_TEXT = True
VOICE_RECOGNIZED_PREVIEW_LIMIT = 500
TTS_MAX_CHARS = 3500
TTS_VOICES: tuple[str, ...] = ("alloy", "verse", "nova", "shimmer")
STT_LANGUAGES: tuple[tuple[str, str | None], ...] = (
    ("авто", None),
    ("ru", "ru"),
    ("en", "en"),
)

_MENU_ASSISTANTS = "Ассистенты"
_MENU_TOPICS = "Темы"
_MENU_SETTINGS = "Настройки"
MAIN_MENU_TEXT = {
    "assistants": "🤖 Ассистенты",
    "topics": "💬 Темы",
    "image": "🖼 Картинка",
    "settings": "⚙️ Настройки",
}
_OPENAI_IMAGE_SIZES: tuple[str, ...] = ("1024x1024", "1024x1792", "1792x1024")
_FLUX_IMAGE_SIZES: tuple[str, ...] = ("1K", "2K")
IMAGE_SIZES: tuple[str, ...] = (
    "1024x1024",
    "1024x1792",
    "1792x1024",
    "1K",
    "2K",
)
_FLOW_KEYS = (
    "flow_name",
    "flow_prompt",
    "flow_edit_id",
    "flow_text_model",
    "flow_image_model",
    "flow_topic_id",
    "flow_topic_name",
    "image_model",
    "image_size",
)
_SCREEN_ASSISTANTS = "screen_assistants"
_SCREEN_TOPICS = "screen_topics"
_SCREEN_SETTINGS = "screen_settings"
_SCREEN_PICKER = "screen_picker"
_SCREEN_IMAGE = "screen_image"
_MENU_ANCHOR = "menu_anchor"
_CHROME = "chrome_ids"
_CHROME_BUCKET: ContextVar[list[int] | None] = ContextVar("chrome_bucket", default=None)
_KEEP_OUTGOING: ContextVar[bool] = ContextVar("keep_outgoing", default=False)
_SCREEN_KEYS = (
    _SCREEN_ASSISTANTS,
    _SCREEN_TOPICS,
    _SCREEN_SETTINGS,
    _SCREEN_PICKER,
    _SCREEN_IMAGE,
)


class NewAssistantSG(StatesGroup):
    """Шаги создания новой персоны."""

    name = State()
    prompt = State()
    models = State()


class EditAssistantSG(StatesGroup):
    """Ввод нового системного промпта существующей персоны."""

    prompt = State()


class NewTopicSG(StatesGroup):
    """Шаг ввода названия новой темы."""

    name = State()


class RenameTopicSG(StatesGroup):
    """Шаг ввода нового названия темы."""

    name = State()


class ImageSG(StatesGroup):
    """Экран выбора модели и размера, затем описание картинки."""

    config = State()
    prompt = State()


MSG_UNAVAILABLE = "⚠️ Сервис временно недоступен. Попробуйте позже."
MSG_TIMEOUT = "⚠️ Превышено время ожидания."
MSG_INTERNAL = "❌ Произошла внутренняя ошибка."
MSG_IMAGE_FAILED = "Не удалось сгенерировать изображение."
MSG_HISTORY = "⚠️ Временные проблемы с историей диалога. Попробуйте позже."
MSG_PROFILE = "⚠️ Не удалось прочитать профиль ассистента. Попробуйте позже."


async def _active_profile(chat_id: int, storage: AssistantStorage) -> AssistantProfile:
    """Возвращает активную персону. Пустой чат получает «Стандартный»."""
    profile = await storage.get_active_assistant(chat_id)
    if profile is None:
        raise AssistantStorageError("Активный ассистент не найден.")
    return profile


def _text_model_of(profile: AssistantProfile, assistant: AssistAIClient) -> str:
    """Модель диалога: выбор профиля или дефолт клиента."""
    return profile.text_model or assistant.config.default_text_model


def _image_model_of(profile: AssistantProfile, assistant: AssistAIClient) -> str:
    """Модель картинки: выбор профиля или дефолт клиента."""
    return profile.image_model or assistant.config.default_image_model


def _menu_button(title: str, name: str) -> str:
    """Подпись нижней кнопки с именем. Telegram режет текст длиннее 64 символов."""
    prefix = f"{title} ("
    suffix = ")"
    room = 64 - len(prefix) - len(suffix)
    if room < 1:
        return title[:64]
    shown = name if len(name) <= room else f"{name[: room - 3]}..."
    return f"{prefix}{shown}{suffix}"


def _reply_menu(assistant_name: str, topic_name: str) -> ReplyKeyboardMarkup:
    """Три кнопки: персоны, темы активной персоны и её настройки."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text=_menu_button(_MENU_ASSISTANTS, assistant_name)),
                KeyboardButton(text=_menu_button(_MENU_TOPICS, topic_name)),
            ],
            [KeyboardButton(text=_MENU_SETTINGS)],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def _is_flux_image_model(model_id: str) -> bool:
    """FLUX просит ярус разрешения, остальные модели из каталога — пиксели."""
    folded = model_id.casefold()
    return "flux" in folded or "black-forest" in folded


def _size_choices(model_id: str) -> tuple[str, ...]:
    if _is_flux_image_model(model_id):
        return _FLUX_IMAGE_SIZES
    return _OPENAI_IMAGE_SIZES


def _default_image_size(model_id: str) -> str:
    return "1K" if _is_flux_image_model(model_id) else "1024x1024"


def _display_image_size(profile: AssistantProfile, assistant: AssistAIClient) -> str:
    """Размер для экрана настроек. Пустой или чужой семье размер показывает дефолт."""
    model_id = _image_model_of(profile, assistant)
    if profile.image_size in _size_choices(model_id):
        return profile.image_size or _default_image_size(model_id)
    return _default_image_size(model_id)


def _stored_image_size(profile: AssistantProfile) -> str:
    """Размер для генерации. Пустое поле персоны остаётся 1024x1024."""
    if isinstance(profile.image_size, str) and profile.image_size.strip():
        return profile.image_size.strip()
    return "1024x1024"


async def _menu_for(
    chat_id: int,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    *,
    ensure_topic: bool = False,
) -> ReplyKeyboardMarkup:
    """Собирает нижнее меню по активной персоне и её теме."""
    assistant_name = DEFAULT_ASSISTANT_NAME
    assistant_id = DEFAULT_ASSISTANT_ID
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        assistant_name = profile.name
        assistant_id = profile.id
    except AssistantStorageError:
        logger.exception("Меню без профиля, chat_id=%s", chat_id)
    topic_name = "General"
    try:
        if ensure_topic:
            topic_id = await topic_storage.ensure_active(
                chat_id, assistant_id, "General"
            )
        else:
            topic_id = await topic_storage.get_active_topic(chat_id, assistant_id)
        topics = await topic_storage.list_topics(chat_id, assistant_id)
        found = next(
            (item["name"] for item in topics if item["id"] == topic_id),
            "",
        )
        if found:
            topic_name = found
    except TopicStorageError:
        logger.exception("Меню без темы, chat_id=%s", chat_id)
    return _reply_menu(assistant_name, topic_name)


async def _prepare_messages(
    chat_id: int,
    assistant_id: str,
    user_text: str,
    topic_id: str,
    storage: TopicHistoryStorage,
    profile: AssistantProfile,
) -> list[dict[str, Any]]:
    """Добавляет реплику пользователя и собирает запрос для активной персоны.

    В модель уходит системный промпт профиля и история темы без старых
    ``system``. ``stream_chat`` дописывает tool-сообщения в переданный список,
    поэтому это копия: черновики не попадают в Redis.
    """
    limit = storage.max_length
    await storage.append_message(
        chat_id,
        assistant_id,
        topic_id,
        {"role": "user", "content": user_text},
        limit,
    )
    stored = await storage.get_history(chat_id, assistant_id, topic_id)
    dialog = [dict(item) for item in stored if item.get("role") != "system"]
    if len(dialog) != len(stored):
        # Промпт персоны подставляется только в запрос. Старый system в теме
        # занимает слот trim и расходится с активным ассистентом — убираем.
        await storage.clear_topic(chat_id, assistant_id, topic_id)
        for item in dialog:
            await storage.append_message(chat_id, assistant_id, topic_id, item, limit)
    return [{"role": "system", "content": profile.system_prompt}, *dialog]


def _compact_tool_content(content: str) -> str:
    """Убирает URL и base64 из результата generate_image, оставляя факт успеха."""
    try:
        payload = json.loads(content)
    except ValueError:
        return content
    if not isinstance(payload, dict):
        return content
    raw_images = payload.get("images")
    if not isinstance(raw_images, list):
        return content
    compacted = dict(payload)
    compacted["images"] = [
        {"type": item.get("type")}
        if isinstance(item, dict) and "type" in item
        else {"type": "unknown"}
        for item in raw_images
    ]
    if "status" not in compacted:
        compacted["status"] = "success"
    if "note" not in compacted:
        compacted["note"] = (
            "Изображение уже передано пользователю в Telegram. "
            "Не вызывай generate_image повторно для этого запроса."
        )
    return json.dumps(compacted, ensure_ascii=False)


def _sanitize_history_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Готовит сообщение раунда модели к записи в тему.

    System и user не сохраняем здесь: user уже записан в ``_prepare_messages``.
    Reasoning убираем, чтобы не тащить его в следующие запросы других моделей.
    """
    role = message.get("role")
    if role not in {"assistant", "tool"}:
        return None
    stored = {
        key: value
        for key, value in message.items()
        if key not in {"reasoning", "reasoning_content"}
    }
    if role == "tool" and isinstance(stored.get("content"), str):
        stored["content"] = _compact_tool_content(stored["content"])
    return stored


def _image_prompts_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Достаёт prompt из коротких JSON-ответов generate_image."""
    prompts: list[str] = []
    for item in messages:
        if item.get("role") != "tool":
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        prompt = payload.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            prompts.append(prompt.strip())
    return prompts


def _image_completion_text(messages: list[dict[str, Any]], image_count: int) -> str:
    """Текст-заглушка, если модель сгенерировала картинку без финальной реплики."""
    prompts = _image_prompts_from_messages(messages)
    if len(prompts) == 1:
        subject = f"по запросу: «{prompts[0]}»"
    elif prompts:
        joined = "; ".join(prompts[:3])
        subject = f"по запросам: «{joined}»"
    else:
        subject = f"({image_count} шт.)"
    return (
        f"Сгенерировано изображение {subject}. "
        "Картинка уже отправлена пользователю в Telegram. "
        "Повторно вызывать generate_image для этого запроса не нужно."
    )


async def _remember_model_turn(
    chat_id: int,
    assistant_id: str,
    topic_id: str,
    storage: TopicHistoryStorage,
    new_messages: list[dict[str, Any]],
    *,
    answer: str,
    images: list[dict[str, Any]],
) -> None:
    """Пишет в тему tool-раунд и финальный ответ без сырых картинок.

    Без этого при возврате к теме модель снова вызывает generate_image:
    в истории был только user, а успех инструмента терялся вместе с копией
    ``stream_chat``.
    """
    limit = storage.max_length
    to_store: list[dict[str, Any]] = []
    for item in new_messages:
        cleaned = _sanitize_history_message(item)
        if cleaned is not None:
            to_store.append(cleaned)

    final_text = answer.strip()
    if not final_text and images:
        final_text = _image_completion_text(new_messages, len(images))

    final_index: int | None = None
    for index in range(len(to_store) - 1, -1, -1):
        item = to_store[index]
        if item.get("role") == "assistant" and not item.get("tool_calls"):
            final_index = index
            break

    if final_index is not None:
        if final_text:
            to_store[final_index]["content"] = final_text
        else:
            # Пустой финал без картинок в историю не пишем.
            to_store.pop(final_index)
    elif final_text:
        to_store.append({"role": "assistant", "content": final_text})

    if not to_store:
        return

    for item in to_store:
        await storage.append_message(chat_id, assistant_id, topic_id, item, limit)


async def _active_topic(
    chat_id: int, assistant_id: str, storage: TopicHistoryStorage
) -> str:
    """Возвращает активную тему персоны. На первом сообщении создаёт General."""
    return await storage.ensure_active(chat_id, assistant_id, "General")


async def _reply_history_error(message: Message) -> None:
    logger.exception("Ошибка хранилища истории, chat_id=%s", message.chat.id)
    await message.answer(MSG_HISTORY)


async def _reply_profile_error(message: Message) -> None:
    logger.exception("Ошибка хранилища профилей, chat_id=%s", message.chat.id)
    await message.answer(MSG_PROFILE)


def _public_error(exc: BaseException) -> str:
    """Текст для пользователя. Подробности исключения наружу не выходят."""
    cause = exc.__cause__
    if isinstance(exc, httpx.TimeoutException) or isinstance(
        cause, httpx.TimeoutException
    ):
        return MSG_TIMEOUT
    if isinstance(exc, AssistAIError):
        return MSG_UNAVAILABLE
    return MSG_INTERNAL


def _log_handler_error(exc: BaseException) -> None:
    if isinstance(exc, AssistAIError):
        logger.error("AssistAI ошибка: %s", exc.message)
        return
    if isinstance(exc, httpx.TimeoutException):
        logger.error("Превышено время ожидания внешнего сервиса")
        return
    logger.exception("Непредвиденная ошибка")


async def _send_text(message: Message, text: str) -> None:
    """Отправляет текст частями: у Telegram лимит 4096 символов на сообщение."""
    payload = text.strip() or "Пустой ответ."
    offset = 0
    while offset < len(payload):
        await message.answer(payload[offset : offset + TELEGRAM_TEXT_LIMIT])
        offset += TELEGRAM_TEXT_LIMIT


async def _edit_or_answer(
    query: CallbackQuery, text: str, keyboard: InlineKeyboardMarkup | None = None
) -> None:
    """Редактирует сообщение кнопки. Повтор того же текста не считается ошибкой."""
    message = query.message
    if not isinstance(message, Message):
        return
    try:
        await message.edit_text(text, reply_markup=keyboard)
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return
        logger.warning("Не удалось отредактировать сообщение: %s", exc)


async def _edit_stream(status: Message, text: str, previous: str) -> str:
    """Обновляет черновик ответа. Повтор того же текста Telegram отвергает."""
    visible = text[:TELEGRAM_TEXT_LIMIT]
    if not visible or visible == previous:
        return previous
    try:
        await status.edit_text(visible)
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return previous
        logger.warning("Не удалось обновить сообщение: %s", exc)
        return previous
    return visible


async def _send_overflow(message: Message, text: str) -> None:
    """Доставляет хвост ответа, который не поместился в редактируемое сообщение."""
    rest = text[TELEGRAM_TEXT_LIMIT:]
    while rest:
        chunk = rest[:TELEGRAM_TEXT_LIMIT]
        rest = rest[TELEGRAM_TEXT_LIMIT:]
        await message.answer(chunk)


def _is_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _extract_generated_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Достаёт картинки из результатов инструмента ``generate_image``.

    Имя функции лежит в ``assistant.tool_calls``, а байты или URL — в
    последующем сообщении ``role: tool`` с тем же ``tool_call_id``.
    Битый JSON пропускается: текстовый ответ модели всё равно уйдёт пользователю.
    """
    names_by_id: dict[str, str] = {}
    for item in messages:
        if item.get("role") != "assistant":
            continue
        tool_calls = item.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            function = call.get("function")
            if not isinstance(call_id, str) or not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str):
                names_by_id[call_id] = name

    images: list[dict[str, Any]] = []
    for item in messages:
        if item.get("role") != "tool":
            continue
        call_id = item.get("tool_call_id")
        if not isinstance(call_id, str) or names_by_id.get(call_id) != "generate_image":
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except ValueError:
            logger.warning("Не удалось разобрать результат generate_image")
            continue
        if not isinstance(payload, dict):
            continue
        raw_images = payload.get("images")
        if not isinstance(raw_images, list):
            continue
        for image in raw_images:
            if not isinstance(image, dict):
                continue
            kind = image.get("type")
            value = image.get("value")
            if kind in {"url", "base64"} and isinstance(value, str) and value:
                images.append({"type": kind, "value": value})
    return images


async def _send_extracted_images(
    message: Message, images: list[dict[str, Any]]
) -> None:
    """Отправляет картинки из инструмента. Сбой одной не отменяет остальные."""
    for index, image in enumerate(images):
        value = image.get("value")
        if not isinstance(value, str):
            continue
        try:
            await _send_generated_image(message, value, index)
        except Exception:
            logger.exception("Не удалось отправить изображение из generate_image")


async def _send_generated_image(message: Message, image: str, index: int) -> None:
    """URL уходит как ссылка, base64 — как файл. Оба варианта отдаёт generate_image."""
    filename = "image.png" if index == 0 else f"image_{index + 1}.png"
    if _is_http_url(image):
        await message.answer_photo(photo=image)
        return
    try:
        raw = base64.b64decode(image, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise AssistAIError("ProxyAPI вернул нечитаемый base64 изображения.") from exc
    if not raw:
        raise AssistAIError("ProxyAPI вернул пустое изображение.")
    await message.answer_photo(photo=BufferedInputFile(raw, filename=filename))


async def on_startup(dispatcher: Dispatcher, bot: Bot) -> None:
    """Создаёт клиент LLM, темы и профили ассистентов. Все живут в workflow_data."""
    topic_storage: TopicHistoryStorage | None = None
    assistant_storage: AssistantStorage | None = None
    try:
        topic_storage = await create_topic_storage(TopicConfig.from_env())
        assistant_storage = await create_assistant_storage(RedisConfig.from_env())
        assistant = AssistAIClient(AssistAIConfig.from_env())
    except ValidationError:
        logger.exception("Некорректная конфигурация бота")
        if assistant_storage is not None:
            await assistant_storage.close()
        if topic_storage is not None:
            await topic_storage.close()
        raise
    dispatcher.workflow_data["topic_storage"] = topic_storage
    dispatcher.workflow_data["assistant_storage"] = assistant_storage
    dispatcher.workflow_data["assistant"] = assistant
    logger.info(
        "AssistAIClient готов, инструментов: %s", len(assistant.get_tools_schema())
    )
    try:
        await bot.set_my_commands(
            [
                BotCommand(command="help", description="Что умеет бот"),
                BotCommand(command="assistants", description="Список персон"),
                BotCommand(command="create_assistant", description="Новая персона"),
                BotCommand(command="model", description="Текстовая модель"),
                BotCommand(command="image_model", description="Модель изображений"),
                BotCommand(command="voice", description="Голосовой режим персоны"),
                BotCommand(command="new_topic", description="Новая тема диалога"),
                BotCommand(command="topics", description="Список тем"),
                BotCommand(command="switch", description="Переключить тему"),
                BotCommand(command="clear_topic", description="Очистить текущую тему"),
                BotCommand(command="image", description="Сгенерировать изображение"),
                BotCommand(command="cancel", description="Отменить ввод"),
            ]
        )
        await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
    except Exception:
        logger.exception("Не удалось записать команды бота")


async def on_shutdown(dispatcher: Dispatcher) -> None:
    """Закрывает клиент LLM, темы и профили, если startup успел их создать."""
    assistant = dispatcher.workflow_data.get("assistant")
    if isinstance(assistant, AssistAIClient):
        await assistant.close()
        logger.info("AssistAIClient закрыт")
    storage = dispatcher.workflow_data.get("topic_storage")
    if isinstance(storage, TopicHistoryStorage):
        await storage.close()
    profiles = dispatcher.workflow_data.get("assistant_storage")
    if isinstance(profiles, AssistantStorage):
        await profiles.close()


async def cmd_start(
    message: Message,
    state: FSMContext,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
) -> None:
    """Приветствие, список возможностей и постоянное меню."""
    await _reset_flow(state)
    menu = await _menu_for(
        message.chat.id,
        assistant_storage,
        topic_storage,
        ensure_topic=True,
    )
    await message.answer(WELCOME, reply_markup=menu)


async def cmd_cancel(
    message: Message,
    state: FSMContext,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
) -> None:
    """Сбрасывает текущий пошаговый ввод. Без шага отвечает, что отменять нечего."""
    menu = await _menu_for(message.chat.id, assistant_storage, topic_storage)
    if await state.get_state() is None:
        await _apply_reply_menu(message, menu, state)
        await message.answer("Нечего отменять.")
        return
    await _reset_flow(state)
    await _apply_reply_menu(message, menu, state)


async def cmd_unknown(message: Message) -> None:
    """Отвечает на команду, для которой нет отдельного хендлера."""
    await message.answer("Неизвестная команда. Список: /help")


async def cmd_new_topic(
    message: Message,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    command: CommandObject,
) -> None:
    """Создаёт тему и делает её активной. Промпт персоны в тему не пишется:

    его подставляет ``_prepare_messages`` из активного ассистента при запросе.
    """
    name = (command.args or "").strip()
    if not name:
        await message.answer("Напишите название: /new_topic Работа")
        return
    if len(name) > 100:
        await message.answer("Название темы длиннее 100 символов.")
        return
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        topic_id = await topic_storage.create_topic(chat_id, profile.id, name)
        await topic_storage.set_active_topic(chat_id, profile.id, topic_id)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    logger.info("Новая тема chat_id=%s topic_id=%s", chat_id, topic_id)
    await message.answer(
        f"✅ Создана новая тема: '{name}'. Контекст обнулен.",
        reply_markup=menu,
    )


async def cmd_topics(
    message: Message,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Показывает темы активной персоны. Та же карточка, что у кнопки «Темы»."""
    await _open_topics(message, topic_storage, assistant_storage, state)


async def cmd_switch(
    message: Message,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    command: CommandObject,
) -> None:
    """Переключает активную тему по id или точному названию."""
    query = (command.args or "").strip()
    if not query:
        await message.answer("Укажите id или название: /switch General")
        return
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        topics = await topic_storage.list_topics(chat_id, profile.id)
        chosen = _match_topic(topics, query)
        if isinstance(chosen, str):
            await message.answer(chosen)
            return
        await topic_storage.set_active_topic(chat_id, profile.id, chosen["id"])
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    logger.info("Переключение темы chat_id=%s topic_id=%s", chat_id, chosen["id"])
    await message.answer(
        f"🔄 Переключено на тему: '{chosen['name']}'.",
        reply_markup=menu,
    )


def _match_topic(topics: list[dict[str, str]], query: str) -> dict[str, str] | str:
    """Находит тему по id или единственному названию. Иначе текст ошибки для пользователя."""
    by_id = next((topic for topic in topics if topic["id"] == query), None)
    if by_id is not None:
        return by_id
    folded = query.casefold()
    by_name = [topic for topic in topics if topic["name"].casefold() == folded]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        return "Несколько тем с таким названием. Укажите id из /topics."
    return "Такой темы нет. Список: /topics"


async def cmd_clear_topic(
    message: Message,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Очищает историю только активной темы активной персоны."""
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        topic_id = await topic_storage.get_active_topic(chat_id, profile.id)
        if not topic_id:
            await message.answer("Нет активной темы. Создайте: /new_topic название")
            return
        topics = await topic_storage.list_topics(chat_id, profile.id)
        await topic_storage.clear_topic(chat_id, profile.id, topic_id)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    name = next(
        (topic["name"] for topic in topics if topic["id"] == topic_id), topic_id
    )
    logger.info("Тема очищена chat_id=%s topic_id=%s", chat_id, topic_id)
    await message.answer(f"История темы '{name}' очищена.")


async def cmd_image(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    command: CommandObject,
) -> None:
    """Генерирует изображение моделью активного ассистента."""
    prompt = (command.args or "").strip()
    if not prompt:
        await message.answer("Напишите описание после команды: /image кот в шляпе")
        return
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    image_model = _image_model_of(profile, assistant)
    await _generate_and_send(
        message,
        assistant,
        prompt,
        image_model,
        size=_stored_image_size(profile),
    )


def _should_send_voice_reply(profile: AssistantProfile, input_kind: InputKind) -> bool:
    """Нужен ли голосовой ответ поверх текста."""
    if profile.voice_mode == "off":
        return False
    if profile.voice_mode == "always":
        return True
    return profile.voice_mode == "auto" and input_kind == "voice"


def _clip_tts_text(text: str) -> str:
    """Обрезает длинный ответ для TTS по последнему пробелу."""
    cleaned = text.strip()
    if len(cleaned) <= TTS_MAX_CHARS:
        return cleaned
    cut = cleaned[:TTS_MAX_CHARS]
    space = cut.rfind(" ")
    if space > TTS_MAX_CHARS // 2:
        cut = cut[:space]
    logger.warning(
        "TTS обрезан: было %s символов, озвучено %s",
        len(cleaned),
        len(cut),
    )
    return cut.rstrip()


async def _send_voice_reply(
    message: Message,
    assistant: AssistAIClient,
    profile: AssistantProfile,
    text: str,
) -> None:
    """Озвучивает ответ персоны. Сбой TTS не откатывает текст и историю."""
    tts_text = _clip_tts_text(text)
    if not tts_text:
        return
    model = profile.tts_model or assistant.config.default_tts_model
    voice = profile.tts_voice or assistant.config.default_tts_voice
    fmt = profile.tts_format or assistant.config.default_tts_format
    try:
        speech = await assistant.synthesize_speech(
            text=tts_text,
            model=model,
            voice=voice,
            response_format=fmt,
        )
    except (AssistAIError, httpx.TimeoutException, ValueError) as exc:
        _log_handler_error(exc)
        await message.answer("⚠️ Не удалось озвучить ответ.")
        return
    except Exception:
        logger.exception("Непредвиденная ошибка TTS")
        await message.answer("⚠️ Не удалось озвучить ответ.")
        return
    audio = BufferedInputFile(speech.audio, filename=speech.filename)
    try:
        await message.answer_voice(voice=audio)
    except TelegramBadRequest as exc:
        logger.warning("Telegram не принял voice, пробую audio: %s", exc)
        try:
            await message.answer_audio(
                audio=BufferedInputFile(speech.audio, filename=speech.filename)
            )
        except TelegramBadRequest as nested:
            logger.warning("Не удалось отправить озвучку: %s", nested)


async def _process_user_message(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    user_text: str,
    input_kind: InputKind = "text",
    status: Message | None = None,
) -> None:
    """Общий pipeline текста и распознанного голоса в активной теме."""
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        topic_id = await _active_topic(chat_id, profile.id, topic_storage)
        history_copy = await _prepare_messages(
            chat_id, profile.id, user_text, topic_id, topic_storage, profile
        )
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    text_model = _text_model_of(profile, assistant)
    logger.info(
        "Сообщение, chat_id=%s topic_id=%s, kind=%s, модель=%s, сообщений в запросе=%s",
        chat_id,
        topic_id,
        input_kind,
        text_model,
        len(history_copy),
    )
    keep: Token[bool] | None = _KEEP_OUTGOING.set(True)
    if status is None:
        status = await message.answer("Готовлю ответ…")
    parts: list[str] = []
    shown = ""
    last_edit = 0.0
    scope = ImageModelScope(profile.image_model, profile.image_size)
    turn_start = len(history_copy)
    try:
        if message.bot is not None:
            await message.bot.send_chat_action(chat_id=chat_id, action="typing")
        with scope:
            async for delta in assistant.stream_chat(
                messages=history_copy, model=text_model
            ):
                if delta.kind != "content" or not delta.text:
                    continue
                parts.append(delta.text)
                now = asyncio.get_running_loop().time()
                if now - last_edit < STREAM_EDIT_INTERVAL:
                    continue
                shown = await _edit_stream(status, "".join(parts), shown)
                last_edit = now
        answer = "".join(parts).strip()
        images = scope.images
        if not answer and not images:
            await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
            await _edit_stream(status, "Пустой ответ.", shown)
            keep = _release_outgoing(keep)
            await _append_chrome(state, [status.message_id])
            return
        if answer:
            visible = await _edit_stream(status, answer, shown)
            head = answer[:TELEGRAM_TEXT_LIMIT]
            if visible == head:
                await _send_overflow(message, answer)
            else:
                await _send_text(message, answer)
        else:
            await _delete_bot_message(status)
        keep = _release_outgoing(keep)
        try:
            await _remember_model_turn(
                chat_id,
                profile.id,
                topic_id,
                topic_storage,
                history_copy[turn_start:],
                answer=answer,
                images=images,
            )
        except TopicStorageError:
            fallback = answer.strip()
            if not fallback and images:
                fallback = _image_completion_text(
                    history_copy[turn_start:], len(images)
                )
            if fallback:
                try:
                    await topic_storage.append_message(
                        chat_id,
                        profile.id,
                        topic_id,
                        {"role": "assistant", "content": fallback},
                        topic_storage.max_length,
                    )
                except TopicStorageError:
                    await _reply_history_error(message)
            else:
                await _reply_history_error(message)
        await _send_extracted_images(message, images)
        logger.info(
            "Ответ, chat_id=%s topic_id=%s, символов=%s, картинок=%s",
            chat_id,
            topic_id,
            len(answer),
            len(images),
        )
        if answer and _should_send_voice_reply(profile, input_kind):
            await _send_voice_reply(message, assistant, profile, answer)
    except (AssistAIError, httpx.TimeoutException) as exc:
        keep = _release_outgoing(keep)
        await _send_extracted_images(message, scope.images)
        _log_handler_error(exc)
        await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
        await _report_stream_error(message, status, parts, _public_error(exc), state)
    except Exception as exc:  # noqa: BLE001
        keep = _release_outgoing(keep)
        await _send_extracted_images(message, scope.images)
        _log_handler_error(exc)
        await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
        await _report_stream_error(message, status, parts, MSG_INTERNAL, state)
    finally:
        _release_outgoing(keep)


async def handle_text(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Обычный диалог в активной теме. Ответ языковой модели идёт потоком."""
    user_text = (message.text or "").strip()
    if not user_text:
        return
    await _process_user_message(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        user_text=user_text,
        input_kind="text",
    )


async def _download_voice(message: Message) -> tuple[bytes, str]:
    """Скачивает голосовое сообщение Telegram в память.

    Returns:
        ``(audio_bytes, filename)``.

    Raises:
        ValueError: нет voice или пустые байты.
        AssistAIError: сбой скачивания.
    """
    voice = message.voice
    if voice is None:
        raise ValueError("В сообщении нет голосового вложения.")
    bot = message.bot
    if bot is None:
        raise AssistAIError("Бот недоступен для скачивания голосового.")
    buffer = io.BytesIO()
    try:
        await bot.download(voice, destination=buffer)
    except Exception as exc:
        logger.exception("Не удалось скачать голосовое")
        raise AssistAIError("Не удалось скачать голосовое сообщение.") from exc
    audio_bytes = buffer.getvalue()
    if not audio_bytes:
        raise ValueError("Голосовое сообщение пустое.")
    return audio_bytes, "voice.ogg"


async def handle_voice(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Распознаёт голосовое и передаёт текст в общий pipeline диалога."""
    voice = message.voice
    if voice is None:
        return
    if voice.duration > MAX_VOICE_DURATION_SECONDS:
        await message.answer("Голосовое слишком длинное. Максимум — 10 минут.")
        return
    if (
        voice.file_size is not None
        and voice.file_size > MAX_VOICE_SIZE_MB * 1024 * 1024
    ):
        await message.answer("Голосовое слишком большое. Максимум — 20 МБ.")
        return
    chat_id = message.chat.id
    logger.info(
        "Голосовое, chat_id=%s duration=%s size=%s",
        chat_id,
        voice.duration,
        voice.file_size,
    )
    keep: Token[bool] | None = _KEEP_OUTGOING.set(True)
    status = await message.answer("🎤 Распознаю речь…")
    keep = _release_outgoing(keep)
    try:
        audio_bytes, filename = await _download_voice(message)
        profile = await _active_profile(chat_id, assistant_storage)
        transcribed = await assistant.transcribe_audio(
            audio_bytes,
            filename=filename,
            language=profile.stt_language,
        )
    except AssistantStorageError:
        await _edit_stream(status, MSG_INTERNAL, "")
        await _append_chrome(state, [status.message_id])
        await _reply_profile_error(message)
        return
    except ValueError as exc:
        await _edit_stream(status, str(exc) or "Не удалось обработать голосовое.", "")
        await _append_chrome(state, [status.message_id])
        return
    except AssistAIError as exc:
        logger.error("AssistAI ошибка STT: %s", exc.message)
        await _edit_stream(
            status, "⚠️ Не удалось распознать речь. Попробуйте ещё раз.", ""
        )
        await _append_chrome(state, [status.message_id])
        return
    except httpx.TimeoutException:
        await _edit_stream(status, "⚠️ Превышено время ожидания распознавания.", "")
        await _append_chrome(state, [status.message_id])
        return
    except Exception:
        logger.exception("Непредвиденная ошибка STT")
        await _edit_stream(status, MSG_INTERNAL, "")
        await _append_chrome(state, [status.message_id])
        return
    logger.info(
        "Распознано, chat_id=%s duration=%s size=%s chars=%s",
        chat_id,
        voice.duration,
        voice.file_size,
        len(transcribed),
    )
    if not transcribed:
        await _edit_stream(status, "🤷 В голосовом не распознано речи.", "")
        await _append_chrome(state, [status.message_id])
        return
    if SHOW_RECOGNIZED_TEXT:
        preview = transcribed[:VOICE_RECOGNIZED_PREVIEW_LIMIT]
        await _edit_stream(
            status,
            f"🗣 Вы сказали:\n«{preview}»\n\nГотовлю ответ…",
            "",
        )
    await _process_user_message(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        user_text=transcribed,
        input_kind="voice",
        status=status,
    )


async def handle_video_note_stub(message: Message) -> None:
    """Кружочки пока не поддерживаются: нужен ffmpeg для извлечения аудио."""
    await message.answer(
        "Кружочки пока не поддерживаются. Отправьте обычное голосовое сообщение."
    )


async def _rollback_user(
    chat_id: int,
    assistant_id: str,
    storage: TopicHistoryStorage,
    topic_id: str,
    message: Message,
) -> None:
    try:
        await storage.rollback_last_user(chat_id, assistant_id, topic_id)
    except TopicStorageError:
        await _reply_history_error(message)


async def _report_stream_error(
    message: Message,
    status: Message,
    parts: list[str],
    text: str,
    state: FSMContext,
) -> None:
    if parts:
        await _delete_bot_message(status)
        await message.answer(text)
        return
    edited = await _edit_stream(status, text, "")
    if edited:
        await _append_chrome(state, [status.message_id])
        return
    await _delete_bot_message(status)
    await message.answer(text)


def _parse_create_args(raw: str) -> tuple[str, str] | str:
    """Разбирает ``Имя "промпт"``. Без двух частей возвращает текст ошибки."""
    try:
        parts = shlex.split(raw, posix=True)
    except ValueError:
        return "Не закрыта кавычка в промпте."
    if len(parts) < 2:
        return 'Формат: /create_assistant Имя "Текст промпта, минимум 10 символов"'
    return parts[0], " ".join(parts[1:])


def _model_keyboard(models: tuple[str, ...], prefix: str) -> InlineKeyboardMarkup:
    """Кнопки моделей и сброс на дефолт клиента. В callback кладётся индекс, не имя."""
    rows = [
        [InlineKeyboardButton(text=model, callback_data=f"{prefix}:{index}")]
        for index, model in enumerate(models)
    ]
    rows.append(
        [InlineKeyboardButton(text="По умолчанию", callback_data=f"{prefix}:default")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _chosen_model_line(selected: str | None, fallback: str) -> str:
    if selected is None:
        return f"{fallback} (по умолчанию)"
    return selected


async def cmd_assistants(
    message: Message, assistant_storage: AssistantStorage, state: FSMContext
) -> None:
    """Показывает персоны чата. Та же карточка, что у кнопки «Ассистенты»."""
    await _open_assistants(message, assistant_storage, state)


async def cmd_create_assistant(
    message: Message,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    command: CommandObject,
) -> None:
    """Создаёт персону из ``/create_assistant Имя "промпт"`` и делает её активной."""
    parsed = _parse_create_args(command.args or "")
    if isinstance(parsed, str):
        await message.answer(parsed)
        return
    name, prompt = parsed
    try:
        profile = AssistantProfile(name=name, system_prompt=prompt)
    except ValidationError:
        await message.answer(
            "Имя: 1–50 символов. Промпт: не короче 10 символов.\n"
            'Формат: /create_assistant Имя "Текст промпта"'
        )
        return
    chat_id = message.chat.id
    try:
        await assistant_storage.save_assistant(chat_id, profile)
        await assistant_storage.set_active_assistant(chat_id, profile.id)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    logger.info("Создан ассистент chat_id=%s id=%s", chat_id, profile.id)
    await message.answer(
        f"✅ Создан ассистент «{profile.name}» ({profile.id}). Он теперь активен.",
        reply_markup=menu,
    )


async def cmd_delete_assistant(
    message: Message,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    command: CommandObject,
) -> None:
    """Удаляет персону по id. Активная заменяется на «Стандартный»."""
    profile_id = (command.args or "").strip()
    if not profile_id:
        await message.answer("Укажите id из /assistants: /delete_assistant abcd1234")
        return
    chat_id = message.chat.id
    try:
        profiles = await assistant_storage.list_assistants(chat_id)
        name = next(
            (item.name for item in profiles if item.id == profile_id), profile_id
        )
        if profile_id != DEFAULT_ASSISTANT_ID:
            await topic_storage.delete_assistant_topics(chat_id, profile_id)
        await assistant_storage.delete_assistant(chat_id, profile_id)
        active = await assistant_storage.get_active_assistant(chat_id)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError as exc:
        await message.answer(exc.message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    active_name = active.name if active is not None else "Стандартный"
    logger.info("Удалён ассистент chat_id=%s id=%s", chat_id, profile_id)
    await message.answer(
        f"Ассистент «{name}» удалён. Активен «{active_name}».",
        reply_markup=menu,
    )


async def cmd_model(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Показывает текстовую модель активной персоны и кнопки замены."""
    await _show_model_picker(
        message,
        assistant,
        assistant_storage,
        kind="text",
    )


async def cmd_image_model(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Показывает модель изображений активной персоны и кнопки замены."""
    await _show_model_picker(
        message,
        assistant,
        assistant_storage,
        kind="image",
    )


async def _show_model_picker(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    *,
    kind: str,
) -> None:
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    text, keyboard = _model_picker_parts(profile, assistant, kind)
    await message.answer(text, reply_markup=keyboard)


async def on_set_text_model(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Записывает текстовую модель в активный профиль."""
    await _apply_model_choice(
        query,
        assistant,
        assistant_storage,
        state,
        catalog=AVAILABLE_TEXT_MODELS,
        prefix="set_text:",
        kind="text",
    )


async def on_set_image_model(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Записывает модель изображений в активный профиль."""
    await _apply_model_choice(
        query,
        assistant,
        assistant_storage,
        state,
        catalog=AVAILABLE_IMAGE_MODELS,
        prefix="set_image:",
        kind="image",
    )


async def _apply_model_choice(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    *,
    catalog: tuple[str, ...],
    prefix: str,
    kind: str,
) -> None:
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    raw = (query.data or "").removeprefix(prefix)
    if raw == "default":
        selected: str | None = None
    else:
        try:
            selected = catalog[int(raw)]
        except (ValueError, IndexError):
            await query.answer("Неизвестная модель", show_alert=True)
            return
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        if kind == "image":
            old_model = _image_model_of(profile, assistant)
            new_model = selected or assistant.config.default_image_model
            fields: dict[str, Any] = {"image_model": selected}
            if _is_flux_image_model(old_model) != _is_flux_image_model(new_model):
                fields["image_size"] = _default_image_size(new_model)
            updated = profile.model_copy(update=fields)
        else:
            updated = profile.model_copy(update={"text_model": selected})
        await assistant_storage.save_assistant(chat_id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer("Модель обновлена")
    await _return_to_settings(message, state, updated, assistant)


async def _return_to_settings(
    message: Message,
    state: FSMContext,
    profile: AssistantProfile,
    assistant: AssistAIClient,
) -> None:
    """Убирает выбор модели и обновляет предыдущий экран настроек."""
    text, markup = _settings_view(profile, assistant)
    try:
        await message.delete()
    except TelegramBadRequest as exc:
        logger.warning("Не удалось удалить выбор модели: %s", exc)
    data = await state.get_data()
    if _SCREEN_PICKER in data:
        data.pop(_SCREEN_PICKER, None)
        await state.set_data(data)
    settings_id = data.get(_SCREEN_SETTINGS)
    bot = message.bot
    if (
        isinstance(settings_id, int)
        and bot is not None
        and settings_id != message.message_id
    ):
        try:
            await bot.edit_message_text(
                text,
                chat_id=message.chat.id,
                message_id=settings_id,
                reply_markup=markup,
            )
            return
        except TelegramBadRequest as exc:
            lowered = str(exc).lower()
            if "message is not modified" in lowered:
                return
            logger.warning("Не удалось вернуть настройки: %s", exc)
    sent = await message.answer(text, reply_markup=markup)
    await state.update_data({_SCREEN_SETTINGS: sent.message_id})


def _clip_label(prefix: str, name: str) -> str:
    """Подпись кнопки не длиннее 64 символов."""
    label = f"{prefix}{name}"
    if len(label) <= 64:
        return label
    return label[:61] + "..."


def _optional_model(value: Any) -> str | None:
    """Пустая строка и не-строка означают модель по умолчанию."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _query_message(query: CallbackQuery) -> Message | None:
    """Сообщение кнопки. Служебные callback без сообщения пропускаются."""
    message = query.message
    if isinstance(message, Message):
        return message
    return None


def _release_outgoing(token: Token[bool] | None) -> Token[bool] | None:
    """Снимает запрет учёта исходящих сообщений, если он ещё включён."""
    if token is not None:
        _KEEP_OUTGOING.reset(token)
    return None


def _is_menu_button(text: str) -> bool:
    """Нижние кнопки и их старые подписи. Это не вопрос к персоне."""
    if text.startswith((f"{_MENU_ASSISTANTS} (", f"{_MENU_TOPICS} (")):
        return True
    return text in {
        _MENU_SETTINGS,
        MAIN_MENU_TEXT["assistants"],
        MAIN_MENU_TEXT["topics"],
        MAIN_MENU_TEXT["settings"],
        MAIN_MENU_TEXT["image"],
    }


async def _append_chrome(state: FSMContext, ids: list[int]) -> None:
    """Запоминает служебные сообщения бота, чтобы позже убрать их из чата."""
    if not ids:
        return
    data = await state.get_data()
    current = data.get(_CHROME)
    merged: list[int] = []
    if isinstance(current, list):
        merged.extend(item for item in current if isinstance(item, int))
    for item in ids:
        if item not in merged:
            merged.append(item)
    if len(merged) > 50:
        merged = merged[-50:]
    await state.update_data({_CHROME: merged})


async def _reset_flow(state: FSMContext) -> None:
    """Сбрасывает шаг мастера и сохраняет id сообщения, которое держит меню."""
    data = await state.get_data()
    anchor = data.get(_MENU_ANCHOR)
    chrome = data.get(_CHROME)
    await state.clear()
    kept: dict[str, Any] = {}
    if isinstance(anchor, int):
        kept[_MENU_ANCHOR] = anchor
    if isinstance(chrome, list):
        kept[_CHROME] = chrome
    if kept:
        await state.update_data(kept)


async def _delete_bot_message(message: Message) -> None:
    """Удаляет сообщение бота. Повторное удаление не считается сбоем."""
    try:
        await message.delete()
    except TelegramBadRequest as exc:
        logger.warning("Не удалось удалить сообщение: %s", exc)


async def _apply_reply_menu(
    message: Message, menu: ReplyKeyboardMarkup, state: FSMContext
) -> None:
    """Ставит нижние кнопки.

    Сообщение с клавиатурой нельзя удалять: вместе с ним клиент убирает меню.
    Текст невидимый, чтобы в чате не было отдельной реплики. Прошлое такое
    сообщение удаляется, когда новое уже держит меню.
    """
    token = _KEEP_OUTGOING.set(True)
    try:
        sent = await message.answer("\u2060", reply_markup=menu)
    finally:
        _KEEP_OUTGOING.reset(token)
    data = await state.get_data()
    previous = data.get(_MENU_ANCHOR)
    await state.update_data({_MENU_ANCHOR: sent.message_id})
    if (
        isinstance(previous, int)
        and previous != sent.message_id
        and message.bot is not None
    ):
        try:
            await message.bot.delete_message(
                chat_id=message.chat.id, message_id=previous
            )
        except TelegramBadRequest as exc:
            logger.warning("Не удалось убрать прежнее сообщение меню: %s", exc)


async def _sweep_chrome(
    message: Message, state: FSMContext, data: dict[str, Any]
) -> None:
    """Удаляет меню, настройки и прочие служебные реплики.

    Перед удалением заново ставит нижние кнопки: иначе клиент прячет меню
    вместе с последним сообщением, на котором оно висело.
    """
    stored = await state.get_data()
    raw_ids = stored.get(_CHROME)
    if not isinstance(raw_ids, list) or not raw_ids:
        return
    profiles = data.get("assistant_storage")
    topics = data.get("topic_storage")
    if isinstance(profiles, AssistantStorage) and isinstance(
        topics, TopicHistoryStorage
    ):
        menu = await _menu_for(message.chat.id, profiles, topics)
        await _apply_reply_menu(message, menu, state)
    stored = await state.get_data()
    anchor = stored.get(_MENU_ANCHOR)
    bot = message.bot
    if bot is not None:
        for item in raw_ids:
            if not isinstance(item, int) or item == anchor:
                continue
            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=item)
            except TelegramBadRequest:
                logger.debug(
                    "Служебное сообщение уже удалено chat_id=%s message_id=%s",
                    message.chat.id,
                    item,
                )
    refreshed = await state.get_data()
    for key in _SCREEN_KEYS:
        refreshed.pop(key, None)
    refreshed[_CHROME] = []
    await state.set_data(refreshed)


async def _drop_user_noise(message: Message, state: FSMContext) -> None:
    """Убирает команды, нажатия меню и шаги мастера. Вопрос к персоне остаётся."""
    text = message.text or ""
    if not text:
        return
    current = await state.get_state()
    if current is None and not text.startswith("/") and not _is_menu_button(text):
        return
    try:
        await message.delete()
    except TelegramBadRequest as exc:
        logger.warning("Не удалось убрать служебную реплику пользователя: %s", exc)


class _CleanChat(BaseMiddleware):
    """Перед новым сообщением убирает из чата всё, кроме диалога и картинок."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        state = data.get("state")
        bucket: list[int] = []
        token = _CHROME_BUCKET.set(bucket)
        try:
            if isinstance(event, Message) and isinstance(state, FSMContext):
                await _sweep_chrome(event, state, data)
                await _drop_user_noise(event, state)
            return await handler(event, data)
        finally:
            if isinstance(state, FSMContext):
                await _append_chrome(state, bucket)
            _CHROME_BUCKET.reset(token)


async def _close_screen(message: Message, state: FSMContext, key: str) -> None:
    """Удаляет экран с кнопками и забывает его id, чтобы не править пустое место."""
    try:
        await message.delete()
    except TelegramBadRequest as exc:
        logger.warning("Не удалось удалить экран: %s", exc)
    data = await state.get_data()
    if data.get(key) == message.message_id:
        data.pop(key, None)
        await state.set_data(data)


async def _edit_inline(
    message: Message, text: str, markup: InlineKeyboardMarkup | None
) -> None:
    """Правит текст и кнопки. Повтор того же текста Telegram не считает ошибкой."""
    try:
        await message.edit_text(text, reply_markup=markup)
    except TelegramBadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            logger.warning("Не удалось обновить сообщение: %s", exc)


async def _drop_flow(state: FSMContext) -> None:
    """Выходит из пошагового ввода и сохраняет id уже открытых экранов."""
    data = await state.get_data()
    kept = {key: value for key, value in data.items() if key not in _FLOW_KEYS}
    await state.set_state(None)
    await state.set_data(kept)


async def _upsert_screen(
    message: Message,
    state: FSMContext,
    key: str,
    text: str,
    markup: InlineKeyboardMarkup,
) -> None:
    """Повторный вызов экрана правит прежнее сообщение, иначе отправляет новое."""
    data = await state.get_data()
    message_id = data.get(key)
    bot = message.bot
    if isinstance(message_id, int) and bot is not None:
        try:
            await bot.edit_message_text(
                text,
                chat_id=message.chat.id,
                message_id=message_id,
                reply_markup=markup,
            )
            return
        except TelegramBadRequest as exc:
            lowered = str(exc).lower()
            if "message is not modified" in lowered:
                return
            logger.warning("Не удалось обновить экран: %s", exc)
    sent = await message.answer(text, reply_markup=markup)
    await state.update_data({key: sent.message_id})


def _page_bounds(count: int, page: int) -> tuple[int, int]:
    """Номер страницы в допустимых пределах и общее число страниц."""
    total = max(1, (count + PAGE_SIZE - 1) // PAGE_SIZE)
    current = max(0, min(page, total - 1))
    return current, total


def _pager_row(page: int, total: int, prefix: str) -> list[InlineKeyboardButton] | None:
    """Строка ◀ / ▶. На одной странице её нет."""
    if total <= 1:
        return None
    row: list[InlineKeyboardButton] = []
    if page > 0:
        row.append(InlineKeyboardButton(text="◀", callback_data=f"{prefix}:{page - 1}"))
    if page < total - 1:
        row.append(InlineKeyboardButton(text="▶", callback_data=f"{prefix}:{page + 1}"))
    return row or None


def _assistants_text(
    profiles: list[AssistantProfile], active_id: str, page: int = 0
) -> str:
    """Текст экрана персон: кто активен и номер страницы. Имена только на кнопках."""
    current, total = _page_bounds(len(profiles), page)
    active = next((item for item in profiles if item.id == active_id), None)
    active_name = active.name if active is not None else "—"
    lines = [f"Активный: ✅ {active_name}"]
    if not profiles:
        lines.append("")
        lines.append("Персон пока нет.")
    lines.append("")
    lines.append(f"Стр. {current + 1}/{total}")
    return "\n".join(lines)


def _assistants_keyboard(
    profiles: list[AssistantProfile], active_id: str, page: int = 0
) -> InlineKeyboardMarkup:
    """Кнопки персон текущей страницы и переход по страницам."""
    current, total = _page_bounds(len(profiles), page)
    chunk = profiles[current * PAGE_SIZE : (current + 1) * PAGE_SIZE]
    rows: list[list[InlineKeyboardButton]] = []
    for profile in chunk:
        mark = "✅ " if profile.id == active_id else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=_clip_label(mark, profile.name),
                    callback_data=f"asst:use:{profile.id}",
                )
            ]
        )
    pager = _pager_row(current, total, "asst_page")
    if pager is not None:
        rows.append(pager)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _assistants_markup(
    profiles: list[AssistantProfile], active_id: str, page: int = 0
) -> tuple[str, InlineKeyboardMarkup]:
    """Текст и кнопки экрана персон. Здесь только переключение."""
    return (
        _assistants_text(profiles, active_id, page),
        _assistants_keyboard(profiles, active_id, page),
    )


async def _assistant_screen(
    chat_id: int, storage: AssistantStorage
) -> tuple[str, InlineKeyboardMarkup]:
    """Собирает экран персон из хранилища."""
    profiles = await storage.list_assistants(chat_id)
    active = await storage.get_active_assistant(chat_id)
    active_id = active.id if active is not None else ""
    return _assistants_markup(profiles, active_id)


async def _open_assistants(
    message: Message, storage: AssistantStorage, state: FSMContext
) -> None:
    """Открывает или обновляет экран персон."""
    try:
        text, markup = await _assistant_screen(message.chat.id, storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    await _upsert_screen(message, state, _SCREEN_ASSISTANTS, text, markup)


def _topics_text(topics: list[dict[str, str]], active: str, page: int = 0) -> str:
    """Текст экрана тем: какая активна и номер страницы. Имена только на кнопках."""
    current, total = _page_bounds(len(topics), page)
    active_name = next((item["name"] for item in topics if item["id"] == active), "нет")
    lines = [f"Активная: 📍 {active_name}"]
    if not topics:
        lines.append("")
        lines.append("Тем пока нет.")
    lines.append("")
    lines.append(f"Стр. {current + 1}/{total}")
    return "\n".join(lines)


def _topics_keyboard(
    topics: list[dict[str, str]], active: str, page: int = 0
) -> InlineKeyboardMarkup:
    """Кнопки тем текущей страницы: переключить, переименовать, удалить."""
    current, total = _page_bounds(len(topics), page)
    chunk = topics[current * PAGE_SIZE : (current + 1) * PAGE_SIZE]
    rows: list[list[InlineKeyboardButton]] = []
    for topic in chunk:
        name = topic["name"]
        topic_id = topic["id"]
        mark = "✅ " if topic_id == active else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=_clip_label(mark, name),
                    callback_data=f"topic:use:{topic_id}",
                ),
                InlineKeyboardButton(
                    text="✏️", callback_data=f"topic:rename:{topic_id}"
                ),
                InlineKeyboardButton(
                    text="🗑", callback_data=f"topic_del_ask:{topic_id}"
                ),
            ]
        )
    rows.append([InlineKeyboardButton(text="Новая тема", callback_data="topic:new")])
    pager = _pager_row(current, total, "topic_page")
    if pager is not None:
        rows.append(pager)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _topics_markup(
    topics: list[dict[str, str]], active_id: str, page: int = 0
) -> tuple[str, InlineKeyboardMarkup]:
    """Текст и кнопки экрана тем."""
    return (
        _topics_text(topics, active_id, page),
        _topics_keyboard(topics, active_id, page),
    )


async def _topics_screen(
    chat_id: int,
    assistant_id: str,
    storage: TopicHistoryStorage,
) -> tuple[str, InlineKeyboardMarkup]:
    """Собирает экран тем активной персоны."""
    topics = await storage.list_topics(chat_id, assistant_id)
    active = await storage.get_active_topic(chat_id, assistant_id)
    return _topics_markup(topics, active)


async def _open_topics(
    message: Message,
    storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает или обновляет экран тем активной персоны."""
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
        text, markup = await _topics_screen(message.chat.id, profile.id, storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    await _upsert_screen(message, state, _SCREEN_TOPICS, text, markup)


def _model_picker_parts(
    profile: AssistantProfile, assistant: AssistAIClient, kind: str
) -> tuple[str, InlineKeyboardMarkup]:
    """Текст и кнопки выбора модели активной персоны."""
    if kind == "image":
        current = _chosen_model_line(
            profile.image_model, assistant.config.default_image_model
        )
        title = "Модель изображений"
        keyboard = _model_keyboard(AVAILABLE_IMAGE_MODELS, "set_image")
    else:
        current = _chosen_model_line(
            profile.text_model, assistant.config.default_text_model
        )
        title = "Текстовая модель"
        keyboard = _model_keyboard(AVAILABLE_TEXT_MODELS, "set_text")
    text = f"{title} ассистента «{profile.name}»:\n{current}\n\nВыберите модель:"
    return text, keyboard


def _voice_settings_lines(
    profile: AssistantProfile, assistant: AssistAIClient
) -> list[str]:
    """Текст блока голосовых настроек персоны."""
    tts_model = profile.tts_model or assistant.config.default_tts_model
    tts_voice = profile.tts_voice or assistant.config.default_tts_voice
    tts_format = profile.tts_format or assistant.config.default_tts_format
    stt_language = profile.stt_language or "авто"
    return [
        "🎙 Голос:",
        f"Режим: {profile.voice_mode}",
        f"TTS модель: {tts_model}",
        f"Голос: {tts_voice}",
        f"Формат: {tts_format}",
        f"STT язык: {stt_language}",
    ]


def _settings_view(
    profile: AssistantProfile, assistant: AssistAIClient
) -> tuple[str, InlineKeyboardMarkup]:
    """Настройки только активной персоны: модели, размер, голос, промпт."""
    image_model = _image_model_of(profile, assistant)
    shown_size = _display_image_size(profile, assistant)
    lines = [
        f"Текущая персона: ✅ {profile.name}",
        (
            "Текстовая модель: "
            f"{_chosen_model_line(profile.text_model, assistant.config.default_text_model)}"
        ),
        (
            "Модель картинок: "
            f"{_chosen_model_line(profile.image_model, assistant.config.default_image_model)}"
        ),
        f"Размер картинки: {shown_size}",
        "",
        *_voice_settings_lines(profile, assistant),
    ]
    text = "\n".join(lines)
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text="Текстовая модель", callback_data="set_text_menu")],
        [InlineKeyboardButton(text="Модель картинок", callback_data="set_image_menu")],
    ]
    size_row: list[InlineKeyboardButton] = []
    for item in _size_choices(image_model):
        mark = "✅ " if item == shown_size else ""
        size_row.append(
            InlineKeyboardButton(text=f"{mark}{item}", callback_data=f"set_size:{item}")
        )
    rows.append(size_row)
    mode_row: list[InlineKeyboardButton] = []
    for mode in ("off", "auto", "always"):
        mark = "✅ " if profile.voice_mode == mode else ""
        mode_row.append(
            InlineKeyboardButton(
                text=f"{mark}🎙 {mode}",
                callback_data=f"voice_mode:{mode}",
            )
        )
    rows.append(mode_row)
    voice_row: list[InlineKeyboardButton] = []
    current_voice = profile.tts_voice or assistant.config.default_tts_voice
    for index, name in enumerate(TTS_VOICES):
        mark = "✅ " if current_voice == name else ""
        voice_row.append(
            InlineKeyboardButton(
                text=f"{mark}{name}",
                callback_data=f"voice_name:{index}",
            )
        )
    rows.append(voice_row)
    lang_row: list[InlineKeyboardButton] = []
    for label, value in STT_LANGUAGES:
        mark = "✅ " if profile.stt_language == value else ""
        lang_row.append(
            InlineKeyboardButton(
                text=f"{mark}STT {label}",
                callback_data=f"stt_lang:{label}",
            )
        )
    rows.append(lang_row)
    rows.append(
        [
            InlineKeyboardButton(
                text="Редактировать промпт",
                callback_data=f"asst_edit:{profile.id}",
            )
        ]
    )
    if profile.id != DEFAULT_ASSISTANT_ID:
        rows.append(
            [
                InlineKeyboardButton(
                    text="Удалить персону",
                    callback_data=f"asst:del_ask:{profile.id}",
                )
            ]
        )
    rows.append([InlineKeyboardButton(text="Новая персона", callback_data="asst:new")])
    rows.append(
        [InlineKeyboardButton(text="Закрыть настройки", callback_data="settings:close")]
    )
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def _image_view(
    selected: str | None, size: str, assistant: AssistAIClient
) -> tuple[str, InlineKeyboardMarkup]:
    """Экран генерации: модель, размер семьи модели и переход к описанию."""
    label = selected or assistant.config.default_image_model
    allowed = _size_choices(label)
    shown = size if size in allowed else _default_image_size(label)
    text = f"Генерация изображения\nМодель: {label}\nРазмер: {shown}"
    rows: list[list[InlineKeyboardButton]] = []
    for index, model in enumerate(AVAILABLE_IMAGE_MODELS):
        mark = "✅ " if selected == model else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=_clip_label(mark, model),
                    callback_data=f"image:model:{index}",
                )
            ]
        )
    default_mark = "✅ " if selected is None else ""
    rows.append(
        [
            InlineKeyboardButton(
                text=f"{default_mark}По умолчанию",
                callback_data="image:model:default",
            )
        ]
    )
    size_row: list[InlineKeyboardButton] = []
    for item in allowed:
        mark = "✅ " if item == shown else ""
        size_row.append(
            InlineKeyboardButton(
                text=f"{mark}{item}", callback_data=f"image:size:{item}"
            )
        )
    rows.append(size_row)
    rows.append(
        [InlineKeyboardButton(text="✍️ Описать картинку", callback_data="image:go")]
    )
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def _draft_models_text(data: dict[str, Any], assistant: AssistAIClient) -> str:
    """Текущий выбор моделей на шаге создания персоны."""
    text_model = _optional_model(data.get("flow_text_model"))
    image_model = _optional_model(data.get("flow_image_model"))
    text_line = (
        f"{assistant.config.default_text_model} (по умолчанию)"
        if text_model is None
        else text_model
    )
    image_line = (
        f"{assistant.config.default_image_model} (по умолчанию)"
        if image_model is None
        else image_model
    )
    return (
        "Выберите модели (необязательно) или нажмите «Готово».\n"
        f"Текст: {text_line}\n"
        f"Картинка: {image_line}"
    )


def _draft_models_keyboard(data: dict[str, Any]) -> InlineKeyboardMarkup:
    """Кнопки временного выбора моделей и подтверждение создания."""
    text_model = _optional_model(data.get("flow_text_model"))
    image_model = _optional_model(data.get("flow_image_model"))
    rows: list[list[InlineKeyboardButton]] = []
    for index, model in enumerate(AVAILABLE_TEXT_MODELS):
        mark = "✅ " if text_model == model else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=_clip_label(mark, model),
                    callback_data=f"asst:new_text:{index}",
                )
            ]
        )
    for index, model in enumerate(AVAILABLE_IMAGE_MODELS):
        mark = "✅ " if image_model == model else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=_clip_label(f"{mark}🖼 ", model),
                    callback_data=f"asst:new_image:{index}",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="По умолчанию", callback_data="asst:new_default")]
    )
    rows.append(
        [InlineKeyboardButton(text="✅ Готово", callback_data="asst:new_commit")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _confirm_row(
    yes_text: str, yes_data: str, cancel_data: str
) -> InlineKeyboardMarkup:
    """Пара кнопок: подтвердить действие или вернуться к списку."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=yes_text, callback_data=yes_data),
                InlineKeyboardButton(text="← Отмена", callback_data=cancel_data),
            ]
        ]
    )


async def _generate_and_send(
    message: Message,
    assistant: AssistAIClient,
    prompt: str,
    model: str | None,
    size: str = "1024x1024",
    *,
    reply_markup: ReplyKeyboardMarkup | None = None,
) -> None:
    """Просит ProxyAPI нарисовать картинку и отправляет её в чат."""
    logger.info(
        "Генерация изображения, chat_id=%s, модель=%s, размер=%s, символов=%s",
        message.chat.id,
        model or assistant.config.default_image_model,
        size,
        len(prompt),
    )
    keep = _KEEP_OUTGOING.set(True)
    try:
        progress = await message.answer("Генерирую изображение…")
    finally:
        _KEEP_OUTGOING.reset(keep)
    try:
        images = await assistant.generate_image(prompt=prompt, model=model, size=size)
        await _delete_bot_message(progress)
        if not images:
            await message.answer(MSG_IMAGE_FAILED, reply_markup=reply_markup)
            return
        for index, image in enumerate(images):
            await _send_generated_image(message, image, index)
    except AssistAIError as exc:
        logger.error("AssistAI ошибка: %s", exc.message)
        await _delete_bot_message(progress)
        await message.answer(MSG_IMAGE_FAILED, reply_markup=reply_markup)
    except httpx.TimeoutException:
        logger.error("Таймаут генерации изображения")
        await _delete_bot_message(progress)
        await message.answer(MSG_TIMEOUT, reply_markup=reply_markup)
    except Exception:
        logger.exception("Непредвиденная ошибка генерации изображения")
        await _delete_bot_message(progress)
        await message.answer(MSG_INTERNAL, reply_markup=reply_markup)


async def on_menu_assistants(
    message: Message, assistant_storage: AssistantStorage, state: FSMContext
) -> None:
    """Кнопка «Ассистенты» открывает список персон."""
    await _drop_flow(state)
    await _open_assistants(message, assistant_storage, state)


async def on_menu_topics(
    message: Message,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Кнопка «Темы» открывает темы активной персоны."""
    await _drop_flow(state)
    await _open_topics(message, topic_storage, assistant_storage, state)


async def on_menu_settings(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Кнопка «Настройки» показывает модели активной персоны."""
    await _drop_flow(state)
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    text, markup = _settings_view(profile, assistant)
    await _upsert_screen(message, state, _SCREEN_SETTINGS, text, markup)


async def on_menu_image(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Кнопка «Картинка» открывает выбор модели и размера."""
    await _drop_flow(state)
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    selected = profile.image_model
    model_id = selected or assistant.config.default_image_model
    size = _default_image_size(model_id)
    await state.set_state(ImageSG.config)
    await state.update_data(image_model=selected, image_size=size)
    text, markup = _image_view(selected, size, assistant)
    await _upsert_screen(message, state, _SCREEN_IMAGE, text, markup)


async def on_asst_list(
    query: CallbackQuery, assistant_storage: AssistantStorage, state: FSMContext
) -> None:
    """Возвращает сообщение к списку персон."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        text, markup = await _assistant_screen(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_ASSISTANTS: message.message_id})
    await query.answer()


async def on_asst_use(
    query: CallbackQuery,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    state: FSMContext,
) -> None:
    """Делает выбранную персону активной, убирает экран и обновляет нижнее меню."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst:use:")
    chat_id = message.chat.id
    try:
        await assistant_storage.set_active_assistant(chat_id, profile_id)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    await _close_screen(message, state, _SCREEN_ASSISTANTS)
    await _apply_reply_menu(message, menu, state)
    logger.info("Активный ассистент chat_id=%s id=%s", chat_id, profile_id)


async def on_assistants_page(
    query: CallbackQuery, assistant_storage: AssistantStorage
) -> None:
    """Открывает другую страницу списка персон."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    try:
        page = int((query.data or "").removeprefix("asst_page:"))
    except ValueError:
        await query.answer("Неверная страница", show_alert=True)
        return
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
        active = await assistant_storage.get_active_assistant(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    active_id = active.id if active is not None else ""
    await query.answer()
    await _edit_or_answer(
        query,
        _assistants_text(profiles, active_id, page),
        _assistants_keyboard(profiles, active_id, page),
    )


async def on_asst_edit_ask(
    query: CallbackQuery, state: FSMContext, assistant_storage: AssistantStorage
) -> None:
    """Показывает текущий промпт и ждёт новый текст."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst_edit:")
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    profile = next((item for item in profiles if item.id == profile_id), None)
    if profile is None:
        await query.answer("Ассистент не найден", show_alert=True)
        return
    await state.set_state(EditAssistantSG.prompt)
    await state.update_data(id=profile.id, name=profile.name)
    await query.answer()
    await _edit_or_answer(
        query,
        f"Промпт ассистента «{profile.name}»:\n\n{profile.system_prompt}\n\n"
        "Пришлите новый промпт (≥10 символов). /cancel — отмена.",
    )


async def sg_edit_prompt(
    message: Message, state: FSMContext, assistant_storage: AssistantStorage
) -> None:
    """Сохраняет новый системный промпт выбранной персоны."""
    prompt = message.text or ""
    if len(prompt) < 10:
        await message.answer("Промпт минимум 10 символов. /cancel — отмена.")
        return
    data = await state.get_data()
    profile_id = data.get("id")
    if not isinstance(profile_id, str) or not profile_id:
        await _reset_flow(state)
        await message.answer("Ассистент не найден.")
        return
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
        old = next((item for item in profiles if item.id == profile_id), None)
        if old is None:
            await _reset_flow(state)
            await message.answer("Ассистент не найден.")
            return
        updated = old.model_copy(update={"system_prompt": prompt})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except ValidationError:
        await message.answer("Промпт минимум 10 символов. /cancel — отмена.")
        return
    await _reset_flow(state)
    logger.info(
        "Обновлён промпт ассистента chat_id=%s id=%s", message.chat.id, updated.id
    )
    await message.answer(f"✅ Промпт ассистента «{updated.name}» обновлён.")


async def on_asst_edit(
    query: CallbackQuery, assistant_storage: AssistantStorage, state: FSMContext
) -> None:
    """Запускает правку промпта существующей персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst:edit:")
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    profile = next((item for item in profiles if item.id == profile_id), None)
    if profile is None:
        await query.answer("Ассистент не найден.", show_alert=True)
        return
    await _drop_flow(state)
    await state.set_state(NewAssistantSG.prompt)
    await state.update_data(
        flow_edit_id=profile.id,
        flow_name=profile.name,
        flow_text_model=profile.text_model,
        flow_image_model=profile.image_model,
    )
    await query.answer()
    await message.answer(
        f"Пришлите системный промпт для «{profile.name}» (мин. 10 символов).\n"
        "/cancel — отмена"
    )


async def on_asst_del_ask(
    query: CallbackQuery, assistant_storage: AssistantStorage
) -> None:
    """Спрашивает подтверждение перед удалением персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst:del_ask:")
    if profile_id == DEFAULT_ASSISTANT_ID:
        await query.answer("Базового ассистента удалить нельзя.", show_alert=True)
        return
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    profile = next((item for item in profiles if item.id == profile_id), None)
    if profile is None:
        await query.answer("Ассистент не найден.", show_alert=True)
        return
    await query.answer()
    await _edit_inline(
        message,
        f"Удалить ассистента «{profile.name}»?",
        _confirm_row("✅ Да, удалить", f"asst:del_yes:{profile.id}", "settings:open"),
    )


async def on_asst_del_yes(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    state: FSMContext,
) -> None:
    """Удаляет персону и её темы, затем показывает настройки оставшейся."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst:del_yes:")
    chat_id = message.chat.id
    try:
        profiles = await assistant_storage.list_assistants(chat_id)
        name = next(
            (item.name for item in profiles if item.id == profile_id), profile_id
        )
        if profile_id != DEFAULT_ASSISTANT_ID:
            await topic_storage.delete_assistant_topics(chat_id, profile_id)
        await assistant_storage.delete_assistant(chat_id, profile_id)
        profile = await _active_profile(chat_id, assistant_storage)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Удалён ассистент chat_id=%s id=%s", chat_id, profile_id)
    await query.answer("Удалено")
    text, markup = _settings_view(profile, assistant)
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_SETTINGS: message.message_id})
    await message.answer(f"Ассистент «{name}» удалён.", reply_markup=menu)


async def on_asst_new_start(query: CallbackQuery, state: FSMContext) -> None:
    """Первый шаг создания персоны: запрос имени."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    await _drop_flow(state)
    await state.set_state(NewAssistantSG.name)
    await query.answer()
    await message.answer("Введите имя персоны (1–50 символов).\n/cancel — отмена")


async def sg_new_assistant_name(message: Message, state: FSMContext) -> None:
    """Принимает имя персоны и переходит к промпту."""
    name = (message.text or "").strip()
    if not name or len(name) > 50:
        await message.answer("Введите имя персоны (1–50 символов).\n/cancel — отмена")
        return
    await state.update_data(flow_name=name, flow_edit_id=None)
    await state.set_state(NewAssistantSG.prompt)
    await message.answer(
        "Пришлите системный промпт (мин. 10 символов).\n/cancel — отмена"
    )


async def sg_new_assistant_prompt(
    message: Message, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Принимает промпт и показывает выбор моделей."""
    prompt = (message.text or "").strip()
    if len(prompt) < 10:
        await message.answer(
            "Пришлите системный промпт (мин. 10 символов).\n/cancel — отмена"
        )
        return
    await state.update_data(flow_prompt=prompt)
    await state.set_state(NewAssistantSG.models)
    data = await state.get_data()
    await message.answer(
        _draft_models_text(data, assistant),
        reply_markup=_draft_models_keyboard(data),
    )


async def sg_new_assistant_models(message: Message) -> None:
    """На шаге моделей текст не уходит в языковую модель."""
    await message.answer(
        "Выберите модели кнопками или нажмите «Готово».\n/cancel — отмена"
    )


async def _refresh_draft_models(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Обновляет текст и кнопки выбора моделей у создаваемой персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    data = await state.get_data()
    await query.answer()
    await _edit_inline(
        message, _draft_models_text(data, assistant), _draft_models_keyboard(data)
    )


async def on_asst_new_text(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Запоминает текстовую модель до нажатия «Готово»."""
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    raw = (query.data or "").removeprefix("asst:new_text:")
    try:
        selected: str | None = AVAILABLE_TEXT_MODELS[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестная модель", show_alert=True)
        return
    await state.update_data(flow_text_model=selected)
    await _refresh_draft_models(query, state, assistant)


async def on_asst_new_image(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Запоминает модель картинок до нажатия «Готово»."""
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    raw = (query.data or "").removeprefix("asst:new_image:")
    try:
        selected: str | None = AVAILABLE_IMAGE_MODELS[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестная модель", show_alert=True)
        return
    await state.update_data(flow_image_model=selected)
    await _refresh_draft_models(query, state, assistant)


async def on_asst_new_default(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Сбрасывает обе модели черновика на дефолт клиента."""
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    await state.update_data(flow_text_model=None, flow_image_model=None)
    await _refresh_draft_models(query, state, assistant)


async def on_asst_new_commit(
    query: CallbackQuery,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    state: FSMContext,
) -> None:
    """Сохраняет персону из шагов мастера и делает её активной."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    data = await state.get_data()
    name = data.get("flow_name")
    prompt = data.get("flow_prompt")
    edit_id = data.get("flow_edit_id")
    if not isinstance(name, str) or not isinstance(prompt, str):
        await query.answer("Имя 1–50, промпт ≥ 10", show_alert=True)
        await state.set_state(NewAssistantSG.name)
        await message.answer("Введите имя персоны (1–50 символов).\n/cancel — отмена")
        return
    try:
        fields: dict[str, Any] = {
            "name": name,
            "system_prompt": prompt,
            "text_model": _optional_model(data.get("flow_text_model")),
            "image_model": _optional_model(data.get("flow_image_model")),
        }
        if isinstance(edit_id, str) and edit_id:
            fields["id"] = edit_id
        profile = AssistantProfile.model_validate(fields)
    except ValidationError:
        await query.answer("Имя 1–50, промпт ≥ 10", show_alert=True)
        await state.set_state(NewAssistantSG.name)
        await message.answer("Введите имя персоны (1–50 символов).\n/cancel — отмена")
        return
    chat_id = message.chat.id
    try:
        await assistant_storage.save_assistant(chat_id, profile)
        await assistant_storage.set_active_assistant(chat_id, profile.id)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await _reset_flow(state)
    logger.info("Сохранён ассистент chat_id=%s id=%s", chat_id, profile.id)
    await query.answer("Сохранено")
    verb = "Обновлён" if isinstance(edit_id, str) and edit_id else "Создан"
    await message.answer(
        f"✅ {verb} ассистент «{profile.name}» ({profile.id}).",
        reply_markup=menu,
    )


async def _persona_id_for_query(
    query: CallbackQuery, message: Message, assistant_storage: AssistantStorage
) -> str | None:
    """Id активной персоны. Ошибку профиля показывает во всплывающем окне."""
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return None
    return profile.id


async def _finish_topic_delete(
    chat_id: int,
    assistant_id: str,
    topic_id: str,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> tuple[str, InlineKeyboardMarkup, ReplyKeyboardMarkup]:
    """Удаляет тему. Пустой каталог получает General, меню берёт её имя."""
    await topic_storage.delete_topic(chat_id, assistant_id, topic_id)
    if not await topic_storage.get_active_topic(chat_id, assistant_id):
        await topic_storage.ensure_active(chat_id, assistant_id, "General")
    text, markup = await _topics_screen(chat_id, assistant_id, topic_storage)
    menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    return text, markup, menu


async def on_topics_refresh(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Перерисовывает список тем с первой страницы в том же сообщении."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
        active = await topic_storage.get_active_topic(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    await _edit_or_answer(
        query,
        _topics_text(topics, active),
        _topics_keyboard(topics, active),
    )


async def on_topics_page(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Открывает другую страницу списка тем."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    try:
        page = int((query.data or "").removeprefix("topic_page:"))
    except ValueError:
        await query.answer("Неверная страница", show_alert=True)
        return
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
        active = await topic_storage.get_active_topic(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    await _edit_or_answer(
        query,
        _topics_text(topics, active, page),
        _topics_keyboard(topics, active, page),
    )


async def on_topic_delete_ask(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Спрашивает удаление темы отдельным сообщением."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic_del_ask:")
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    name = next((item["name"] for item in topics if item["id"] == topic_id), None)
    if name is None:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    await query.answer()
    await message.answer(
        f"Удалить тему «{name}» со всей историей?",
        reply_markup=_confirm_row(
            "✅ Да, удалить", f"topic_del_yes:{topic_id}", "topic_list"
        ),
    )


async def on_topic_delete_yes(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Удаляет тему и историю, затем показывает список с первой страницы."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic_del_yes:")
    chat_id = message.chat.id
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        text, markup, menu = await _finish_topic_delete(
            chat_id, assistant_id, topic_id, topic_storage, assistant_storage
        )
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Удалена тема chat_id=%s topic_id=%s", chat_id, topic_id)
    await query.answer("Тема удалена")
    await _edit_or_answer(query, text, markup)
    await message.answer("Тема удалена.", reply_markup=menu)


async def on_topic_list(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Возвращает сообщение к списку тем."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        text, markup = await _topics_screen(
            message.chat.id, assistant_id, topic_storage
        )
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_TOPICS: message.message_id})
    await query.answer()


async def on_topic_use(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Делает тему активной, убирает экран и обновляет нижнее меню."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:use:")
    chat_id = message.chat.id
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        await topic_storage.set_active_topic(chat_id, assistant_id, topic_id)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Переключение темы chat_id=%s topic_id=%s", chat_id, topic_id)
    await query.answer()
    await _close_screen(message, state, _SCREEN_TOPICS)
    await _apply_reply_menu(message, menu, state)


async def on_topic_clear_ask(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Спрашивает подтверждение очистки истории темы."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:clear_ask:")
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    name = next((item["name"] for item in topics if item["id"] == topic_id), None)
    if name is None:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    await query.answer()
    await _edit_inline(
        message,
        f"Очистить историю темы «{name}»?",
        _confirm_row("✅ Да, очистить", f"topic:clear_yes:{topic_id}", "topic:list"),
    )


async def on_topic_clear_yes(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Очищает историю темы и показывает список."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:clear_yes:")
    chat_id = message.chat.id
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        await topic_storage.clear_topic(chat_id, assistant_id, topic_id)
        text, markup = await _topics_screen(chat_id, assistant_id, topic_storage)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Тема очищена chat_id=%s topic_id=%s", chat_id, topic_id)
    await query.answer("История очищена")
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_TOPICS: message.message_id})


async def on_topic_rename(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Запрашивает новое название темы."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:rename:")
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    name = next((item["name"] for item in topics if item["id"] == topic_id), None)
    if name is None:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    await _drop_flow(state)
    await state.set_state(RenameTopicSG.name)
    await state.update_data(flow_topic_id=topic_id, flow_topic_name=name)
    await query.answer()
    await message.answer(f"Введите новое название для темы «{name}».\n/cancel — отмена")


async def on_topic_del_ask(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Спрашивает подтверждение удаления темы."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:del_ask:")
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    name = next((item["name"] for item in topics if item["id"] == topic_id), None)
    if name is None:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    await query.answer()
    await _edit_inline(
        message,
        f"Удалить тему «{name}»?",
        _confirm_row("✅ Да, удалить", f"topic:del_yes:{topic_id}", "topic:list"),
    )


async def on_topic_del_yes(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Удаляет тему. Если активных не осталось, создаёт тему по умолчанию."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:del_yes:")
    chat_id = message.chat.id
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        text, markup, menu = await _finish_topic_delete(
            chat_id, assistant_id, topic_id, topic_storage, assistant_storage
        )
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Удалена тема chat_id=%s topic_id=%s", chat_id, topic_id)
    await query.answer("Тема удалена")
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_TOPICS: message.message_id})
    await message.answer("Тема удалена.", reply_markup=menu)


async def on_topic_new(query: CallbackQuery, state: FSMContext) -> None:
    """Запрашивает название новой темы."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    await _drop_flow(state)
    await state.set_state(NewTopicSG.name)
    await query.answer()
    await message.answer("Введите название темы (до 100 символов).\n/cancel — отмена")


async def sg_new_topic_name(
    message: Message,
    state: FSMContext,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Создаёт тему с введённым названием и делает её активной."""
    name = (message.text or "").strip()
    if not name or len(name) > 100:
        await message.answer(
            "Введите название темы (до 100 символов).\n/cancel — отмена"
        )
        return
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        topic_id = await topic_storage.create_topic(chat_id, profile.id, name)
        await topic_storage.set_active_topic(chat_id, profile.id, topic_id)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    await _reset_flow(state)
    logger.info("Новая тема chat_id=%s topic_id=%s", chat_id, topic_id)
    await message.answer(
        f"✅ Создана новая тема: '{name}'. Контекст обнулен.",
        reply_markup=menu,
    )
    await _open_topics(message, topic_storage, assistant_storage, state)


async def sg_rename_topic_name(
    message: Message,
    state: FSMContext,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает новое название выбранной темы."""
    name = (message.text or "").strip()
    if not name or len(name) > 100:
        await message.answer(
            "Введите новое название (до 100 символов).\n/cancel — отмена"
        )
        return
    data = await state.get_data()
    topic_id = data.get("flow_topic_id")
    chat_id = message.chat.id
    if not isinstance(topic_id, str) or not topic_id:
        await _reset_flow(state)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
        await message.answer("Тема для переименования не выбрана.", reply_markup=menu)
        return
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        await topic_storage.rename_topic(chat_id, profile.id, topic_id, name)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    await _reset_flow(state)
    logger.info("Тема переименована chat_id=%s topic_id=%s", chat_id, topic_id)
    await message.answer(f"Тема переименована в «{name}».", reply_markup=menu)
    await _open_topics(message, topic_storage, assistant_storage, state)


async def _refresh_image_screen(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Обновляет экран картинки после выбора модели или размера."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    data = await state.get_data()
    selected = _optional_model(data.get("image_model"))
    model_id = selected or assistant.config.default_image_model
    size = data.get("image_size")
    allowed = _size_choices(model_id)
    if not isinstance(size, str) or size not in allowed:
        size = _default_image_size(model_id)
        await state.update_data(image_size=size)
    await state.set_state(ImageSG.config)
    await query.answer()
    text, markup = _image_view(selected, size, assistant)
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_IMAGE: message.message_id})


async def on_image_model(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Меняет модель картинки на экране генерации, не записывая её в персону."""
    raw = (query.data or "").removeprefix("image:model:")
    if raw == "default":
        selected: str | None = None
    else:
        try:
            selected = AVAILABLE_IMAGE_MODELS[int(raw)]
        except (ValueError, IndexError):
            await query.answer("Неизвестная модель", show_alert=True)
            return
    data = await state.get_data()
    old_model = (
        _optional_model(data.get("image_model")) or assistant.config.default_image_model
    )
    new_model = selected or assistant.config.default_image_model
    fields: dict[str, Any] = {"image_model": selected}
    if _is_flux_image_model(old_model) != _is_flux_image_model(new_model):
        fields["image_size"] = _default_image_size(new_model)
    await state.update_data(fields)
    await _refresh_image_screen(query, state, assistant)


async def on_image_size(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Меняет размер картинки на экране генерации."""
    size = (query.data or "").removeprefix("image:size:")
    data = await state.get_data()
    selected = _optional_model(data.get("image_model"))
    model_id = selected or assistant.config.default_image_model
    if size not in _size_choices(model_id):
        await query.answer("Этот размер не подходит модели", show_alert=True)
        return
    await state.update_data(image_size=size)
    await _refresh_image_screen(query, state, assistant)


async def on_image_go(query: CallbackQuery, state: FSMContext) -> None:
    """Переходит к вводу описания картинки."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    await state.set_state(ImageSG.prompt)
    await query.answer()
    await message.answer("Опишите изображение.\n/cancel — отмена")


async def sg_image_prompt(
    message: Message,
    state: FSMContext,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
) -> None:
    """Генерирует картинку по описанию и выбору с экрана."""
    prompt = (message.text or "").strip()
    if not prompt:
        await message.answer("Опишите изображение.\n/cancel — отмена")
        return
    data = await state.get_data()
    model = _optional_model(data.get("image_model"))
    model_id = model or assistant.config.default_image_model
    size = data.get("image_size")
    if not isinstance(size, str) or size not in _size_choices(model_id):
        size = _default_image_size(model_id)
    await _reset_flow(state)
    menu = await _menu_for(message.chat.id, assistant_storage, topic_storage)
    await _generate_and_send(message, assistant, prompt, model, size, reply_markup=menu)


async def on_set_text_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает выбор текстовой модели отдельным сообщением."""
    await _open_model_menu(query, assistant, assistant_storage, state, kind="text")


async def on_set_image_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает выбор модели картинок отдельным сообщением."""
    await _open_model_menu(query, assistant, assistant_storage, state, kind="image")


async def on_settings_open(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Возвращает сообщение к настройкам активной персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    text, markup = _settings_view(profile, assistant)
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_SETTINGS: message.message_id})


async def on_settings_close(query: CallbackQuery, state: FSMContext) -> None:
    """Убирает настройки и открытый список моделей."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    await query.answer()
    data = await state.get_data()
    picker_id = data.get(_SCREEN_PICKER)
    bot = message.bot
    if (
        isinstance(picker_id, int)
        and bot is not None
        and picker_id != message.message_id
    ):
        try:
            await bot.delete_message(chat_id=message.chat.id, message_id=picker_id)
        except TelegramBadRequest as exc:
            logger.warning("Не удалось закрыть выбор модели: %s", exc)
        data.pop(_SCREEN_PICKER, None)
        await state.set_data(data)
    await _close_screen(message, state, _SCREEN_SETTINGS)


async def on_set_persona_size(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает размер картинки активной персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    size = (query.data or "").removeprefix("set_size:")
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
        model_id = _image_model_of(profile, assistant)
        if size not in _size_choices(model_id):
            await query.answer("Этот размер не подходит модели", show_alert=True)
            return
        updated = profile.model_copy(update={"image_size": size})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer("Размер обновлён")
    text, markup = _settings_view(updated, assistant)
    await _edit_or_answer(query, text, markup)


async def cmd_voice(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    state: FSMContext,
    command: CommandObject,
) -> None:
    """Открывает голосовые настройки или сразу ставит режим off/auto/always."""
    raw = (command.args or "").strip().casefold()
    if raw in {"off", "auto", "always"}:
        try:
            profile = await _active_profile(message.chat.id, assistant_storage)
            updated = profile.model_copy(update={"voice_mode": raw})
            await assistant_storage.save_assistant(message.chat.id, updated)
        except AssistantStorageError:
            await _reply_profile_error(message)
            return
        menu = await _menu_for(message.chat.id, assistant_storage, topic_storage)
        await message.answer(f"Голосовой режим: {raw}.", reply_markup=menu)
        return
    await on_menu_settings(message, assistant, assistant_storage, state)


async def on_voice_mode(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Переключает voice_mode активной персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    mode = (query.data or "").removeprefix("voice_mode:")
    if mode not in {"off", "auto", "always"}:
        await query.answer("Неизвестный режим", show_alert=True)
        return
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
        updated = profile.model_copy(update={"voice_mode": mode})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer(f"Режим: {mode}")
    text, markup = _settings_view(updated, assistant)
    await _edit_or_answer(query, text, markup)


async def on_voice_name(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Выбирает голос TTS активной персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    raw = (query.data or "").removeprefix("voice_name:")
    try:
        voice = TTS_VOICES[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестный голос", show_alert=True)
        return
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
        updated = profile.model_copy(update={"tts_voice": voice})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer(f"Голос: {voice}")
    text, markup = _settings_view(updated, assistant)
    await _edit_or_answer(query, text, markup)


async def on_stt_language(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Задаёт язык распознавания речи для активной персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    label = (query.data or "").removeprefix("stt_lang:")
    mapping = {item[0]: item[1] for item in STT_LANGUAGES}
    if label not in mapping:
        await query.answer("Неизвестный язык", show_alert=True)
        return
    language = mapping[label]
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
        updated = profile.model_copy(update={"stt_language": language})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer(f"STT: {label}")
    text, markup = _settings_view(updated, assistant)
    await _edit_or_answer(query, text, markup)


async def _open_model_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    *,
    kind: str,
) -> None:
    """Открывает выбор модели отдельным сообщением. Настройки остаются выше."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await state.update_data({_SCREEN_SETTINGS: message.message_id})
    text, markup = _model_picker_parts(profile, assistant, kind)
    await query.answer()
    sent = await message.answer(text, reply_markup=markup)
    await state.update_data({_SCREEN_PICKER: sent.message_id})


class _CommandLeavesWizard(BaseMiddleware):
    """Любая команда, кроме /cancel, выходит из мастера до своего хендлера."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, Message):
            text = event.text or ""
            state = data.get("state")
            if (
                isinstance(state, FSMContext)
                and text.startswith("/")
                and not text.startswith("/cancel")
            ):
                await _drop_flow(state)
        return await handler(event, data)


def _register_handlers(dispatcher: Dispatcher) -> None:
    """Команды регистрируются раньше общего текста, чтобы /image не ушёл в LLM.

    ``aiogram.filters.Text`` удалён в aiogram 3.31. Любой некомандный текст
    ловит ``F.text``.
    """
    dispatcher.message.middleware(_CleanChat())
    dispatcher.callback_query.middleware(_CleanChat())
    dispatcher.message.middleware(_CommandLeavesWizard())
    dispatcher.message.register(cmd_start, CommandStart())
    dispatcher.message.register(cmd_start, Command("help"))
    dispatcher.message.register(cmd_cancel, Command("cancel"))
    dispatcher.message.register(cmd_assistants, Command("assistants"))
    dispatcher.message.register(cmd_assistants, Command("profiles"))
    dispatcher.message.register(cmd_create_assistant, Command("create_assistant"))
    dispatcher.message.register(cmd_delete_assistant, Command("delete_assistant"))
    dispatcher.message.register(cmd_model, Command("model"))
    dispatcher.message.register(cmd_image_model, Command("image_model"))
    dispatcher.message.register(cmd_voice, Command("voice"))
    dispatcher.message.register(cmd_new_topic, Command("new_topic"))
    dispatcher.message.register(cmd_new_topic, Command("topic"))
    dispatcher.message.register(cmd_topics, Command("topics"))
    dispatcher.message.register(cmd_switch, Command("switch"))
    dispatcher.message.register(cmd_clear_topic, Command("clear_topic"))
    dispatcher.message.register(cmd_clear_topic, Command("clear"))
    dispatcher.message.register(cmd_image, Command("image"))
    dispatcher.message.register(
        on_menu_assistants,
        F.text.startswith(f"{_MENU_ASSISTANTS} (")
        | (F.text == MAIN_MENU_TEXT["assistants"]),
    )
    dispatcher.message.register(
        on_menu_topics,
        F.text.startswith(f"{_MENU_TOPICS} (") | (F.text == MAIN_MENU_TEXT["topics"]),
    )
    dispatcher.message.register(on_menu_image, F.text == MAIN_MENU_TEXT["image"])
    dispatcher.message.register(
        on_menu_settings,
        (F.text == _MENU_SETTINGS) | (F.text == MAIN_MENU_TEXT["settings"]),
    )
    dispatcher.callback_query.register(
        on_assistants_page, F.data.startswith("asst_page:")
    )
    dispatcher.callback_query.register(on_topics_page, F.data.startswith("topic_page:"))
    dispatcher.callback_query.register(
        on_asst_edit_ask, F.data.startswith("asst_edit:")
    )
    dispatcher.callback_query.register(on_topics_refresh, F.data == "topic_list")
    dispatcher.callback_query.register(
        on_topic_delete_ask, F.data.startswith("topic_del_ask:")
    )
    dispatcher.callback_query.register(
        on_topic_delete_yes, F.data.startswith("topic_del_yes:")
    )
    dispatcher.message.register(cmd_unknown, F.text.startswith("/"))
    dispatcher.message.register(sg_new_assistant_name, NewAssistantSG.name)
    dispatcher.message.register(sg_edit_prompt, EditAssistantSG.prompt)
    dispatcher.message.register(sg_new_assistant_prompt, NewAssistantSG.prompt)
    dispatcher.message.register(sg_new_assistant_models, NewAssistantSG.models)
    dispatcher.message.register(sg_new_topic_name, NewTopicSG.name)
    dispatcher.message.register(sg_rename_topic_name, RenameTopicSG.name)
    dispatcher.message.register(sg_image_prompt, ImageSG.prompt)
    dispatcher.callback_query.register(on_asst_list, F.data == "asst:list")
    dispatcher.callback_query.register(on_asst_use, F.data.startswith("asst:use:"))
    dispatcher.callback_query.register(on_asst_edit, F.data.startswith("asst:edit:"))
    dispatcher.callback_query.register(
        on_asst_del_ask, F.data.startswith("asst:del_ask:")
    )
    dispatcher.callback_query.register(
        on_asst_del_yes, F.data.startswith("asst:del_yes:")
    )
    dispatcher.callback_query.register(on_asst_new_start, F.data == "asst:new")
    dispatcher.callback_query.register(
        on_asst_new_text, F.data.startswith("asst:new_text:")
    )
    dispatcher.callback_query.register(
        on_asst_new_image, F.data.startswith("asst:new_image:")
    )
    dispatcher.callback_query.register(
        on_asst_new_default, F.data == "asst:new_default"
    )
    dispatcher.callback_query.register(on_asst_new_commit, F.data == "asst:new_commit")
    dispatcher.callback_query.register(on_topic_list, F.data == "topic:list")
    dispatcher.callback_query.register(on_topic_use, F.data.startswith("topic:use:"))
    dispatcher.callback_query.register(
        on_topic_clear_ask, F.data.startswith("topic:clear_ask:")
    )
    dispatcher.callback_query.register(
        on_topic_clear_yes, F.data.startswith("topic:clear_yes:")
    )
    dispatcher.callback_query.register(
        on_topic_rename, F.data.startswith("topic:rename:")
    )
    dispatcher.callback_query.register(
        on_topic_del_ask, F.data.startswith("topic:del_ask:")
    )
    dispatcher.callback_query.register(
        on_topic_del_yes, F.data.startswith("topic:del_yes:")
    )
    dispatcher.callback_query.register(on_topic_new, F.data == "topic:new")
    dispatcher.callback_query.register(
        on_image_model, F.data.startswith("image:model:")
    )
    dispatcher.callback_query.register(on_image_size, F.data.startswith("image:size:"))
    dispatcher.callback_query.register(on_image_go, F.data == "image:go")
    dispatcher.callback_query.register(on_settings_open, F.data == "settings:open")
    dispatcher.callback_query.register(on_settings_close, F.data == "settings:close")
    dispatcher.callback_query.register(on_set_text_menu, F.data == "set_text_menu")
    dispatcher.callback_query.register(on_set_image_menu, F.data == "set_image_menu")
    dispatcher.callback_query.register(
        on_set_persona_size, F.data.startswith("set_size:")
    )
    dispatcher.callback_query.register(on_voice_mode, F.data.startswith("voice_mode:"))
    dispatcher.callback_query.register(on_voice_name, F.data.startswith("voice_name:"))
    dispatcher.callback_query.register(on_stt_language, F.data.startswith("stt_lang:"))
    dispatcher.callback_query.register(
        on_set_text_model, F.data.startswith("set_text:")
    )
    dispatcher.callback_query.register(
        on_set_image_model, F.data.startswith("set_image:")
    )
    dispatcher.message.register(handle_voice, F.voice)
    dispatcher.message.register(handle_video_note_stub, F.video_note)
    dispatcher.message.register(handle_text, F.text & ~F.text.startswith("/"))


class _ChromeBot(Bot):
    """Запоминает исходящий текст, чтобы убрать его из чата.

    ``message.answer`` вызывает бота напрямую, минуя ``send_message``.
    Ответ персоны и невидимое сообщение с нижним меню отправляются с
    выключенным учётом и в чате остаются.
    """

    async def __call__(
        self,
        method: TelegramMethod[Any],
        request_timeout: int | None = None,
    ) -> Any:
        result = await super().__call__(method, request_timeout=request_timeout)
        if (
            not _KEEP_OUTGOING.get()
            and isinstance(method, SendMessage)
            and isinstance(result, Message)
        ):
            bucket = _CHROME_BUCKET.get()
            if bucket is not None and result.message_id not in bucket:
                bucket.append(result.message_id)
        return result


async def main() -> None:
    """Запускает long polling. Токен бота читается из TELEGRAM_BOT_TOKEN."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не задан")
        raise SystemExit(1)

    bot = _ChromeBot(token=token, session=AiohttpSession())
    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.startup.register(on_startup)
    dispatcher.shutdown.register(on_shutdown)
    _register_handlers(dispatcher)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
