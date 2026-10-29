"""
Production-ready Telegram-бот с интеграцией AssistAIClient.

Клиент LLM создаётся один раз в startup и живёт в ``workflow_data``.
Цикл Function Calling уже внутри ``chat_completion`` и здесь не повторяется.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    BotCommand,
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from dotenv import load_dotenv
from pydantic import ValidationError

from Assist_ai import AssistAIClient, AssistAIConfig, AssistAIError
from assistant_storage import (
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
WELCOME = (
    "Привет. Я ассистент: отвечаю на обычные вопросы и умею получать актуальные данные.\n\n"
    "Погода: текущая, прогноз, качество воздуха, сравнение городов.\n"
    "Валюты: конвертация суммы из одной валюты в другую.\n"
    "Изображения: генерация по текстовому описанию.\n\n"
    "Команды:\n"
    "/assistants — персоны и кнопка «Использовать»\n"
    '/create_assistant Имя "промпт" — новая персона\n'
    "/delete_assistant <id> — удалить персону\n"
    "/model — текстовая модель активного ассистента\n"
    "/image_model — модель картинок\n"
    "/new_topic <название> — новая тема, контекст с нуля\n"
    "/topics — список тем\n"
    "/switch <id или название> — вернуться к теме\n"
    "/clear_topic — очистить только текущую тему\n"
    "/image <описание> — картинка\n"
    "/stream <текст> — ответ потоком\n"
    "/help — эта справка"
)
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


async def _prepare_messages(
    chat_id: int,
    user_text: str,
    topic_id: str,
    storage: TopicHistoryStorage,
    profile: AssistantProfile,
) -> list[dict[str, Any]]:
    """Добавляет реплику пользователя и собирает запрос для активной персоны.

    В модель уходит системный промпт профиля и история темы без старых
    ``system``. ``chat_completion`` и ``stream_chat`` дописывают tool-сообщения
    в переданный список, поэтому это копия: черновики не попадают в Redis.
    """
    limit = storage.max_length
    await storage.append_message(
        chat_id,
        topic_id,
        {"role": "user", "content": user_text},
        limit,
    )
    stored = await storage.get_history(chat_id, topic_id)
    dialog = [dict(item) for item in stored if item.get("role") != "system"]
    return [{"role": "system", "content": profile.system_prompt}, *dialog]


async def _remember_assistant(
    chat_id: int,
    content: str,
    topic_id: str,
    storage: TopicHistoryStorage,
) -> None:
    """Сохраняет итоговый текст ассистента в историю активной темы."""
    await storage.append_message(
        chat_id,
        topic_id,
        {"role": "assistant", "content": content},
        storage.max_length,
    )


async def _active_topic(chat_id: int, storage: TopicHistoryStorage) -> str:
    """Возвращает активную тему. На первом сообщении создаёт тему General."""
    return await storage.ensure_active(chat_id, "General")


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
                BotCommand(command="new_topic", description="Новая тема диалога"),
                BotCommand(command="topics", description="Список тем"),
                BotCommand(command="switch", description="Переключить тему"),
                BotCommand(command="clear_topic", description="Очистить текущую тему"),
                BotCommand(command="image", description="Сгенерировать изображение"),
                BotCommand(command="stream", description="Ответ потоком"),
            ]
        )
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


async def cmd_start(message: Message) -> None:
    """Приветствие и список возможностей."""
    await message.answer(WELCOME)


async def cmd_unknown(message: Message) -> None:
    """Отвечает на команду, для которой нет отдельного хендлера."""
    await message.answer("Неизвестная команда. Список: /help")


async def cmd_new_topic(
    message: Message,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    command: CommandObject,
) -> None:
    """Создаёт тему, делает её активной и кладёт в неё промпт активной персоны."""
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
        topic_id = await topic_storage.create_topic(chat_id, name)
        await topic_storage.set_active_topic(chat_id, topic_id)
        await topic_storage.append_message(
            chat_id,
            topic_id,
            {"role": "system", "content": profile.system_prompt},
            topic_storage.max_length,
        )
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    logger.info("Новая тема chat_id=%s topic_id=%s", chat_id, topic_id)
    await message.answer(f"✅ Создана новая тема: '{name}'. Контекст обнулен.")


async def cmd_topics(message: Message, topic_storage: TopicHistoryStorage) -> None:
    """Показывает темы чата и отмечает активную."""
    chat_id = message.chat.id
    try:
        topics = await topic_storage.list_topics(chat_id)
        active = await topic_storage.get_active_topic(chat_id)
    except TopicStorageError:
        await _reply_history_error(message)
        return
    if not topics:
        await message.answer("Тем пока нет. Создайте: /new_topic название")
        return
    lines = ["Темы:"]
    for topic in topics:
        mark = "📍 " if topic["id"] == active else ""
        lines.append(f"{mark}{topic['id']} — {topic['name']}")
    await message.answer("\n".join(lines))


async def cmd_switch(
    message: Message,
    topic_storage: TopicHistoryStorage,
    command: CommandObject,
) -> None:
    """Переключает активную тему по id или точному названию."""
    query = (command.args or "").strip()
    if not query:
        await message.answer("Укажите id или название: /switch General")
        return
    chat_id = message.chat.id
    try:
        topics = await topic_storage.list_topics(chat_id)
        chosen = _match_topic(topics, query)
        if isinstance(chosen, str):
            await message.answer(chosen)
            return
        await topic_storage.set_active_topic(chat_id, chosen["id"])
    except TopicStorageError:
        await _reply_history_error(message)
        return
    logger.info("Переключение темы chat_id=%s topic_id=%s", chat_id, chosen["id"])
    await message.answer(f"🔄 Переключено на тему: '{chosen['name']}'.")


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


async def cmd_clear_topic(message: Message, topic_storage: TopicHistoryStorage) -> None:
    """Очищает историю только активной темы."""
    chat_id = message.chat.id
    try:
        topic_id = await topic_storage.get_active_topic(chat_id)
        if not topic_id:
            await message.answer("Нет активной темы. Создайте: /new_topic название")
            return
        topics = await topic_storage.list_topics(chat_id)
        await topic_storage.clear_topic(chat_id, topic_id)
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
    logger.info(
        "Генерация изображения, chat_id=%s, модель=%s, символов=%s",
        chat_id,
        image_model,
        len(prompt),
    )
    try:
        await message.answer("Генерирую изображение…")
        images = await assistant.generate_image(prompt=prompt, model=image_model)
        if not images:
            await message.answer(MSG_IMAGE_FAILED)
            return
        for index, image in enumerate(images):
            await _send_generated_image(message, image, index)
    except AssistAIError as exc:
        logger.error("AssistAI ошибка: %s", exc.message)
        await message.answer(MSG_IMAGE_FAILED)
    except httpx.TimeoutException:
        logger.error("Таймаут генерации изображения")
        await message.answer(MSG_TIMEOUT)
    except Exception:
        logger.exception("Непредвиденная ошибка генерации изображения")
        await message.answer(MSG_INTERNAL)


async def handle_text(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Обычный диалог в активной теме. Инструменты вызывает сам AssistAIClient."""
    user_text = (message.text or "").strip()
    if not user_text:
        return
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        topic_id = await _active_topic(chat_id, topic_storage)
        history_copy = await _prepare_messages(
            chat_id, user_text, topic_id, topic_storage, profile
        )
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    text_model = _text_model_of(profile, assistant)
    logger.info(
        "Текст, chat_id=%s topic_id=%s, модель=%s, сообщений в запросе=%s",
        chat_id,
        topic_id,
        text_model,
        len(history_copy),
    )
    try:
        if message.bot is not None:
            await message.bot.send_chat_action(chat_id=chat_id, action="typing")
        reply = await assistant.chat_completion(messages=history_copy, model=text_model)
    except (AssistAIError, httpx.TimeoutException) as exc:
        _log_handler_error(exc)
        await _rollback_user(chat_id, topic_storage, topic_id, message)
        await message.answer(_public_error(exc))
        return
    except Exception as exc:  # noqa: BLE001
        _log_handler_error(exc)
        await _rollback_user(chat_id, topic_storage, topic_id, message)
        await message.answer(MSG_INTERNAL)
        return
    try:
        await _remember_assistant(chat_id, reply, topic_id, topic_storage)
    except TopicStorageError:
        await _send_text(message, reply)
        await _reply_history_error(message)
        return
    await _send_text(message, reply)


async def _rollback_user(
    chat_id: int,
    storage: TopicHistoryStorage,
    topic_id: str,
    message: Message,
) -> None:
    try:
        await storage.rollback_last_user(chat_id, topic_id)
    except TopicStorageError:
        await _reply_history_error(message)


async def cmd_stream(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    command: CommandObject,
) -> None:
    """Показывает ответ по мере поступления и пишет его в активную тему."""
    prompt = (command.args or "").strip()
    if not prompt:
        await message.answer(
            "Напишите текст после команды: /stream сколько будет 17 * 23"
        )
        return
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        topic_id = await _active_topic(chat_id, topic_storage)
        history_copy = await _prepare_messages(
            chat_id, prompt, topic_id, topic_storage, profile
        )
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    text_model = _text_model_of(profile, assistant)
    status = await message.answer("Готовлю ответ…")
    parts: list[str] = []
    shown = ""
    last_edit = 0.0
    try:
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
        if not answer:
            await _rollback_user(chat_id, topic_storage, topic_id, message)
            await _edit_stream(status, "Пустой ответ.", shown)
            return
        visible = await _edit_stream(status, answer, shown)
        head = answer[:TELEGRAM_TEXT_LIMIT]
        if visible == head:
            await _send_overflow(message, answer)
        else:
            await _send_text(message, answer)
        try:
            await _remember_assistant(chat_id, answer, topic_id, topic_storage)
        except TopicStorageError:
            await _reply_history_error(message)
            return
        logger.info(
            "Стрим, chat_id=%s topic_id=%s, символов=%s", chat_id, topic_id, len(answer)
        )
    except (AssistAIError, httpx.TimeoutException) as exc:
        _log_handler_error(exc)
        await _rollback_user(chat_id, topic_storage, topic_id, message)
        await _report_stream_error(message, status, parts, _public_error(exc))
    except Exception as exc:  # noqa: BLE001
        _log_handler_error(exc)
        await _rollback_user(chat_id, topic_storage, topic_id, message)
        await _report_stream_error(message, status, parts, MSG_INTERNAL)


async def _report_stream_error(
    message: Message,
    status: Message,
    parts: list[str],
    text: str,
) -> None:
    if parts:
        await message.answer(text)
        return
    edited = await _edit_stream(status, text, "")
    if not edited:
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


def _assistants_keyboard(
    profiles: list[AssistantProfile], active_id: str
) -> InlineKeyboardMarkup:
    """Кнопка на каждую персону. У активной в подписи стоит отметка."""
    rows: list[list[InlineKeyboardButton]] = []
    for profile in profiles:
        mark = "✅ " if profile.id == active_id else ""
        label = f"{mark}Использовать: {profile.name}"
        if len(label) > 64:
            label = label[:61] + "..."
        rows.append(
            [InlineKeyboardButton(text=label, callback_data=f"use_asst:{profile.id}")]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


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


def _assistants_text(profiles: list[AssistantProfile], active_id: str) -> str:
    lines = ["Ассистенты:"]
    for profile in profiles:
        mark = "✅ " if profile.id == active_id else ""
        lines.append(f"{mark}{profile.id} — {profile.name}")
    lines.append("")
    lines.append("Нажмите «Использовать», чтобы переключить персону.")
    return "\n".join(lines)


def _chosen_model_line(selected: str | None, fallback: str) -> str:
    if selected is None:
        return f"{fallback} (по умолчанию)"
    return selected


async def cmd_assistants(message: Message, assistant_storage: AssistantStorage) -> None:
    """Показывает персоны чата и кнопки переключения."""
    chat_id = message.chat.id
    try:
        profiles = await assistant_storage.list_assistants(chat_id)
        active = await assistant_storage.get_active_assistant(chat_id)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    active_id = active.id if active is not None else ""
    await message.answer(
        _assistants_text(profiles, active_id),
        reply_markup=_assistants_keyboard(profiles, active_id),
    )


async def on_use_assistant(
    query: CallbackQuery, assistant_storage: AssistantStorage
) -> None:
    """Делает выбранную персону активной и обновляет список."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("use_asst:")
    chat_id = message.chat.id
    try:
        await assistant_storage.set_active_assistant(chat_id, profile_id)
        profiles = await assistant_storage.list_assistants(chat_id)
        active = await assistant_storage.get_active_assistant(chat_id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    active_id = active.id if active is not None else profile_id
    name = next((item.name for item in profiles if item.id == active_id), profile_id)
    await query.answer(f"Активен: {name}")
    try:
        await message.edit_text(
            _assistants_text(profiles, active_id),
            reply_markup=_assistants_keyboard(profiles, active_id),
        )
    except TelegramBadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            logger.warning("Не удалось обновить список ассистентов: %s", exc)


async def cmd_create_assistant(
    message: Message,
    assistant_storage: AssistantStorage,
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
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    logger.info("Создан ассистент chat_id=%s id=%s", chat_id, profile.id)
    await message.answer(
        f"✅ Создан ассистент «{profile.name}» ({profile.id}). Он теперь активен."
    )


async def cmd_delete_assistant(
    message: Message,
    assistant_storage: AssistantStorage,
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
        await assistant_storage.delete_assistant(chat_id, profile_id)
        active = await assistant_storage.get_active_assistant(chat_id)
    except AssistantStorageError as exc:
        await message.answer(exc.message)
        return
    active_name = active.name if active is not None else "Стандартный"
    logger.info("Удалён ассистент chat_id=%s id=%s", chat_id, profile_id)
    await message.answer(f"Ассистент «{name}» удалён. Активен «{active_name}».")


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
    await message.answer(
        f"{title} ассистента «{profile.name}»:\n{current}\n\nВыберите модель:",
        reply_markup=keyboard,
    )


async def on_set_text_model(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает текстовую модель в активный профиль."""
    await _apply_model_choice(
        query,
        assistant,
        assistant_storage,
        catalog=AVAILABLE_TEXT_MODELS,
        prefix="set_text:",
        kind="text",
    )


async def on_set_image_model(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает модель изображений в активный профиль."""
    await _apply_model_choice(
        query,
        assistant,
        assistant_storage,
        catalog=AVAILABLE_IMAGE_MODELS,
        prefix="set_image:",
        kind="image",
    )


async def _apply_model_choice(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
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
            updated = profile.model_copy(update={"image_model": selected})
            fallback = assistant.config.default_image_model
            title = "Модель изображений"
        else:
            updated = profile.model_copy(update={"text_model": selected})
            fallback = assistant.config.default_text_model
            title = "Текстовая модель"
        await assistant_storage.save_assistant(chat_id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    current = _chosen_model_line(selected, fallback)
    keyboard = _model_keyboard(catalog, prefix[:-1])
    await query.answer("Модель обновлена")
    try:
        await message.edit_text(
            f"{title} ассистента «{updated.name}»:\n{current}\n\nВыберите модель:",
            reply_markup=keyboard,
        )
    except TelegramBadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            logger.warning("Не удалось обновить выбор модели: %s", exc)


def _register_handlers(dispatcher: Dispatcher) -> None:
    """Команды регистрируются раньше общего текста, чтобы /image не ушёл в LLM.

    ``aiogram.filters.Text`` удалён в aiogram 3.31. Любой некомандный текст
    ловит ``F.text``.
    """
    dispatcher.message.register(cmd_start, CommandStart())
    dispatcher.message.register(cmd_start, Command("help"))
    dispatcher.message.register(cmd_assistants, Command("assistants"))
    dispatcher.message.register(cmd_assistants, Command("profiles"))
    dispatcher.message.register(cmd_create_assistant, Command("create_assistant"))
    dispatcher.message.register(cmd_delete_assistant, Command("delete_assistant"))
    dispatcher.message.register(cmd_model, Command("model"))
    dispatcher.message.register(cmd_image_model, Command("image_model"))
    dispatcher.message.register(cmd_new_topic, Command("new_topic"))
    dispatcher.callback_query.register(on_use_assistant, F.data.startswith("use_asst:"))
    dispatcher.callback_query.register(
        on_set_text_model, F.data.startswith("set_text:")
    )
    dispatcher.callback_query.register(
        on_set_image_model, F.data.startswith("set_image:")
    )
    dispatcher.message.register(cmd_new_topic, Command("topic"))
    dispatcher.message.register(cmd_topics, Command("topics"))
    dispatcher.message.register(cmd_switch, Command("switch"))
    dispatcher.message.register(cmd_clear_topic, Command("clear_topic"))
    dispatcher.message.register(cmd_clear_topic, Command("clear"))
    dispatcher.message.register(cmd_image, Command("image"))
    dispatcher.message.register(cmd_stream, Command("stream"))
    dispatcher.message.register(handle_text, F.text & ~F.text.startswith("/"))
    dispatcher.message.register(cmd_unknown, F.text.startswith("/"))


async def main() -> None:
    """Запускает long polling. Токен бота читается из TELEGRAM_BOT_TOKEN."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не задан")
        raise SystemExit(1)

    bot = Bot(token=token, session=AiohttpSession())
    dispatcher = Dispatcher()
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
