"""Хранилище истории диалогов: Redis или память процесса.

Сообщения лежат в формате OpenAI: ``{"role": "...", "content": "..."}``.
``AssistAIClient`` получает их как обычный ``list[dict]`` и сам список не
является хранилищем: в модель уходит копия.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis
from redis.exceptions import RedisError, WatchError

logger = logging.getLogger("HistoryStorage")

_APPEND_ATTEMPTS = 64


async def _redis_list(value: Awaitable[list[Any]] | list[Any]) -> list[Any]:
    """redis-py типизирует ответ pipeline и как список, и как awaitable."""
    if isinstance(value, Awaitable):
        return await value
    return value


class HistoryStorageError(Exception):
    """Сбой хранилища истории. Бот остаётся живым и показывает текст пользователю."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RedisConfig(BaseModel):
    """Настройки истории. Пустой ``redis_url`` означает память процесса."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    redis_url: str = ""
    history_ttl_days: int = Field(default=7, ge=1)
    max_history_length: int = Field(default=20, ge=1)

    @classmethod
    def from_env(cls) -> RedisConfig:
        """Читает ``REDIS_URL``, ``HISTORY_TTL_DAYS`` и ``MAX_HISTORY_LENGTH``."""
        load_dotenv(Path(__file__).resolve().parent / ".env")
        load_dotenv()
        return cls(
            redis_url=os.getenv("REDIS_URL", ""),
            history_ttl_days=os.getenv("HISTORY_TTL_DAYS", "7"),  # type: ignore[arg-type]
            max_history_length=os.getenv("MAX_HISTORY_LENGTH", "20"),  # type: ignore[arg-type]
        )


def trim_messages(
    messages: list[dict[str, Any]], max_length: int
) -> list[dict[str, Any]]:
    """Оставляет не больше ``max_length`` сообщений, самые новые.

    Системный промпт в начале списка не вытесняется. Если после среза
    сразу за ним оказывается ``assistant``, эта реплика отбрасывается:
    у модели не должно быть ответа без вопроса.
    """
    if max_length < 1:
        raise ValueError("max_length должен быть >= 1.")
    if len(messages) <= max_length:
        return [dict(item) for item in messages]

    has_system = bool(messages) and messages[0].get("role") == "system"
    system = dict(messages[0]) if has_system else None
    body = [dict(item) for item in (messages[1:] if has_system else messages)]
    max_body = max_length - (1 if has_system else 0)
    if len(body) > max_body:
        body = body[-max_body:]
    while body and body[0].get("role") == "assistant":
        body = body[1:]
    if system is None:
        return body
    return [system, *body]


def _dump_message(message: dict[str, Any]) -> str:
    if not isinstance(message, dict) or not isinstance(message.get("role"), str):
        raise HistoryStorageError(
            "Сообщение истории должно быть объектом с полем role."
        )
    try:
        return json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise HistoryStorageError("Сообщение истории нельзя записать в JSON.") from exc


def _load_message(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except ValueError:
        logger.error("Пропущена повреждённая запись истории")
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("role"), str):
        logger.error("Пропущена запись истории без role")
        return None
    return payload


class ChatHistoryStorage(ABC):
    """Общий интерфейс истории одного чата."""

    max_length: int

    @abstractmethod
    async def get_messages(self, chat_id: int) -> list[dict[str, Any]]:
        """Возвращает копию истории. Пустой чат — пустой список."""

    @abstractmethod
    async def append_message(
        self,
        chat_id: int,
        message: dict[str, Any],
        max_length: int,
    ) -> None:
        """Добавляет одно сообщение и обрезает историю до ``max_length``."""

    @abstractmethod
    async def append_many(
        self,
        chat_id: int,
        messages: list[dict[str, Any]],
        max_length: int,
    ) -> None:
        """Добавляет несколько сообщений одной операцией и обрезает историю."""

    @abstractmethod
    async def rollback_last_user(self, chat_id: int) -> None:
        """Удаляет последнюю реплику ``user``, если ответ модели не сохранился."""

    @abstractmethod
    async def clear(self, chat_id: int) -> None:
        """Удаляет историю чата."""

    @abstractmethod
    async def close(self) -> None:
        """Освобождает соединение. Повторный вызов ничего не делает."""


class InMemoryChatHistoryStorage(ChatHistoryStorage):
    """История в памяти процесса. Тот же лимит, что у Redis, без TTL.

    Подходит для локального запуска без Redis. Между инстансами бота
    эта история не общая и пропадает при остановке процесса.
    """

    def __init__(self, max_length: int = 20) -> None:
        if max_length < 1:
            raise ValueError("max_length должен быть >= 1.")
        self.max_length = max_length
        self._items: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise HistoryStorageError("Хранилище истории уже закрыто.")

    async def get_messages(self, chat_id: int) -> list[dict[str, Any]]:
        self._check_open()
        return [dict(item) for item in self._items.get(chat_id, [])]

    async def append_message(
        self,
        chat_id: int,
        message: dict[str, Any],
        max_length: int,
    ) -> None:
        await self.append_many(chat_id, [message], max_length)

    async def append_many(
        self,
        chat_id: int,
        messages: list[dict[str, Any]],
        max_length: int,
    ) -> None:
        self._check_open()
        dumped = [_dump_message(message) for message in messages]
        bucket = self._items[chat_id]
        for raw in dumped:
            loaded = _load_message(raw)
            if loaded is not None:
                bucket.append(loaded)
        self._items[chat_id] = trim_messages(bucket, max_length)
        logger.info(
            "Память: chat_id=%s, добавлено=%s, всего=%s",
            chat_id,
            len(messages),
            len(self._items[chat_id]),
        )

    async def rollback_last_user(self, chat_id: int) -> None:
        self._check_open()
        bucket = self._items.get(chat_id)
        if not bucket or bucket[-1].get("role") != "user":
            return
        bucket.pop()
        if not bucket:
            self._items.pop(chat_id, None)
        logger.info("Память: откат последней реплики user, chat_id=%s", chat_id)

    async def clear(self, chat_id: int) -> None:
        self._check_open()
        self._items.pop(chat_id, None)
        logger.info("Память: история очищена, chat_id=%s", chat_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._items.clear()
        logger.info("Память: хранилище истории закрыто")


class RedisChatHistoryStorage(ChatHistoryStorage):
    """История в Redis List ``chat:history:{chat_id}``.

    Добавление, обрезка и TTL выполняются одним ``MULTI/EXEC`` под ``WATCH``.
    Так два инстанса бота не теряют реплики и не оставляют список без срока жизни.

    ``LTRIM 0 max-1`` здесь не используется: при ``RPUSH`` он сохранил бы
    самые старые записи и вытеснил новые, вместе с ними пропал бы и системный
    промпт. Обрезка оставляет хвост списка и первое сообщение ``system``.
    """

    def __init__(self, config: RedisConfig, redis: Redis | None = None) -> None:
        self.max_length = config.max_history_length
        self._ttl_seconds = config.history_ttl_days * 24 * 60 * 60
        self._owns_client = redis is None
        self._redis: Redis = redis or Redis.from_url(
            config.redis_url,
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        self._closed = False

    @staticmethod
    def _key(chat_id: int) -> str:
        return f"chat:history:{chat_id}"

    def _check_open(self) -> None:
        if self._closed:
            raise HistoryStorageError("Хранилище истории уже закрыто.")

    async def ping(self) -> None:
        """Проверяет соединение. Недоступный Redis становится ``HistoryStorageError``."""
        self._check_open()
        try:
            await self._redis.ping()
        except RedisError as exc:
            raise HistoryStorageError("Redis недоступен.") from exc

    async def _read_modify(
        self,
        chat_id: int,
        mutator: Any,
    ) -> None:
        """Атомарно читает список, меняет его и записывает вместе с TTL."""
        self._check_open()
        key = self._key(chat_id)
        for _attempt in range(_APPEND_ATTEMPTS):
            try:
                async with self._redis.pipeline() as pipe:
                    await pipe.watch(key)
                    raw_items = await _redis_list(pipe.lrange(key, 0, -1))
                    current: list[dict[str, Any]] = []
                    for raw in raw_items:
                        loaded = _load_message(
                            raw if isinstance(raw, str) else raw.decode("utf-8")
                        )
                        if loaded is not None:
                            current.append(loaded)
                    updated = mutator(current)
                    pipe.multi()
                    pipe.delete(key)
                    if updated:
                        pipe.rpush(key, *[_dump_message(item) for item in updated])
                        pipe.expire(key, self._ttl_seconds)
                    await pipe.execute()
                    return
            except WatchError:
                continue
            except HistoryStorageError:
                raise
            except RedisError as exc:
                logger.exception("Ошибка Redis при записи истории, chat_id=%s", chat_id)
                raise HistoryStorageError(
                    "История диалога временно недоступна."
                ) from exc
        logger.error("Не удалось записать историю после повторов, chat_id=%s", chat_id)
        raise HistoryStorageError("История диалога временно недоступна.")

    async def get_messages(self, chat_id: int) -> list[dict[str, Any]]:
        self._check_open()
        try:
            raw_items = await _redis_list(self._redis.lrange(self._key(chat_id), 0, -1))
        except RedisError as exc:
            logger.exception("Ошибка Redis при чтении истории, chat_id=%s", chat_id)
            raise HistoryStorageError("История диалога временно недоступна.") from exc
        messages: list[dict[str, Any]] = []
        for raw in raw_items:
            loaded = _load_message(raw if isinstance(raw, str) else raw.decode("utf-8"))
            if loaded is not None:
                messages.append(loaded)
        logger.info("Redis: чтение chat_id=%s, сообщений=%s", chat_id, len(messages))
        return messages

    async def append_message(
        self,
        chat_id: int,
        message: dict[str, Any],
        max_length: int,
    ) -> None:
        await self.append_many(chat_id, [message], max_length)

    async def append_many(
        self,
        chat_id: int,
        messages: list[dict[str, Any]],
        max_length: int,
    ) -> None:
        prepared = [_load_message(_dump_message(message)) for message in messages]
        incoming = [item for item in prepared if item is not None]

        def mutate(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return trim_messages([*current, *incoming], max_length)

        await self._read_modify(chat_id, mutate)
        logger.info("Redis: chat_id=%s, добавлено=%s", chat_id, len(incoming))

    async def rollback_last_user(self, chat_id: int) -> None:
        def mutate(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if current and current[-1].get("role") == "user":
                return current[:-1]
            return current

        await self._read_modify(chat_id, mutate)
        logger.info("Redis: откат последней реплики user, chat_id=%s", chat_id)

    async def clear(self, chat_id: int) -> None:
        self._check_open()
        try:
            await self._redis.delete(self._key(chat_id))
        except RedisError as exc:
            logger.exception("Ошибка Redis при очистке истории, chat_id=%s", chat_id)
            raise HistoryStorageError("История диалога временно недоступна.") from exc
        logger.info("Redis: история очищена, chat_id=%s", chat_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            try:
                await self._redis.aclose()
            except RedisError:
                logger.exception("Ошибка при закрытии клиента Redis")
        logger.info("Redis: хранилище истории закрыто")


async def create_storage(config: RedisConfig) -> ChatHistoryStorage:
    """Возвращает Redis, если ``REDIS_URL`` задан и сервер отвечает.

    Пустой URL или недоступный Redis дают память процесса: бот стартует
    и без общей истории.
    """
    if not config.redis_url:
        logger.info("REDIS_URL не задан, история диалогов хранится в памяти процесса")
        return InMemoryChatHistoryStorage(config.max_history_length)

    storage = RedisChatHistoryStorage(config)
    try:
        await storage.ping()
    except HistoryStorageError as exc:
        logger.warning(
            "Redis недоступен, история диалогов переключена на память процесса: %s",
            exc.message,
        )
        await storage.close()
        return InMemoryChatHistoryStorage(config.max_history_length)
    logger.info(
        "История диалогов хранится в Redis, ttl_days=%s", config.history_ttl_days
    )
    return storage


async def _demo() -> None:
    """Проверяет обрезку и очистку на in-memory хранилище."""
    storage = InMemoryChatHistoryStorage(max_length=4)
    await storage.append_message(1, {"role": "system", "content": "sys"}, 4)
    for index in range(6):
        role = "user" if index % 2 == 0 else "assistant"
        await storage.append_message(1, {"role": role, "content": str(index)}, 4)
    messages = await storage.get_messages(1)
    print("roles:", [item["role"] for item in messages], "count:", len(messages))
    await storage.clear(1)
    print("after clear:", len(await storage.get_messages(1)))
    await storage.close()


if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(_demo())
