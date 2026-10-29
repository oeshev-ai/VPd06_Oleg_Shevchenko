"""История диалогов по темам: Redis или память процесса.

У одного чата несколько независимых контекстов. В модель уходит только
история активной темы, в формате OpenAI ``{"role": "...", "content": "..."}``.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis
from redis.exceptions import RedisError, WatchError

from history_storage import trim_messages

logger = logging.getLogger("TopicHistory")

_APPEND_ATTEMPTS = 64
_DEFAULT_TOPIC_NAME = "General"


async def _redis_list(value: Awaitable[list[Any]] | list[Any]) -> list[Any]:
    """redis-py типизирует ответ pipeline и как список, и как awaitable."""
    if isinstance(value, Awaitable):
        return await value
    return value


class TopicStorageError(Exception):
    """Сбой хранилища тем. Бот остаётся живым и показывает текст пользователю."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TopicConfig(BaseModel):
    """Настройки тем. Пустой ``redis_url`` означает память процесса."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    redis_url: str = ""
    topic_ttl_days: int = Field(default=30, ge=1)
    max_history_length: int = Field(default=20, ge=1)

    @classmethod
    def from_env(cls) -> TopicConfig:
        """Читает ``REDIS_URL``, ``TOPIC_TTL_DAYS`` и ``MAX_HISTORY_LENGTH``."""
        load_dotenv(Path(__file__).resolve().parent / ".env")
        load_dotenv()
        return cls(
            redis_url=os.getenv("REDIS_URL", ""),
            topic_ttl_days=os.getenv("TOPIC_TTL_DAYS", "30"),  # type: ignore[arg-type]
            max_history_length=os.getenv("MAX_HISTORY_LENGTH", "20"),  # type: ignore[arg-type]
        )


def _new_topic_id(existing: set[str]) -> str:
    for _ in range(10):
        topic_id = uuid.uuid4().hex[:8]
        if topic_id not in existing:
            return topic_id
    raise TopicStorageError("Не удалось создать идентификатор темы.")


def _clean_topic_name(topic_name: str) -> str:
    if not isinstance(topic_name, str):
        raise TopicStorageError("Название темы должно быть строкой.")
    name = topic_name.strip()
    if not name:
        raise TopicStorageError("Название темы не может быть пустым.")
    if len(name) > 100:
        raise TopicStorageError("Название темы длиннее 100 символов.")
    return name


def _dump_message(message: dict[str, Any]) -> str:
    if not isinstance(message, dict) or not isinstance(message.get("role"), str):
        raise TopicStorageError("Сообщение истории должно быть объектом с полем role.")
    try:
        return json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TopicStorageError("Сообщение истории нельзя записать в JSON.") from exc


def _load_message(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except ValueError:
        logger.error("Пропущена повреждённая запись истории темы")
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("role"), str):
        logger.error("Пропущена запись истории темы без role")
        return None
    return payload


def _dump_topic(topic_id: str, name: str) -> str:
    return json.dumps(
        {"id": topic_id, "name": name}, ensure_ascii=False, separators=(",", ":")
    )


def _load_topic(raw: str) -> dict[str, str] | None:
    try:
        payload = json.loads(raw)
    except ValueError:
        logger.error("Пропущена повреждённая запись темы")
        return None
    if not isinstance(payload, dict):
        return None
    topic_id = payload.get("id")
    name = payload.get("name")
    if (
        not isinstance(topic_id, str)
        or not isinstance(name, str)
        or not topic_id
        or not name
    ):
        logger.error("Пропущена тема без id или названия")
        return None
    return {"id": topic_id, "name": name}


class _ChatTopics:
    """Темы и истории одного чата в памяти процесса."""

    def __init__(self) -> None:
        self.active = ""
        self.order: list[str] = []
        self.names: dict[str, str] = {}
        self.histories: dict[str, list[dict[str, Any]]] = {}

    def as_list(self) -> list[dict[str, str]]:
        return [
            {"id": topic_id, "name": self.names[topic_id]} for topic_id in self.order
        ]


class TopicHistoryStorage(ABC):
    """История нескольких тем внутри одного чата."""

    max_length: int

    @abstractmethod
    async def get_active_topic(self, chat_id: int) -> str:
        """Возвращает id активной темы. Если темы нет, строка пустая."""

    @abstractmethod
    async def set_active_topic(self, chat_id: int, topic_id: str) -> None:
        """Делает тему активной. Неизвестная тема — ``TopicStorageError``."""

    @abstractmethod
    async def create_topic(self, chat_id: int, topic_name: str) -> str:
        """Создаёт тему и возвращает её id. Активную тему не меняет."""

    @abstractmethod
    async def list_topics(self, chat_id: int) -> list[dict[str, str]]:
        """Возвращает темы чата в порядке создания: ``id`` и ``name``."""

    @abstractmethod
    async def get_history(self, chat_id: int, topic_id: str) -> list[dict[str, Any]]:
        """Возвращает копию истории темы. Неизвестная тема — пустой список."""

    @abstractmethod
    async def append_message(
        self,
        chat_id: int,
        topic_id: str,
        message: dict[str, Any],
        max_length: int,
    ) -> None:
        """Добавляет сообщение в тему и обрезает историю до ``max_length``."""

    @abstractmethod
    async def clear_topic(self, chat_id: int, topic_id: str) -> None:
        """Удаляет сообщения темы. Сама тема и остальные темы остаются."""

    @abstractmethod
    async def rollback_last_user(self, chat_id: int, topic_id: str) -> None:
        """Убирает последнюю реплику ``user``, если ответ модели не сохранился."""

    @abstractmethod
    async def ensure_active(
        self, chat_id: int, default_name: str = _DEFAULT_TOPIC_NAME
    ) -> str:
        """Возвращает активную тему, а если её нет — создаёт тему по умолчанию."""

    @abstractmethod
    async def close(self) -> None:
        """Освобождает соединение. Повторный вызов ничего не делает."""


class InMemoryTopicStorage(TopicHistoryStorage):
    """Темы в памяти процесса. Между инстансами бота они не общие."""

    def __init__(self, max_length: int = 20) -> None:
        if max_length < 1:
            raise ValueError("max_length должен быть >= 1.")
        self.max_length = max_length
        self._chats: dict[int, _ChatTopics] = {}
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise TopicStorageError("Хранилище тем уже закрыто.")

    def _chat(self, chat_id: int) -> _ChatTopics:
        return self._chats.setdefault(chat_id, _ChatTopics())

    async def get_active_topic(self, chat_id: int) -> str:
        self._check_open()
        chat = self._chats.get(chat_id)
        if chat is None or chat.active not in chat.names:
            return ""
        return chat.active

    async def set_active_topic(self, chat_id: int, topic_id: str) -> None:
        self._check_open()
        chat = self._chat(chat_id)
        if topic_id not in chat.names:
            raise TopicStorageError("Тема не найдена.")
        chat.active = topic_id
        logger.info("Память: активная тема chat_id=%s topic_id=%s", chat_id, topic_id)

    async def create_topic(self, chat_id: int, topic_name: str) -> str:
        self._check_open()
        name = _clean_topic_name(topic_name)
        chat = self._chat(chat_id)
        topic_id = _new_topic_id(set(chat.names))
        chat.names[topic_id] = name
        chat.order.append(topic_id)
        chat.histories[topic_id] = []
        logger.info("Память: создана тема chat_id=%s topic_id=%s", chat_id, topic_id)
        return topic_id

    async def list_topics(self, chat_id: int) -> list[dict[str, str]]:
        self._check_open()
        chat = self._chats.get(chat_id)
        if chat is None:
            return []
        return chat.as_list()

    async def get_history(self, chat_id: int, topic_id: str) -> list[dict[str, Any]]:
        self._check_open()
        chat = self._chats.get(chat_id)
        if chat is None:
            return []
        return [dict(item) for item in chat.histories.get(topic_id, [])]

    async def append_message(
        self,
        chat_id: int,
        topic_id: str,
        message: dict[str, Any],
        max_length: int,
    ) -> None:
        self._check_open()
        chat = self._chat(chat_id)
        if topic_id not in chat.names:
            raise TopicStorageError("Тема не найдена.")
        loaded = _load_message(_dump_message(message))
        if loaded is None:
            raise TopicStorageError("Сообщение истории нельзя сохранить.")
        bucket = chat.histories.setdefault(topic_id, [])
        bucket.append(loaded)
        chat.histories[topic_id] = trim_messages(bucket, max_length)
        logger.info(
            "Память: chat_id=%s topic_id=%s всего=%s",
            chat_id,
            topic_id,
            len(chat.histories[topic_id]),
        )

    async def clear_topic(self, chat_id: int, topic_id: str) -> None:
        self._check_open()
        chat = self._chats.get(chat_id)
        if chat is None or topic_id not in chat.names:
            raise TopicStorageError("Тема не найдена.")
        chat.histories[topic_id] = []
        logger.info(
            "Память: история темы очищена chat_id=%s topic_id=%s", chat_id, topic_id
        )

    async def rollback_last_user(self, chat_id: int, topic_id: str) -> None:
        self._check_open()
        chat = self._chats.get(chat_id)
        if chat is None:
            return
        bucket = chat.histories.get(topic_id)
        if not bucket or bucket[-1].get("role") != "user":
            return
        bucket.pop()
        logger.info("Память: откат user chat_id=%s topic_id=%s", chat_id, topic_id)

    async def ensure_active(
        self, chat_id: int, default_name: str = _DEFAULT_TOPIC_NAME
    ) -> str:
        self._check_open()
        chat = self._chat(chat_id)
        if chat.active in chat.names:
            return chat.active
        name = _clean_topic_name(default_name)
        topic_id = _new_topic_id(set(chat.names))
        chat.names[topic_id] = name
        chat.order.append(topic_id)
        chat.histories[topic_id] = []
        chat.active = topic_id
        logger.info(
            "Память: тема по умолчанию chat_id=%s topic_id=%s", chat_id, topic_id
        )
        return topic_id

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._chats.clear()
        logger.info("Память: хранилище тем закрыто")


class RedisTopicStorage(TopicHistoryStorage):
    """Темы в Redis.

    Список тем: ``chat:{chat_id}:topics``.
    Активная тема: ``chat:{chat_id}:active``.
    История: ``chat:{chat_id}:topic:{topic_id}:history``.

    Запись истории и каталога тем идёт одним ``MULTI/EXEC`` под ``WATCH``.
    У ключей стоит TTL, чтобы неактивные темы удалялись сами.
    """

    def __init__(self, config: TopicConfig, redis: Redis | None = None) -> None:
        self.max_length = config.max_history_length
        self._ttl_seconds = config.topic_ttl_days * 24 * 60 * 60
        self._owns_client = redis is None
        self._redis: Redis = redis or Redis.from_url(
            config.redis_url,
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        self._closed = False

    @staticmethod
    def _topics_key(chat_id: int) -> str:
        return f"chat:{chat_id}:topics"

    @staticmethod
    def _active_key(chat_id: int) -> str:
        return f"chat:{chat_id}:active"

    @staticmethod
    def _history_key(chat_id: int, topic_id: str) -> str:
        return f"chat:{chat_id}:topic:{topic_id}:history"

    def _check_open(self) -> None:
        if self._closed:
            raise TopicStorageError("Хранилище тем уже закрыто.")

    async def ping(self) -> None:
        """Проверяет соединение. Недоступный Redis становится ``TopicStorageError``."""
        self._check_open()
        try:
            await self._redis.ping()
        except RedisError as exc:
            raise TopicStorageError("Redis недоступен.") from exc

    def _decode_topics(self, raw_items: list[Any]) -> list[dict[str, str]]:
        topics: list[dict[str, str]] = []
        for raw in raw_items:
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            loaded = _load_topic(text)
            if loaded is not None:
                topics.append(loaded)
        return topics

    def _decode_messages(self, raw_items: list[Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for raw in raw_items:
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            loaded = _load_message(text)
            if loaded is not None:
                messages.append(loaded)
        return messages

    async def _update_catalog(self, chat_id: int, mutator: Any) -> Any:
        """Атомарно меняет список тем и активную тему."""
        self._check_open()
        topics_key = self._topics_key(chat_id)
        active_key = self._active_key(chat_id)
        for _attempt in range(_APPEND_ATTEMPTS):
            try:
                async with self._redis.pipeline() as pipe:
                    await pipe.watch(topics_key, active_key)
                    raw_topics = await _redis_list(pipe.lrange(topics_key, 0, -1))
                    active_raw = await pipe.get(active_key)
                    active = active_raw if isinstance(active_raw, str) else ""
                    topics = self._decode_topics(raw_topics)
                    new_topics, new_active, result = mutator(topics, active)
                    pipe.multi()
                    pipe.delete(topics_key)
                    if new_topics:
                        pipe.rpush(
                            topics_key,
                            *[
                                _dump_topic(item["id"], item["name"])
                                for item in new_topics
                            ],
                        )
                        pipe.expire(topics_key, self._ttl_seconds)
                    if new_active:
                        pipe.set(active_key, new_active, ex=self._ttl_seconds)
                    else:
                        pipe.delete(active_key)
                    await pipe.execute()
                    return result
            except WatchError:
                continue
            except TopicStorageError:
                raise
            except RedisError as exc:
                logger.exception("Ошибка Redis при записи тем, chat_id=%s", chat_id)
                raise TopicStorageError("История диалога временно недоступна.") from exc
        logger.error("Не удалось записать темы после повторов, chat_id=%s", chat_id)
        raise TopicStorageError("История диалога временно недоступна.")

    async def _update_history(self, chat_id: int, topic_id: str, mutator: Any) -> None:
        """Атомарно меняет историю одной темы и обновляет TTL каталога."""
        self._check_open()
        history_key = self._history_key(chat_id, topic_id)
        topics_key = self._topics_key(chat_id)
        active_key = self._active_key(chat_id)
        for _attempt in range(_APPEND_ATTEMPTS):
            try:
                async with self._redis.pipeline() as pipe:
                    await pipe.watch(history_key, topics_key)
                    raw_topics = await _redis_list(pipe.lrange(topics_key, 0, -1))
                    known = {item["id"] for item in self._decode_topics(raw_topics)}
                    if topic_id not in known:
                        raise TopicStorageError("Тема не найдена.")
                    raw_items = await _redis_list(pipe.lrange(history_key, 0, -1))
                    updated = mutator(self._decode_messages(raw_items))
                    pipe.multi()
                    pipe.delete(history_key)
                    if updated:
                        pipe.rpush(
                            history_key, *[_dump_message(item) for item in updated]
                        )
                        pipe.expire(history_key, self._ttl_seconds)
                    pipe.expire(topics_key, self._ttl_seconds)
                    pipe.expire(active_key, self._ttl_seconds)
                    await pipe.execute()
                    return
            except WatchError:
                continue
            except TopicStorageError:
                raise
            except RedisError as exc:
                logger.exception(
                    "Ошибка Redis при записи истории темы, chat_id=%s topic_id=%s",
                    chat_id,
                    topic_id,
                )
                raise TopicStorageError("История диалога временно недоступна.") from exc
        logger.error(
            "Не удалось записать историю темы после повторов, chat_id=%s topic_id=%s",
            chat_id,
            topic_id,
        )
        raise TopicStorageError("История диалога временно недоступна.")

    async def get_active_topic(self, chat_id: int) -> str:
        self._check_open()
        try:
            active = await self._redis.get(self._active_key(chat_id))
            raw_topics = await _redis_list(
                self._redis.lrange(self._topics_key(chat_id), 0, -1)
            )
        except RedisError as exc:
            logger.exception(
                "Ошибка Redis при чтении активной темы, chat_id=%s", chat_id
            )
            raise TopicStorageError("История диалога временно недоступна.") from exc
        if not isinstance(active, str) or not active:
            return ""
        known = {item["id"] for item in self._decode_topics(raw_topics)}
        return active if active in known else ""

    async def set_active_topic(self, chat_id: int, topic_id: str) -> None:
        def mutate(
            topics: list[dict[str, str]],
            _active: str,
        ) -> tuple[list[dict[str, str]], str, None]:
            if topic_id not in {item["id"] for item in topics}:
                raise TopicStorageError("Тема не найдена.")
            return topics, topic_id, None

        await self._update_catalog(chat_id, mutate)
        logger.info("Redis: активная тема chat_id=%s topic_id=%s", chat_id, topic_id)

    async def create_topic(self, chat_id: int, topic_name: str) -> str:
        name = _clean_topic_name(topic_name)

        def mutate(
            topics: list[dict[str, str]],
            active: str,
        ) -> tuple[list[dict[str, str]], str, str]:
            topic_id = _new_topic_id({item["id"] for item in topics})
            return [*topics, {"id": topic_id, "name": name}], active, topic_id

        topic_id = await self._update_catalog(chat_id, mutate)
        logger.info("Redis: создана тема chat_id=%s topic_id=%s", chat_id, topic_id)
        return str(topic_id)

    async def list_topics(self, chat_id: int) -> list[dict[str, str]]:
        self._check_open()
        try:
            raw_topics = await _redis_list(
                self._redis.lrange(self._topics_key(chat_id), 0, -1)
            )
        except RedisError as exc:
            logger.exception("Ошибка Redis при чтении списка тем, chat_id=%s", chat_id)
            raise TopicStorageError("История диалога временно недоступна.") from exc
        return self._decode_topics(raw_topics)

    async def get_history(self, chat_id: int, topic_id: str) -> list[dict[str, Any]]:
        self._check_open()
        try:
            raw_items = await _redis_list(
                self._redis.lrange(self._history_key(chat_id, topic_id), 0, -1)
            )
        except RedisError as exc:
            logger.exception(
                "Ошибка Redis при чтении истории темы, chat_id=%s topic_id=%s",
                chat_id,
                topic_id,
            )
            raise TopicStorageError("История диалога временно недоступна.") from exc
        messages = self._decode_messages(raw_items)
        logger.info(
            "Redis: чтение chat_id=%s topic_id=%s сообщений=%s",
            chat_id,
            topic_id,
            len(messages),
        )
        return messages

    async def append_message(
        self,
        chat_id: int,
        topic_id: str,
        message: dict[str, Any],
        max_length: int,
    ) -> None:
        loaded = _load_message(_dump_message(message))
        if loaded is None:
            raise TopicStorageError("Сообщение истории нельзя сохранить.")

        def mutate(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return trim_messages([*current, loaded], max_length)

        await self._update_history(chat_id, topic_id, mutate)
        logger.info(
            "Redis: добавлено сообщение chat_id=%s topic_id=%s", chat_id, topic_id
        )

    async def clear_topic(self, chat_id: int, topic_id: str) -> None:
        def mutate(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
            del current
            return []

        await self._update_history(chat_id, topic_id, mutate)
        logger.info(
            "Redis: история темы очищена chat_id=%s topic_id=%s", chat_id, topic_id
        )

    async def rollback_last_user(self, chat_id: int, topic_id: str) -> None:
        def mutate(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if current and current[-1].get("role") == "user":
                return current[:-1]
            return current

        await self._update_history(chat_id, topic_id, mutate)
        logger.info("Redis: откат user chat_id=%s topic_id=%s", chat_id, topic_id)

    async def ensure_active(
        self, chat_id: int, default_name: str = _DEFAULT_TOPIC_NAME
    ) -> str:
        name = _clean_topic_name(default_name)

        def mutate(
            topics: list[dict[str, str]],
            active: str,
        ) -> tuple[list[dict[str, str]], str, str]:
            known = {item["id"] for item in topics}
            if active in known:
                return topics, active, active
            topic_id = _new_topic_id(known)
            return [*topics, {"id": topic_id, "name": name}], topic_id, topic_id

        topic_id = await self._update_catalog(chat_id, mutate)
        logger.info(
            "Redis: активная тема готова chat_id=%s topic_id=%s", chat_id, topic_id
        )
        return str(topic_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            try:
                await self._redis.aclose()
            except RedisError:
                logger.exception("Ошибка при закрытии клиента Redis")
        logger.info("Redis: хранилище тем закрыто")


async def create_topic_storage(config: TopicConfig) -> TopicHistoryStorage:
    """Возвращает Redis, если ``REDIS_URL`` задан и сервер отвечает.

    Пустой URL или недоступный Redis дают память процесса.
    """
    if not config.redis_url:
        logger.info("REDIS_URL не задан, темы диалогов хранятся в памяти процесса")
        return InMemoryTopicStorage(config.max_history_length)

    storage = RedisTopicStorage(config)
    try:
        await storage.ping()
    except TopicStorageError as exc:
        logger.warning(
            "Redis недоступен, темы диалогов переключены на память процесса: %s",
            exc.message,
        )
        await storage.close()
        return InMemoryTopicStorage(config.max_history_length)
    logger.info("Темы диалогов хранятся в Redis, ttl_days=%s", config.topic_ttl_days)
    return storage


async def _demo() -> None:
    """Показывает, что две темы одного чата не смешивают историю."""
    storage = InMemoryTopicStorage(max_length=20)
    first = await storage.create_topic(1, "General")
    second = await storage.create_topic(1, "Работа")
    await storage.set_active_topic(1, first)
    await storage.append_message(1, first, {"role": "user", "content": "погода"}, 20)
    await storage.append_message(1, second, {"role": "user", "content": "отчёт"}, 20)
    print("active:", await storage.get_active_topic(1))
    print("topics:", await storage.list_topics(1))
    print("general:", len(await storage.get_history(1, first)))
    print("work:", len(await storage.get_history(1, second)))
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
