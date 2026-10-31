"""Профили ассистентов: имя, системный промпт и выбранные модели.

У одного чата несколько персон. В запрос к модели уходит промпт активной.
Пустой ``REDIS_URL`` или недоступный Redis оставляют профили в памяти процесса.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis
from redis.exceptions import RedisError

from history_storage import RedisConfig

logger = logging.getLogger("AssistantStorage")

DEFAULT_ASSISTANT_ID = "standard"
DEFAULT_ASSISTANT_NAME = "Стандартный"
DEFAULT_SYSTEM_PROMPT = (
    "Ты полезный, вежливый и компетентный ассистент. "
    "Отвечай кратко, по делу, на русском языке. "
    "Если пользователь спрашивает о погоде или курсах валют, обязательно используй доступные функции для получения актуальных данных, а не полагайся на память. "
    "Если пользователь просит нарисовать, сгенерировать или создать изображение, схему или иллюстрацию, вызови generate_image и не отказывайся. "
    "Не выдумывай факты. Если информации недостаточно, честно скажи об этом."
)
_ASSISTANT_TTL_DAYS = 30
_T = TypeVar("_T")


class AssistantStorageError(Exception):
    """Сбой хранилища профилей. Бот остаётся живым и показывает текст пользователю."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AssistantProfile(BaseModel):
    """Персона чата. Пустые модели означают дефолт ``AssistAIConfig``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:8], min_length=1, max_length=32
    )
    name: str = Field(min_length=1, max_length=50)
    system_prompt: str = Field(min_length=10)
    text_model: str | None = None
    image_model: str | None = None
    image_size: str | None = None
    voice_mode: Literal["off", "auto", "always"] = "auto"
    tts_model: str | None = None
    tts_voice: str | None = None
    tts_format: str | None = None
    stt_language: str | None = None


def default_assistant_profile() -> AssistantProfile:
    """Базовый ассистент «Стандартный». Его id фиксирован, чтобы на него можно было вернуться."""
    return AssistantProfile(
        id=DEFAULT_ASSISTANT_ID,
        name=DEFAULT_ASSISTANT_NAME,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )


def _dump_profile(profile: AssistantProfile) -> str:
    return json.dumps(profile.model_dump(), ensure_ascii=False, separators=(",", ":"))


def _load_profile(raw: str) -> AssistantProfile | None:
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            logger.error("Пропущен профиль ассистента не-объект")
            return None
        return AssistantProfile.model_validate(payload)
    except (ValueError, TypeError):
        logger.error("Пропущен повреждённый профиль ассистента")
        return None


async def _redis_value(value: Awaitable[_T] | _T) -> _T:
    """redis-py типизирует ответ клиента и как значение, и как awaitable."""
    if isinstance(value, Awaitable):
        return await value
    return value


class AssistantStorage(ABC):
    """Общий интерфейс профилей одного чата."""

    @abstractmethod
    async def get_active_assistant(self, chat_id: int) -> AssistantProfile | None:
        """Возвращает активный профиль. Если профилей ещё не было, создаёт «Стандартный»."""

    @abstractmethod
    async def set_active_assistant(self, chat_id: int, profile_id: str) -> None:
        """Делает профиль активным. Неизвестный id — ``AssistantStorageError``."""

    @abstractmethod
    async def save_assistant(self, chat_id: int, profile: AssistantProfile) -> None:
        """Создаёт или заменяет профиль с тем же id. Активного сам не меняет."""

    @abstractmethod
    async def list_assistants(self, chat_id: int) -> list[AssistantProfile]:
        """Возвращает профили чата. «Стандартный» есть всегда и идёт первым."""

    @abstractmethod
    async def delete_assistant(self, chat_id: int, profile_id: str) -> None:
        """Удаляет профиль. «Стандартный» удалить нельзя.

        Если удалён активный, активным становится «Стандартный».
        """

    @abstractmethod
    async def close(self) -> None:
        """Освобождает соединение. Повторный вызов ничего не делает."""


def _ordered(profiles: dict[str, AssistantProfile]) -> list[AssistantProfile]:
    rest = [profiles[key] for key in profiles if key != DEFAULT_ASSISTANT_ID]
    rest.sort(key=lambda item: item.name.casefold())
    default = profiles.get(DEFAULT_ASSISTANT_ID)
    if default is None:
        return rest
    return [default, *rest]


class InMemoryAssistantStorage(AssistantStorage):
    """Профили в памяти процесса. Между инстансами бота не общие.

    При создании хранилища готовится шаблон «Стандартный». В чат он копируется
    при первом обращении, чтобы у каждого ``chat_id`` был свой профиль.
    """

    def __init__(self) -> None:
        self._default = default_assistant_profile()
        self._profiles: dict[int, dict[str, AssistantProfile]] = {}
        self._active: dict[int, str] = {}
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise AssistantStorageError("Хранилище профилей уже закрыто.")

    def _bucket(self, chat_id: int) -> dict[str, AssistantProfile]:
        bucket = self._profiles.get(chat_id)
        if bucket is None:
            bucket = {self._default.id: self._default.model_copy(deep=True)}
            self._profiles[chat_id] = bucket
            self._active[chat_id] = self._default.id
        elif DEFAULT_ASSISTANT_ID not in bucket:
            bucket[DEFAULT_ASSISTANT_ID] = self._default.model_copy(deep=True)
        if self._active.get(chat_id) not in bucket:
            self._active[chat_id] = DEFAULT_ASSISTANT_ID
        return bucket

    async def get_active_assistant(self, chat_id: int) -> AssistantProfile | None:
        """Возвращает копию активного профиля и создаёт «Стандартный», если чат пуст."""
        self._check_open()
        bucket = self._bucket(chat_id)
        active_id = self._active.get(chat_id, DEFAULT_ASSISTANT_ID)
        profile = bucket.get(active_id)
        if profile is None:
            return None
        return profile.model_copy(deep=True)

    async def set_active_assistant(self, chat_id: int, profile_id: str) -> None:
        """Переключает активный профиль. Чужой id в этом чате не принимается."""
        self._check_open()
        bucket = self._bucket(chat_id)
        if profile_id not in bucket:
            raise AssistantStorageError("Ассистент не найден.")
        self._active[chat_id] = profile_id
        logger.info("Память: активный ассистент chat_id=%s id=%s", chat_id, profile_id)

    async def save_assistant(self, chat_id: int, profile: AssistantProfile) -> None:
        """Записывает профиль. Повторный id заменяет имя, промпт и модели."""
        self._check_open()
        bucket = self._bucket(chat_id)
        bucket[profile.id] = profile.model_copy(deep=True)
        logger.info("Память: сохранён ассистент chat_id=%s id=%s", chat_id, profile.id)

    async def list_assistants(self, chat_id: int) -> list[AssistantProfile]:
        """Копии профилей чата. Базовый профиль всегда первый."""
        self._check_open()
        bucket = self._bucket(chat_id)
        return [item.model_copy(deep=True) for item in _ordered(bucket)]

    async def delete_assistant(self, chat_id: int, profile_id: str) -> None:
        """Удаляет профиль. Базовый остаётся, активный сбрасывается на него."""
        self._check_open()
        bucket = self._bucket(chat_id)
        if profile_id == DEFAULT_ASSISTANT_ID:
            raise AssistantStorageError(
                "Базового ассистента «Стандартный» удалить нельзя."
            )
        if profile_id not in bucket:
            raise AssistantStorageError("Ассистент не найден.")
        del bucket[profile_id]
        if self._active.get(chat_id) == profile_id:
            self._active[chat_id] = DEFAULT_ASSISTANT_ID
        logger.info("Память: удалён ассистент chat_id=%s id=%s", chat_id, profile_id)

    async def close(self) -> None:
        """Сбрасывает профили процесса. Повторный вызов ничего не делает."""
        if self._closed:
            return
        self._closed = True
        self._profiles.clear()
        self._active.clear()
        logger.info("Память: хранилище профилей закрыто")


class RedisAssistantStorage(AssistantStorage):
    """Профили в Redis Hash ``chat:{chat_id}:assistants``.

    Активный id лежит в ``chat:{chat_id}:active_assistant``.
    TTL обновляется при чтении и записи, чтобы живой чат не терял персон.
    """

    def __init__(self, config: RedisConfig, redis: Redis | None = None) -> None:
        self._ttl_seconds = _ASSISTANT_TTL_DAYS * 24 * 60 * 60
        self._owns_client = redis is None
        self._redis: Redis = redis or Redis.from_url(
            config.redis_url,
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        self._closed = False

    @staticmethod
    def _hash_key(chat_id: int) -> str:
        return f"chat:{chat_id}:assistants"

    @staticmethod
    def _active_key(chat_id: int) -> str:
        return f"chat:{chat_id}:active_assistant"

    def _check_open(self) -> None:
        if self._closed:
            raise AssistantStorageError("Хранилище профилей уже закрыто.")

    async def ping(self) -> None:
        """Проверяет соединение. Недоступный Redis становится ``AssistantStorageError``."""
        self._check_open()
        try:
            await self._redis.ping()
        except RedisError as exc:
            raise AssistantStorageError("Redis недоступен.") from exc

    async def _touch(self, chat_id: int) -> None:
        try:
            await self._redis.expire(self._hash_key(chat_id), self._ttl_seconds)
            await self._redis.expire(self._active_key(chat_id), self._ttl_seconds)
        except RedisError as exc:
            logger.exception(
                "Ошибка Redis при обновлении TTL профилей, chat_id=%s", chat_id
            )
            raise AssistantStorageError(
                "Профили ассистентов временно недоступны."
            ) from exc

    async def _load(self, chat_id: int) -> tuple[dict[str, AssistantProfile], str]:
        """Читает профили и гарантирует базового ассистента."""
        self._check_open()
        hash_key = self._hash_key(chat_id)
        active_key = self._active_key(chat_id)
        try:
            raw_map = await _redis_value(self._redis.hgetall(hash_key))
            active_raw = await _redis_value(self._redis.get(active_key))
        except RedisError as exc:
            logger.exception("Ошибка Redis при чтении профилей, chat_id=%s", chat_id)
            raise AssistantStorageError(
                "Профили ассистентов временно недоступны."
            ) from exc

        profiles: dict[str, AssistantProfile] = {}
        if isinstance(raw_map, dict):
            for field, payload in raw_map.items():
                text = payload if isinstance(payload, str) else str(payload)
                loaded = _load_profile(text)
                if loaded is None:
                    continue
                profile_id = field if isinstance(field, str) else str(field)
                profiles[profile_id] = loaded

        if DEFAULT_ASSISTANT_ID not in profiles:
            default = default_assistant_profile()
            profiles[default.id] = default
            try:
                await _redis_value(
                    self._redis.hset(hash_key, default.id, _dump_profile(default))
                )
            except RedisError as exc:
                logger.exception(
                    "Ошибка Redis при создании базового профиля, chat_id=%s", chat_id
                )
                raise AssistantStorageError(
                    "Профили ассистентов временно недоступны."
                ) from exc

        active = active_raw if isinstance(active_raw, str) else ""
        if active not in profiles:
            active = DEFAULT_ASSISTANT_ID
            try:
                await self._redis.set(active_key, active)
            except RedisError as exc:
                logger.exception(
                    "Ошибка Redis при записи активного профиля, chat_id=%s", chat_id
                )
                raise AssistantStorageError(
                    "Профили ассистентов временно недоступны."
                ) from exc
        await self._touch(chat_id)
        return profiles, active

    async def get_active_assistant(self, chat_id: int) -> AssistantProfile | None:
        """Возвращает активный профиль. Пустой чат получает «Стандартный»."""
        profiles, active = await self._load(chat_id)
        profile = profiles.get(active)
        if profile is None:
            return None
        return profile.model_copy(deep=True)

    async def set_active_assistant(self, chat_id: int, profile_id: str) -> None:
        """Записывает активный id, если такой профиль уже сохранён."""
        profiles, _active = await self._load(chat_id)
        if profile_id not in profiles:
            raise AssistantStorageError("Ассистент не найден.")
        try:
            await self._redis.set(
                self._active_key(chat_id), profile_id, ex=self._ttl_seconds
            )
            await self._redis.expire(self._hash_key(chat_id), self._ttl_seconds)
        except RedisError as exc:
            logger.exception("Ошибка Redis при смене ассистента, chat_id=%s", chat_id)
            raise AssistantStorageError(
                "Профили ассистентов временно недоступны."
            ) from exc
        logger.info("Redis: активный ассистент chat_id=%s id=%s", chat_id, profile_id)

    async def save_assistant(self, chat_id: int, profile: AssistantProfile) -> None:
        """Кладёт JSON профиля в hash. Активный id не переключается."""
        await self._load(chat_id)
        try:
            await _redis_value(
                self._redis.hset(
                    self._hash_key(chat_id), profile.id, _dump_profile(profile)
                )
            )
            await self._touch(chat_id)
        except RedisError as exc:
            logger.exception("Ошибка Redis при сохранении профиля, chat_id=%s", chat_id)
            raise AssistantStorageError(
                "Профили ассистентов временно недоступны."
            ) from exc
        logger.info("Redis: сохранён ассистент chat_id=%s id=%s", chat_id, profile.id)

    async def list_assistants(self, chat_id: int) -> list[AssistantProfile]:
        """Профили чата. Базовый создаётся при первом чтении и стоит первым."""
        profiles, _active = await self._load(chat_id)
        return [item.model_copy(deep=True) for item in _ordered(profiles)]

    async def delete_assistant(self, chat_id: int, profile_id: str) -> None:
        """Удаляет поле hash. Базовый профиль и его роль запасного активным не отдаются."""
        profiles, active = await self._load(chat_id)
        if profile_id == DEFAULT_ASSISTANT_ID:
            raise AssistantStorageError(
                "Базового ассистента «Стандартный» удалить нельзя."
            )
        if profile_id not in profiles:
            raise AssistantStorageError("Ассистент не найден.")
        try:
            await _redis_value(self._redis.hdel(self._hash_key(chat_id), profile_id))
            if active == profile_id:
                await self._redis.set(
                    self._active_key(chat_id),
                    DEFAULT_ASSISTANT_ID,
                    ex=self._ttl_seconds,
                )
            await self._touch(chat_id)
        except RedisError as exc:
            logger.exception("Ошибка Redis при удалении профиля, chat_id=%s", chat_id)
            raise AssistantStorageError(
                "Профили ассистентов временно недоступны."
            ) from exc
        logger.info("Redis: удалён ассистент chat_id=%s id=%s", chat_id, profile_id)

    async def close(self) -> None:
        """Закрывает свой клиент Redis. Чужой клиент, переданный снаружи, не закрывает."""
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            try:
                await self._redis.aclose()
            except RedisError:
                logger.exception("Ошибка при закрытии клиента Redis")
        logger.info("Redis: хранилище профилей закрыто")


async def create_assistant_storage(config: RedisConfig) -> AssistantStorage:
    """Возвращает Redis, если ``REDIS_URL`` задан и сервер отвечает.

    Пустой URL или недоступный Redis дают память процесса: бот стартует
    и без общих профилей.
    """
    if not config.redis_url:
        logger.info(
            "REDIS_URL не задан, профили ассистентов хранятся в памяти процесса"
        )
        return InMemoryAssistantStorage()

    storage = RedisAssistantStorage(config)
    try:
        await storage.ping()
    except AssistantStorageError as exc:
        logger.warning(
            "Redis недоступен, профили ассистентов переключены на память процесса: %s",
            exc.message,
        )
        await storage.close()
        return InMemoryAssistantStorage()
    logger.info(
        "Профили ассистентов хранятся в Redis, ttl_days=%s", _ASSISTANT_TTL_DAYS
    )
    return storage


__all__ = [
    "DEFAULT_ASSISTANT_ID",
    "DEFAULT_ASSISTANT_NAME",
    "DEFAULT_SYSTEM_PROMPT",
    "AssistantProfile",
    "AssistantStorage",
    "AssistantStorageError",
    "InMemoryAssistantStorage",
    "RedisAssistantStorage",
    "create_assistant_storage",
    "default_assistant_profile",
]
