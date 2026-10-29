"""Профили ассистентов: память процесса и Redis через fakeredis."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from assistant_storage import (
    DEFAULT_ASSISTANT_ID,
    DEFAULT_ASSISTANT_NAME,
    AssistantProfile,
    AssistantStorage,
    AssistantStorageError,
    InMemoryAssistantStorage,
    RedisAssistantStorage,
)
from history_storage import RedisConfig

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def memory_storage() -> AsyncIterator[InMemoryAssistantStorage]:
    storage = InMemoryAssistantStorage()
    yield storage
    await storage.close()


@pytest_asyncio.fixture
async def redis_storage() -> AsyncIterator[RedisAssistantStorage]:
    client = FakeAsyncRedis(decode_responses=True)
    config = RedisConfig(redis_url="redis://localhost:6379/0")
    storage = RedisAssistantStorage(config, redis=client)
    yield storage
    await storage.close()
    await client.aclose()


def _pick(
    name: str,
    memory_storage: InMemoryAssistantStorage,
    redis_storage: RedisAssistantStorage,
) -> AssistantStorage:
    if name == "memory":
        return memory_storage
    return redis_storage


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_empty_chat_gets_standard_assistant(
    storage_name: str,
    memory_storage: InMemoryAssistantStorage,
    redis_storage: RedisAssistantStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    active = await storage.get_active_assistant(7)
    assert active is not None
    assert active.id == DEFAULT_ASSISTANT_ID
    assert active.name == DEFAULT_ASSISTANT_NAME
    listed = await storage.list_assistants(7)
    assert [item.id for item in listed] == [DEFAULT_ASSISTANT_ID]


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_save_switch_and_models_stay_on_one_chat(
    storage_name: str,
    memory_storage: InMemoryAssistantStorage,
    redis_storage: RedisAssistantStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    profile = AssistantProfile(
        name="Синоптик",
        system_prompt="Ты отвечаешь только про погоду и кратко.",
        text_model="openai/gpt-4o-mini",
        image_model="openai/gpt-image-1-mini",
    )
    await storage.save_assistant(1, profile)
    await storage.set_active_assistant(1, profile.id)

    active = await storage.get_active_assistant(1)
    assert active is not None
    assert active.name == "Синоптик"
    assert active.text_model == "openai/gpt-4o-mini"
    assert active.image_model == "openai/gpt-image-1-mini"

    other = await storage.get_active_assistant(2)
    assert other is not None
    assert other.id == DEFAULT_ASSISTANT_ID


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_delete_active_falls_back_to_standard(
    storage_name: str,
    memory_storage: InMemoryAssistantStorage,
    redis_storage: RedisAssistantStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    profile = AssistantProfile(
        name="Черновик",
        system_prompt="Отвечай одной короткой фразой.",
    )
    await storage.save_assistant(3, profile)
    await storage.set_active_assistant(3, profile.id)
    await storage.delete_assistant(3, profile.id)

    active = await storage.get_active_assistant(3)
    assert active is not None
    assert active.id == DEFAULT_ASSISTANT_ID
    assert profile.id not in {item.id for item in await storage.list_assistants(3)}


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_standard_assistant_cannot_be_deleted(
    storage_name: str,
    memory_storage: InMemoryAssistantStorage,
    redis_storage: RedisAssistantStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    await storage.get_active_assistant(4)
    with pytest.raises(AssistantStorageError, match="Стандартный"):
        await storage.delete_assistant(4, DEFAULT_ASSISTANT_ID)
    active = await storage.get_active_assistant(4)
    assert active is not None
    assert active.id == DEFAULT_ASSISTANT_ID


async def test_redis_profile_ttl_is_set(redis_storage: RedisAssistantStorage) -> None:
    await redis_storage.get_active_assistant(5)
    hash_ttl = await redis_storage._redis.ttl("chat:5:assistants")
    active_ttl = await redis_storage._redis.ttl("chat:5:active_assistant")
    day = 24 * 60 * 60
    assert 29 * day <= hash_ttl <= 30 * day
    assert 29 * day <= active_ttl <= 30 * day
