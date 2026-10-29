"""Тесты истории диалогов: память процесса и Redis через fakeredis."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from history_storage import (
    ChatHistoryStorage,
    InMemoryChatHistoryStorage,
    RedisChatHistoryStorage,
    RedisConfig,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def memory_storage() -> AsyncIterator[InMemoryChatHistoryStorage]:
    storage = InMemoryChatHistoryStorage(max_length=20)
    yield storage
    await storage.close()


@pytest_asyncio.fixture
async def redis_storage() -> AsyncIterator[RedisChatHistoryStorage]:
    client = FakeAsyncRedis(decode_responses=True)
    config = RedisConfig(
        redis_url="redis://localhost:6379/0",
        history_ttl_days=7,
        max_history_length=20,
    )
    storage = RedisChatHistoryStorage(config, redis=client)
    yield storage
    await storage.close()
    await client.aclose()


@pytest.fixture(params=["memory", "redis"])
def storage_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


async def _storage(
    name: str,
    memory_storage: InMemoryChatHistoryStorage,
    redis_storage: RedisChatHistoryStorage,
) -> ChatHistoryStorage:
    if name == "memory":
        return memory_storage
    return redis_storage


async def test_append_truncates_to_max_length(
    storage_name: str,
    memory_storage: InMemoryChatHistoryStorage,
    redis_storage: RedisChatHistoryStorage,
) -> None:
    storage = await _storage(storage_name, memory_storage, redis_storage)
    for index in range(25):
        await storage.append_message(7, {"role": "user", "content": f"m{index}"}, 20)
    messages = await storage.get_messages(7)
    assert len(messages) == 20
    assert messages[0]["content"] == "m5"
    assert messages[-1]["content"] == "m24"


async def test_system_prompt_preserved(
    storage_name: str,
    memory_storage: InMemoryChatHistoryStorage,
    redis_storage: RedisChatHistoryStorage,
) -> None:
    storage = await _storage(storage_name, memory_storage, redis_storage)
    await storage.append_message(3, {"role": "system", "content": "sys"}, 20)
    for index in range(30):
        role = "user" if index % 2 == 0 else "assistant"
        await storage.append_message(3, {"role": role, "content": str(index)}, 20)
    messages = await storage.get_messages(3)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "sys"
    assert len(messages) <= 20
    assert messages[1]["role"] != "assistant"


async def test_clear_removes_all(
    storage_name: str,
    memory_storage: InMemoryChatHistoryStorage,
    redis_storage: RedisChatHistoryStorage,
) -> None:
    storage = await _storage(storage_name, memory_storage, redis_storage)
    await storage.append_many(
        9,
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "привет"},
        ],
        20,
    )
    await storage.clear(9)
    assert await storage.get_messages(9) == []


async def test_ttl_is_set(redis_storage: RedisChatHistoryStorage) -> None:
    await redis_storage.append_message(4, {"role": "user", "content": "привет"}, 20)
    ttl = await redis_storage._redis.ttl("chat:history:4")
    assert 6 * 24 * 60 * 60 < ttl <= 7 * 24 * 60 * 60


async def test_concurrent_appends(
    storage_name: str,
    memory_storage: InMemoryChatHistoryStorage,
    redis_storage: RedisChatHistoryStorage,
) -> None:
    storage = await _storage(storage_name, memory_storage, redis_storage)

    async def send(index: int) -> None:
        await storage.append_message(11, {"role": "user", "content": str(index)}, 50)

    await asyncio.gather(*(send(index) for index in range(20)))
    messages = await storage.get_messages(11)
    assert len(messages) == 20
    assert {item["content"] for item in messages} == {str(index) for index in range(20)}


async def test_get_missing_chat_is_empty(
    memory_storage: InMemoryChatHistoryStorage,
) -> None:
    assert await memory_storage.get_messages(404) == []
