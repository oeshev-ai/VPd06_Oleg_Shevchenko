"""Темы диалога: память процесса и Redis через fakeredis."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from topic_history import (
    InMemoryTopicStorage,
    RedisTopicStorage,
    TopicConfig,
    TopicHistoryStorage,
    TopicStorageError,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def memory_storage() -> AsyncIterator[InMemoryTopicStorage]:
    storage = InMemoryTopicStorage(max_length=20)
    yield storage
    await storage.close()


@pytest_asyncio.fixture
async def redis_storage() -> AsyncIterator[RedisTopicStorage]:
    client = FakeAsyncRedis(decode_responses=True)
    config = TopicConfig(
        redis_url="redis://localhost:6379/0", topic_ttl_days=30, max_history_length=20
    )
    storage = RedisTopicStorage(config, redis=client)
    yield storage
    await storage.close()
    await client.aclose()


def _pick(
    name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> TopicHistoryStorage:
    if name == "memory":
        return memory_storage
    return redis_storage


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_topics_do_not_share_history(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    general = await storage.create_topic(1, "General")
    work = await storage.create_topic(1, "Работа")
    await storage.set_active_topic(1, work)
    await storage.append_message(1, general, {"role": "user", "content": "погода"}, 20)
    await storage.append_message(1, work, {"role": "user", "content": "отчёт"}, 20)

    assert await storage.get_active_topic(1) == work
    assert [item["content"] for item in await storage.get_history(1, general)] == [
        "погода"
    ]
    assert [item["content"] for item in await storage.get_history(1, work)] == ["отчёт"]


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_clear_topic_keeps_the_other(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    general = await storage.create_topic(2, "General")
    work = await storage.create_topic(2, "Работа")
    await storage.append_message(2, general, {"role": "user", "content": "a"}, 20)
    await storage.append_message(2, work, {"role": "user", "content": "b"}, 20)
    await storage.clear_topic(2, general)

    assert await storage.get_history(2, general) == []
    assert [item["content"] for item in await storage.get_history(2, work)] == ["b"]
    assert {item["id"] for item in await storage.list_topics(2)} == {general, work}


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_system_prompt_preserved_inside_topic(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    topic_id = await storage.create_topic(3, "General")
    await storage.append_message(3, topic_id, {"role": "system", "content": "sys"}, 20)
    for index in range(30):
        role = "user" if index % 2 == 0 else "assistant"
        await storage.append_message(
            3, topic_id, {"role": role, "content": str(index)}, 20
        )
    messages = await storage.get_history(3, topic_id)
    assert messages[0] == {"role": "system", "content": "sys"}
    assert len(messages) <= 20
    assert messages[1]["role"] != "assistant"


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_ensure_active_creates_general_once(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    first = await storage.ensure_active(4, "General")
    second = await storage.ensure_active(4, "General")
    topics = await storage.list_topics(4)
    assert first == second
    assert topics == [{"id": first, "name": "General"}]
    assert await storage.get_active_topic(4) == first


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_switch_unknown_topic(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    await storage.create_topic(5, "General")
    with pytest.raises(TopicStorageError):
        await storage.set_active_topic(5, "missing")


async def test_topic_ttl_is_set(redis_storage: RedisTopicStorage) -> None:
    topic_id = await redis_storage.create_topic(6, "General")
    await redis_storage.set_active_topic(6, topic_id)
    await redis_storage.append_message(
        6, topic_id, {"role": "user", "content": "привет"}, 20
    )
    history_ttl = await redis_storage._redis.ttl(f"chat:6:topic:{topic_id}:history")
    topics_ttl = await redis_storage._redis.ttl("chat:6:topics")
    assert 29 * 24 * 60 * 60 < history_ttl <= 30 * 24 * 60 * 60
    assert 29 * 24 * 60 * 60 < topics_ttl <= 30 * 24 * 60 * 60
