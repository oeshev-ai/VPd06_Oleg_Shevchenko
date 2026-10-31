"""Темы диалога: память процесса и Redis через fakeredis."""

from __future__ import annotations

import json
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

AID = "standard"
OTHER = "other"


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
    general = await storage.create_topic(1, AID, "General")
    work = await storage.create_topic(1, AID, "Работа")
    await storage.set_active_topic(1, AID, work)
    await storage.append_message(
        1, AID, general, {"role": "user", "content": "погода"}, 20
    )
    await storage.append_message(1, AID, work, {"role": "user", "content": "отчёт"}, 20)

    assert await storage.get_active_topic(1, AID) == work
    assert [item["content"] for item in await storage.get_history(1, AID, general)] == [
        "погода"
    ]
    assert [item["content"] for item in await storage.get_history(1, AID, work)] == [
        "отчёт"
    ]


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_clear_topic_keeps_the_other(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    general = await storage.create_topic(2, AID, "General")
    work = await storage.create_topic(2, AID, "Работа")
    await storage.append_message(2, AID, general, {"role": "user", "content": "a"}, 20)
    await storage.append_message(2, AID, work, {"role": "user", "content": "b"}, 20)
    await storage.clear_topic(2, AID, general)

    assert await storage.get_history(2, AID, general) == []
    assert [item["content"] for item in await storage.get_history(2, AID, work)] == [
        "b"
    ]
    assert {item["id"] for item in await storage.list_topics(2, AID)} == {general, work}


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_system_prompt_preserved_inside_topic(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    topic_id = await storage.create_topic(3, AID, "General")
    await storage.append_message(
        3, AID, topic_id, {"role": "system", "content": "sys"}, 20
    )
    for index in range(30):
        role = "user" if index % 2 == 0 else "assistant"
        await storage.append_message(
            3, AID, topic_id, {"role": role, "content": str(index)}, 20
        )
    messages = await storage.get_history(3, AID, topic_id)
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
    first = await storage.ensure_active(4, AID, "General")
    second = await storage.ensure_active(4, AID, "General")
    topics = await storage.list_topics(4, AID)
    assert first == second
    assert topics == [{"id": first, "name": "General"}]
    assert await storage.get_active_topic(4, AID) == first


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_switch_unknown_topic(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    await storage.create_topic(5, AID, "General")
    with pytest.raises(TopicStorageError):
        await storage.set_active_topic(5, AID, "missing")


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_delete_topic_moves_active_and_drops_history(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    general = await storage.create_topic(7, AID, "General")
    work = await storage.create_topic(7, AID, "Работа")
    await storage.set_active_topic(7, AID, work)
    await storage.append_message(7, AID, work, {"role": "user", "content": "x"}, 20)

    await storage.delete_topic(7, AID, work)

    assert await storage.get_active_topic(7, AID) == general
    assert await storage.list_topics(7, AID) == [{"id": general, "name": "General"}]
    assert await storage.get_history(7, AID, work) == []
    if isinstance(storage, RedisTopicStorage):
        assert (
            await storage._redis.get(f"chat:7:asst:{AID}:topic:{work}:history") is None
        )


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_delete_last_topic_clears_active(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    only = await storage.create_topic(8, AID, "Одна")
    await storage.set_active_topic(8, AID, only)

    await storage.delete_topic(8, AID, only)

    assert await storage.get_active_topic(8, AID) == ""
    assert await storage.list_topics(8, AID) == []


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_rename_topic_and_unknown(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    topic_id = await storage.create_topic(9, AID, "Старое")
    await storage.rename_topic(9, AID, topic_id, "  Новое  ")
    assert await storage.list_topics(9, AID) == [{"id": topic_id, "name": "Новое"}]
    with pytest.raises(TopicStorageError):
        await storage.rename_topic(9, AID, "missing", "Имя")
    with pytest.raises(TopicStorageError):
        await storage.delete_topic(9, AID, "missing")


async def test_topic_ttl_is_set(redis_storage: RedisTopicStorage) -> None:
    topic_id = await redis_storage.create_topic(6, AID, "General")
    await redis_storage.set_active_topic(6, AID, topic_id)
    await redis_storage.append_message(
        6, AID, topic_id, {"role": "user", "content": "привет"}, 20
    )
    history_ttl = await redis_storage._redis.ttl(
        f"chat:6:asst:{AID}:topic:{topic_id}:history"
    )
    topics_ttl = await redis_storage._redis.ttl(f"chat:6:asst:{AID}:topics")
    assert 29 * 24 * 60 * 60 < history_ttl <= 30 * 24 * 60 * 60
    assert 29 * 24 * 60 * 60 < topics_ttl <= 30 * 24 * 60 * 60


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_assistants_do_not_share_topics(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    own = await storage.create_topic(11, AID, "Своя")
    await storage.create_topic(11, OTHER, "Чужая")
    await storage.set_active_topic(11, AID, own)

    assert [item["name"] for item in await storage.list_topics(11, AID)] == ["Своя"]
    assert [item["name"] for item in await storage.list_topics(11, OTHER)] == ["Чужая"]
    assert await storage.get_active_topic(11, OTHER) == ""


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_delete_assistant_topics_keeps_the_other(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    topic_id = await storage.create_topic(12, AID, "Уйдёт")
    kept = await storage.create_topic(12, OTHER, "Останется")
    await storage.append_message(
        12, AID, topic_id, {"role": "user", "content": "x"}, 20
    )
    await storage.delete_assistant_topics(12, AID)

    assert await storage.list_topics(12, AID) == []
    assert await storage.get_history(12, AID, topic_id) == []
    assert await storage.list_topics(12, OTHER) == [{"id": kept, "name": "Останется"}]


async def test_legacy_catalog_moves_to_the_reader(
    redis_storage: RedisTopicStorage,
) -> None:
    """Старый каталог чата забирает первая персона, которая его прочитала."""
    client = redis_storage._redis
    await client.rpush(
        "chat:10:topics",
        json.dumps({"id": "abc", "name": "Старая"}, ensure_ascii=False),
    )
    await client.set("chat:10:active", "abc")
    await client.rpush(
        "chat:10:topic:abc:history",
        json.dumps({"role": "user", "content": "привет"}, ensure_ascii=False),
    )

    topics = await redis_storage.list_topics(10, AID)
    assert topics == [{"id": "abc", "name": "Старая"}]
    assert await redis_storage.get_active_topic(10, AID) == "abc"
    assert await redis_storage.get_history(10, AID, "abc") == [
        {"role": "user", "content": "привет"}
    ]
    assert await client.get("chat:10:topics") is None
    assert await client.lrange("chat:10:topic:abc:history", 0, -1) == []
    assert await redis_storage.list_topics(10, OTHER) == []
