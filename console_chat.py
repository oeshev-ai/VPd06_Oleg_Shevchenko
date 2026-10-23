"""Консольный чат с AssistAIClient.

Только текстовые модели. Генерации изображений здесь нет.
DeepSeek отвечает потоком и с включёнными рассуждениями.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from pydantic import ValidationError

from Assist_ai import AssistAIClient, AssistAIError, ChatDelta

DEEPSEEK_MODEL = "deepseek/deepseek-v4.1-flash"
GPT_MODEL = "openai/gpt-4o"
# ProxyAPI включает рассуждения полем reasoning_effort. high — рабочий уровень DeepSeek.
REASONING_EFFORT = "high"

MODELS: tuple[tuple[str, str], ...] = (
    (DEEPSEEK_MODEL, "рассуждения и стриминг"),
    (GPT_MODEL, "обычный ответ целиком"),
)

# Инструменты остаются в запросе, но не задают роль модели.
SYSTEM_PROMPT = (
    "Ты универсальный помощник. Отвечай на любые темы своими знаниями: "
    "объяснения, текст, код, перевод, расчёты и советы. "
    "Не отказывайся от обычного вопроса и не говори, что умеешь только погоду или валюты.\n"
    "Инструменты погоды и курса валют вызывай только когда пользователь прямо "
    "спрашивает погоду, прогноз, качество воздуха или конвертацию валют. "
    "В остальных ответах инструменты не вызывай и не перечисляй их. "
    "Не представляйся специалистом по погоде или валютам."
)


def _new_history() -> list[dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def _configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    logging.getLogger("AssistAI").setLevel(logging.WARNING)
    logging.getLogger("OpenWeatherAPI").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def _prompt(text: str) -> str:
    return (await asyncio.to_thread(input, text)).strip()


def _print_help(model: str) -> None:
    print(
        "\nКоманды: /help  /model  /clear  /exit\n"
        f"Текущая модель: {model}\n"
        "Можно спросить текущую погоду, прогноз, качество воздуха, "
        "сравнение городов или курс валют — модель вызовет нужный инструмент сама.\n"
    )


def _print_models() -> None:
    print("\nМодели:")
    for index, (model_id, note) in enumerate(MODELS, start=1):
        print(f"  {index}. {model_id} — {note}")


async def _choose_model(current: str | None = None) -> str:
    _print_models()
    while True:
        hint = "Номер или id модели"
        if current:
            hint += f" (сейчас {current}, Enter — оставить)"
        else:
            hint += " (Enter — 1)"
        raw = await _prompt(f"{hint}: ")
        if not raw:
            return current or MODELS[0][0]
        if raw.isdigit():
            number = int(raw)
            if 1 <= number <= len(MODELS):
                return MODELS[number - 1][0]
        known = {model_id for model_id, _note in MODELS}
        if raw in known:
            return raw
        print("Нет такой модели. Введите 1, 2 или полный id.")


async def _print_stream(
    client: AssistAIClient,
    messages: list[dict[str, Any]],
    model: str,
) -> None:
    started_reasoning = False
    started_answer = False
    async for delta in client.stream_chat(
        messages,
        model=model,
        reasoning_effort=REASONING_EFFORT,
    ):
        _print_delta(delta, started_reasoning, started_answer)
        if delta.kind == "reasoning":
            started_reasoning = True
        elif delta.kind == "content":
            started_answer = True
        elif delta.kind == "tool":
            started_reasoning = False
            started_answer = False
    if not started_answer and not started_reasoning:
        print("\nАссистент: (пустой ответ)")
    print()


def _print_delta(delta: ChatDelta, started_reasoning: bool, started_answer: bool) -> None:
    if delta.kind == "tool":
        if started_reasoning or started_answer:
            print()
        print(f"\n[инструмент] {delta.text}")
        return
    if delta.kind == "reasoning":
        if not started_reasoning:
            print("\nРассуждение: ", end="", flush=True)
        print(delta.text, end="", flush=True)
        return
    if not started_answer:
        if started_reasoning:
            print()
        print("\nАссистент: ", end="", flush=True)
    print(delta.text, end="", flush=True)


async def _ask(
    client: AssistAIClient,
    model: str,
    messages: list[dict[str, Any]],
    prompt: str,
) -> None:
    checkpoint = len(messages)
    messages.append({"role": "user", "content": prompt})
    try:
        if model == DEEPSEEK_MODEL:
            await _print_stream(client, messages, model)
            return
        print("\nЖду ответ...", flush=True)
        reply = await client.chat_completion(messages, model=model)
    except AssistAIError as exc:
        del messages[checkpoint:]
        print(f"\nОшибка: {exc.message}\n")
        return
    print(f"\nАссистент: {reply}\n")


async def main() -> None:
    """Читает реплики из консоли и отвечает выбранной текстовой моделью."""
    _configure_logging()
    try:
        client = AssistAIClient()
    except ValidationError as exc:
        print(f"Некорректная конфигурация: {exc}")
        raise SystemExit(1) from exc

    histories: dict[str, list[dict[str, Any]]] = {
        model_id: _new_history() for model_id, _note in MODELS
    }
    async with client:
        model = await _choose_model()
        print(
            f"\nМодель: {model}\n"
            "Команды: /help  /model  /clear  /exit"
        )
        if model == DEEPSEEK_MODEL:
            print(f"Рассуждения: {REASONING_EFFORT}, ответ идёт потоком.")
        while True:
            try:
                user_text = await _prompt("\nВы: ")
            except (EOFError, KeyboardInterrupt):
                print("\nВыход.")
                return
            if not user_text:
                continue
            command = user_text.lower()
            if command in {"/exit", "/quit", "exit", "quit"}:
                print("Выход.")
                return
            if command == "/help":
                _print_help(model)
                continue
            if command == "/clear":
                histories[model] = _new_history()
                print("История этой модели очищена.")
                continue
            if command == "/model":
                model = await _choose_model(model)
                print(f"Модель: {model}")
                if model == DEEPSEEK_MODEL:
                    print(f"Рассуждения: {REASONING_EFFORT}, ответ идёт потоком.")
                continue
            await _ask(client, model, histories[model], user_text)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nВыход.")
