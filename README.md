# VPd06 — асинхронные клиенты для Telegram-бота

Python-модули для погоды, курсов валют и LLM через [ProxyAPI](https://proxyapi.ru/) с Function Calling.

## Состав

| Файл | Назначение |
|------|------------|
| `OpenWeatherAPI.py` | Асинхронный клиент OpenWeatherMap: текущая погода, прогноз, воздух, сравнение городов |
| `ExchangerateAPI.py` | Конвертация валют через `api.exchangerate.host` |
| `Assist_ai.py` | Единый клиент LLM: chat + tools + генерация изображений |
| `bot.py` | Telegram-бот на aiogram 3: диалог, стриминг, картинки |
| `history_storage.py` | История одного списка сообщений: Redis или память процесса |
| `topic_history.py` | Несколько тем диалога в одном чате |

## Требования

- Python 3.11+
- Ключи API (см. `.env.example`)

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Заполните `.env` реальными ключами. Файл `.env` в git не попадает.

## Переменные окружения

| Переменная | Для чего |
|------------|----------|
| `PROXYAPI_API_KEY` | Текст, tools и картинки через ProxyAPI |
| `OPENWEATHER_API_KEY` | Инструменты погоды |
| `EXCHANGERATE_API_KEY` | Конвертация валют |
| `TELEGRAM_BOT_TOKEN` | Токен бота от @BotFather, только для `bot.py` |
| `REDIS_URL` | Общая история диалогов. Пусто — память одного процесса |
| `HISTORY_TTL_DAYS` | TTL для старого хранилища одного списка сообщений |
| `TOPIC_TTL_DAYS` | Через сколько дней Redis удаляет неактивную тему. По умолчанию 30 |
| `MAX_HISTORY_LENGTH` | Сколько последних сообщений хранить в одной теме |

Без ключа погоды или курса соответствующий инструмент просто не регистрируется. Без `PROXYAPI_API_KEY` клиент LLM не стартует. Без `TELEGRAM_BOT_TOKEN` не стартует только Telegram-бот. Если Redis недоступен, бот пишет историю в память процесса.

## Быстрый старт

### Демо клиента LLM

```powershell
python Assist_ai.py
```

Проверяет Function Calling (погода, валюта) и генерацию изображения.

### Telegram-бот

```powershell
python bot.py
```

Нужен `TELEGRAM_BOT_TOKEN`. Один `AssistAIClient` создаётся при старте и закрывается при остановке. Темы диалогов хранятся в Redis (`docker compose up`), а без `REDIS_URL` — в памяти процесса. В одной теме остаётся не больше `MAX_HISTORY_LENGTH` сообщений, и в модель уходит только активная тема.

Команды: `/help`, `/assistants`, `/create_assistant`, `/delete_assistant`, `/model`, `/image_model`, `/new_topic`, `/topics`, `/switch`, `/clear_topic`, `/image <описание>`. Обычный текст идёт в `stream_chat`: ответ языковой модели печатается в чат по мере генерации, с промптом и моделью активного ассистента. Профили лежат в Redis или в памяти процесса, отдельно от тем.

### Только погода

```powershell
python OpenWeatherAPI.py
```

## AssistAIClient

```python
import asyncio
from Assist_ai import AssistAIClient


async def main() -> None:
    async with AssistAIClient() as client:
        answer = await client.chat_completion(
            [{"role": "user", "content": "Какая погода в Казани?"}]
        )
        print(answer)

        async for delta in client.stream_chat(
            [{"role": "user", "content": "17 * 23?"}],
            model="deepseek/deepseek-v4.1-flash",
        ):
            if delta.kind == "content":
                print(delta.text, end="", flush=True)


asyncio.run(main())
```

### Инструменты модели

- `get_weather_by_city` / `get_weather_by_coords` — текущая погода и суточный прогноз
- `get_forecast_by_city` / `get_forecast_by_coords` — прогноз с шагом 3 часа
- `get_air_pollution` — качество воздуха
- `compare_cities` — сравнение двух городов
- `format_notification` — готовый текст уведомления
- `convert_currency` — конвертация суммы

Инструменты погоды и курса вызывает модель, когда они нужны для ответа.

## Зависимости

См. `requirements.txt`: `httpx`, `pydantic`, `python-dotenv`, `tenacity`, `aiogram` 3.x, `redis` 5.x, `openai` (опционально для совместимости окружения).

Запросы к ProxyAPI, погоде и курсу идут через `httpx`. Telegram-транспорт у aiogram — `aiohttp`.
