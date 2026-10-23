# VPd06 — асинхронные клиенты для Telegram-бота

Python-модули для погоды, курсов валют и LLM через [ProxyAPI](https://proxyapi.ru/) с Function Calling.

## Состав

| Файл | Назначение |
|------|------------|
| `OpenWeatherAPI.py` | Асинхронный клиент OpenWeatherMap: текущая погода, прогноз, воздух, сравнение городов |
| `ExchangerateAPI.py` | Конвертация валют через `api.exchangerate.host` |
| `Assist_ai.py` | Единый клиент LLM: chat + tools + генерация изображений |
| `console_chat.py` | Консольный чат с выбором модели (только текст) |

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

Без ключа погоды или курса соответствующий инструмент просто не регистрируется. Без `PROXYAPI_API_KEY` клиент LLM не стартует.

## Быстрый старт

### Консольный чат

```powershell
python console_chat.py
```

Доступны модели:

1. `deepseek/deepseek-v4.1-flash` — стриминг и рассуждения
2. `openai/gpt-4o` — обычный ответ целиком

Команды в чате: `/help`, `/model`, `/clear`, `/exit`.

### Демо клиента LLM

```powershell
python Assist_ai.py
```

Проверяет Function Calling (погода, валюта) и генерацию изображения.

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
        answer = await client.chat_completion([
            {"role": "user", "content": "Какая погода в Казани?"}
        ])
        print(answer)

        # DeepSeek: поток + рассуждения
        async for delta in client.stream_chat(
            [{"role": "user", "content": "17 * 23?"}],
            model="deepseek/deepseek-v4.1-flash",
            reasoning_effort="high",
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

Инструменты вызываются только когда пользователь прямо спрашивает погоду или курс. В консольном чате это задано системным промптом.

## Зависимости

См. `requirements.txt`: `httpx`, `pydantic`, `python-dotenv`, `tenacity`, `openai` (опционально для совместимости окружения).

Сетевые запросы к API идут через `httpx`, без `requests` / `aiohttp`.
