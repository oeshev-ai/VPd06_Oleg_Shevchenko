"""Асинхронный клиент LLM для Telegram-бота.

``AssistAIClient`` ходит в ProxyAPI (совместимый с OpenAI формат):
текстовые ответы с Function Calling и генерация изображений.
Погода и конвертация валют подключены как инструменты, которые модель
запрашивает сама, а выполняет этот модуль.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import ssl
import sys
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Ключи лежат рядом с модулем. Загружаем их до импорта клиентов,
# чтобы те увидели то же окружение, даже если процесс запущен не из каталога проекта.
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()

from ExchangerateAPI import Conversion, ExchangeAPIError, get_exchange_rate
from OpenWeatherAPI import (
    AirPollutionData,
    CurrentWeather,
    DailyForecast,
    OpenWeatherAPIError,
    OpenWeatherClient,
    OWMConfig,
)

# URL с access_key не должен попадать в лог при уровне INFO.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("AssistAI")

_CHAT_PATH = "chat/completions"
_IMAGES_PATH = "images/generations"
_TRANSCRIBE_PATH = "audio/transcriptions"
_SPEECH_PATH = "audio/speech"
# Картинка и длинный потоковый ответ могут занимать до двух минут.
_IMAGE_READ_TIMEOUT = 180.0
_STREAM_READ_TIMEOUT = 180.0
_TRANSCRIBE_READ_TIMEOUT = 120.0
_SPEECH_READ_TIMEOUT = 120.0


def _ssl_context() -> ssl.SSLContext:
    """Контекст проверки TLS по хранилищу ОС, а не только по certifi.

    Kaspersky подменяет сертификат своим корнем из хранилища Windows.
    httpx с certifi такой цепочке не доверяет и рвёт запрос.
    """
    return ssl.create_default_context()


def _error_json(message: str) -> str:
    """JSON-строка, которую можно безопасно вернуть модели вместо падения диалога."""
    return json.dumps({"error": message}, ensure_ascii=False)


def _exception_text(exc: Exception) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _as_text(content: Any) -> str:
    """Приводит ``message.content`` к строке. Мультимодальный content сериализуется в JSON."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _response_error_text(response: httpx.Response) -> str:
    """Достаёт человекочитаемое сообщение об ошибке из тела ответа ProxyAPI."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)[:500]
        if isinstance(error, str) and error:
            return error[:500]
    return response.text[:500]


_SSE_SKIP = object()
_SSE_DONE = object()


def _parse_sse_data(line: str) -> Any:
    """Разбирает одну строку SSE.

    ``_SSE_SKIP`` — строку нужно пропустить, ``_SSE_DONE`` — поток закончился,
    словарь — очередной JSON-фрагмент.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return _SSE_SKIP
    if not stripped.startswith("data:"):
        return _SSE_SKIP
    data = stripped[5:].strip()
    if data == "[DONE]":
        return _SSE_DONE
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        logger.error("Пропущен не-JSON фрагмент потока")
        return _SSE_SKIP
    return payload


def _reasoning_piece(delta: dict[str, Any]) -> str:
    """Достаёт кусок рассуждения. DeepSeek кладёт его в ``reasoning_content``."""
    for key in ("reasoning_content", "reasoning"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _assistant_stream_message(
    reasoning: str,
    content: str,
    tool_calls: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Сообщение ассистента для следующего запроса после потокового круга."""
    payload: dict[str, Any] = {"role": "assistant"}
    if reasoning:
        payload["reasoning_content"] = reasoning
    if tool_calls:
        payload["tool_calls"] = tool_calls
        if content:
            payload["content"] = content
        return payload
    payload["content"] = content
    return payload


class _ToolCallBuilder:
    """Склеивает ``delta.tool_calls``, которые в потоке приходят по кускам."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}

    def add(self, delta_call: dict[str, Any]) -> None:
        raw_index = delta_call.get("index", 0)
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = 0
        slot = self._calls.setdefault(
            index,
            {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        call_id = delta_call.get("id")
        if isinstance(call_id, str) and call_id:
            slot["id"] = call_id
        call_type = delta_call.get("type")
        if isinstance(call_type, str) and call_type:
            slot["type"] = call_type
        function = delta_call.get("function")
        if not isinstance(function, dict):
            return
        name = function.get("name")
        if isinstance(name, str) and name:
            slot["function"]["name"] += name
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            slot["function"]["arguments"] += arguments

    def calls(self) -> list[dict[str, Any]]:
        return [self._calls[index] for index in sorted(self._calls)]


ChatDeltaKind = Literal["content", "tool"]


@dataclass(frozen=True)
class ChatDelta:
    """Фрагмент потокового ответа ``stream_chat``.

    ``content`` — кусок итогового текста.
    ``tool`` — инструмент уже выполнен, в ``text`` его имя и аргументы.
    Рассуждения модели в чат не отдаются.
    """

    kind: ChatDeltaKind
    text: str


class AssistAIError(Exception):
    """Ошибка запроса к ProxyAPI или исчерпанный лимит вызовов инструментов.

    Сбой самого инструмента сюда не попадает: он возвращается модели
    как JSON ``{"error": "..."}``.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AssistAIConfig(BaseModel):
    """Конфигурация клиента. Ключи не попадают в ``repr``.

    ``from_env`` читает ``PROXYAPI_API_KEY``, ``OPENWEATHER_API_KEY``
    и ``EXCHANGERATE_API_KEY``. Пустой ключ погоды или курса валют
    не мешает запустить клиент: соответствующий инструмент просто не регистрируется.

    Голос: ``PROXYAPI_TRANSCRIBE_MODEL``, ``PROXYAPI_TTS_MODEL``,
    ``PROXYAPI_TTS_VOICE``, ``PROXYAPI_TTS_FORMAT``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    proxyapi_key: str = Field(min_length=1, repr=False)
    openweather_api_key: str = Field(default="", repr=False)
    exchangerate_api_key: str = Field(default="", repr=False)
    base_url: str = "https://api.proxyapi.ru/v1"
    default_text_model: str = "openai/gpt-4o"
    default_image_model: str = "openai/gpt-image-1-mini"
    default_transcribe_model: str = "openai/whisper-1"
    default_tts_model: str = "openai/gpt-4o-mini-tts"
    default_tts_voice: str = "alloy"
    default_tts_format: str = "opus"
    timeout: float = Field(default=30.0, gt=0)
    max_tool_iterations: int = Field(default=5, ge=1)

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("base_url должен начинаться с http:// или https://")
        return normalized

    @classmethod
    def from_env(cls) -> AssistAIConfig:
        """Собирает конфигурацию из переменных окружения и файла ``.env``."""
        load_dotenv(Path(__file__).resolve().parent / ".env")
        load_dotenv()
        return cls(
            proxyapi_key=os.getenv("PROXYAPI_API_KEY", ""),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY", ""),
            exchangerate_api_key=os.getenv("EXCHANGERATE_API_KEY", ""),
            default_transcribe_model=os.getenv(
                "PROXYAPI_TRANSCRIBE_MODEL", "openai/whisper-1"
            ),
            default_tts_model=os.getenv("PROXYAPI_TTS_MODEL", "openai/gpt-4o-mini-tts"),
            default_tts_voice=os.getenv("PROXYAPI_TTS_VOICE", "alloy"),
            default_tts_format=os.getenv("PROXYAPI_TTS_FORMAT", "opus"),
        )


class BaseTool(ABC):
    """Инструмент, который модель может запросить через Function Calling.

    ``execute`` всегда возвращает JSON-строку. Исключение внутри инструмента
    не должно выходить наружу: его текст кладётся в поле ``error``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя функции, которое модель указывает в ``tool_calls``."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Когда вызывать функцию. По этому тексту модель выбирает инструмент."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema аргументов в формате OpenAI tools."""

    def to_openai_format(self) -> dict[str, Any]:
        """Собирает описание инструмента в объект поля ``tools``."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Выполняет инструмент и возвращает JSON-строку.

        При любой ошибке реализация возвращает ``{"error": "..."}``,
        а не пробрасывает исключение.
        """


def _parse_city(raw: Any, key: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Параметр {key} обязателен и должен быть непустой строкой.")
    return raw.strip()


def _parse_country(raw: Any) -> str | None:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise ValueError("country_code должен быть строкой, например RU.")  # noqa: TRY004
    stripped = raw.strip()
    return stripped or None


def _parse_bool(raw: Any, *, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes", "да"}:
            return True
        if lowered in {"false", "0", "no", "нет"}:
            return False
    raise ValueError("with_forecast должен быть true или false.")


def _parse_coord(raw: Any, name: str, low: float, high: float) -> float:
    if raw is None or isinstance(raw, bool):
        raise ValueError(f"Параметр {name} обязателен и должен быть числом.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Параметр {name} должен быть числом.") from exc
    if not low <= value <= high:
        raise ValueError(f"Параметр {name} должен быть от {low} до {high}.")
    return value


def _parse_days(raw: Any) -> int:
    if raw is None:
        return 5
    if isinstance(raw, bool):
        raise ValueError("days должен быть целым числом от 1 до 5.")  # noqa: TRY004
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError("days должен быть целым числом от 1 до 5.")
    try:
        days = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("days должен быть целым числом от 1 до 5.") from exc
    if days < 1 or days > 5:
        raise ValueError("days должен быть от 1 до 5.")
    return days


def _local_stamp(timestamp: int, timezone_offset: int, with_time: bool) -> str:
    moment = datetime.fromtimestamp(
        timestamp, timezone(timedelta(seconds=timezone_offset))
    )
    if with_time:
        return moment.strftime("%Y-%m-%d %H:%M")
    return moment.date().isoformat()


def _weather_payload(weather: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Текущая погода и, если клиент её приложил, суточный прогноз."""
    current_raw = weather.get("current")
    current = CurrentWeather.from_owm(
        current_raw if isinstance(current_raw, dict) else {}
    )
    payload: dict[str, Any] = {
        "latitude": weather.get("lat"),
        "longitude": weather.get("lon"),
        "temperature": current.temp,
        "feels_like": current.feels_like,
        "description": current.description,
        "humidity": current.humidity,
        "wind_speed": current.wind_speed,
    }
    daily = weather.get("daily")
    if isinstance(daily, list):
        offset = int(weather.get("timezone") or 0)
        payload["daily"] = [
            _daily_payload(item, offset) for item in daily if isinstance(item, dict)
        ]
    payload.update(extra)
    return payload


def _daily_payload(item: dict[str, Any], timezone_offset: int) -> dict[str, Any]:
    day = DailyForecast.from_owm(item)
    return {
        "date": _local_stamp(day.dt, timezone_offset, with_time=False),
        "temp_day": day.temp_day,
        "temp_night": day.temp_night,
        "description": day.description,
    }


def _forecast_payload(forecast: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Сжимает 3-часовой прогноз до полей, которые нужны ответу модели."""
    offset = int(forecast.get("timezone") or 0)
    hours: list[dict[str, Any]] = []
    for item in forecast.get("list") or []:
        if not isinstance(item, dict):
            continue
        main_raw = item.get("main")
        main = main_raw if isinstance(main_raw, dict) else {}
        wind_raw = item.get("wind")
        wind = wind_raw if isinstance(wind_raw, dict) else {}
        weather_raw = item.get("weather")
        weather_list = weather_raw if isinstance(weather_raw, list) else []
        first_weather = weather_list[0] if weather_list else None
        weather_info = first_weather if isinstance(first_weather, dict) else {}
        hours.append(
            {
                "time_local": _local_stamp(
                    int(item.get("dt") or 0), offset, with_time=True
                ),
                "temperature": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "humidity": main.get("humidity"),
                "wind_speed": wind.get("speed"),
                "description": weather_info.get("description"),
                "precipitation_probability": item.get("pop"),
            }
        )
    return {
        "latitude": forecast.get("lat"),
        "longitude": forecast.get("lon"),
        "hours": hours,
        **extra,
    }


def _air_payload(data: dict[str, Any]) -> dict[str, Any]:
    items = data.get("list")
    first = (
        items[0]
        if isinstance(items, list) and items and isinstance(items[0], dict)
        else {}
    )
    parsed = AirPollutionData.from_owm(first)
    components = (
        first.get("components") if isinstance(first.get("components"), dict) else {}
    )
    return {
        "aqi": parsed.aqi,
        "aqi_scale": "1 — хорошо, 5 — опасно",
        "pm2_5": parsed.pm2_5,
        "pm10": parsed.pm10,
        "components": components,
    }


_CITY_NAME_SCHEMA: dict[str, Any] = {
    "type": "string",
    "description": "Название города, например Москва или Irkutsk.",
}
_COUNTRY_CODE_SCHEMA: dict[str, Any] = {
    "type": "string",
    "description": (
        "Необязательный код страны ISO 3166-1 alpha-2, например RU. "
        "Передавай, если город встречается в нескольких странах."
    ),
}
_LAT_SCHEMA: dict[str, Any] = {
    "type": "number",
    "minimum": -90,
    "maximum": 90,
    "description": "Широта в градусах.",
}
_LON_SCHEMA: dict[str, Any] = {
    "type": "number",
    "minimum": -180,
    "maximum": 180,
    "description": "Долгота в градусах.",
}
_FORECAST_FLAG_SCHEMA: dict[str, Any] = {
    "type": "boolean",
    "description": (
        "true — текущая погода и суточный прогноз до 5 дней. "
        "false — только текущая погода. По умолчанию true."
    ),
}


def _object_schema(
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class _OpenWeatherTool(BaseTool):
    """Общий контур ошибок для инструментов OpenWeather."""

    def __init__(self, client: OpenWeatherClient) -> None:
        self._client = client

    async def _guard(self, action: Callable[[], Awaitable[dict[str, Any]]]) -> str:
        try:
            payload = await action()
        except OpenWeatherAPIError as exc:
            logger.error("%s: %s", self.name, exc.message)
            return _error_json(exc.message)
        except Exception as exc:
            logger.exception("%s: непредвиденная ошибка", self.name)
            return _error_json(_exception_text(exc))
        logger.info("%s: ответ готов", self.name)
        return json.dumps(payload, ensure_ascii=False)


class WeatherByCityTool(_OpenWeatherTool):
    """Текущая погода и суточный прогноз по названию города."""

    @property
    def name(self) -> str:
        return "get_weather_by_city"

    @property
    def description(self) -> str:
        return (
            "Погода по названию города: температура в °C, ощущение, описание, "
            "влажность в процентах, ветер в м/с, координаты. "
            "При with_forecast=true добавляется суточный прогноз до 5 дней "
            "(дневная и ночная температура). "
            "Для прогноза на конкретный час используй get_forecast_by_city. "
            "Вызывай только если пользователь прямо спрашивает погоду в городе. "
            "Цифры бери из ответа функции, не из памяти."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {
                "city_name": _CITY_NAME_SCHEMA,
                "country_code": _COUNTRY_CODE_SCHEMA,
                "with_forecast": _FORECAST_FLAG_SCHEMA,
            },
            ["city_name"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.get_weather_by_city``."""
        try:
            city_name = _parse_city(kwargs.get("city_name"), "city_name")
            country_code = _parse_country(kwargs.get("country_code"))
            with_forecast = _parse_bool(kwargs.get("with_forecast"), default=True)
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            weather = await self._client.get_weather_by_city(
                city_name,
                country_code,
                with_forecast=with_forecast,
            )
            return _weather_payload(
                weather,
                city_name=city_name,
                country_code=country_code,
                with_forecast=with_forecast,
            )

        return await self._guard(_load)


class WeatherByCoordsTool(_OpenWeatherTool):
    """Текущая погода и суточный прогноз по координатам."""

    @property
    def name(self) -> str:
        return "get_weather_by_coords"

    @property
    def description(self) -> str:
        return (
            "Погода по широте и долготе. Возвращает те же поля, что "
            "get_weather_by_city. Параметр days задаёт длину суточного прогноза "
            "от 1 до 5 и работает только при with_forecast=true. "
            "Вызывай только если пользователь прямо спрашивает погоду по координатам."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {
                "lat": _LAT_SCHEMA,
                "lon": _LON_SCHEMA,
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Число суток прогноза, от 1 до 5. По умолчанию 5.",
                },
                "with_forecast": _FORECAST_FLAG_SCHEMA,
            },
            ["lat", "lon"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.get_weather_by_coords``."""
        try:
            lat = _parse_coord(kwargs.get("lat"), "lat", -90, 90)
            lon = _parse_coord(kwargs.get("lon"), "lon", -180, 180)
            days = _parse_days(kwargs.get("days"))
            with_forecast = _parse_bool(kwargs.get("with_forecast"), default=True)
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            weather = await self._client.get_weather_by_coords(
                lat,
                lon,
                days,
                with_forecast=with_forecast,
            )
            return _weather_payload(
                weather,
                latitude=lat,
                longitude=lon,
                days=days,
                with_forecast=with_forecast,
            )

        return await self._guard(_load)


class ForecastByCityTool(_OpenWeatherTool):
    """Прогноз с шагом 3 часа по названию города."""

    @property
    def name(self) -> str:
        return "get_forecast_by_city"

    @property
    def description(self) -> str:
        return (
            "Прогноз по городу на 5 дней с шагом 3 часа: местное время, "
            "температура, влажность, ветер и вероятность осадков. "
            "Вызывай, только если пользователь просит прогноз по часам или на часть дня. "
            "Для сводки по суткам достаточно get_weather_by_city."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {
                "city_name": _CITY_NAME_SCHEMA,
                "country_code": _COUNTRY_CODE_SCHEMA,
            },
            ["city_name"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.get_forecast_by_city``."""
        try:
            city_name = _parse_city(kwargs.get("city_name"), "city_name")
            country_code = _parse_country(kwargs.get("country_code"))
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            forecast = await self._client.get_forecast_by_city(city_name, country_code)
            return _forecast_payload(
                forecast,
                city_name=city_name,
                country_code=country_code,
            )

        return await self._guard(_load)


class ForecastByCoordsTool(_OpenWeatherTool):
    """Прогноз с шагом 3 часа по координатам."""

    @property
    def name(self) -> str:
        return "get_forecast_by_coords"

    @property
    def description(self) -> str:
        return (
            "Прогноз по координатам на 5 дней с шагом 3 часа. "
            "Поля те же, что у get_forecast_by_city. "
            "Вызывай только если пользователь прямо просит такой прогноз."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {"lat": _LAT_SCHEMA, "lon": _LON_SCHEMA},
            ["lat", "lon"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.get_forecast_by_coords``."""
        try:
            lat = _parse_coord(kwargs.get("lat"), "lat", -90, 90)
            lon = _parse_coord(kwargs.get("lon"), "lon", -180, 180)
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            forecast = await self._client.get_forecast_by_coords(lat, lon)
            return _forecast_payload(forecast, latitude=lat, longitude=lon)

        return await self._guard(_load)


class AirPollutionTool(_OpenWeatherTool):
    """Качество воздуха по координатам."""

    @property
    def name(self) -> str:
        return "get_air_pollution"

    @property
    def description(self) -> str:
        return (
            "Качество воздуха по координатам: индекс AQI от 1 (хорошо) до 5 (опасно), "
            "PM2.5, PM10 и остальные компоненты. "
            "Координаты можно взять из get_weather_by_city или get_weather_by_coords. "
            "Вызывай только если пользователь прямо спрашивает качество воздуха."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {"lat": _LAT_SCHEMA, "lon": _LON_SCHEMA},
            ["lat", "lon"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.get_air_pollution``."""
        try:
            lat = _parse_coord(kwargs.get("lat"), "lat", -90, 90)
            lon = _parse_coord(kwargs.get("lon"), "lon", -180, 180)
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            data = await self._client.get_air_pollution(lat, lon)
            payload = _air_payload(data)
            payload["latitude"] = lat
            payload["longitude"] = lon
            return payload

        return await self._guard(_load)


class CompareCitiesTool(_OpenWeatherTool):
    """Сравнение текущей погоды в двух городах."""

    @property
    def name(self) -> str:
        return "compare_cities"

    @property
    def description(self) -> str:
        return (
            "Сравнить текущую погоду в двух городах: температура, ощущение, "
            "описание и скорость ветра. Города запрашиваются параллельно. "
            "Вызывай только если пользователь прямо просит сравнить погоду."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {
                "city1": {
                    "type": "string",
                    "description": "Первый город.",
                },
                "city2": {
                    "type": "string",
                    "description": "Второй город.",
                },
            },
            ["city1", "city2"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.compare_cities``."""
        try:
            city1 = _parse_city(kwargs.get("city1"), "city1")
            city2 = _parse_city(kwargs.get("city2"), "city2")
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            comparison = await self._client.compare_cities(city1, city2)
            return (
                comparison if isinstance(comparison, dict) else {"result": comparison}
            )

        return await self._guard(_load)


class WeatherNotificationTool(_OpenWeatherTool):
    """Готовый текст погодного уведомления."""

    @property
    def name(self) -> str:
        return "format_notification"

    @property
    def description(self) -> str:
        return (
            "Собрать готовый текст уведомления о погоде в городе: температура, "
            "ветер, влажность, описание и качество воздуха. "
            "Вызывай только если пользователь прямо просит готовый текст уведомления о погоде."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {
                "city_name": _CITY_NAME_SCHEMA,
                "country_code": _COUNTRY_CODE_SCHEMA,
            },
            ["city_name"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.format_notification`` и кладёт текст в JSON."""
        try:
            city_name = _parse_city(kwargs.get("city_name"), "city_name")
            country_code = _parse_country(kwargs.get("country_code"))
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            text = await self._client.format_notification(city_name, country_code)
            return {
                "city_name": city_name,
                "country_code": country_code,
                "notification": text,
            }

        return await self._guard(_load)


_WEATHER_TOOLS: tuple[type[_OpenWeatherTool], ...] = (
    WeatherByCityTool,
    WeatherByCoordsTool,
    ForecastByCityTool,
    ForecastByCoordsTool,
    AirPollutionTool,
    CompareCitiesTool,
    WeatherNotificationTool,
)


class ExchangeRateTool(BaseTool):
    """Конвертация суммы между двумя валютами через ``get_exchange_rate``."""

    def __init__(self, api_key: str) -> None:
        key = api_key.strip()
        if not key:
            raise ValueError("Для ExchangeRateTool нужен непустой API-ключ.")
        self._api_key = key

    @property
    def name(self) -> str:
        return "convert_currency"

    @property
    def description(self) -> str:
        return (
            "Конвертировать сумму из одной валюты в другую по актуальному курсу. "
            "Коды валют — ISO 4217 из трёх букв, например USD, EUR, RUB. "
            "Возвращает курс за 1 единицу и итоговую сумму. "
            "Вызывай только если пользователь прямо просит курс или конвертацию, "
            "и не считай курс сам."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "from_currency": {
                    "type": "string",
                    "description": "Исходная валюта, код ISO 4217. Пример: USD.",
                },
                "to_currency": {
                    "type": "string",
                    "description": "Целевая валюта, код ISO 4217. Пример: RUB.",
                },
                "amount": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Положительная сумма в исходной валюте.",
                },
            },
            "required": ["from_currency", "to_currency", "amount"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        """Конвертирует сумму и сериализует ``Conversion`` в JSON.

        ``ExchangeAPIError`` возвращается модели текстом ``user_message``.
        Остальные исключения тоже становятся ``{"error": "..."}``.
        """
        try:
            conversion: Conversion = await get_exchange_rate(
                from_currency=str(kwargs["from_currency"]),
                to_currency=str(kwargs["to_currency"]),
                amount=float(kwargs["amount"]),
                api_key=self._api_key,
            )
        except ExchangeAPIError as exc:
            logger.error("convert_currency: %s", exc.user_message)
            return _error_json(exc.user_message)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "параметр"
            logger.error("convert_currency: не передан параметр %s", missing)
            return _error_json(f"Не передан обязательный параметр: {missing}")
        except (TypeError, ValueError) as exc:
            logger.error("convert_currency: %s", exc)
            return _error_json(_exception_text(exc))
        except Exception as exc:
            logger.exception("convert_currency: непредвиденная ошибка")
            return _error_json(_exception_text(exc))

        logger.info(
            "convert_currency: %s %s -> %s",
            conversion.amount,
            conversion.from_currency,
            conversion.to_currency,
        )
        return json.dumps(asdict(conversion), ensure_ascii=False)


_REQUEST_IMAGE_MODEL: ContextVar[str | None] = ContextVar(
    "request_image_model", default=None
)
_REQUEST_IMAGE_SIZE: ContextVar[str | None] = ContextVar(
    "request_image_size", default=None
)
_REQUEST_IMAGES: ContextVar[list[dict[str, str]] | None] = ContextVar(
    "request_generated_images", default=None
)

# FLUX принимает ярус разрешения, а не пиксели вроде 1024x1024.
_FLUX_SIZE_TIERS: tuple[tuple[int, str], ...] = (
    (768, "768"),
    (1024, "1K"),
    (1536, "1.5K"),
    (2048, "2K"),
    (4096, "4K"),
)
_FLUX_TIER_NAMES: dict[str, str] = {name.upper(): name for _, name in _FLUX_SIZE_TIERS}
_FLUX_ASPECTS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (5, 4),
    (4, 5),
    (4, 3),
    (3, 4),
    (3, 2),
    (2, 3),
    (16, 9),
    (9, 16),
    (21, 9),
    (9, 21),
)
_PIXEL_SIZE = re.compile(r"(\d+)\s*[xX×]\s*(\d+)")


class ImageModelScope:
    """На время запроса подставляет image-модель персоны и собирает картинки.

    Если модель сама передала ``model`` или ``size``, её значение важнее.
    Пустая строка означает дефолт клиента. Размер без явного ``size``
    берётся из персоны, а если его нет — ``1024x1024``.
    Байты и URL из ``generate_image`` копятся в ``images`` и не уходят
    обратно в контекст текстовой модели.
    """

    def __init__(self, model: str | None, size: str | None = None) -> None:
        self._model = (
            model.strip() if isinstance(model, str) and model.strip() else None
        )
        self._size = size.strip() if isinstance(size, str) and size.strip() else None
        self.images: list[dict[str, str]] = []
        self._token: Token[str | None] | None = None
        self._size_token: Token[str | None] | None = None
        self._images_token: Token[list[dict[str, str]] | None] | None = None

    def __enter__(self) -> Self:
        self._token = _REQUEST_IMAGE_MODEL.set(self._model)
        self._size_token = _REQUEST_IMAGE_SIZE.set(self._size)
        self._images_token = _REQUEST_IMAGES.set(self.images)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._images_token is not None:
            _REQUEST_IMAGES.reset(self._images_token)
        if self._size_token is not None:
            _REQUEST_IMAGE_SIZE.reset(self._size_token)
        if self._token is not None:
            _REQUEST_IMAGE_MODEL.reset(self._token)


class GenerateImageTool(BaseTool):
    """Генерация изображения через ``AssistAIClient.generate_image``.

    Модель вызывает инструмент, когда пользователь просит картинку.
    Клиент передаётся снаружи: HTTP и выбор модели по умолчанию остаются в нём.
    """

    def __init__(self, client: AssistAIClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return (
            "Сгенерировать изображение по текстовому описанию. "
            "Вызывай, когда пользователь просит нарисовать, сгенерировать или "
            "создать изображение, картинку, схему или иллюстрацию. "
            "Не отвечай, что не умеешь рисовать: используй этот инструмент. "
            "Если модель генерации не указана, берётся модель по умолчанию клиента."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Детальное описание изображения на русском или английском."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Модель генерации, например openai/gpt-image-1-mini. "
                        "Необязательно."
                    ),
                },
                "size": {
                    "type": "string",
                    "description": (
                        "Размер. Для OpenAI — пиксели, например 1024x1024. "
                        "Для FLUX пиксели автоматически становятся ярусом "
                        "768, 1K, 1.5K, 2K или 4K. По умолчанию 1024x1024."
                    ),
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        """Генерирует картинку и возвращает короткий JSON для текстовой модели.

        Сами URL и base64 складываются в ``ImageModelScope`` и в ответ модели
        не попадают: иначе следующий запрос превышает лимит контекста.
        Ошибки аргументов и ProxyAPI становятся ``{"error": "..."}``.
        """
        prompt = kwargs.get("prompt")
        model = kwargs.get("model")
        raw_size = kwargs.get("size")
        if not isinstance(prompt, str) or not prompt.strip():
            return _error_json(
                "Параметр prompt обязателен и должен быть непустой строкой."
            )
        if model is not None and not isinstance(model, str):
            return _error_json("Параметр model должен быть строкой.")
        if raw_size is None:
            scoped = _REQUEST_IMAGE_SIZE.get()
            size = scoped if isinstance(scoped, str) and scoped.strip() else "1024x1024"
        elif not isinstance(raw_size, str) or not raw_size.strip():
            return _error_json("Параметр size должен быть строкой, например 1024x1024.")
        else:
            size = raw_size
        selected_model = (
            model.strip() if isinstance(model, str) and model.strip() else None
        )
        if selected_model is None:
            selected_model = _REQUEST_IMAGE_MODEL.get()
        try:
            images = await self._client.generate_image(
                prompt=prompt.strip(),
                model=selected_model,
                size=size.strip(),
            )
        except AssistAIError as exc:
            logger.error("generate_image: %s", exc.message)
            return _error_json(exc.message)
        except (TypeError, ValueError) as exc:
            logger.error("generate_image: %s", exc)
            return _error_json(_exception_text(exc))
        except Exception as exc:
            logger.exception("generate_image: непредвиденная ошибка")
            return _error_json(_exception_text(exc))

        packed = [
            {
                "type": "url"
                if image.startswith(("http://", "https://"))
                else "base64",
                "value": image,
            }
            for image in images
        ]
        bucket = _REQUEST_IMAGES.get()
        if bucket is None:
            logger.warning(
                "generate_image: нет ImageModelScope, картинки некому отдать в чат"
            )
        else:
            bucket.extend(packed)
        logger.info("generate_image: получено изображений=%s", len(packed))
        return json.dumps(
            {
                "status": "success",
                "image_count": len(packed),
                "images": [{"type": item["type"]} for item in packed],
                "prompt": prompt.strip(),
                "note": (
                    "Изображение уже передаётся пользователю в Telegram. "
                    "Не вставляй base64 и не вызывай инструмент повторно. "
                    "Кратко подтверди, что картинка готова."
                ),
            },
            ensure_ascii=False,
        )


def _is_flux_model(model: str) -> bool:
    """FLUX не принимает пиксельный ``size`` вроде ``1024x1024``."""
    lowered = model.lower()
    return "flux" in lowered or "black-forest" in lowered


def _snap_aspect_ratio(width: int, height: int) -> str:
    """Ближайшее соотношение сторон из набора, который понимает FLUX."""
    if width <= 0 or height <= 0:
        return "1:1"
    target = width / height
    numer, denom = min(
        _FLUX_ASPECTS,
        key=lambda item: abs((item[0] / item[1]) - target),
    )
    return f"{numer}:{denom}"


def _image_generation_body(model: str, prompt: str, size: str) -> dict[str, Any]:
    """Тело ``images/generations``. Для FLUX пиксели становятся ярусом разрешения."""
    fields: dict[str, Any] = {"model": model, "prompt": prompt, "n": 1}
    cleaned = size.strip()
    if not _is_flux_model(model):
        fields["size"] = cleaned
        return fields
    known = _FLUX_TIER_NAMES.get(cleaned.upper())
    if known is not None:
        fields["size"] = known
        return fields
    match = _PIXEL_SIZE.fullmatch(cleaned)
    if match is None:
        fields["size"] = cleaned
        return fields
    width = int(match.group(1))
    height = int(match.group(2))
    long_side = max(width, height)
    tier = min(
        _FLUX_SIZE_TIERS,
        key=lambda item: (abs(item[0] - long_side), item[0]),
    )[1]
    fields["size"] = tier
    ratio = _snap_aspect_ratio(width, height)
    if ratio != "1:1":
        fields["aspect_ratio"] = ratio
    return fields


@dataclass(frozen=True)
class SpeechResult:
    """Готовый аудиоответ TTS: байты, имя файла и MIME."""

    audio: bytes
    filename: str
    mime_type: str


def _audio_mime(filename: str) -> str:
    """MIME по расширению аудиофайла для multipart STT."""
    lowered = filename.rsplit(".", 1)[-1].casefold() if "." in filename else ""
    return {
        "ogg": "audio/ogg",
        "oga": "audio/ogg",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
        "webm": "audio/webm",
    }.get(lowered, "application/octet-stream")


def _speech_file(response_format: str) -> tuple[str, str]:
    """Имя файла и MIME для ответа TTS."""
    fmt = response_format.strip().casefold()
    if fmt == "mp3":
        return "answer.mp3", "audio/mpeg"
    return "answer.ogg", "audio/ogg"


class AssistAIClient:
    """Единый асинхронный клиент ProxyAPI: чат с инструментами и картинки.

    Экземпляр владеет ``httpx.AsyncClient`` и, если задан ключ OpenWeather,
    собственным ``OpenWeatherClient``. Закрывайте их через ``close``
    или ``async with``.

    ``chat_completion`` дописывает реплики ассистента и результаты инструментов
    в переданный список ``messages``. Один и тот же список не стоит обрабатывать
    двумя вызовами одновременно. Сам HTTP-клиент такие вызовы переносит,
    если у каждого свой список сообщений.
    """

    def __init__(self, config: AssistAIConfig | None = None) -> None:
        self.config: AssistAIConfig = config or AssistAIConfig.from_env()
        self._http: httpx.AsyncClient = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            follow_redirects=False,
            verify=_ssl_context(),
            headers={
                "Authorization": f"Bearer {self.config.proxyapi_key}",
                "Content-Type": "application/json",
                "User-Agent": "AssistAIClient/1.0",
            },
        )
        self._tools_registry: dict[str, BaseTool] = {}
        self._weather_client: OpenWeatherClient | None = None
        self._closed: bool = False
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        if self.config.openweather_api_key:
            self._weather_client = OpenWeatherClient(
                OWMConfig(api_key=self.config.openweather_api_key)
            )
            for tool_type in _WEATHER_TOOLS:
                self.register_tool(tool_type(self._weather_client))
        else:
            logger.info("OPENWEATHER_API_KEY не задан, инструменты погоды отключены")

        if self.config.exchangerate_api_key:
            self.register_tool(ExchangeRateTool(self.config.exchangerate_api_key))
        else:
            logger.info(
                "EXCHANGERATE_API_KEY не задан, инструмент курса валют отключён"
            )
        self.register_tool(GenerateImageTool(self))

    def register_tool(self, tool: BaseTool) -> None:
        """Добавляет инструмент в реестр. Повторное имя заменяет предыдущий инструмент."""
        if not tool.name:
            raise ValueError("У инструмента должно быть непустое имя.")
        self._tools_registry[tool.name] = tool
        logger.info("Зарегистрирован инструмент %s", tool.name)

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Возвращает схемы всех зарегистрированных инструментов для поля ``tools``."""
        return [tool.to_openai_format() for tool in self._tools_registry.values()]

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Закрывает сессию ProxyAPI и ``OpenWeatherClient``, если он был создан.

        Повторный вызов ничего не делает.
        """
        if self._closed:
            return
        self._closed = True
        await self._http.aclose()
        if self._weather_client is not None:
            await self._weather_client.close()
            self._weather_client = None
        logger.info("AssistAIClient закрыт")

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        """Ведёт диалог с моделью и сама выполняет запрошенные инструменты.

        В каждый запрос уходит поле ``tools``. Пока ``finish_reason`` равен
        ``tool_calls``, метод разбирает ``function.arguments`` как JSON-строку,
        вызывает ``execute`` нужного инструмента и добавляет в ``messages``
        ответ ассистента и сообщение ``role: "tool"`` с тем же ``tool_call_id``.
        Цикл останавливается на обычном текстовом ответе и возвращает его ``content``.

        Больше ``max_tool_iterations`` кругов не выполняется: это защита от
        модели, которая запрашивает инструменты без остановки. В этом случае
        выбрасывается ``AssistAIError``, уже полученные результаты остаются в ``messages``.

        Args:
            messages: История диалога в формате OpenAI. Метод дополняет её на месте.
            model: Идентификатор модели ProxyAPI. По умолчанию ``default_text_model``.

        Returns:
            Итоговый текст ответа модели.

        Raises:
            ValueError: Список сообщений пуст.
            AssistAIError: Ошибка HTTP, неразборчивый ответ или лимит итераций.
        """
        if not messages:
            raise ValueError("Список messages не может быть пустым.")

        selected_model = model or self.config.default_text_model
        tools = self.get_tools_schema()

        for iteration in range(1, self.config.max_tool_iterations + 1):
            payload: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            logger.info(
                "chat_completion: модель=%s, итерация=%s/%s, сообщений=%s",
                selected_model,
                iteration,
                self.config.max_tool_iterations,
                len(messages),
            )
            data = await self._post_json(_CHAT_PATH, payload)
            message, finish_reason = self._extract_choice(data)
            raw_tool_calls = message.get("tool_calls") or []
            tool_calls = [
                self._normalize_tool_call(call)
                for call in raw_tool_calls
                if isinstance(call, dict)
            ]

            # Модель иногда прикладывает tool_calls при другом finish_reason.
            # Такой ответ тоже нужно выполнить, иначе вызов потеряется.
            if finish_reason == "tool_calls" or tool_calls:
                if not tool_calls:
                    raise AssistAIError(
                        "Модель вернула finish_reason=tool_calls без списка tool_calls."
                    )
                names = ", ".join(call["function"]["name"] for call in tool_calls)
                logger.info("Модель запросила инструменты: %s", names)
                messages.append(self._assistant_tool_message(message, tool_calls))
                results = await asyncio.gather(
                    *(self._execute_registered_tool(call) for call in tool_calls)
                )
                for call, result in zip(tool_calls, results, strict=True):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result,
                        }
                    )
                if iteration >= self.config.max_tool_iterations:
                    logger.error(
                        "Достигнут лимит итераций инструментов: %s",
                        self.config.max_tool_iterations,
                    )
                    raise AssistAIError(
                        "Превышен лимит вызовов инструментов: "
                        f"{self.config.max_tool_iterations}"
                    )
                continue

            break

        content = _as_text(message.get("content"))
        messages.append({"role": "assistant", "content": content})
        logger.info(
            "chat_completion: финальный ответ, finish_reason=%s, символов=%s",
            finish_reason,
            len(content),
        )
        return content

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> AsyncIterator[ChatDelta]:
        """Потоковый диалог с теми же инструментами, что и ``chat_completion``.

        В запрос добавляется ``stream: true``. Наружу выходят только текст
        ответа и отметки об инструментах. Куски рассуждения в поток не попадают.

        Вызовы инструментов собираются из ``delta.tool_calls`` и выполняются
        после закрытия потока. Для DeepSeek рассуждение сохраняется в истории
        запроса как ``reasoning_content``: при включённых tools его нужно
        вернуть в следующем круге, иначе сервис отвечает 400.

        Таймаут чтения не короче 180 секунд. Соединение ограничено ``config.timeout``.

        Args:
            messages: История диалога. Метод дополняет её на месте.
            model: Модель ProxyAPI. По умолчанию ``default_text_model``.

        Yields:
            Фрагменты ответа и отметки о выполненных инструментах.

        Raises:
            ValueError: Список сообщений пуст.
            AssistAIError: Ошибка HTTP, обрыв потока или лимит итераций.
        """
        if not messages:
            raise ValueError("Список messages не может быть пустым.")

        selected_model = model or self.config.default_text_model
        tools = self.get_tools_schema()

        for iteration in range(1, self.config.max_tool_iterations + 1):
            payload: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "stream": True,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            logger.info(
                "stream_chat: модель=%s, итерация=%s/%s, сообщений=%s",
                selected_model,
                iteration,
                self.config.max_tool_iterations,
                len(messages),
            )
            reasoning_parts: list[str] = []
            content_parts: list[str] = []
            tool_builder = _ToolCallBuilder()
            finish_reason = ""

            async for chunk in self._iter_chat_chunks(payload):
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                reason = choice.get("finish_reason")
                if isinstance(reason, str) and reason:
                    finish_reason = reason
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue

                reasoning = _reasoning_piece(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)

                content = delta.get("content")
                if isinstance(content, str) and content:
                    content_parts.append(content)
                    yield ChatDelta("content", content)

                raw_calls = delta.get("tool_calls")
                if isinstance(raw_calls, list):
                    for call in raw_calls:
                        if isinstance(call, dict):
                            tool_builder.add(call)

            tool_calls = [
                self._normalize_tool_call(call) for call in tool_builder.calls()
            ]
            reasoning_text = "".join(reasoning_parts)
            content_text = "".join(content_parts)
            wants_tools = (finish_reason == "tool_calls" or bool(tool_calls)) and bool(
                tool_calls
            )
            if finish_reason == "tool_calls" and not tool_calls:
                raise AssistAIError(
                    "Модель вернула finish_reason=tool_calls без списка tool_calls."
                )
            if wants_tools:
                names = ", ".join(call["function"]["name"] for call in tool_calls)
                logger.info("Модель запросила инструменты: %s", names)
                messages.append(
                    _assistant_stream_message(reasoning_text, content_text, tool_calls)
                )
                results = await asyncio.gather(
                    *(self._execute_registered_tool(call) for call in tool_calls)
                )
                for call, result in zip(tool_calls, results, strict=True):
                    name = str(call["function"]["name"])
                    arguments = str(call["function"]["arguments"])
                    preview = (
                        arguments if len(arguments) <= 120 else arguments[:117] + "..."
                    )
                    yield ChatDelta("tool", f"{name} {preview}".rstrip())
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result,
                        }
                    )
                if iteration >= self.config.max_tool_iterations:
                    logger.error(
                        "Достигнут лимит итераций инструментов: %s",
                        self.config.max_tool_iterations,
                    )
                    raise AssistAIError(
                        "Превышен лимит вызовов инструментов: "
                        f"{self.config.max_tool_iterations}"
                    )
                continue

            messages.append(
                _assistant_stream_message(reasoning_text, content_text, None)
            )
            logger.info(
                "stream_chat: финальный ответ, finish_reason=%s, символов=%s",
                finish_reason,
                len(content_text),
            )
            return

        raise AssistAIError(
            f"Превышен лимит вызовов инструментов: {self.config.max_tool_iterations}"
        )

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        size: str = "1024x1024",
    ) -> list[str]:
        """Генерирует изображение и возвращает URL или строки base64.

        Запрос уходит на ``{base_url}/images/generations``. Если элемент ответа
        содержит ``url``, в список попадает он. Иначе, при наличии ``b64_json``,
        возвращается сама строка base64 без префикса data-URL.

        Таймаут чтения для этого запроса не короче 180 секунд: картинка
        готовится дольше текстового ответа. Таймаут соединения остаётся
        ``config.timeout``. Остальные запросы клиента ограничены им целиком.

        Args:
            prompt: Текстовое описание изображения.
            model: Модель генерации. По умолчанию ``default_image_model``.
            size: Размер. Для OpenAI — пиксели, например ``1024x1024``.
                Для FLUX пиксели переводятся в ярус ``1K`` и соседние.

        Returns:
            Список URL или base64-строк, по одному элементу на картинку.

        Raises:
            ValueError: Пустой prompt или size.
            AssistAIError: Ошибка HTTP или ответ без изображения.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt не может быть пустым.")
        if not isinstance(size, str) or not size.strip():
            raise ValueError("size не может быть пустым.")

        selected_model = model or self.config.default_image_model
        payload = _image_generation_body(selected_model, prompt, size.strip())
        read_timeout = max(self.config.timeout, _IMAGE_READ_TIMEOUT)
        timeout = httpx.Timeout(read_timeout, connect=self.config.timeout)
        logger.info(
            "generate_image: модель=%s, size=%s, символов в prompt=%s",
            selected_model,
            payload.get("size"),
            len(prompt),
        )
        data = await self._post_json(_IMAGES_PATH, payload, timeout=timeout)

        images: list[str] = []
        raw_items = data.get("data")
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                url = item.get("url")
                if isinstance(url, str) and url:
                    images.append(url)
                    continue
                encoded = item.get("b64_json")
                if isinstance(encoded, str) and encoded:
                    images.append(encoded)

        if not images:
            logger.error("generate_image: в ответе нет url и b64_json")
            raise AssistAIError("ProxyAPI не вернул изображение.")

        logger.info("generate_image: получено изображений: %s", len(images))
        return images

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "voice.ogg",
        model: str | None = None,
        language: str | None = None,
    ) -> str:
        """Распознаёт речь через ProxyAPI ``/audio/transcriptions``.

        Args:
            audio_bytes: байты аудиофайла.
            filename: имя файла с расширением. Telegram voice обычно OGG/Opus.
            model: модель распознавания. Если None — ``default_transcribe_model``.
            language: язык речи, например ``ru``. Если None — автоопределение.

        Returns:
            Распознанный текст. Если речи нет — пустая строка.

        Raises:
            ValueError: если ``audio_bytes`` пустой.
            AssistAIError: ошибка HTTP, таймаут, некорректный ответ ProxyAPI.
        """
        if not audio_bytes:
            raise ValueError("audio_bytes не может быть пустым.")
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        selected_model = model or self.config.default_transcribe_model
        url = f"{self.config.base_url.rstrip('/')}/{_TRANSCRIBE_PATH}"
        form: dict[str, str] = {"model": selected_model}
        if language:
            form["language"] = language
        mime_type = _audio_mime(filename)
        files = {"file": (filename, audio_bytes, mime_type)}
        # Базовый клиент шлёт Content-Type: application/json. Для multipart
        # нужен отдельный клиент без этого заголовка, иначе boundary пропадёт.
        read_timeout = max(self.config.timeout, _TRANSCRIBE_READ_TIMEOUT)
        timeout = httpx.Timeout(read_timeout, connect=self.config.timeout)
        logger.info(
            "transcribe_audio: модель=%s, байт=%s, язык=%s",
            selected_model,
            len(audio_bytes),
            language or "auto",
        )
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                verify=_ssl_context(),
                headers={
                    "Authorization": f"Bearer {self.config.proxyapi_key}",
                    "User-Agent": "AssistAIClient/1.0",
                },
            ) as client:
                response = await client.post(url, data=form, files=files)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _response_error_text(exc.response)
            logger.error(
                "ProxyAPI HTTP %s для %s: %s",
                exc.response.status_code,
                _TRANSCRIBE_PATH,
                detail,
            )
            raise AssistAIError(
                f"ProxyAPI вернул HTTP {exc.response.status_code}: {detail}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("Таймаут запроса к %s", _TRANSCRIBE_PATH)
            raise AssistAIError(
                f"Превышено время ожидания ответа ProxyAPI ({_TRANSCRIBE_PATH})."
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Сетевая ошибка при запросе к %s: %s", _TRANSCRIBE_PATH, exc)
            raise AssistAIError(
                f"Сетевая ошибка при обращении к ProxyAPI: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("ProxyAPI вернул не-JSON для %s", _TRANSCRIBE_PATH)
            raise AssistAIError("ProxyAPI вернул ответ не в формате JSON.") from exc
        if not isinstance(data, dict):
            raise AssistAIError("ProxyAPI вернул неожиданный формат ответа.")
        text = data.get("text")
        if not isinstance(text, str):
            return ""
        recognized = text.strip()
        logger.info("transcribe_audio: символов=%s", len(recognized))
        return recognized

    async def synthesize_speech(
        self,
        text: str,
        model: str | None = None,
        voice: str | None = None,
        response_format: str | None = None,
    ) -> SpeechResult:
        """Синтезирует речь через ProxyAPI ``/audio/speech``.

        Args:
            text: текст ответа ассистента.
            model: TTS-модель.
            voice: голос.
            response_format: формат аудио. По умолчанию opus.

        Returns:
            SpeechResult с байтами аудио, filename и mime_type.

        Raises:
            ValueError: пустой текст.
            AssistAIError: ошибка HTTP, таймаут или пустой аудиоответ.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text не может быть пустым.")
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        selected_model = model or self.config.default_tts_model
        selected_voice = voice or self.config.default_tts_voice
        selected_format = response_format or self.config.default_tts_format
        payload = {
            "model": selected_model,
            "input": text,
            "voice": selected_voice,
            "response_format": selected_format,
        }
        url = f"{self.config.base_url.rstrip('/')}/{_SPEECH_PATH}"
        read_timeout = max(self.config.timeout, _SPEECH_READ_TIMEOUT)
        timeout = httpx.Timeout(read_timeout, connect=self.config.timeout)
        logger.info(
            "synthesize_speech: модель=%s, голос=%s, формат=%s, символов=%s",
            selected_model,
            selected_voice,
            selected_format,
            len(text),
        )
        try:
            response = await self._http.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _response_error_text(exc.response)
            logger.error(
                "ProxyAPI HTTP %s для %s: %s",
                exc.response.status_code,
                _SPEECH_PATH,
                detail,
            )
            raise AssistAIError(
                f"ProxyAPI вернул HTTP {exc.response.status_code}: {detail}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("Таймаут запроса к %s", _SPEECH_PATH)
            raise AssistAIError(
                f"Превышено время ожидания ответа ProxyAPI ({_SPEECH_PATH})."
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Сетевая ошибка при запросе к %s: %s", _SPEECH_PATH, exc)
            raise AssistAIError(
                f"Сетевая ошибка при обращении к ProxyAPI: {exc}"
            ) from exc

        audio = response.content
        if not audio:
            raise AssistAIError("ProxyAPI вернул пустой аудиоответ.")
        filename, mime_type = _speech_file(selected_format)
        logger.info("synthesize_speech: байт=%s", len(audio))
        return SpeechResult(audio=audio, filename=filename, mime_type=mime_type)

    async def _iter_chat_chunks(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Читает SSE-поток ``/chat/completions`` и отдаёт JSON каждого ``data:``."""
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        url = f"{self.config.base_url.rstrip('/')}/{_CHAT_PATH}"
        read_timeout = max(self.config.timeout, _STREAM_READ_TIMEOUT)
        timeout = httpx.Timeout(read_timeout, connect=self.config.timeout)
        logger.info("POST %s (stream)", url)
        try:
            async with self._http.stream(
                "POST", url, json=payload, timeout=timeout
            ) as response:
                await self._raise_for_stream_status(response, _CHAT_PATH)
                async for line in response.aiter_lines():
                    chunk = _parse_sse_data(line)
                    if chunk is _SSE_SKIP:
                        continue
                    if chunk is _SSE_DONE:
                        break
                    if isinstance(chunk, dict):
                        error = chunk.get("error")
                        if error:
                            if isinstance(error, dict):
                                message = str(error.get("message") or error)
                            else:
                                message = str(error)
                            logger.error("ProxyAPI вернул ошибку в потоке: %s", message)
                            raise AssistAIError(f"ProxyAPI: {message}")
                        yield chunk
        except AssistAIError:
            raise
        except httpx.TimeoutException as exc:
            logger.error("Таймаут потока %s", _CHAT_PATH)
            raise AssistAIError(
                f"Превышено время ожидания ответа ProxyAPI ({_CHAT_PATH})."
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Сетевая ошибка потока %s: %s", _CHAT_PATH, exc)
            raise AssistAIError(
                f"Сетевая ошибка при обращении к ProxyAPI: {exc}"
            ) from exc

    @staticmethod
    async def _raise_for_stream_status(response: httpx.Response, path: str) -> None:
        if response.status_code < 400:
            return
        raw = await response.aread()
        detail = raw.decode("utf-8", errors="replace")[:500]
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and error.get("message"):
                detail = str(error["message"])[:500]
            elif isinstance(error, str) and error:
                detail = error[:500]
        logger.error("ProxyAPI HTTP %s для %s: %s", response.status_code, path, detail)
        raise AssistAIError(
            f"ProxyAPI вернул HTTP {response.status_code}: {detail}",
            status_code=response.status_code,
        )

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: httpx.Timeout | None = None,
    ) -> dict[str, Any]:
        """POST JSON на ProxyAPI. Сетевые и HTTP-ошибки становятся ``AssistAIError``."""
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        logger.info("POST %s", url)
        try:
            response = await self._http.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _response_error_text(exc.response)
            logger.error(
                "ProxyAPI HTTP %s для %s: %s",
                exc.response.status_code,
                path,
                detail,
            )
            raise AssistAIError(
                f"ProxyAPI вернул HTTP {exc.response.status_code}: {detail}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("Таймаут запроса к %s", path)
            raise AssistAIError(
                f"Превышено время ожидания ответа ProxyAPI ({path})."
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Сетевая ошибка при запросе к %s: %s", path, exc)
            raise AssistAIError(
                f"Сетевая ошибка при обращении к ProxyAPI: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("ProxyAPI вернул не-JSON для %s", path)
            raise AssistAIError("ProxyAPI вернул ответ не в формате JSON.") from exc

        if not isinstance(data, dict):
            raise AssistAIError("ProxyAPI вернул неожиданный формат ответа.")

        error = data.get("error")
        if error:
            if isinstance(error, dict):
                message = str(error.get("message") or error)
            else:
                message = str(error)
            logger.error("ProxyAPI вернул ошибку в теле ответа %s: %s", path, message)
            raise AssistAIError(f"ProxyAPI: {message}")
        return data

    @staticmethod
    def _extract_choice(data: dict[str, Any]) -> tuple[dict[str, Any], str]:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AssistAIError("ProxyAPI не вернул choices.")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise AssistAIError("ProxyAPI вернул некорректный элемент choices.")
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise AssistAIError("ProxyAPI вернул некорректное сообщение модели.")
        finish_reason = choice.get("finish_reason")
        return message, finish_reason if isinstance(finish_reason, str) else ""

    @staticmethod
    def _normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
        """Приводит вызов к виду, который ProxyAPI принимает в следующем запросе.

        ``function.arguments`` в ответе модели — строка JSON. Если провайдер
        уже отдал объект, он снова сериализуется в строку.
        """
        function = call.get("function")
        if not isinstance(function, dict):
            function = {}

        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            arguments = raw_arguments
        else:
            arguments = json.dumps(raw_arguments, ensure_ascii=False)

        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{uuid.uuid4().hex}"

        name = function.get("name")
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name if isinstance(name, str) else "",
                "arguments": arguments,
            },
        }

    @staticmethod
    def _assistant_tool_message(
        message: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            payload["content"] = content
        # DeepSeek thinking + tools: без reasoning_content следующий запрос даёт 400.
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            payload["reasoning_content"] = reasoning
        else:
            reasoning_alt = message.get("reasoning")
            if isinstance(reasoning_alt, str) and reasoning_alt:
                payload["reasoning_content"] = reasoning_alt
        return payload

    async def _execute_registered_tool(self, call: dict[str, Any]) -> str:
        """Выполняет один tool call. Любой сбой возвращается модели как JSON-ошибка."""
        function = call.get("function")
        if not isinstance(function, dict):
            return _error_json("В вызове инструмента нет объекта function.")

        name = function.get("name")
        if not isinstance(name, str) or not name:
            return _error_json("Модель не указала имя инструмента.")

        try:
            arguments = self._parse_tool_arguments(function.get("arguments"))
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Не удалось разобрать аргументы %s: %s", name, exc)
            return _error_json(
                f"Аргументы инструмента {name} не являются JSON-объектом."
            )

        tool = self._tools_registry.get(name)
        if tool is None:
            logger.error("Неизвестный инструмент: %s", name)
            return _error_json(f"Неизвестный инструмент: {name}")

        logger.info("Вызов инструмента %s, аргументы=%s", name, arguments)
        try:
            result = await tool.execute(**arguments)
        except Exception as exc:
            # Запасной контур: инструмент обязан гасить ошибки сам, но диалог
            # не прерывается, даже если реализация пробросила исключение.
            logger.exception("Инструмент %s пробросил исключение", name)
            return _error_json(_exception_text(exc))

        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)
        if not result:
            logger.error("Инструмент %s вернул пустой ответ", name)
            return _error_json(f"Инструмент {name} вернул пустой ответ.")

        logger.info("Инструмент %s вернул %s символов", name, len(result))
        return result

    @staticmethod
    def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
        """Разбирает ``function.arguments``. ProxyAPI передаёт их строкой JSON."""
        if raw is None or raw == "":
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("arguments должны быть JSON-объектом.")  # noqa: TRY004
            return parsed
        raise ValueError("Неподдерживаемый тип arguments.")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _describe_image(image: str) -> str:
    if image.startswith(("http://", "https://")):
        return image
    return f"base64, {len(image)} символов"


async def main() -> None:
    """Проверяет Function Calling на погоде и валюте и один запрос картинки."""
    _configure_logging()
    try:
        client = AssistAIClient()
    except ValidationError as exc:
        logger.error("Некорректная конфигурация AssistAI: %s", exc)
        raise SystemExit(1) from exc

    async with client:
        print("\n--- Погода (Function Calling) ---")
        weather_reply = await client.chat_completion(
            [
                {
                    "role": "user",
                    "content": (
                        "Узнай текущую погоду в Москве через доступную функцию "
                        "и ответь кратко: температура, описание, влажность и ветер."
                    ),
                }
            ]
        )
        print(weather_reply)

        print("\n--- Конвертация валют (Function Calling) ---")
        exchange_reply = await client.chat_completion(
            [
                {
                    "role": "user",
                    "content": (
                        "Конвертируй 100 USD в RUB через доступную функцию. "
                        "Назови курс и итоговую сумму."
                    ),
                }
            ]
        )
        print(exchange_reply)

        print("\n--- Генерация изображения ---")
        images = await client.generate_image(
            "Минималистичный красный маяк на скале, вечерний свет, цифровая иллюстрация"
        )
        for index, image in enumerate(images, start=1):
            print(f"{index}: {_describe_image(image)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        asyncio.run(main())
    except AssistAIError as exc:
        logger.error("AssistAI: %s", exc.message)
        raise SystemExit(1) from exc
