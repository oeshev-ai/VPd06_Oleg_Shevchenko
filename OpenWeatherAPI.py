"""
Production-ready асинхронный клиент для OpenWeatherMap API.
Предназначен для использования в крупных погодных ботах.

"""

import asyncio
import logging
import os
import ssl
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

# ==============================================================================
# 1. Конфигурация и Логирование
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("OpenWeatherAPI")


@dataclass
class OWMConfig:
    """Конфигурация клиента OpenWeatherMap."""

    api_key: str = field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", ""))
    base_url_weather: str = "https://api.openweathermap.org/data/2.5"
    base_url_geo: str = "https://api.openweathermap.org/geo/1.0"
    base_url_air: str = "https://api.openweathermap.org/data/2.5"
    timeout: float = 10.0
    max_retries: int = 3

    def __post_init__(self):
        if not self.api_key:
            logger.warning("OPENWEATHER_API_KEY не найден в переменных окружения!")


# ==============================================================================
# 2. Исключения
# ==============================================================================


class OpenWeatherAPIError(Exception):
    """Кастомное исключение для ошибок OpenWeatherMap API."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_json: dict | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_json = response_json


# ==============================================================================
# 3. Pydantic Модели Данных
# ==============================================================================


class WeatherCondition(BaseModel):
    description: str = Field(alias="description", default="Неизвестно")
    icon: str = Field(alias="icon", default="")


class CurrentWeather(BaseModel):
    temp: float = Field(alias="temp")
    feels_like: float = Field(alias="feels_like")
    humidity: int = Field(alias="humidity")
    wind_speed: float = Field(alias="wind_speed")
    description: str = Field(default="Неизвестно")
    icon: str = Field(default="")

    @classmethod
    def from_owm(cls, data: dict) -> "CurrentWeather":
        weather_list = data.get("weather", [{}])
        weather_info = weather_list[0] if weather_list else {}
        return cls(
            temp=data.get("temp", 0.0),
            feels_like=data.get("feels_like", 0.0),
            humidity=data.get("humidity", 0),
            wind_speed=data.get("wind_speed", 0.0),
            description=weather_info.get("description", "Неизвестно"),
            icon=weather_info.get("icon", ""),
        )


class DailyForecast(BaseModel):
    dt: int = Field(alias="dt")
    temp_day: float = Field(alias="temp_day")
    temp_night: float = Field(alias="temp_night")
    description: str = Field(default="Неизвестно")
    icon: str = Field(default="")

    @classmethod
    def from_owm(cls, data: dict) -> "DailyForecast":
        weather_list = data.get("weather", [{}])
        weather_info = weather_list[0] if weather_list else {}
        temp_data = data.get("temp", {})
        return cls(
            dt=data.get("dt", 0),
            temp_day=temp_data.get("day", 0.0),
            temp_night=temp_data.get("night", 0.0),
            description=weather_info.get("description", "Неизвестно"),
            icon=weather_info.get("icon", ""),
        )


class AirPollutionData(BaseModel):
    aqi: int = Field(alias="aqi", description="1-5 (1=Хорошо, 5=Опасно)")
    pm2_5: float = Field(alias="pm2_5", default=0.0)
    pm10: float = Field(alias="pm10", default=0.0)

    @classmethod
    def from_owm(cls, data: dict) -> "AirPollutionData":
        main = data.get("main", {})
        components = data.get("components", {})
        return cls(
            aqi=main.get("aqi", 1),
            pm2_5=components.get("pm2_5", 0.0),
            pm10=components.get("pm10", 0.0),
        )


# ==============================================================================
# 4. Основной Клиент
# ==============================================================================


def _normalize_current(data: dict) -> dict:
    """Приводит ответ Current Weather 2.5 к полям, которые читает остальной клиент."""
    main = data.get("main", {})
    wind = data.get("wind", {})
    coord = data.get("coord", {})
    return {
        "lat": coord.get("lat", 0.0),
        "lon": coord.get("lon", 0.0),
        "timezone": data.get("timezone", 0),
        "current": {
            "dt": data.get("dt", 0),
            "temp": main.get("temp", 0.0),
            "feels_like": main.get("feels_like", 0.0),
            "humidity": main.get("humidity", 0),
            "wind_speed": wind.get("speed", 0.0),
            "weather": data.get("weather", []),
        },
    }


def _daily_from_forecast(forecast: dict, days: int, timezone_offset: int) -> list[dict]:
    """Собирает суточные точки из 3-часового прогноза на 5 дней."""
    if days <= 0:
        return []

    tz = timezone(timedelta(seconds=timezone_offset))
    buckets: dict[str, list[tuple[datetime, dict]]] = {}
    for item in forecast.get("list", []):
        local_dt = datetime.fromtimestamp(item.get("dt", 0), tz)
        buckets.setdefault(local_dt.date().isoformat(), []).append((local_dt, item))

    daily: list[dict] = []
    for day_key in sorted(buckets)[:days]:
        slots = buckets[day_key]
        day_items = [item for local_dt, item in slots if 9 <= local_dt.hour <= 18]
        night_items = [
            item
            for local_dt, item in slots
            if local_dt.hour <= 6 or local_dt.hour >= 21
        ]
        if not day_items:
            day_items = [item for _, item in slots]
        if not night_items:
            night_items = [item for _, item in slots]

        representative = min(
            day_items,
            key=lambda item: abs(
                datetime.fromtimestamp(item.get("dt", 0), tz).hour - 15
            ),
        )
        daily.append(
            {
                "dt": slots[0][1].get("dt", 0),
                "temp": {
                    "day": max(
                        item.get("main", {}).get("temp", 0.0) for item in day_items
                    ),
                    "night": min(
                        item.get("main", {}).get("temp", 0.0) for item in night_items
                    ),
                },
                "weather": representative.get("weather", []),
            }
        )
    return daily


def is_retryable_exception(e: BaseException) -> bool:
    """Определяет, стоит ли повторять запрос при данной ошибке."""
    if isinstance(e, httpx.HTTPStatusError):
        # Повторяем только при 429 (Too Many Requests) и 5xx ошибках сервера
        return e.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(e, (httpx.TimeoutException, httpx.RequestError))


class OpenWeatherClient:
    """Асинхронный клиент для взаимодействия с OpenWeatherMap API."""

    def __init__(self, config: OWMConfig | None = None):
        self.config = config or OWMConfig()
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout,
            verify=ssl.create_default_context(),
            headers={"User-Agent": "ProductionWeatherBot/1.0"},
        )

    async def close(self):
        """Закрытие сессии httpx."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @retry(
        retry=retry_if_exception(is_retryable_exception),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            "Попытка %s не удалась. Повтор через %sс...",
            retry_state.attempt_number,
            retry_state.next_action.sleep if retry_state.next_action is not None else 0,
        ),
    )
    async def _make_request(self, url: str, params: dict[str, Any]) -> dict:
        """Внутренний метод для выполнения HTTP-запросов с retry-логикой."""
        params["appid"] = self.config.api_key
        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP ошибка %s для %s", e.response.status_code, url)
            raise OpenWeatherAPIError(
                message=f"API вернул ошибку {e.response.status_code}",
                status_code=e.response.status_code,
                response_json=e.response.json()
                if e.response.headers.get("content-type") == "application/json"
                else None,
            )
        except httpx.TimeoutException:
            logger.error("Таймаут запроса к %s", url)
            raise OpenWeatherAPIError(message="Превышено время ожидания ответа от API")
        except httpx.RequestError:
            logger.error("Ошибка сети при запросе к %s", url)
            raise OpenWeatherAPIError(message="Ошибка сети при запросе к API")

    async def get_weather_by_coords(
        self,
        lat: float,
        lon: float,
        days: int = 5,
        *,
        with_forecast: bool = True,
    ) -> dict:
        """Текущая погода и, если нужно, суточный прогноз по бесплатным эндпоинтам."""
        params = {
            "lat": lat,
            "lon": lon,
            "units": "metric",
            "lang": "ru",
        }
        logger.info(
            "Запрос погоды по координатам: lat=%s, lon=%s, forecast=%s",
            lat,
            lon,
            with_forecast,
        )
        if not with_forecast:
            current_raw = await self._make_request(
                f"{self.config.base_url_weather}/weather", dict(params)
            )
            return _normalize_current(current_raw)

        current_raw, forecast_raw = await asyncio.gather(
            self._make_request(f"{self.config.base_url_weather}/weather", dict(params)),
            self._make_request(
                f"{self.config.base_url_weather}/forecast", dict(params)
            ),
        )
        weather = _normalize_current(current_raw)
        weather["daily"] = _daily_from_forecast(
            forecast_raw, min(days, 5), weather["timezone"]
        )
        return weather

    async def get_weather_by_city(
        self,
        city_name: str,
        country_code: str | None = None,
        *,
        with_forecast: bool = True,
    ) -> dict:
        """Погода по названию города: одно геокодирование и один запрос текущей погоды."""
        location = await self._geocode(city_name, country_code)
        return await self.get_weather_by_coords(
            location["lat"],
            location["lon"],
            with_forecast=with_forecast,
        )

    async def get_forecast_by_coords(self, lat: float, lon: float) -> dict:
        """Сырой 5-дневный прогноз с шагом 3 часа."""
        params = {
            "lat": lat,
            "lon": lon,
            "units": "metric",
            "lang": "ru",
        }
        logger.info("Запрос 3-часового прогноза: lat=%s, lon=%s", lat, lon)
        raw = await self._make_request(
            f"{self.config.base_url_weather}/forecast", dict(params)
        )
        city = raw.get("city") or {}
        return {
            "lat": float(city.get("coord", {}).get("lat", lat)),
            "lon": float(city.get("coord", {}).get("lon", lon)),
            "timezone": int(city.get("timezone", 0)),
            "list": list(raw.get("list") or []),
        }

    async def get_forecast_by_city(
        self, city_name: str, country_code: str | None = None
    ) -> dict:
        """3-часовой прогноз по названию города."""
        location = await self._geocode(city_name, country_code)
        return await self.get_forecast_by_coords(location["lat"], location["lon"])

    async def _geocode(self, city_name: str, country_code: str | None = None) -> dict:
        query = f"{city_name},{country_code}" if country_code else city_name
        geo_url = f"{self.config.base_url_geo}/direct"
        geo_params = {"q": query, "limit": 1, "lang": "ru"}
        logger.info("Геокодирование города: %s", query)
        geo_data = await self._make_request(geo_url, geo_params)
        if not geo_data:
            raise OpenWeatherAPIError(message=f"Город '{city_name}' не найден")
        return geo_data[0]

    async def get_air_pollution(self, lat: float, lon: float) -> dict:
        """Получение данных о качестве воздуха."""
        url = f"{self.config.base_url_air}/air_pollution"
        params = {"lat": lat, "lon": lon}
        logger.info(f"Запрос данных о загрязнении воздуха: lat={lat}, lon={lon}")
        return await self._make_request(url, params)

    async def compare_cities(self, city1: str, city2: str) -> dict:
        """Асинхронное сравнение погоды в двух городах."""
        logger.info(f"Сравнение погоды в городах: {city1} и {city2}")
        try:
            # Запускаем запросы параллельно
            task1 = self.get_weather_by_city(city1, with_forecast=False)
            task2 = self.get_weather_by_city(city2, with_forecast=False)

            # Используем return_exceptions=True, чтобы один упавший запрос не отменял другой
            results = await asyncio.gather(task1, task2, return_exceptions=True)

            data: dict[str, dict[str, Any]] = {}
            comparison: dict[str, Any] = {"city1": city1, "city2": city2, "data": data}

            for i, (city, result) in enumerate(zip([city1, city2], results), start=1):
                key = f"city{i}"
                if isinstance(result, BaseException):
                    data[key] = {"error": str(result)}
                else:
                    current = CurrentWeather.from_owm(result.get("current", {}))
                    data[key] = {
                        "temp": current.temp,
                        "feels_like": current.feels_like,
                        "description": current.description,
                        "wind_speed": current.wind_speed,
                    }
            return comparison
        except Exception as e:
            logger.error(f"Ошибка при сравнении городов: {e}")
            raise

    async def format_notification(
        self, city_name: str, country_code: str | None = None
    ) -> str:
        """Формирование красивой строки уведомления для бота."""
        try:
            weather_data = await self.get_weather_by_city(city_name, country_code)
            current = CurrentWeather.from_owm(weather_data.get("current", {}))
            lat = weather_data.get("lat", 0.0)
            lon = weather_data.get("lon", 0.0)

            aqi_text = "Нет данных"
            if lat and lon:
                try:
                    air_data = await self.get_air_pollution(lat, lon)
                    if air_data.get("list"):
                        aqi = AirPollutionData.from_owm(air_data["list"][0])
                        aqi_emojis = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "🟣"}
                        aqi_text = f"{aqi_emojis.get(aqi.aqi, '⚪')} AQI: {aqi.aqi} (PM2.5: {aqi.pm2_5})"
                except OpenWeatherAPIError as e:
                    logger.warning(
                        f"Не удалось получить данные о воздухе для {city_name}: {e}"
                    )

            # Формируем красивое сообщение
            icon_map = {
                "01d": "☀️",
                "01n": "🌙",
                "02d": "⛅",
                "02n": "☁️",
                "03d": "☁️",
                "03n": "☁️",
                "04d": "☁️",
                "04n": "☁️",
                "09d": "🌧️",
                "09n": "🌧️",
                "10d": "🌦️",
                "10n": "🌧️",
                "11d": "⛈️",
                "11n": "⛈️",
                "13d": "❄️",
                "13n": "❄️",
                "50d": "🌫️",
                "50n": "🌫️",
            }
            emoji = icon_map.get(current.icon, "🌡️")

            updated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
            notification = (
                f"{emoji} **Погода в городе {city_name.title()}**\n\n"
                f"🌡️ Температура: **{current.temp:.1f}°C** (ощущается как {current.feels_like:.1f}°C)\n"
                f"💨 Ветер: {current.wind_speed} м/с\n"
                f"💧 Влажность: {current.humidity}%\n"
                f"📝 Описание: {current.description.capitalize()}\n"
                f"🌬️ Качество воздуха: {aqi_text}\n\n"
                f"🕒 Обновлено: {updated_at}"
            )
            return notification

        except OpenWeatherAPIError as e:
            return f"❌ Ошибка при получении погоды для '{city_name}': {e.message}"
        except Exception:
            logger.exception(
                "Непредвиденная ошибка при форматировании уведомления для %s",
                city_name,
            )
            return "❌ Произошла непредвиденная ошибка при обработке данных."


# ==============================================================================
# 5. Пример использования (Entry Point)
# ==============================================================================


async def main():
    """Демонстрация работы клиента."""
    # Убедитесь, что в .env файле или в окружении задан OPENWEATHER_API_KEY
    config = OWMConfig()

    # Используем контекстный менеджер для автоматического закрытия соединения
    async with OpenWeatherClient(config) as client:
        city = "Москва"

        print(f"\n--- 1. Прогноз по городу ({city}) ---")
        try:
            weather = await client.get_weather_by_city(city)
            current = CurrentWeather.from_owm(weather.get("current", {}))
            print(f"Температура: {current.temp}°C, {current.description}")
        except OpenWeatherAPIError as e:
            print(f"Ошибка: {e}")

        print(f"\n--- 2. Форматированное уведомление ({city}) ---")
        notification = await client.format_notification(city)
        print(notification)

        print("\n--- 3. Сравнение городов (Москва и Санкт-Петербург) ---")
        try:
            comparison = await client.compare_cities("Москва", "Санкт-Петербург")
            print(
                f"Москва: {comparison['data']['city1'].get('temp', 'N/A')}°C, {comparison['data']['city1'].get('description', 'N/A')}"
            )
            print(
                f"Санкт-Петербург: {comparison['data']['city2'].get('temp', 'N/A')}°C, {comparison['data']['city2'].get('description', 'N/A')}"
            )
        except OpenWeatherAPIError as e:
            print(f"Ошибка сравнения: {e}")


if __name__ == "__main__":
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")
    asyncio.run(main())
