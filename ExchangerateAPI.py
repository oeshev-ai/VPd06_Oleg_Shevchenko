"""Асинхронный клиент конвертации валют.

Запросы уходят только на ``http://api.exchangerate.host/convert``
и всегда содержат параметр ``access_key``.
"""

import ssl
from dataclasses import dataclass

import httpx

# Единственный разрешённый адрес. Другие API и пути не используются.
CONVERT_URL = "http://api.exchangerate.host/convert"

MSG_INVALID_ACCESS_KEY = (
    "⚙️ Ошибка конфигурации API. Проверьте параметр access_key в .env файле."
)
MSG_SERVICE_UNAVAILABLE = (
    "🌐 Сервис курсов валют временно недоступен. "
    "Попробуйте ввести курс вручную или повторите позже."
)
MSG_INVALID_CURRENCY = (
    "Не удалось получить курс. Проверьте название страны/валюты или попробуйте позже."
)

# 101 — нет или неверный ключ, 102 — неактивный аккаунт.
_ACCESS_KEY_CODES = {101, 102}
# 201/202 — коды exchangeratesapi, 402 — код api.exchangerate.host
# для неверной валюты (тип при этом invalid_from_currency / invalid_to_currency).
_CURRENCY_CODES = {201, 202, 402}
_CURRENCY_TYPES = {
    "invalid_from_currency",
    "invalid_to_currency",
    "invalid_currency",
    "invalid_base_currency",
}


class ExchangeAPIError(Exception):
    """Базовая ошибка клиента. ``user_message`` можно показать в чате."""

    def __init__(self, user_message: str, *, code: int | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.code = code


class InvalidAccessKeyError(ExchangeAPIError):
    """Ключ access_key отсутствует или отклонён сервисом."""


class InvalidCurrencyError(ExchangeAPIError):
    """Сервис не распознал код валюты."""


class ExchangeServiceError(ExchangeAPIError):
    """Сеть, таймаут или временный отказ сервиса курсов."""


@dataclass(frozen=True)
class Conversion:
    """Результат успешного запроса ``/convert``.

    ``rate`` — сколько единиц ``to_currency`` даёт 1 единица ``from_currency``.
    ``result`` — сумма ``amount``, уже пересчитанная в ``to_currency``.
    """

    from_currency: str
    to_currency: str
    amount: float
    rate: float
    result: float


def _error_code(payload: dict) -> int | None:
    """Достаёт числовой код из ``error.code``, если он есть."""
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if isinstance(code, int):
        return code
    if isinstance(code, str) and code.isdigit():
        return int(code)
    return None


def _error_type(payload: dict) -> str:
    """Достаёт строковый тип ошибки, например ``invalid_access_key``."""
    error = payload.get("error")
    if not isinstance(error, dict):
        return ""
    return str(error.get("type") or "")


def _raise_for_api_error(payload: dict) -> None:
    """Превращает ``success: false`` в понятное исключение."""
    code = _error_code(payload)
    error_type = _error_type(payload).lower()
    access_key_failed = code in _ACCESS_KEY_CODES or "access_key" in error_type
    if access_key_failed:
        raise InvalidAccessKeyError(MSG_INVALID_ACCESS_KEY, code=code)
    # Код 402 бывает и у других отказов, поэтому смотрим ещё и тип ошибки.
    currency_failed = error_type in _CURRENCY_TYPES or (
        code in _CURRENCY_CODES and "currency" in error_type
    )
    if currency_failed or code in {201, 202}:
        raise InvalidCurrencyError(MSG_INVALID_CURRENCY, code=code)
    raise ExchangeServiceError(MSG_SERVICE_UNAVAILABLE, code=code)


def _parse_conversion(
    payload: dict,
    from_currency: str,
    to_currency: str,
    amount: float,
) -> Conversion:
    """Собирает курс и сумму из успешного JSON."""
    if payload.get("success") is not True:
        _raise_for_api_error(payload)

    raw_result = payload.get("result")
    info = payload.get("info")
    if not isinstance(info, dict):
        info = {}
    raw_rate = info.get("rate", info.get("quote"))

    if raw_result is None:
        raise ExchangeServiceError(MSG_SERVICE_UNAVAILABLE)
    try:
        result = float(raw_result)
    except (TypeError, ValueError) as exc:
        raise ExchangeServiceError(MSG_SERVICE_UNAVAILABLE) from exc

    if raw_rate is None:
        # В ответе иногда нет info.rate: курс равен результату на 1 единицу суммы.
        raw_rate = result / amount
    try:
        rate = float(raw_rate)
    except (TypeError, ValueError) as exc:
        raise ExchangeServiceError(MSG_SERVICE_UNAVAILABLE) from exc

    if rate <= 0 or result < 0:
        raise ExchangeServiceError(MSG_SERVICE_UNAVAILABLE)

    return Conversion(
        from_currency=from_currency,
        to_currency=to_currency,
        amount=amount,
        rate=rate,
        result=result,
    )


async def get_exchange_rate(
    from_currency: str,
    to_currency: str,
    amount: float,
    api_key: str,
) -> Conversion:
    """Конвертирует ``amount`` из ``from_currency`` в ``to_currency``.

    Параметры запроса: ``access_key``, ``from``, ``to``, ``amount``.
    """
    source = from_currency.strip().upper()
    target = to_currency.strip().upper()
    key = api_key.strip()

    if not key:
        raise InvalidAccessKeyError(MSG_INVALID_ACCESS_KEY)
    if not source or not target:
        raise InvalidCurrencyError(MSG_INVALID_CURRENCY)
    if amount <= 0:
        raise ValueError("Сумма должна быть положительным числом.")

    params: dict[str, str | float] = {
        "access_key": key,
        "from": source,
        "to": target,
        "amount": amount,
    }

    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=False,
            verify=ssl.create_default_context(),
        ) as client:
            response = await client.get(CONVERT_URL, params=params)
    except httpx.HTTPError as exc:
        raise ExchangeServiceError(MSG_SERVICE_UNAVAILABLE) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ExchangeServiceError(MSG_SERVICE_UNAVAILABLE) from exc

    if not isinstance(payload, dict):
        raise ExchangeServiceError(MSG_SERVICE_UNAVAILABLE)

    # Ошибка ключа может прийти и с HTTP 401, но JSON всё равно разбираем.
    if response.status_code >= 400 and payload.get("success") is not False:
        raise ExchangeServiceError(MSG_SERVICE_UNAVAILABLE)

    return _parse_conversion(payload, source, target, float(amount))
