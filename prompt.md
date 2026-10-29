Действуй как Senior Python Backend Engineer, специализирующийся на aiogram 3.x и проектировании stateful-систем.

## 🎯 Цель
Реализовать две новые функции:
1. Выбор пользователем текстовой и image-модели из списка доступных в ProxyAPI.
2. Создание, сохранение и переключение между различными "ассистентами" (персонами), каждый из которых имеет свое имя, системный промпт и (опционально) предпочтительные модели.

НЕ меняй логику `AssistAIClient` (он уже поддерживает передачу `model` и `image_model` в методы). Меняй только слой хранения и роутинг в `bot.py`.

## 🏗️ Архитектура решения

### 1. Новый модуль `assistant_storage.py`
Создай файл для управления профилями ассистентов.

**Модель данных (Pydantic):**
```python
class AssistantProfile(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = Field(min_length=1, max_length=50)
    system_prompt: str = Field(min_length=10)
    text_model: str | None = None  # Если None, используется дефолт из AssistAIConfig
    image_model: str | None = None  # Если None, используется дефолт
```

**Абстрактный интерфейс `AssistantStorage` (ABC):**
- `async def get_active_assistant(self, chat_id: int) -> AssistantProfile | None`
- `async def set_active_assistant(self, chat_id: int, profile_id: str) -> None`
- `async def save_assistant(self, chat_id: int, profile: AssistantProfile) -> None`
- `async def list_assistants(self, chat_id: int) -> list[AssistantProfile]`
- `async def delete_assistant(self, chat_id: int, profile_id: str) -> None`
- `async def close(self) -> None`

**Реализации:**
1. `InMemoryAssistantStorage`: Словарь `dict[int, dict[str, AssistantProfile]]` + `dict[int, str]` для активного ID. При инициализации создавай один дефолтный ассистент "Стандартный" с текущим `SYSTEM_PROMPT` из `bot.py`.
2. `RedisAssistantStorage`: Используй Redis Hash. Ключ `chat:{chat_id}:assistants` (hset по `profile.id`), и ключ `chat:{chat_id}:active_assistant` для хранения текущего ID.

**Фабрика:** `create_assistant_storage(config: RedisConfig) -> AssistantStorage` (с тем же graceful degradation, что и в `history_storage.py`).

### 2. Конфигурация доступных моделей
В `assistant_storage.py` или отдельном файле `models_config.py` определи константы со списком популярных моделей ProxyAPI (чтобы не делать лишний HTTP-запрос при каждом нажатии кнопки):
```python
AVAILABLE_TEXT_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "anthropic/claude-3-5-sonnet",
    "deepseek/deepseek-chat",
]
AVAILABLE_IMAGE_MODELS = ["openai/dall-e-3", "midjourney/midjourney-v6"]
```

### 3. Обновление `bot.py`
- **Инициализация**: В `on_startup` создавай `assistant_storage` через фабрику и клади в `dispatcher.workflow_data["assistant_storage"]`. В `on_shutdown` вызывай `await storage.close()`.
- **Глобальная переменная**: Удали жестко заданный `SYSTEM_PROMPT` из глобальной области видимости. Он теперь будет браться из активного ассистента.

**Новые хендлеры (с использованием InlineKeyboardMarkup):**
1. `/assistants` (или `/profiles`): Показывает список сохраненных ассистентов. Рядом с каждым — Inline-кнопка "Использовать". Если ассистент активен, помечай его ✅.
2. `/create_assistant <имя> <промпт>`: Парсит аргументы (например, через `shlex` или простое разделение, если промпт в кавычках, или просит ввести промпт следующим сообщением через FSM, но для простоты начни с формата: `/create_assistant Имя "Твой промпт здесь"`). Сохраняет профиль, делает его активным.
3. `/delete_assistant <id>`: Удаляет профиль. Если удалялся активный, переключает на "Стандартный".
4. `/model`: Показывает текущие выбранные модели для активного ассистента и предлагает Inline-клавиатуру для смены текстовой модели.
5. `/image_model`: Аналогично, но для генерации изображений.

**Модификация существующих хендлеров (`handle_text`, `cmd_stream`, `cmd_image`):**
Перед вызовом `AssistAIClient`:
1. Получи активный профиль: `profile = await assistant_storage.get_active_assistant(chat_id)`.
2. Если профиля нет, используй дефолтный (создай "Стандартный" на лету или гарантируй его наличие в storage).
3. Определи модели: 
   `text_model = profile.text_model or assistant.config.default_text_model`
   `image_model = profile.image_model or assistant.config.default_image_model`
4. При вызове `chat_completion` или `stream_chat` передавай:
   - `messages`: где первое сообщение — `{"role": "system", "content": profile.system_prompt}` (а затем история из `history_storage`).
   - `model=text_model`.
5. При вызове `generate_image` передавай `model=image_model`.

### 4. UX и Обработка ошибок
- Используй `aiogram.types.InlineKeyboardMarkup` и `InlineKeyboardButton` для выбора моделей и ассистентов. Это намного удобнее, чем печатать ID.
- При создании ассистента валидируй длину промпта.
- Если `assistant_storage` падает (Redis недоступен), логируй WARNING, используй in-memory fallback и продолжай работу.

## 🛡️ Требования к коду
- Строгая типизация (type hints) везде.
- Использование `uuid` для генерации коротких ID ассистентов.
- Все публичные методы с docstrings.
- Не ломай существующую логику `history_storage` и `AssistAIClient`.
- Раздели код логически.

## 📝 Формат вывода
1. Полный код нового файла `assistant_storage.py` (включая модели, интерфейсы, реализации и фабрику).
2. Полный код нового файла `models_config.py` (или включи константы в `assistant_storage.py`).
3. Полный обновленный код `bot.py` с новыми хендлерами, Inline-клавиатурами и интегрированной логикой выбора промпта/модели.
4. Краткая инструкция по использованию новых команд для пользователя.

Раздели файлы чёткими заголовками. Не используй заглушки (`...` или `pass`) в основной логике.
```
