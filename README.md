# 📊 EduStat AI

**Интеллектуальная платформа для анализа и прогнозирования образовательных данных**

EduStat AI — это веб-приложение на основе Streamlit для комплексного анализа, очистки и моделирования данных с поддержкой ИИ-помощника.

---

## ✨ Основные возможности

- 📥 **Загрузка и управление данными** — Импорт CSV, Excel и других форматов
- 🛡️ **Автоматическая обработка** — Быстрая очистка пропусков и выбросов
- ⚙️ **Ручная очистка** — Тонкая настройка обработки аномалий и пропущенных значений
- 📊 **Визуальный анализ** — Интерактивные графики и диаграммы
- 📈 **Сводные таблицы и группировка** — Исследование данных по разным измерениям
- ⚖️ **Статистические тесты** — T-тесты, ANOVA и другие методы сравнения групп
- 🤖 **Машинное обучение** — Логистическая регрессия и CatBoost моделирование
- 💬 **ИИ-помощник** — Интерактивный чат для получения советов по анализу
- 🗂️ **Управление проектами** — Сохранение и восстановление проектов анализа

---

## 🛠️ Стек технологий

| Компонент | Технология |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.12+ |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Modeling** | CatBoost, Logistic Regression |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Database** | Supabase (PostgreSQL) |
| **LLM** | OpenAI / OpenRouter |
| **Statistics** | SciPy, StatsModels |

---

## 📁 Структура проекта

```
edudata-analytics/
├── Home.py                          # Главная страница приложения
├── requirements.txt                 # Зависимости Python
├── run.sh                          # Скрипт для запуска приложения
├── .env.example                    # Пример переменных окружения
├── .gitignore                      # Git исключения
├── supabase_schema.sql             # SQL схема базы данных
│
├── .streamlit/                     # Конфигурация Streamlit
│   └── config.toml                # Настройки UI и сервера
│
├── Utils/                          # Вспомогательные модули
│   ├── ui_config.py               # Инициализация UI и сеансов
│   ├── i18n.py                    # Многоязычная поддержка (en, ru, tg)
│   ├── AI_helper.py               # Работа с OpenAI/LLM
│   ├── chat.py                    # Функции чата с ИИ
│   ├── database_utils.py          # Работа с Supabase
│   ├── upload_utils.py            # Обработка загрузок файлов
│   ├── automatic_data_processing.py # Автоматическая очистка данных
│   ├── outlier_utils.py           # Обработка выбросов
│   ├── project_state.py           # Сохранение состояния проектов
│   ├── modeling_utils.py          # Подготовка данных для моделей
│   ├── stats_tests.py             # Статистические тесты
│   ├── catboost_modeling.py       # CatBoost моделирование
│   ├── visualization.py           # Создание графиков
│   └── supabase_client.py         # Клиент Supabase
│
├── pages/                          # Страницы приложения Streamlit
│   ├── 00_Projects.py             # Управление проектами
│   ├── 01_Data_Upload.py          # Загрузка данных
│   ├── 02_Auto_Processing.py      # Автоматическая обработка
│   ├── 03_Missing_Values.py       # Обработка пропусков
│   ├── 04_Outlier_Handling.py     # Обработка выбросов
│   ├── 05_Visual_Analysis.py      # Визуальный анализ
│   ├── 06_Pivot_Tables.py         # Сводные таблицы
│   ├── 07_Group_Comparison.py     # Сравнение групп
│   ├── 08_Logistic_Regression.py  # Логистическая регрессия
│   ├── 09_CatBoost_Modeling.py    # CatBoost моделирование
│   └── 10_AI_Chat.py              # Чат с ИИ-помощником
│
└── catboost_info/                 # Логи обучения CatBoost
    ├── learn/
    └── test/
```

---

## ⚙️ Установка и настройка

### Предварительные требования

- **Python 3.12+** (проверка: `python3 --version`)
- **Git** (для работы с проектом)
- **Интернет-соединение** (для API OpenAI/Supabase)

### Шаг 1: Клонируем репозиторий

```bash
git clone <repository-url>
cd edudata-analytics
```

### Шаг 2: Создаем виртуальное окружение

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**Проверка**: Вы должны увидеть `(venv)` в начале приглашения терминала.

### Шаг 3: Устанавливаем зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Проверка**: `pip list` должна показать все необходимые пакеты.

### Шаг 4: Настраиваем переменные окружения

#### Вариант A: Использование файла `.env` (локально)

```bash
# Копируем пример
cp .env.example .env

# Редактируем .env и добавляем реальные значения
nano .env  # или используйте ваш редактор
```

Файл `.env` должен содержать:

```env
OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxx
SUPABASE_URL=https://your-project.supabase.co/rest/v1/
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Вариант B: Использование Streamlit Secrets (в production)

Для развертывания на Streamlit Cloud:

```bash
# Редактируем .streamlit/secrets.toml локально
nano .streamlit/secrets.toml
```

Затем синхронизируем с Streamlit Cloud через веб-интерфейс.

**⚠️ Важно**: `.env` и `.streamlit/secrets.toml` **НЕ** должны добавляться в Git!

### Шаг 5: Запускаем приложение

#### Вариант A: Используя скрипт

```bash
chmod +x run.sh
./run.sh
```

#### Вариант B: Вручную

```bash
source venv/bin/activate
streamlit run Home.py
```

Приложение откроется в браузере на `http://localhost:8501`

---

## 🔌 Настройка API и баз данных

### OpenAI / OpenRouter API

1. **Получить API ключ**:
   - Для OpenRouter: https://openrouter.ai/keys
   - Для OpenAI: https://platform.openai.com/api-keys

2. **Добавить в `.env`**:
   ```env
   OPENAI_API_KEY=sk-or-v1-xxxxx
   OPENROUTER_MODEL=cognitivecomputations/dolphin-mistral-24b-venice-edition:free
   ```

### Supabase (БД и Auth)

1. **Создать проект**: https://app.supabase.com
2. **Получить credentials**:
   - URL проекта
   - Anon/Public ключ
3. **Инициализировать БД**: Запустить SQL-скрипт `supabase_schema.sql` в Supabase SQL Editor
4. **Добавить в `.env`**:
   ```env
   SUPABASE_URL=https://your-project.supabase.co/rest/v1/
   SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

---

## 🚀 Использование приложения

### Основной workflow

1. **Создать проект** (страница "Проекты")
2. **Загрузить данные** (страница "Загрузка данных")
3. **Обработать данные** (автоматическая или ручная очистка)
4. **Анализировать** (визуальный анализ, статистические тесты)
5. **Моделировать** (LogReg или CatBoost)
6. **Получить советы** (чат с ИИ)

### Примеры данных

Приложение принимает CSV и Excel файлы с таблицами данных:
- Числовые и категориальные переменные
- Пропущенные значения (NaN, пустые ячейки)
- До ~1M строк (в зависимости от памяти)

---

## 🔒 Безопасность

### Секреты и конфигурация

- **Никогда** не коммитьте `.env` или `.streamlit/secrets.toml`
- Используйте `.env.example` как шаблон
- На production используйте переменные окружения сервера или Streamlit Secrets

### Проверка Git

```bash
# Убедитесь что секреты не отслеживаются
git status

# Не должно быть .env, .streamlit/secrets.toml и .DS_Store
```

---

## 🧪 Тестирование

```bash
# Активируем виртуальное окружение
source venv/bin/activate

# Проверяем импорты
python3 -c "import streamlit; import pandas; print('✓ Все зависимости установлены')"

# Запускаем автоматические тесты проекта
PYTHONPYCACHEPREFIX=/tmp/edudata_pycache python3 -m unittest discover -s tests

# Запускаем приложение
streamlit run Home.py
```

---

## 📊 Дополнительные материалы

- [Streamlit документация](https://docs.streamlit.io/)
- [Supabase руководство](https://supabase.com/docs)
- [Scikit-learn ML гайд](https://scikit-learn.org/stable/)
- [CatBoost документация](https://catboost.ai/en/docs/)

---

## 🤝 Разработка

### Структура кода

- **Utils/** — переиспользуемые функции и классы
- **pages/** — Streamlit страницы (нумерованы для порядка)
- **Home.py** — главная страница и навигация

### Добавление новых страниц

Создать файл `pages/XX_Page_Name.py` и добавить в навигацию в `Home.py`.

### Переводы (i18n)

Переводы управляются в `Utils/i18n.py`. Поддерживаемые языки:
- 🇷🇺 Русский (Russian)
- 🇺🇸 Английский (English)  
- 🇹🇯 Таджикский (Tajik)

---

## 🐛 Решение проблем

### Ошибка: `ModuleNotFoundError: No module named 'streamlit'`

```bash
# Активируйте виртуальное окружение
source venv/bin/activate

# Переустановите зависимости
pip install -r requirements.txt
```

### Ошибка: `OPENAI_API_KEY не найден`

1. Проверьте что `.env` файл создан
2. Убедитесь что содержит `OPENAI_API_KEY=...`
3. Перезагрузите приложение: Нажмите R в Streamlit

### Ошибка: `Supabase не инициализирован`

1. Проверьте `SUPABASE_URL` и `SUPABASE_KEY` в `.env`
2. Убедитесь что Supabase проект создан и активен
3. Проверьте что БД инициализирована (запустите `supabase_schema.sql`)


---

## 👨‍💻 Разработчик

Для вопросов и предложений обратитесь к разработчику проекта.

---

**Версия**: 1.0  
**Последнее обновление**: 2026  
**Python**: 3.12+  
**Status**: Active ✓
