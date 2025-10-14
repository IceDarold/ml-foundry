# ==============================================================================
# Makefile для Фреймворка ML-экспериментов
# ==============================================================================

# --- Переменные ---
# Используйте `make train E=exp002` для переопределения.
E ?= exp001_lgbm_baseline    # `E` - сокращение для `EXPERIMENT`
I ?= inf_exp001             # `I` - сокращение для `INFERENCE`
S ?= lgbm_xgb_catboost      # `S` - сокращение для `STACKING`
T ?= lgbm_default            # `T` - сокращение для `TUNING`

.PHONY: help install features train tune stack predict select fulltrain clean

# ==============================================================================
# --- Основные команды Workflow ---
# ==============================================================================

help:
	@echo "------------------------------------------------------------------"
	@echo "  Доступные команды:"
	@echo "------------------------------------------------------------------"
	@echo "  setup:"
	@echo "    install          - Установить все зависимости проекта (poetry install)"
	@echo ""
	@echo "  workflow:"
	@echo "    features E=<exp>   - Сгенерировать признаки для эксперимента (по умолч. E=$(E))"
	@echo "    select E=<exp>   - Выполнить отбор признаков для эксперимента (по умолч. E=$(E))"
	@echo "    train E=<exp>    - Обучить модель на CV для эксперимента (по умолч. E=$(E))"
	@echo "    fulltrain E=<exp>- Обучить модель на 100% данных для эксперимента (по умолч. E=$(E))"
	@echo "    tune T=<tune>    - Запустить подбор гиперпараметров (по умолч. T=$(T))"
	@echo "    stack S=<stack>  - Запустить стекинг (по умолч. S=$(S))"
	@echo "    predict I=<inf>  - Сделать инференс (по умолч. I=$(I))"
	@echo ""
	@echo "  cleanup:"
	@echo "    clean            - Удалить временные файлы Python (__pycache__)"
	@echo ""
	@echo "  Пример использования: make train E=exp002_catboost"
	@echo "------------------------------------------------------------------"


# --- Команды настройки окружения ---
install:
	@echo ">>> Установка зависимостей через Poetry..."
	poetry install
	@echo ">>> Не забудьте выполнить 'wandb login', если делаете это впервые."


# --- Команды основного рабочего цикла ---

# Генерирует признаки, используя конфигурацию из `experiment`
features:
	@echo ">>> Генерация признаков для эксперимента: $(E)..."
	python src/scripts/make_features.py experiment=$(E)

# Выполняет отбор признаков
select:
	@echo ">>> Отбор признаков для эксперимента: $(E)..."
	python src/scripts/select_features.py experiment=$(E)

# Обучает модель на кросс-валидации
train:
	@echo ">>> Обучение (CV) для эксперимента: $(E)..."
	python src/scripts/train.py experiment=$(E)

# Обучает модель на 100% данных
fulltrain:
	@echo ">>> Обучение (на 100% данных) для эксперимента: $(E)..."
	python src/scripts/train.py experiment=$(E) training.full_data=true

# Запускает подбор гиперпараметров
tune:
	@echo ">>> Подбор гиперпараметров с конфигурацией: $(T)..."
	python src/scripts/tune.py tuning=$(T)

# Запускает стекинг
stack:
	@echo ">>> Запуск стекинга с конфигурацией: $(S)..."
	python src/scripts/stack.py stacking=$(S)

# Делает предсказания на основе обученной модели
predict:
	@echo ">>> Инференс с конфигурацией: $(I)..."
	python src/scripts/predict.py inference=$(I)


# --- Команды очистки ---

clean:
	@echo ">>> Очистка временных файлов..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	@echo ">>> Очистка завершена."