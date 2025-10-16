# ==============================================================================
# Makefile — удобные шорткаты для основного ML workflow
# ==============================================================================

# --- Инструменты и окружение --------------------------------------------------
PY := $(shell if command -v poetry >/dev/null 2>&1; then echo "poetry run python"; else echo "python3"; fi)

EXPERIMENT ?= $(E)
EXPERIMENT ?= exp001_baseline_lgbm
INFERENCE ?= $(I)
INFERENCE ?= inf_exp001
STACKING ?= $(S)
STACKING ?= titanic_stack
TUNING ?= $(T)
TUNING ?= titanic_lgbm
SELECTION ?= $(SEL)
SELECTION ?= default

FEATURE_PIPELINE ?= v1_baseline
FEATURE_SET ?= v1_all
FEATURE_STORAGE ?= processed/titanic

ENSEMBLE_EXPERIMENT ?= exp005_ensemble
TITANIC_EXPERIMENTS ?= exp001_baseline_lgbm exp002_baseline_catboost exp003_tuned_lgbm exp004_feature_engineering

WANDB_MODE ?= disabled
OUTPUT_ROOT ?= outputs
OOF_ARTIFACT_DIR ?= 05_oof
SUBMISSION_ARTIFACT_DIR ?= 07_submissions

export WANDB_MODE
export PYTHONPATH ?= $(abspath src)

MAKEFLAGS += --warn-undefined-variables

.PHONY: help install features select train fulltrain tune stack predict pseudo \
        baseline titanic-all ensemble stacking clean clean-oof clean-submissions \
        reset-outputs

# --- Утилитарные макросы ------------------------------------------------------
define _maybe_override
$(if $(strip $2),$1=$2,)
endef

COMMON_OVERRIDES := $(call _maybe_override,data.features_path,$(FEATURE_STORAGE))
FEATURE_OVERRIDES := $(call _maybe_override,feature_engineering,$(FEATURE_PIPELINE)) \
                     $(call _maybe_override,features,$(FEATURE_SET))

# ==============================================================================
# --- Справка ------------------------------------------------------------------
# ==============================================================================
help:
	@echo "=============================================================================="
	@echo "  ML Foundry Workflow — основные цели Makefile"
	@echo "=============================================================================="
	@echo ""
	@echo "  НАСТРОЙКА:"
	@echo "    make install [TOOL=poetry|pip]          Установка зависимостей."
	@echo ""
	@echo "  ОСНОВНОЙ PIPELINE:"
	@echo "    make features [EXPERIMENT=...]          Генерация признаков."
	@echo "    make select [EXPERIMENT=... SEL=...]    Отбор признаков."
	@echo "    make train [EXPERIMENT=...]             Обучение с CV."
	@echo "    make fulltrain [EXPERIMENT=...]         Обучение на 100% данных."
	@echo ""
	@echo "  РАСШИРЕННЫЕ ШАГИ:"
	@echo "    make tune [EXPERIMENT=... TUNING=...]   Подбор гиперпараметров."
	@echo "    make ensemble                           Обучить weighted-ensemble (exp005)."
	@echo "    make stacking [STACKING=...]            Собрать стекинг (stack.py)."
	@echo "    make predict [INFERENCE=...]            Инференс по конфигу."
	@echo "    make pseudo                             Пайплайн псевдо-лейблинга."
	@echo ""
	@echo "  ГОТОВЫЕ СЦЕНАРИИ:"
	@echo "    make baseline                           v1_baseline + exp001."
	@echo "    make titanic-all                        Эксперименты exp001–exp004."
	@echo ""
	@echo "  ОБСЛУЖИВАНИЕ:"
	@echo "    make clean                              Удалить кеш Python."
	@echo "    make clean-oof                          Очистить $(OOF_ARTIFACT_DIR)/."
	@echo "    make clean-submissions                  Очистить $(SUBMISSION_ARTIFACT_DIR)/."
	@echo "    make reset-outputs                      Полностью очистить артефакты."
	@echo ""
	@echo "  ПОЛЕЗНЫЕ ПЕРЕМЕННЫЕ:"
	@echo "    EXPERIMENT (по умолчанию 'exp001_baseline_lgbm'), FEATURE_PIPELINE ('v1_baseline'),"
	@echo "    FEATURE_SET ('v1_all'), FEATURE_STORAGE ('processed/titanic')."
	@echo "    WANDB_MODE (по умолчанию 'disabled'), OUTPUT_ROOT."
	@echo "=============================================================================="

# ==============================================================================
# --- Установка зависимостей ---------------------------------------------------
# ==============================================================================
install:
	@echo ">>> Установка зависимостей..."
	@if [ "$(TOOL)" = "pip" ]; then \
		echo ">>> Используется pip..."; \
		python3 -m pip install -r requirements.txt; \
	elif [ "$(TOOL)" = "poetry" ]; then \
		if command -v poetry >/dev/null 2>&1; then \
			echo ">>> Используется Poetry..."; \
			poetry install; \
		else \
			echo ">>> Ошибка: Poetry не найден, но выбран TOOL=poetry."; \
			exit 1; \
		fi; \
	else \
		if command -v poetry >/dev/null 2>&1; then \
			echo ">>> Используется Poetry..."; \
			poetry install; \
		else \
			echo ">>> Poetry не найден. Используется pip..."; \
			python3 -m pip install -r requirements.txt; \
		fi; \
	fi
	@echo ">>> ГОТОВО. WANDB_MODE=$(WANDB_MODE)"

# ==============================================================================
# --- Основной workflow --------------------------------------------------------
# ==============================================================================
features:
	@echo ">>> Генерация признаков для эксперимента: $(EXPERIMENT)..."
	@$(PY) src/scripts/make_features.py experiment=$(EXPERIMENT) $(COMMON_OVERRIDES) $(FEATURE_OVERRIDES)

select:
	@echo ">>> Отбор признаков: $(EXPERIMENT) (selection=$(SELECTION))..."
	@$(PY) src/scripts/select_features.py experiment=$(EXPERIMENT) selection=$(SELECTION) $(COMMON_OVERRIDES)

train:
	@echo ">>> Обучение (CV) для эксперимента: $(EXPERIMENT)..."
	@$(PY) src/scripts/train.py experiment=$(EXPERIMENT) $(COMMON_OVERRIDES)

fulltrain:
	@echo ">>> Обучение на полном датасете для: $(EXPERIMENT)..."
	@$(PY) src/scripts/train.py experiment=$(EXPERIMENT) training.full_data=true $(COMMON_OVERRIDES)

tune:
	@echo ">>> Подбор гиперпараметров: tuning=$(TUNING), experiment=$(EXPERIMENT)..."
	@$(PY) src/scripts/tune.py experiment=$(EXPERIMENT) tuning=$(TUNING) $(COMMON_OVERRIDES)

stack:
stacking:
	@echo ">>> Стекинг с конфигурацией: $(STACKING)..."
	@$(PY) src/scripts/stack.py stacking=$(STACKING) $(COMMON_OVERRIDES)

predict:
	@echo ">>> Инференс по конфигурации: $(INFERENCE)..."
	@$(PY) src/scripts/predict.py inference=$(INFERENCE) $(COMMON_OVERRIDES)

pseudo:
	@echo ">>> Запуск пайплайна псевдо-лейблинга..."
	@$(PY) src/scripts/pseudo_label.py $(COMMON_OVERRIDES)

# ==============================================================================
# --- Готовые сценарии ---------------------------------------------------------
# ==============================================================================
baseline:
	@$(MAKE) features --no-print-directory EXPERIMENT=exp001_baseline_lgbm FEATURE_PIPELINE=v1_baseline FEATURE_SET=v1_all
	@$(MAKE) train --no-print-directory EXPERIMENT=exp001_baseline_lgbm

titanic-all:
	@echo ">>> Полный прогон Titanic (эксперименты: $(TITANIC_EXPERIMENTS))..."
	@set -e; \
	for exp in $(TITANIC_EXPERIMENTS); do \
		case "$${exp}" in \
			exp001_baseline_lgbm|exp002_baseline_catboost) \
				feat_pipeline=v1_baseline; \
				feat_set=v1_all; \
				;; \
			exp003_tuned_lgbm|exp004_feature_engineering) \
				feat_pipeline=v2_advanced; \
				feat_set=v2_selected; \
				;; \
			*) \
				feat_pipeline="$(FEATURE_PIPELINE)"; \
				feat_set="$(FEATURE_SET)"; \
				;; \
		esac; \
		echo ""; \
		echo ">>> === $${exp}: генерация признаков (pipeline=$${feat_pipeline}, features=$${feat_set}) ==="; \
		$(MAKE) --no-print-directory features EXPERIMENT=$${exp} FEATURE_PIPELINE=$${feat_pipeline} FEATURE_SET=$${feat_set}; \
		echo ">>> === $${exp}: обучение ==="; \
		$(MAKE) --no-print-directory train EXPERIMENT=$${exp}; \
	done

ensemble:
	@echo ">>> Weighted ensemble эксперимент: $(ENSEMBLE_EXPERIMENT)..."
	@$(MAKE) train --no-print-directory EXPERIMENT=$(ENSEMBLE_EXPERIMENT)

# ==============================================================================
# --- Обслуживание -------------------------------------------------------------
# ==============================================================================
clean:
	@echo ">>> Очистка временных файлов Python..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	@echo ">>> Очистка завершена."

clean-oof:
	@echo ">>> Удаление артефактов OOF в $(OOF_ARTIFACT_DIR)..."
	@rm -rf $(OOF_ARTIFACT_DIR)

clean-submissions:
	@echo ">>> Удаление сабмишенов в $(SUBMISSION_ARTIFACT_DIR)..."
	@rm -rf $(SUBMISSION_ARTIFACT_DIR)

reset-outputs: clean-oof clean-submissions
	@echo ">>> Очистка каталога $(OUTPUT_ROOT)..."
	@rm -rf $(OUTPUT_ROOT)
	@echo ">>> Готово."
