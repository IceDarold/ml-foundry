# src/tune.py

import hydra
from omegaconf import DictConfig, OmegaConf
import optuna
from optuna_integration import WandbCallback
import wandb

# Импортируем нашу основную функцию обучения
from src.train import train


# Создаем W&B callback для визуализации
wandb_callback = WandbCallback(
    metric_name="oof_score_mean", # Метрика, которую Optuna будет максимизировать
    wandb_kwargs={"project": "tuning-project-name"} # Укажите ваш проект для тюнинга
)

# ==================================================================================
# Objective Function
# ==================================================================================
@wandb_callback.track_in_wandb() # Автоматически логирует параметры и результат
def objective(trial: optuna.trial.Trial, cfg: DictConfig) -> float:
    """
    "Целевая функция" для Optuna.

    Optuna вызывает эту функцию на каждой итерации, передавая объект `trial`,
    который предлагает новые значения для гиперпараметров.
    Функция запускает полный пайплайн обучения и возвращает метрику качества.
    """
    
    # 1. Предлагаем новые гиперпараметры
    # Мы будем определять пространство поиска в YAML-конфиге
    # и "инжектировать" его в `cfg` перед вызовом.
    # Optuna предлагает значения, а мы перезаписываем ими конфиг.
    
    # Пример:
    # cfg.model.params.learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    # cfg.model.params.max_depth = trial.suggest_int('max_depth', 3, 10)
    
    # --- Универсальная реализация через конфиг ---
    for param, search_space in cfg.tuning.search_space.items():
        # param будет, например, 'model.params.learning_rate'
        # search_space - словарь {'type': 'float', 'low': 0.01, 'high': 0.1, 'log': True}
        
        param_type = search_space.pop('type')
        
        if param_type == 'float':
            value = trial.suggest_float(param.split('.')[-1], **search_space)
        elif param_type == 'int':
            value = trial.suggest_int(param.split('.')[-1], **search_space)
        elif param_type == 'categorical':
            value = trial.suggest_categorical(param.split('.')[-1], **search_space)
        else:
            raise ValueError(f"Неподдерживаемый тип параметра: {param_type}")
        
        # Обновляем значение в объекте конфига с помощью OmegaConf
        OmegaConf.update(cfg, param, value)
    
    # 2. Запускаем обучение с новыми параметрами
    # Мы вызываем нашу основную функцию `train`, которая вернет oof_score.
    # W&B будет инициализирован внутри `train`, но callback его "подхватит".
    try:
        oof_score = train(cfg)
    except Exception as e:
        print(f"Ошибка во время trial: {e}. Optuna обработает это как неудачный запуск.")
        # Если произошла ошибка (например, из-за нехватки памяти),
        # Optuna должна знать, что этот trial провалился.
        raise optuna.exceptions.TrialPruned()
        
    return oof_score

# ==================================================================================
# Main Tuning Function
# ==================================================================================
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def tune(cfg: DictConfig) -> None:
    """
    Главный скрипт для запуска подбора гиперпараметров.
    """
    
    print("--- Запуск подбора гиперпараметров ---")
    print(OmegaConf.to_yaml(cfg.tuning))
    
    # Обновляем имя проекта в W&B callback из конфига
    wandb_callback.wandb_kwargs['project'] = cfg.wandb.project
    wandb_callback.wandb_kwargs['entity'] = cfg.wandb.entity

    # Создаем "исследование" Optuna
    study = optuna.create_study(
        direction=cfg.tuning.direction, # "maximize" или "minimize"
        study_name=f"{cfg.model._target_.split('.')[-1]}-{cfg.features.name}-tuning"
    )

    # Запускаем оптимизацию
    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=cfg.tuning.n_trials,
        callbacks=[wandb_callback]
    )

    print("\n--- Подбор гиперпараметров завершен ---")
    print(f"Количество завершенных trials: {len(study.trials)}")
    print("Лучший trial:")
    trial = study.best_trial

    print(f"  Value (OOF Score): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    print("\n💡 Скопируйте эти параметры в ваш `conf/model/*.yaml` для создания 'tuned' версии модели.")

if __name__ == "__main__":
    tune()