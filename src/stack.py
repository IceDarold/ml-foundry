# src/stack.py

import warnings
from pathlib import Path
import time
from typing import List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import wandb

# Импортируем наши собственные модули
from src import utils
from src.models.base import ModelInterface # Для мета-модели
from src.metrics.base import MetricInterface # Для оценки стекинга

warnings.filterwarnings("ignore")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def stack(cfg: DictConfig) -> float:
    """
    Пайплайн для стекинга.

    Обучает мета-модель на Out-of-Fold (OOF) предсказаниях базовых моделей.
    """
    start_time = time.time()
    
    # === 1. Инициализация и подготовка ===
    utils.seed_everything(cfg.globals.seed)
    output_dir = Path(hydra.utils.get_original_cwd()) / "outputs/stacking" / utils.get_timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"stacking-{cfg.stacking.name}-{utils.get_timestamp()}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["stacking"],
        dir=output_dir,
    )
    
    print("--- Запуск пайплайна стекинга ---")
    print(OmegaConf.to_yaml(cfg.stacking))
    
    # === 2. Загрузка и объединение предсказаний ===
    data_path = Path(hydra.utils.get_original_cwd()) / "data"
    id_col = cfg.globals.id_col
    target_col = cfg.globals.target_col
    
    # --- 2.1. Загрузка OOF-предсказаний (для обучения мета-модели) ---
    oof_dfs = []
    print("\n--- Загрузка OOF-предсказаний ---")
    for model_cfg in cfg.stacking.base_models:
        oof_path = data_path / model_cfg.oof_path
        if not oof_path.exists():
            raise FileNotFoundError(f"OOF-файл не найден: {oof_path}")
        
        oof_df = pd.read_csv(oof_path)
        # Переименовываем колонку с предсказаниями, чтобы избежать конфликтов
        oof_df.rename(columns={'oof_preds': f"oof_{model_cfg.name}"}, inplace=True)
        oof_dfs.append(oof_df)
        print(f"  - Загружен OOF от модели: {model_cfg.name}")
        
    # Объединяем все OOF-файлы в один датафрейм
    meta_train_df = oof_dfs[0]
    for i in range(1, len(oof_dfs)):
        meta_train_df = pd.merge(meta_train_df, oof_dfs[i], on=id_col, how='left')
        
    # --- 2.2. Загрузка Test-предсказаний (для инференса) ---
    test_dfs = []
    print("\n--- Загрузка Test-предсказаний ---")
    for model_cfg in cfg.stacking.base_models:
        test_path = data_path / model_cfg.test_preds_path
        if not test_path.exists():
            raise FileNotFoundError(f"Файл с предсказаниями на тесте не найден: {test_path}")
            
        test_df = pd.read_csv(test_path)
        test_df.rename(columns={target_col: f"oof_{model_cfg.name}"}, inplace=True)
        test_dfs.append(test_df)
        print(f"  - Загружены предсказания на тесте от модели: {model_cfg.name}")
        
    meta_test_df = test_dfs[0]
    for i in range(1, len(test_dfs)):
        meta_test_df = pd.merge(meta_test_df, test_dfs[i], on=id_col, how='left')

    # --- 2.3. Добавляем исходный таргет ---
    raw_train_df = pd.read_csv(data_path / cfg.data.processed_path / cfg.data.train_file)
    meta_train_df = pd.merge(meta_train_df, raw_train_df[[id_col, target_col]], on=id_col, how='left')
    
    feature_cols = [col for col in meta_train_df.columns if col.startswith('oof_')]
    
    X = meta_train_df[feature_cols]
    y = meta_train_df[target_col]
    X_test = meta_test_df[feature_cols]
    
    print(f"\nСоздан мета-датасет с {len(feature_cols)} признаками (предсказаниями).")

    # === 3. Обучение мета-модели на кросс-валидации ===
    cv_splitter = hydra.utils.instantiate(cfg.validation.strategy)
    main_metric: MetricInterface = hydra.utils.instantiate(cfg.metric.main)
    main_metric_name = main_metric.__class__.__name__

    oof_preds = np.zeros(len(meta_train_df))
    test_preds = np.zeros(len(meta_test_df))
    fold_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(cv_splitter.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        # Инстанциируем мета-модель
        meta_model: ModelInterface = hydra.utils.instantiate(cfg.stacking.meta_model)
        
        # Убираем параметры, которые не нужны для простых моделей
        fit_params = cfg.training.fit_params.copy()
        fit_params.pop('early_stopping_rounds', None)
        fit_params.pop('verbose', None)

        meta_model.fit(X_train, y_train, **fit_params)
        
        valid_preds_proba = meta_model.predict_proba(X_valid)
        oof_preds[valid_idx] = valid_preds_proba
        test_preds += meta_model.predict_proba(X_test) / cv_splitter.get_n_splits()
        
        fold_score = main_metric(y_valid.values, valid_preds_proba)
        fold_scores.append(fold_score)
        wandb.log({f"fold_score/{main_metric_name}": fold_score, "fold": fold + 1})
        print(f"Скор стекинга на фолде {fold + 1}: {fold_score:.5f}")

    # === 4. Оценка и финализация ===
    oof_score_mean = np.mean(fold_scores)
    wandb.summary[f"final_stack_oof_score"] = oof_score_mean
    print(f"\nИтоговый OOF-скор стекинга: {oof_score_mean:.5f}")

    # === 5. Сохранение финального сабмишена ===
    submission_path = data_path / cfg.data.submissions_path
    submission_filepath = submission_path / f"submission_stack_{cfg.stacking.name}.csv"
    
    submission_df = pd.DataFrame({id_col: meta_test_df[id_col], target_col: test_preds})
    submission_df.to_csv(submission_filepath, index=False)
    
    wandb.save(str(submission_filepath))
    print(f"Финальный сабмишен сохранен в: {submission_filepath}")
    
    end_time = time.time()
    print(f"Пайплайн стекинга завершен за {end_time - start_time:.2f} секунд.")
    wandb.finish()
    
    return oof_score_mean


if __name__ == "__main__":
    stack()