# src/select_features.py

import time
from pathlib import Path
from typing import Dict, Any

import hydra
import lightgbm as lgb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.exceptions import ConfigurationError, DataLoadError, FileOperationError


def validate_config(cfg: DictConfig) -> None:
    """
    Validate configuration parameters.
    """
    if cfg.selection.top_n <= 0:
        raise ConfigurationError("top_n must be positive.")
    if cfg.selection.top_n > len(cfg.features.cols):
        raise ConfigurationError("top_n cannot exceed the number of available features.")
    if cfg.globals.id_col not in cfg.features.cols:
        raise ConfigurationError("ID column must be in feature columns.")
    if cfg.globals.target_col in cfg.features.cols:
        raise ConfigurationError("Target column must not be in feature columns.")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def select_features(cfg: DictConfig) -> None:
    """
    Скрипт для отбора признаков.

    Обучает модель LightGBM на кросс-валидации, агрегирует важность
    признаков по всем фолдам и сохраняет список `top_n` лучших признаков.
    """
    start_time = time.time()

    validate_config(cfg)

    print("--- Запуск отбора признаков ---")
    print(OmegaConf.to_yaml(cfg.selection))
    
    # --- 1. Загрузка данных ---
    data_path = Path(hydra.utils.get_original_cwd()) / "data"
    features_path = data_path / cfg.data.features_path

    train_features_path = features_path / f"train_{cfg.feature_engineering.name}.parquet"
    try:
        train_df = pd.read_parquet(train_features_path)
    except Exception as e:
        raise DataLoadError("Failed to load training features file.") from e
    
    print(f"Загружены признаки: {train_features_path.name} (shape: {train_df.shape})")

    # --- 2. Подготовка данных и CV ---
    target_col = cfg.globals.target_col
    # Исключаем ID и таргет, чтобы получить полный список исходных признаков
    feature_cols = train_df.columns.drop([cfg.globals.id_col, target_col]).tolist()
    
    X = train_df[feature_cols]
    y = train_df[target_col]
    
    cv_splitter = hydra.utils.instantiate(cfg.validation.strategy)
    
    # --- 3. Обучение на CV и сбор важности признаков ---
    feature_importances = pd.DataFrame(index=feature_cols)
    
    for fold, (train_idx, valid_idx) in tqdm(
        enumerate(cv_splitter.split(X, y)), 
        total=cv_splitter.get_n_splits(),
        desc="Обучение на фолдах"
    ):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        
        # Используем простую модель LGBM для скорости
        model = lgb.LGBMClassifier(random_state=cfg.globals.seed, **cfg.selection.model_params)
        model.fit(X_train, y_train)
        
        # Сохраняем важность признаков для текущего фолда
        feature_importances[f'fold_{fold+1}'] = model.feature_importances_

    # --- 4. Агрегация и отбор лучших ---
    # Усредняем важность по всем фолдам
    feature_importances['mean'] = feature_importances.mean(axis=1)
    feature_importances.sort_values(by='mean', ascending=False, inplace=True)
    
    top_n = cfg.selection.top_n
    top_features = feature_importances.head(top_n).index.tolist()
    
    print(f"\n--- Топ 10 самых важных признаков ---")
    print(feature_importances['mean'].head(10))
    print("--------------------------------------")
    
    # --- 5. Сохранение артефакта ---
    output_path = data_path / cfg.data.feature_lists_path
    output_path.mkdir(parents=True, exist_ok=True)

    # Создаем информативное имя для файла
    output_filename = f"{cfg.feature_engineering.name}_top_{top_n}.txt"
    output_filepath = output_path / output_filename

    try:
        with open(output_filepath, 'w') as f:
            for feature in top_features:
                f.write(f"{feature}\n")
    except Exception as e:
        raise FileOperationError("Failed to save feature list file.") from e
            
    end_time = time.time()
    print(f"\nОтбор завершен. Список из {len(top_features)} признаков сохранен в:")
    print(output_filepath)
    print(f"Общее время выполнения: {end_time - start_time:.2f} секунд.")

if __name__ == "__main__":
    select_features()