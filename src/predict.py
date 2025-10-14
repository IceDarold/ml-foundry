# src/predict.py

import glob
import time
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

# Импортируем базовый класс, чтобы можно было загружать любую модель
from src.models.base import ModelInterface


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def predict(cfg: DictConfig) -> None:
    """
    Скрипт для инференса (предсказаний).

    Загружает обученные модели из директории, соответствующей `run_id`,
    применяет их к указанному набору признаков и сохраняет файл сабмишена.
    """
    start_time = time.time()
    
    print("--- Запуск инференса ---")
    
    # --- 1. Проверка и подготовка путей ---
    run_id = cfg.inference.run_id
    if not run_id:
        raise ValueError("Необходимо указать ID запуска (`inference.run_id`) для инференса!")
        
    print(f"Используются модели из W&B run_id: {run_id}")
    
    # Ищем директорию с результатами по всему проекту
    # Это более надежно, чем жестко заданный путь.
    try:
        search_path = Path(hydra.utils.get_original_cwd()) / "outputs"
        models_path = next(search_path.glob(f"**/{run_id}"))
    except StopIteration:
        raise FileNotFoundError(f"Не удалось найти директорию для run_id '{run_id}'. "
                                f"Проверьте, что такой запуск существует в папке 'outputs'.")

    print(f"Путь к моделям: {models_path}")
    
    # --- 2. Загрузка данных ---
    data_path = Path(hydra.utils.get_original_cwd()) / "data"
    features_path = data_path / cfg.data.features_path
    
    input_filename = cfg.inference.input_features_filename
    test_features_path = features_path / input_filename
    
    if not test_features_path.exists():
        raise FileNotFoundError(f"Файл с признаками не найден: {test_features_path}")
        
    test_df = pd.read_parquet(test_features_path)
    
    # Загружаем список колонок из конфига, соответствующего эксперименту
    feature_cols = cfg.features.cols
    X_test = test_df[feature_cols]
    
    print(f"Загружены данные для инференса: {test_features_path.name} (shape: {X_test.shape})")

    # --- 3. Загрузка моделей и инференс ---
    # Сначала ищем модели, обученные на CV
    model_files = sorted(glob.glob(str(models_path / "model_fold_*.pkl")))
    
    # Если их нет, ищем одну модель, обученную на всех данных
    if not model_files:
        model_files = sorted(glob.glob(str(models_path / "model_full_train.pkl")))

    if not model_files:
        raise FileNotFoundError(f"В директории {models_path} не найдены обученные модели.")
        
    print(f"Найдено {len(model_files)} моделей для предсказания.")
    
    final_preds = None
    
    for model_path in tqdm(model_files, desc="Предсказание по моделям"):
        # Загружаем модель
        model: ModelInterface = ModelInterface.load(model_path)
        
        fold_preds = model.predict_proba(X_test)
        
        if final_preds is None:
            final_preds = fold_preds
        else:
            final_preds += fold_preds
            
    # Усредняем предсказания, если моделей было несколько
    if len(model_files) > 1:
        final_preds /= len(model_files)

    # --- 4. Сохранение сабмишена ---
    output_path = data_path / cfg.data.submissions_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Используем ID колонку из конфига инференса
    id_col = cfg.inference.id_col
    if id_col not in test_df.columns:
        raise ValueError(f"ID колонка '{id_col}', указанная в конфиге, не найдена в данных для инференса.")
        
    submission_df = pd.DataFrame({
        id_col: test_df[id_col],
        cfg.globals.target_col: final_preds
    })
    
    submission_filepath = output_path / f"submission_from_{run_id}.csv"
    submission_df.to_csv(submission_filepath, index=False)
    
    end_time = time.time()
    print("\n--- Инференс завершен ---")
    print(f"Файл сабмишена сохранен в: {submission_filepath}")
    print(f"Общее время выполнения: {end_time - start_time:.2f} секунд.")


if __name__ == "__main__":
    predict()