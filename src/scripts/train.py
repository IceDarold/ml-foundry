# src/train.py

import warnings
from pathlib import Path
import time
import os
from typing import List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import wandb

# Импортируем наши собственные модули
from src import utils
from src.models.base import ModelInterface
from src.metrics.base import MetricInterface
from src.validation.base import BaseSplitter # Наш новый интерфейс для валидации
from src.utils import get_logger, setup_logging, performance_monitor
from src.utils import validate_type, validate_non_empty
from hydra.core.hydra_config import HydraConfig

LOGGER = get_logger(__name__)

warnings.filterwarnings("ignore")


@hydra.main(config_path="../../conf/projects/titanic", config_name="titanic", version_base=None)
@validate_type(DictConfig)
@performance_monitor
def train(cfg: DictConfig) -> float:
    """
    Главный пайплайн для обучения модели.

    1. Инициализирует W&B run.
    2. Скачивает версионированный набор признаков из W&B Artifacts.
    3. Выполняет обучение в одном из двух режимов (CV или Full Data), используя
       гибкий модуль валидации.
    4. Сохраняет и логирует артефакты (модели, OOF-предсказания, сабмишен).
    """
    start_time = time.time()

    # === 0. Setup logging ===
    setup_logging(cfg)

    # === 1. Инициализация W&B и подготовка ===
    utils.seed_everything(cfg.globals.seed)
    # Hydra создает уникальную директорию для каждого запуска.
    # Мы будем использовать ее для временного хранения артефактов.
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    experiment_cfg = OmegaConf.select(cfg, "experiment")
    if experiment_cfg:
        OmegaConf.set_struct(cfg, False)
        for key, value in experiment_cfg.items():
            cfg[key] = value
        OmegaConf.set_struct(cfg, True)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger = LOGGER

    model_target = OmegaConf.select(cfg, "model._target_")
    model_name = (model_target.split(".")[-1] if model_target else "Model").replace("Model", "")
    feature_name = OmegaConf.select(cfg, "feature_engineering.name") or "unknown"
    is_full = bool(OmegaConf.select(cfg, "training.full_data"))
    run_name = f"{model_name}-{feature_name}{'-FULL' if is_full else '-CV'}"

    wandb_mode = os.environ.get("WANDB_MODE", "").lower()
    use_wandb = wandb_mode not in {"offline", "disabled"}
    wandb_tags = OmegaConf.select(cfg, "wandb.tags") or []
    run = None
    if use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=cfg_dict,
            tags=wandb_tags,
            job_type="training",
        )
    else:
        logger.info("W&B disabled via WANDB_MODE, skipping remote logging.")
    logger.info("--- Конфигурация эксперимента ---")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("-----------------------------------")
    
    # === 2. Загрузка данных из артефакта W&B ===
    logger.info("\n--- Загрузка набора признаков ---")

    feature_artifact_name = feature_name
    if use_wandb:
        artifact_to_use = f"{cfg.wandb.entity}/{cfg.wandb.project}/{feature_artifact_name}:latest"
        logger.info(f"Используется артефакт: {artifact_to_use}")

        try:
            artifact = run.use_artifact(artifact_to_use)
        except wandb.errors.CommError as e:
            logger.error(f"\nНе удалось найти артефакт '{artifact_to_use}'.")
            logger.error("Убедитесь, что вы сначала запустили `make_features.py` с соответствующим конфигом.")
            raise e

        artifact_dir = Path(artifact.download())
    else:
        artifact_dir = Path(hydra.utils.get_original_cwd()) / "data" / cfg.data.features_path
        logger.info(f"Используются локальные файлы признаков из: {artifact_dir}")
    
    train_features_path = artifact_dir / f"train_{feature_artifact_name}.parquet"
    test_features_path = artifact_dir / f"test_{feature_artifact_name}.parquet"
    
    train_df = pd.read_parquet(train_features_path)
    test_df = pd.read_parquet(test_features_path)

    logger.info("Признаки успешно загружены.")
    
    feature_cols = OmegaConf.select(cfg, "features.cols")
    if not feature_cols:
        excluded = {cfg.globals.id_col, cfg.globals.target_col}
        feature_cols = [col for col in train_df.columns if col not in excluded]
    target_col = cfg.globals.target_col
    
    X = train_df[feature_cols]
    y = train_df[target_col]
    X_test = test_df[feature_cols]
    
    logger.info(f"Используется {len(feature_cols)} признаков. train.shape={X.shape}, test.shape={X_test.shape}")

    overrides = []
    try:
        overrides = HydraConfig.get().overrides.task or []
    except Exception:
        overrides = []
    experiment_override = next((ov for ov in overrides if ov.startswith("experiment=")), None)
    experiment_name = experiment_override.split("=", 1)[1] if experiment_override else feature_artifact_name

    # ==========================================================================
    # ❗️ ВЫБОР РЕЖИМА ОБУЧЕНИЯ
    # ==========================================================================
    if cfg.training.full_data:
        # --- РЕЖИМ 1: ОБУЧЕНИЕ НА ВСЕХ ДАННЫХ ---
        logger.info("\n--- Режим: Обучение на 100% данных ---")
        
        model: ModelInterface = hydra.utils.instantiate(cfg.model)
        fit_params_cfg = OmegaConf.select(cfg, "training.fit_params")
        fit_params = OmegaConf.to_container(fit_params_cfg, resolve=True) if fit_params_cfg else {}
        if getattr(model, "_using_fallback", False):
            fit_params.pop("early_stopping_rounds", None)
            fit_params.pop("eval_metric", None)

        model.fit(X, y, **fit_params)
        test_preds = model.predict_proba(X_test)
        
        model_path = output_dir / "model_full_train.pkl"
        model.save(model_path)
        logger.info(f"Модель, обученная на всех данных, сохранена в: {model_path}")

        oof_score_mean = -1.0
        
    else:
        # --- РЕЖИМ 2: ОБУЧЕНИЕ НА КРОСС-ВАЛИДАЦИИ ---
        logger.info("\n--- Режим: Обучение на кросс-валидации ---")

        # Инстанциируем сплиттер из нашего нового модуля валидации
        splitter: BaseSplitter = hydra.utils.instantiate(cfg.validation)
        logger.info(f"Стратегия валидации: {splitter.__class__.__name__} ({splitter.get_n_splits()} фолдов)")
        
        # Подготовка групп, если они требуются для сплиттера (например, GroupKFold)
        groups = None
        group_col = cfg.validation.get("group_col")
        if group_col:
            if group_col not in train_df.columns:
                raise ValueError(f"Колонка для группировки '{group_col}' не найдена в данных.")
            groups = train_df[group_col]
            logger.info(f"Используется группировка по колонке: {group_col}")
        
        # Инициализация метрик
        main_metric: MetricInterface = hydra.utils.instantiate(cfg.metric.main)
        main_metric_name = main_metric.__class__.__name__.replace("Metric", "")
        
        additional_metrics: List[MetricInterface] = []
        additional_cfg = OmegaConf.select(cfg, "metric.additional")
        if additional_cfg:
            if isinstance(additional_cfg, DictConfig) and "_target_" in additional_cfg:
                iterable = [additional_cfg]
            elif isinstance(additional_cfg, DictConfig):
                iterable = list(additional_cfg.values())
            else:
                iterable = list(additional_cfg)

            for metric_cfg in iterable:
                metric_obj = hydra.utils.instantiate(metric_cfg)
                metric_obj.name = getattr(metric_obj, "name", metric_obj.__class__.__name__.replace("Metric", ""))
                additional_metrics.append(metric_obj)

        oof_preds = np.zeros(len(train_df))
        test_preds = np.zeros(len(test_df))
        fold_scores = []
        
        # Получаем итератор для разделения данных
        split_iterator = splitter.split(data=train_df, y=y, groups=groups)
        
        for fold, (train_idx, valid_idx) in enumerate(split_iterator):
            fold_start_time = time.time()
            logger.info(f"\n--- Фолд {fold + 1}/{splitter.get_n_splits()} ---")
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            
            model: ModelInterface = hydra.utils.instantiate(cfg.model)
            LOGGER.info(
                "Model instance: %s, fallback flag: %s, backend type: %s",
                model.__class__.__name__,
                getattr(model, '_using_fallback', None),
                getattr(getattr(model, 'model', None), '__class__', type(None)).__name__,
            )

            fit_params_cfg = OmegaConf.select(cfg, "training.fit_params")
            fit_params = OmegaConf.to_container(fit_params_cfg, resolve=True) if fit_params_cfg else {}
            use_eval_set = True
            if getattr(model, "_using_fallback", False):
                fit_params.pop("early_stopping_rounds", None)
                fit_params.pop("eval_metric", None)
                use_eval_set = False
                LOGGER.info(f"Fallback fit parameters (after cleanup): {fit_params}")

            fit_kwargs = dict(fit_params)
            if use_eval_set:
                fit_kwargs["eval_set"] = [(X_valid, y_valid)]

            model.fit(X_train, y_train, **fit_kwargs)
            
            valid_preds_proba = model.predict_proba(X_valid)
            oof_preds[valid_idx] = valid_preds_proba
            test_preds += model.predict_proba(X_test) / splitter.get_n_splits()
            
            # Подготовка дополнительных данных для метрики (например, групп)
            metric_kwargs = {}
            if groups is not None:
                metric_kwargs['groups'] = groups.iloc[valid_idx]
            
            # Логирование
            log_dict = {"fold": fold + 1}
            fold_score = main_metric(y_valid.values, valid_preds_proba, **metric_kwargs)
            fold_scores.append(fold_score)
            log_dict[f"fold_score/{main_metric_name}"] = fold_score
            
            for metric_obj in additional_metrics:
                add_score = metric_obj(y_valid.values, valid_preds_proba, **metric_kwargs)
                log_dict[f"fold_score/{metric_obj.name}"] = add_score
            if run is not None:
                run.log(log_dict)
            
            # Сохранение модели
            model_path = output_dir / f"model_fold_{fold + 1}.pkl"
            model.save(model_path)
            
            fold_end_time = time.time()
            logger.info(f"Скор на фолде {fold + 1} ({main_metric_name}): {fold_score:.5f} (за {fold_end_time - fold_start_time:.2f} с)")

        oof_score_mean = np.mean(fold_scores)
        oof_score_std = np.std(fold_scores)
        
        logger.info(f"\n--- Итоговый результат CV ---")
        logger.info(f"Средний OOF-скор ({main_metric_name}): {oof_score_mean:.5f} (Std: {oof_score_std:.5f})")
        
        if run is not None:
            run.summary[f"oof_score_mean"] = oof_score_mean
            run.summary[f"oof_score_std"] = oof_score_std

        # Сохранение OOF-предсказаний
        oof_df = pd.DataFrame({cfg.globals.id_col: train_df[cfg.globals.id_col], 'oof_preds': oof_preds})
        oof_path = output_dir / "oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)

    # === ФИНАЛЬНЫЙ ШАГ: СОХРАНЕНИЕ САБМИШЕНА И АРТЕФАКТОВ ===
    logger.info("\n--- Сохранение артефактов ---")
    submission_df = pd.DataFrame({cfg.globals.id_col: test_df[cfg.globals.id_col], target_col: test_preds})
    submission_path = output_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    
    project_root = Path(hydra.utils.get_original_cwd())
    submissions_root = project_root / "07_submissions"
    submissions_root.mkdir(parents=True, exist_ok=True)
    canonical_submission = submissions_root / f"submission_{experiment_name}.csv"
    submission_df.to_csv(canonical_submission, index=False)

    if not cfg.training.full_data:
        oof_root = project_root / "05_oof"
        oof_root.mkdir(parents=True, exist_ok=True)
        canonical_oof = oof_root / f"oof_{experiment_name}.csv"
        oof_df.to_csv(canonical_oof, index=False)

    if run is not None:
        output_artifact = wandb.Artifact(name=f"output-{run.id}", type="output")
        output_artifact.add_file(str(submission_path))
        if not cfg.training.full_data:
            output_artifact.add_file(str(oof_path))
        output_artifact.add_dir(str(output_dir), name="models")
        run.log_artifact(output_artifact)
    
    end_time = time.time()
    logger.info(f"Все результаты сохранены в: {output_dir}")
    logger.info(f"Пайплайн завершен за {end_time - start_time:.2f} секунд.")
    
    if run is not None:
        run.finish()
    
    return oof_score_mean


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        LOGGER.error(f"\nКритическая ошибка во время выполнения: {e}")
        if wandb.run:
            LOGGER.error("Завершение W&B run с ошибкой...")
            wandb.finish(exit_code=1)
        raise
class TrainingContextManager:
    """Context manager for training scripts to handle resource cleanup."""

    def __init__(self, run=None):
        self.run = run

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # Cleanup W&B run if it exists
        if self.run and exc_type is not None:
            # Finish the run with error status if an exception occurred
            self.run.finish(exit_code=1)
        elif self.run:
            self.run.finish()
