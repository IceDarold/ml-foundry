# src/models/lgbm.py

from typing import Any, Dict
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import pandas as pd

from .base import ModelInterface # Импортируем наш базовый "контракт"

# ==================================================================================
# LGBMModel
# ==================================================================================
@dataclass
class LGBMModel(ModelInterface):
    """Wrapper class for LightGBM Classifier and Regressor models.

    This class provides a unified interface for LightGBM models, automatically
    determining whether to use classification or regression based on the objective
    parameter. It supports both binary/multiclass classification and regression tasks.

    Attributes:
        params (Dict[str, Any]): Model parameters passed to LightGBM.
        is_regressor (bool): Whether the model is configured for regression.
        model: The underlying LightGBM model instance.

    Example:
        >>> params = {'objective': 'binary', 'num_leaves': 31}
        >>> model = LGBMModel(params)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    params: Dict[str, Any]
    is_regressor: bool = field(init=False, default=False)
    model = field(init=False)

    def __post_init__(self):
        """
        Инициализирует модель LightGBM.

        Args:
            params (Dict[str, Any]): Словарь с параметрами для модели.
                                      Например, {'objective': 'binary', ...}
        """
        # Выбираем класс в зависимости от задачи (классификация или регрессия)
        objective = self.params.get('objective', '').lower()

        if 'regression' in objective or 'mae' in objective or 'mse' in objective:
            self.is_regressor = True
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            self.is_regressor = False
            self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """Train the LightGBM model.

        Accepts eval_set and other fit parameters directly, enabling features
        like early stopping and validation monitoring.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target values.
            **kwargs: Additional arguments passed to the model's fit method,
                such as eval_set, early_stopping_rounds, eval_metric, etc.

        Note:
            Use eval_set parameter to monitor validation performance during training.
            Early stopping can be enabled with early_stopping_rounds parameter.
        """
        print("Обучение модели LightGBM...")
        self.model.fit(X_train, y_train, **kwargs)

    def predict(self, X: pd.DataFrame) -> Any:
        """Generate predictions for the input data.

        Args:
            X (pd.DataFrame): Input features for prediction.

        Returns:
            Any: Predicted class labels (classification) or continuous values (regression).
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """Generate probability predictions for classification or numeric predictions for regression.

        Args:
            X (pd.DataFrame): Input features for prediction.

        Returns:
            Any: For classification, returns probabilities for the positive class.
                For regression, returns numeric predictions (same as predict()).

        Raises:
            AttributeError: If the underlying model doesn't support probability predictions
                (should not occur for properly configured models).
        """
        if self.is_regressor:
            # Для регрессора predict_proba не существует, возвращаем обычные предсказания
            return self.model.predict(X)
        else:
            # Для классификации возвращаем только вероятности для класса "1"
            return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str) -> None:
        """Save the trained model to disk using joblib.

        Args:
            filepath (str): Path where the model should be saved.
                Should include the file extension (e.g., '.pkl' or '.joblib').

        Raises:
            IOError: If the file cannot be written to the specified path.
        """
        print(f"Saving model to {filepath}")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'LGBMModel':
        """Load a saved model from disk using joblib.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            LGBMModel: Loaded model instance ready for prediction.

        Raises:
            IOError: If the file cannot be read.
            ValueError: If the file format is invalid or corrupted.
        """
        print(f"Loading model from {filepath}")
        return joblib.load(filepath)
    def __enter__(self):
        """Enter the context manager.

        Returns:
            LGBMModel: The model instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # Cleanup LightGBM model resources
        if hasattr(self, 'model') and self.model is not None:
            self.model = None