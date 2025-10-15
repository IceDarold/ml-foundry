# src/models/sklearn_model.py

import os
import re
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn import ensemble, linear_model, svm

from .base import ModelInterface

# Whitelist of allowed sklearn model classes for security
ALLOWED_MODELS = {
    'sklearn.ensemble.ExtraTreesClassifier': ensemble.ExtraTreesClassifier,
    'sklearn.ensemble.RandomForestClassifier': ensemble.RandomForestClassifier,
    'sklearn.ensemble.RandomForestRegressor': ensemble.RandomForestRegressor,
    'sklearn.linear_model.LogisticRegression': linear_model.LogisticRegression,
    'sklearn.linear_model.Ridge': linear_model.Ridge,
    'sklearn.svm.SVC': svm.SVC,
}

# ==================================================================================
# SklearnModel
# ==================================================================================
class SklearnModel(ModelInterface):
    """
    Универсальный класс-обертка для моделей из библиотеки scikit-learn.
    """
    
    def __init__(self, model_class: str, params: Dict[str, Any]):
        """
        Инициализирует sklearn-совместимую модель.

        Args:
            model_class (str): Полный путь к классу модели в scikit-learn.
                                Например, 'sklearn.linear_model.LogisticRegression'.
            params (Dict[str, Any]): Словарь с параметрами для конструктора модели.
        """
        # Validate model_class string for security
        if not isinstance(model_class, str):
            raise ValueError("model_class must be a string")
        if not re.match(r'^[a-zA-Z_.]+$', model_class):
            raise ValueError("model_class contains invalid characters. Only letters, dots, and underscores are allowed.")
        if '..' in model_class or model_class.startswith('.') or model_class.endswith('.'):
            raise ValueError("model_class has invalid format")

        # Validate params for security (basic check)
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")
        for key, value in params.items():
            if not isinstance(key, str):
                raise ValueError("Parameter keys must be strings")
            # Prevent potentially dangerous values (basic check)
            if isinstance(value, str) and ('..' in value or value.startswith('/') or value.startswith('\\')):
                raise ValueError(f"Parameter '{key}' contains potentially dangerous path-like value")

        self.model_class_path = model_class
        self.params = params

        try:
            # Validate and get model class from whitelist
            if model_class not in ALLOWED_MODELS:
                raise ValueError(f"Model class '{model_class}' is not in the allowed list for security reasons")
            model_constructor = ALLOWED_MODELS[model_class]
            self.model: BaseEstimator = model_constructor(**self.params)
        except (KeyError, TypeError) as e:
            raise ImportError(f"Не удалось импортировать или создать класс модели: {model_class}") from e

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """Обучает модель. `kwargs` игнорируются для совместимости."""
        print(f"Обучение модели {self.model.__class__.__name__}...")
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> Any:
        """Возвращает предсказания классов или значения."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """Возвращает предсказания вероятностей или значения."""
        if hasattr(self.model, 'predict_proba'):
            # Для классификаторов возвращаем вероятности класса "1"
            return self.model.predict_proba(X)[:, 1]
        else:
            # Для регрессоров возвращаем просто предсказание
            return self.model.predict(X)

    def save(self, filepath: str) -> None:
        """Сохраняет модель с помощью joblib."""
        print(f"Сохранение модели в {filepath}")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'SklearnModel':
        """Загружает модель с помощью joblib."""
        print(f"Загрузка модели из {filepath}")
        return joblib.load(filepath)