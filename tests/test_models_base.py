import pytest
import pandas as pd
from abc import ABC
from unittest.mock import MagicMock

from src.models.base import ModelInterface


class TestModelInterface:
    """Test cases for ModelInterface abstract base class."""

    def test_model_interface_is_abstract(self):
        """Test that ModelInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelInterface()

    def test_abstract_methods_must_be_implemented(self):
        """Test that concrete classes must implement all abstract methods."""

        class IncompleteModel1(ModelInterface):
            # Missing fit method
            def predict(self, X: pd.DataFrame):
                return None

            def predict_proba(self, X: pd.DataFrame):
                return None

            def save(self, filepath: str) -> None:
                pass

            @classmethod
            def load(cls, filepath: str):
                return cls()

        with pytest.raises(TypeError):
            IncompleteModel1()

        class IncompleteModel2(ModelInterface):
            def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
                pass

            # Missing predict method
            def predict_proba(self, X: pd.DataFrame):
                return None

            def save(self, filepath: str) -> None:
                pass

            @classmethod
            def load(cls, filepath: str):
                return cls()

        with pytest.raises(TypeError):
            IncompleteModel2()

        class IncompleteModel3(ModelInterface):
            def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
                pass

            def predict(self, X: pd.DataFrame):
                return None

            # Missing predict_proba method
            def save(self, filepath: str) -> None:
                pass

            @classmethod
            def load(cls, filepath: str):
                return cls()

        with pytest.raises(TypeError):
            IncompleteModel3()

        class IncompleteModel4(ModelInterface):
            def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
                pass

            def predict(self, X: pd.DataFrame):
                return None

            def predict_proba(self, X: pd.DataFrame):
                return None

            # Missing save method
            @classmethod
            def load(cls, filepath: str):
                return cls()

        with pytest.raises(TypeError):
            IncompleteModel4()

        class IncompleteModel5(ModelInterface):
            def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
                pass

            def predict(self, X: pd.DataFrame):
                return None

            def predict_proba(self, X: pd.DataFrame):
                return None

            def save(self, filepath: str) -> None:
                pass

            # Missing load method

        with pytest.raises(TypeError):
            IncompleteModel5()

    def test_concrete_model_implementation(self):
        """Test that a properly implemented concrete model works."""

        class ConcreteModel(ModelInterface):
            def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
                self.is_fitted = True

            def predict(self, X: pd.DataFrame):
                return pd.Series([1] * len(X))

            def predict_proba(self, X: pd.DataFrame):
                return pd.DataFrame({'prob_0': [0.5] * len(X), 'prob_1': [0.5] * len(X)})

            def save(self, filepath: str) -> None:
                with open(filepath, 'w') as f:
                    f.write('model_data')

            @classmethod
            def load(cls, filepath: str):
                instance = cls()
                instance.loaded = True
                return instance

        # Should be able to instantiate
        model = ConcreteModel()
        assert isinstance(model, ModelInterface)

        # Test fit
        X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y_train = pd.Series([0, 1, 0])
        model.fit(X_train, y_train)
        assert model.is_fitted

        # Test predict
        X_test = pd.DataFrame({'feature1': [7, 8], 'feature2': [9, 10]})
        predictions = model.predict(X_test)
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 2

        # Test predict_proba
        probas = model.predict_proba(X_test)
        assert isinstance(probas, pd.DataFrame)
        assert len(probas) == 2

        # Test save and load
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            model.save(tmp.name)

            # Verify file was written
            with open(tmp.name, 'r') as f:
                content = f.read()
                assert content == 'model_data'

            # Test load
            loaded_model = ConcreteModel.load(tmp.name)
            assert loaded_model.loaded

        # Clean up
        os.unlink(tmp.name)

    def test_fit_with_kwargs(self):
        """Test that fit method accepts additional keyword arguments."""

        class ConcreteModel(ModelInterface):
            def __init__(self):
                self.fit_kwargs = None

            def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
                self.fit_kwargs = kwargs

            def predict(self, X: pd.DataFrame):
                return pd.Series([1] * len(X))

            def predict_proba(self, X: pd.DataFrame):
                return pd.DataFrame({'prob': [0.5] * len(X)})

            def save(self, filepath: str) -> None:
                pass

            @classmethod
            def load(cls, filepath: str):
                return cls()

        model = ConcreteModel()
        X_train = pd.DataFrame({'feature': [1, 2]})
        y_train = pd.Series([0, 1])

        # Test with various kwargs
        model.fit(X_train, y_train, learning_rate=0.01, epochs=100, verbose=True)
        assert model.fit_kwargs == {'learning_rate': 0.01, 'epochs': 100, 'verbose': True}

    def test_predict_return_types(self):
        """Test that predict methods return appropriate types."""

        class ConcreteModel(ModelInterface):
            def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
                pass

            def predict(self, X: pd.DataFrame):
                # Return numpy array
                import numpy as np
                return np.array([0, 1])

            def predict_proba(self, X: pd.DataFrame):
                # Return list of lists
                return [[0.7, 0.3], [0.4, 0.6]]

            def save(self, filepath: str) -> None:
                pass

            @classmethod
            def load(cls, filepath: str):
                return cls()

        model = ConcreteModel()
        X = pd.DataFrame({'feature': [1, 2]})

        # The interface doesn't enforce specific return types, just that they exist
        predictions = model.predict(X)
        assert predictions is not None

        probas = model.predict_proba(X)
        assert probas is not None

    def test_save_load_file_operations(self):
        """Test save and load with file operations."""

        class ConcreteModel(ModelInterface):
            def __init__(self):
                self.data = "model_state"

            def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
                pass

            def predict(self, X: pd.DataFrame):
                return pd.Series([1] * len(X))

            def predict_proba(self, X: pd.DataFrame):
                return pd.DataFrame({'prob': [0.5] * len(X)})

            def save(self, filepath: str) -> None:
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(self.data, f)

            @classmethod
            def load(cls, filepath: str):
                import pickle
                instance = cls()
                with open(filepath, 'rb') as f:
                    instance.data = pickle.load(f)
                return instance

        model = ConcreteModel()

        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Test save
            model.save(tmp.name)

            # Test load
            loaded_model = ConcreteModel.load(tmp.name)
            assert loaded_model.data == "model_state"

        # Clean up
        os.unlink(tmp.name)

    def test_empty_dataframe_handling(self):
        """Test model methods with empty DataFrames."""

        class ConcreteModel(ModelInterface):
            def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
                pass

            def predict(self, X: pd.DataFrame):
                return pd.Series(dtype=float)

            def predict_proba(self, X: pd.DataFrame):
                return pd.DataFrame()

            def save(self, filepath: str) -> None:
                pass

            @classmethod
            def load(cls, filepath: str):
                return cls()

        model = ConcreteModel()
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)

        # Should handle empty data gracefully
        model.fit(empty_X, empty_y)
        predictions = model.predict(empty_X)
        assert len(predictions) == 0

        probas = model.predict_proba(empty_X)
        assert probas.empty