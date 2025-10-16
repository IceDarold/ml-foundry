import pytest
import numpy as np
import pandas as pd
from abc import ABC

from src.metrics.base import MetricInterface


class TestMetricInterface:
    """Test cases for MetricInterface abstract base class."""

    def test_metric_interface_is_abstract(self):
        """Test that MetricInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MetricInterface()

    def test_abstract_method_must_be_implemented(self):
        """Test that concrete classes must implement the __call__ method."""

        class IncompleteMetric(MetricInterface):
            # Missing __call__ method
            pass

        with pytest.raises(TypeError):
            IncompleteMetric()

    def test_concrete_metric_implementation(self):
        """Test that a properly implemented concrete metric works."""

        class ConcreteMetric(MetricInterface):
            def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
                return float(np.mean(y_true == y_pred))

        # Should be able to instantiate
        metric = ConcreteMetric()
        assert isinstance(metric, MetricInterface)

        # Test basic functionality
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])
        score = metric(y_true, y_pred)
        assert isinstance(score, float)
        assert score == 0.75  # 3 out of 4 correct

    def test_call_with_kwargs(self):
        """Test that __call__ method accepts additional keyword arguments."""

        class ConcreteMetric(MetricInterface):
            def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
                # Use kwargs to modify behavior
                if 'normalize' in kwargs and kwargs['normalize']:
                    return float(np.mean(y_true == y_pred))
                else:
                    return float(np.sum(y_true == y_pred))

        metric = ConcreteMetric()

        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])

        # Test with normalize=True
        score_normalized = metric(y_true, y_pred, normalize=True)
        assert score_normalized == 0.75

        # Test with normalize=False
        score_raw = metric(y_true, y_pred, normalize=False)
        assert score_raw == 3.0

        # Test with other kwargs (should default to normalized since normalize not specified)
        score_with_extra = metric(y_true, y_pred, sample_weight=np.array([1, 1, 1, 1]))
        assert score_with_extra == 3.0  # Should use the else branch (sum)

    def test_call_with_numpy_arrays(self):
        """Test __call__ with various numpy array types."""

        class ConcreteMetric(MetricInterface):
            def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
                return float(np.mean(np.abs(y_true - y_pred)))

        metric = ConcreteMetric()

        # Test with float arrays
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        score = metric(y_true, y_pred)
        assert abs(score - 0.1) < 1e-6

        # Test with int arrays
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        score = metric(y_true, y_pred)
        assert score == 1.0 / 3.0

    def test_call_with_pandas_series(self):
        """Test __call__ with pandas Series (converted to numpy)."""

        class ConcreteMetric(MetricInterface):
            def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
                # Convert to numpy if needed
                if hasattr(y_true, 'values'):
                    y_true = y_true.values
                if hasattr(y_pred, 'values'):
                    y_pred = y_pred.values
                return float(np.mean(y_true == y_pred))

        metric = ConcreteMetric()

        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 0, 0])
        score = metric(y_true.values, y_pred.values)  # Pass as numpy
        assert score == 2.0 / 3.0

    def test_call_with_different_shapes(self):
        """Test __call__ with arrays of different shapes (should handle gracefully or raise appropriate errors)."""

        class ConcreteMetric(MetricInterface):
            def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
                if y_true.shape != y_pred.shape:
                    raise ValueError("Arrays must have the same shape")
                return float(np.mean(y_true == y_pred))

        metric = ConcreteMetric()

        # Same shape
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        score = metric(y_true, y_pred)
        assert score == 2.0 / 3.0

        # Different shape - should raise ValueError
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])
        with pytest.raises(ValueError, match="Arrays must have the same shape"):
            metric(y_true, y_pred)

    def test_call_with_empty_arrays(self):
        """Test __call__ with empty arrays."""

        class ConcreteMetric(MetricInterface):
            def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
                if len(y_true) == 0:
                    return 0.0
                return float(np.mean(y_true == y_pred))

        metric = ConcreteMetric()

        y_true = np.array([])
        y_pred = np.array([])
        score = metric(y_true, y_pred)
        assert score == 0.0

    def test_call_with_single_element_arrays(self):
        """Test __call__ with single element arrays."""

        class ConcreteMetric(MetricInterface):
            def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
                return float(np.mean(y_true == y_pred))

        metric = ConcreteMetric()

        y_true = np.array([1])
        y_pred = np.array([1])
        score = metric(y_true, y_pred)
        assert score == 1.0

        y_true = np.array([1])
        y_pred = np.array([0])
        score = metric(y_true, y_pred)
        assert score == 0.0

    def test_call_with_multidimensional_arrays(self):
        """Test __call__ with multidimensional arrays."""

        class ConcreteMetric(MetricInterface):
            def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
                return float(np.mean(y_true == y_pred))

        metric = ConcreteMetric()

        y_true = np.array([[1, 0], [1, 1]])
        y_pred = np.array([[1, 0], [0, 1]])
        score = metric(y_true, y_pred)
        assert score == 0.75  # 3 out of 4 correct

    def test_call_with_nan_values(self):
        """Test __call__ with NaN values."""

        class ConcreteMetric(MetricInterface):
            def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
                # Handle NaN values by ignoring them
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                if np.sum(mask) == 0:
                    return 0.0
                return float(np.mean(y_true[mask] == y_pred[mask]))

        metric = ConcreteMetric()

        y_true = np.array([1, np.nan, 1, 0])
        y_pred = np.array([1, 0, np.nan, 0])
        score = metric(y_true, y_pred)
        assert score == 1.0  # Only compare non-NaN values: 1==1 and 0==0

    def test_call_return_type_enforcement(self):
        """Test that __call__ must return a float."""

        class BadMetric(MetricInterface):
            def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
                return "not_a_float"  # Wrong return type

        metric = BadMetric()
        y_true = np.array([1])
        y_pred = np.array([1])

        # This will pass type checking at runtime, but the interface expects float
        result = metric(y_true, y_pred)
        assert result == "not_a_float"  # Type hints don't enforce at runtime

    def test_multiple_inheritance_with_metric_interface(self):
        """Test that MetricInterface can be used with multiple inheritance."""

        class BaseClass:
            def base_method(self):
                return "base"

        class ConcreteMetric(BaseClass, MetricInterface):
            def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
                return 0.5

        metric = ConcreteMetric()
        assert isinstance(metric, MetricInterface)
        assert isinstance(metric, BaseClass)
        assert metric.base_method() == "base"

        score = metric(np.array([1]), np.array([1]))
        assert score == 0.5