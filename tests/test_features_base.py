import pytest
import pandas as pd
from abc import ABC
from unittest.mock import MagicMock

from src.features.base import FeatureGenerator, FitStrategy


class TestFeatureGenerator:
    """Test cases for FeatureGenerator abstract base class."""

    def test_feature_generator_is_abstract(self):
        """Test that FeatureGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FeatureGenerator("test")

    def test_feature_generator_init(self):
        """Test FeatureGenerator initialization."""

        class ConcreteFeatureGenerator(FeatureGenerator):
            def fit(self, data: pd.DataFrame) -> None:
                pass

            def transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return data

        generator = ConcreteFeatureGenerator("test_name")
        assert generator.name == "test_name"
        assert generator.fit_strategy == "train_only"

    def test_feature_generator_custom_fit_strategy(self):
        """Test FeatureGenerator with custom fit strategy."""

        class ConcreteFeatureGenerator(FeatureGenerator):
            fit_strategy: FitStrategy = "combined"

            def fit(self, data: pd.DataFrame) -> None:
                pass

            def transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return data

        generator = ConcreteFeatureGenerator("test_name")
        assert generator.fit_strategy == "combined"

    def test_fit_transform_method(self):
        """Test the fit_transform convenience method."""
        mock_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

        class ConcreteFeatureGenerator(FeatureGenerator):
            def fit(self, data: pd.DataFrame) -> None:
                self.fitted_ = True

            def transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return data * 2

        generator = ConcreteFeatureGenerator("test")
        result = generator.fit_transform(mock_data)

        # Check that fit was called
        assert hasattr(generator, 'fitted_')
        assert generator.fitted_

        # Check that transform was called and returned correct result
        expected = pd.DataFrame({'col1': [2, 4, 6], 'col2': [8, 10, 12]})
        pd.testing.assert_frame_equal(result, expected)

    def test_fit_transform_with_empty_dataframe(self):
        """Test fit_transform with empty DataFrame."""
        empty_data = pd.DataFrame()

        class ConcreteFeatureGenerator(FeatureGenerator):
            def fit(self, data: pd.DataFrame) -> None:
                pass

            def transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return data

        generator = ConcreteFeatureGenerator("test")
        result = generator.fit_transform(empty_data)
        assert result.empty

    def test_fit_transform_with_single_row(self):
        """Test fit_transform with single row DataFrame."""
        single_row_data = pd.DataFrame({'col1': [1], 'col2': [2]})

        class ConcreteFeatureGenerator(FeatureGenerator):
            def fit(self, data: pd.DataFrame) -> None:
                pass

            def transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return data + 10

        generator = ConcreteFeatureGenerator("test")
        result = generator.fit_transform(single_row_data)
        expected = pd.DataFrame({'col1': [11], 'col2': [12]})
        pd.testing.assert_frame_equal(result, expected)

    def test_fit_transform_preserves_index(self):
        """Test that fit_transform preserves DataFrame index."""
        data = pd.DataFrame({'col1': [1, 2]}, index=['a', 'b'])

        class ConcreteFeatureGenerator(FeatureGenerator):
            def fit(self, data: pd.DataFrame) -> None:
                pass

            def transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return data

        generator = ConcreteFeatureGenerator("test")
        result = generator.fit_transform(data)
        pd.testing.assert_index_equal(result.index, data.index)

    def test_fit_transform_preserves_columns(self):
        """Test that fit_transform preserves DataFrame columns."""
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        class ConcreteFeatureGenerator(FeatureGenerator):
            def fit(self, data: pd.DataFrame) -> None:
                pass

            def transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return data

        generator = ConcreteFeatureGenerator("test")
        result = generator.fit_transform(data)
        assert list(result.columns) == list(data.columns)

    def test_abstract_methods_must_be_implemented(self):
        """Test that concrete classes must implement abstract methods."""

        class IncompleteFeatureGenerator(FeatureGenerator):
            # Missing fit method
            def transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return data

        with pytest.raises(TypeError):
            IncompleteFeatureGenerator("test")

        class AnotherIncompleteFeatureGenerator(FeatureGenerator):
            def fit(self, data: pd.DataFrame) -> None:
                pass
            # Missing transform method

        with pytest.raises(TypeError):
            AnotherIncompleteFeatureGenerator("test")

    def test_fit_strategy_literal_values(self):
        """Test that fit_strategy accepts only valid literal values."""
        # This is more of a type hint test, but we can verify the default
        class ConcreteFeatureGenerator(FeatureGenerator):
            def fit(self, data: pd.DataFrame) -> None:
                pass

            def transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return data

        generator = ConcreteFeatureGenerator("test")
        # Should be able to set valid values
        generator.fit_strategy = "train_only"
        generator.fit_strategy = "combined"

        # But this is runtime, so we can't enforce literals strictly
        # The type hints will catch this at static analysis time