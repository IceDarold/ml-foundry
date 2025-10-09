from abc import ABC, abstractmethod
import pandas as pd


class Feature(ABC):
    """Abstract base class for feature engineering."""

    @abstractmethod
    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from the input dataframe.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with new features added
        """
        pass


class IdentityFeature(Feature):
    """Example feature that returns the input dataframe unchanged."""

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


def create_feature(name: str) -> Feature:
    """Factory function to instantiate feature classes by name.

    Args:
        name: Name of the feature class

    Returns:
        Instance of the feature class

    Raises:
        ValueError: If the feature name is not recognized
    """
    features = {
        "IdentityFeature": IdentityFeature,
    }

    if name not in features:
        raise ValueError(f"Unknown feature: {name}")

    return features[name]()