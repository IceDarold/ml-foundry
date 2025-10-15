from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import pandas as pd


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def load_train_data(self) -> pd.DataFrame:
        """Load training data.

        Returns:
            pd.DataFrame: Training data
        """
        pass

    @abstractmethod
    def load_test_data(self) -> pd.DataFrame:
        """Load test data.

        Returns:
            pd.DataFrame: Test data
        """
        pass

    @abstractmethod
    def ensure_data_available(self) -> None:
        """Ensure that data is available locally, downloading if necessary."""
        pass