import logging
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from .base import DataLoader

logger = logging.getLogger(__name__)


class LocalDataLoader(DataLoader):
    """Data loader for local files."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.train_path = Path(config.get('train_path', 'data/train.csv'))
        self.test_path = Path(config.get('test_path', 'data/test.csv'))

    def ensure_data_available(self) -> None:
        """Check if local data files exist."""
        if not self.train_path.exists():
            raise FileNotFoundError(f"Training data file not found: {self.train_path}")
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test data file not found: {self.test_path}")
        logger.info("Local data files verified")

    def load_train_data(self) -> pd.DataFrame:
        """Load training data from local file."""
        try:
            return pd.read_csv(self.train_path)
        except Exception as e:
            logger.error(f"Failed to read training data from {self.train_path}: {e}")
            raise

    def load_test_data(self) -> pd.DataFrame:
        """Load test data from local file."""
        try:
            return pd.read_csv(self.test_path)
        except Exception as e:
            logger.error(f"Failed to read test data from {self.test_path}: {e}")
            raise