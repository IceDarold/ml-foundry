import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from .base import DataLoader

# Import kaggle only when needed to avoid authentication errors at module level
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kaggle.rest import ApiException
    KAGGLE_AVAILABLE = True
except (ImportError, OSError):
    KAGGLE_AVAILABLE = False
    KaggleApi = None
    ApiException = None

logger = logging.getLogger(__name__)


class KaggleDataLoader(DataLoader):
    """Data loader for Kaggle datasets and competitions."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api = None
        if not KAGGLE_AVAILABLE:
            logger.warning("Kaggle package not available, data loading will be skipped")
            return

        try:
            self.api = KaggleApi()
            self.api.authenticate()
            logger.info("Successfully authenticated with Kaggle API")
        except Exception as e:
            logger.warning(f"Failed to authenticate with Kaggle API: {e}")
            logger.warning("Kaggle data loading will be skipped. Ensure you have a valid kaggle.json file in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
            # Don't raise exception - allow fallback to other loaders

        self.dataset_name = config.get('dataset_name')
        self.competition_name = config.get('competition_name')
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.cache_dir = self.data_dir / 'cache' / 'kaggle'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.dataset_name and not self.competition_name:
            raise ValueError("Either 'dataset_name' or 'competition_name' must be specified in config")

    def ensure_data_available(self) -> None:
        """Download data if not already cached."""
        if self.api is None:
            raise RuntimeError("Kaggle API not available - authentication failed during initialization")
        if self.dataset_name:
            self._download_dataset()
        elif self.competition_name:
            self._download_competition_data()

    def _download_dataset(self) -> None:
        """Download Kaggle dataset."""
        cache_path = self.cache_dir / self.dataset_name.replace('/', '_')
        if cache_path.exists():
            logger.info(f"Dataset {self.dataset_name} already cached at {cache_path}")
            return

        try:
            logger.info(f"Downloading dataset {self.dataset_name}")
            self.api.dataset_download_files(self.dataset_name, path=str(cache_path), unzip=True)
            logger.info(f"Dataset downloaded to {cache_path}")
        except ApiException as e:
            if e.status == 404:
                logger.error(f"Dataset {self.dataset_name} not found. Please check the dataset name and your access permissions.")
            elif e.status == 403:
                logger.error(f"Access denied for dataset {self.dataset_name}. You may need to accept competition rules or dataset terms.")
            else:
                logger.error(f"Kaggle API error while downloading dataset {self.dataset_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to download dataset {self.dataset_name}: {e}")
            raise

    def _download_competition_data(self) -> None:
        """Download Kaggle competition data."""
        cache_path = self.cache_dir / self.competition_name
        if cache_path.exists():
            logger.info(f"Competition data {self.competition_name} already cached at {cache_path}")
            return

        try:
            logger.info(f"Downloading competition data {self.competition_name}")
            self.api.competition_download_files(self.competition_name, path=str(cache_path))
            logger.info(f"Competition data downloaded to {cache_path}")
        except ApiException as e:
            if e.status == 404:
                logger.error(f"Competition {self.competition_name} not found. Please check the competition name.")
            elif e.status == 403:
                logger.error(f"Access denied for competition {self.competition_name}. You may need to join the competition first.")
            else:
                logger.error(f"Kaggle API error while downloading competition data {self.competition_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to download competition data {self.competition_name}: {e}")
            raise

    def load_train_data(self) -> pd.DataFrame:
        """Load training data."""
        self.ensure_data_available()
        train_path = self._find_file('train')
        if not train_path:
            raise FileNotFoundError("Training data file not found")
        try:
            return pd.read_csv(train_path)
        except Exception as e:
            logger.error(f"Failed to read training data from {train_path}: {e}")
            raise

    def load_test_data(self) -> pd.DataFrame:
        """Load test data."""
        self.ensure_data_available()
        test_path = self._find_file('test')
        if not test_path:
            raise FileNotFoundError("Test data file not found")
        try:
            return pd.read_csv(test_path)
        except Exception as e:
            logger.error(f"Failed to read test data from {test_path}: {e}")
            raise

    def _find_file(self, file_type: str) -> Optional[Path]:
        """Find file of specified type in cache directory."""
        base_dir = self.cache_dir / (self.dataset_name.replace('/', '_') if self.dataset_name else self.competition_name)

        # Common patterns for train/test files
        patterns = {
            'train': ['train.csv', 'training.csv', 'train_data.csv'],
            'test': ['test.csv', 'testing.csv', 'test_data.csv']
        }

        for pattern in patterns[file_type]:
            file_path = base_dir / pattern
            if file_path.exists():
                return file_path

        # Fallback: search recursively
        for file_path in base_dir.rglob('*.csv'):
            if file_type in file_path.name.lower():
                return file_path

        return None