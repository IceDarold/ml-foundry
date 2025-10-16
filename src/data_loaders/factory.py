from typing import Any, Dict
import logging

from .base import DataLoader
from .kaggle import KaggleDataLoader
from .local import LocalDataLoader

logger = logging.getLogger(__name__)


def create_data_loader(config: Dict[str, Any]) -> DataLoader:
    """Create appropriate data loader based on configuration.

    Attempts to create loaders in order: Kaggle -> Local
    Raises an error if no data sources are available.

    Args:
        config: Configuration dictionary containing data source settings

    Returns:
        DataLoader: Instance of appropriate data loader

    Raises:
        ValueError: If no data loader options are available
    """
    data_source = config.get('data_source', {})

    if not isinstance(data_source, dict):
        raise ValueError("data_source must be a dictionary")

    source_type = data_source.get('type', 'local')

    # Try requested source type first
    if source_type == 'kaggle':
        try:
            loader = KaggleDataLoader(data_source)
            # Test if Kaggle API is available
            if loader.api is not None:
                logger.info("Using Kaggle data loader")
                return loader
            else:
                raise ValueError("Kaggle API not available")
        except Exception as e:
            logger.warning(f"Kaggle data loader failed: {e}")

    # Try local data
    if source_type in ['local', 'kaggle']:  # Allow fallback from kaggle to local
        try:
            loader = LocalDataLoader(data_source)
            loader.ensure_data_available()  # Test if local files exist
            logger.info("Using local data loader")
            return loader
        except Exception as e:
            logger.warning(f"Local data loader failed: {e}")

    # No data sources available
    raise ValueError(
        "No data sources available. Please ensure data is downloaded locally or Kaggle API is configured. "
        "For local data, place files in the expected directory. For Kaggle data, set up API credentials."
    )