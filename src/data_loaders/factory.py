from typing import Any, Dict

from .base import DataLoader
from .kaggle import KaggleDataLoader
from .local import LocalDataLoader


def create_data_loader(config: Dict[str, Any]) -> DataLoader:
    """Create appropriate data loader based on configuration.

    Args:
        config: Configuration dictionary containing data source settings

    Returns:
        DataLoader: Instance of appropriate data loader

    Raises:
        ValueError: If data source type is not supported or configuration is invalid
    """
    try:
        data_source = config.get('data_source', {})

        if not isinstance(data_source, dict):
            raise ValueError("data_source must be a dictionary")

        source_type = data_source.get('type', 'local')

        if source_type == 'kaggle':
            return KaggleDataLoader(data_source)
        elif source_type == 'local':
            return LocalDataLoader(data_source)
        else:
            raise ValueError(f"Unsupported data source type: {source_type}. Supported types: 'local', 'kaggle'")
    except Exception as e:
        raise ValueError(f"Failed to create data loader: {e}") from e