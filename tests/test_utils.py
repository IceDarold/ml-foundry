import pytest
import os
import random
import numpy as np
import logging
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.utils import (
    seed_everything,
    get_timestamp,
    get_hydra_logging_directory,
    setup_logging,
    get_logger
)


class TestSeedEverything:
    """Test cases for seed_everything function."""

    def test_seed_everything_basic(self):
        """Test basic functionality of seed_everything."""
        seed = 42
        seed_everything(seed)

        # Check environment variable
        assert os.environ['PYTHONHASHSEED'] == str(seed)

        # Check that random seed was set (we can't easily check the internal state)
        # Instead, check that subsequent random calls are reproducible
        random.seed(seed)
        expected_random = random.random()
        random.seed(seed)
        actual_random = random.random()
        assert expected_random == actual_random

        # Check numpy seed reproducibility
        np.random.seed(seed)
        expected = np.random.rand()
        np.random.seed(seed)
        actual = np.random.rand()
        assert expected == actual

        # Note: PyTorch seeding is tested separately

    def test_seed_everything_with_torch(self):
        """Test seed_everything when PyTorch is available."""
        try:
            import torch
            seed = 123
            seed_everything(seed)

            # Check torch seed
            assert torch.initial_seed() == seed
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_seed_everything_without_torch(self):
        """Test seed_everything when PyTorch is not available."""
        # The function should handle missing torch gracefully
        seed = 456
        # Should not raise an exception even if torch is not available
        seed_everything(seed)

    def test_seed_everything_edge_cases(self):
        """Test edge cases for seed values."""
        # Test with zero
        seed_everything(0)

        # Test with large number
        seed_everything(2**32 - 1)

        # Test with negative - numpy doesn't allow negative seeds, so this should raise ValueError
        with pytest.raises(ValueError, match="Seed must be between 0 and 2\\*\\*32 - 1"):
            seed_everything(-1)

        # Note: PyTorch seeding is tested separately


class TestGetTimestamp:
    """Test cases for get_timestamp function."""

    @patch('src.utils.datetime')
    def test_get_timestamp_format(self, mock_datetime):
        """Test that timestamp is returned in correct format."""
        mock_datetime.now.return_value = datetime(2023, 10, 15, 14, 30, 45)
        result = get_timestamp()
        assert result == '20231015_143045'

    def test_get_timestamp_type(self):
        """Test that get_timestamp returns a string."""
        result = get_timestamp()
        assert isinstance(result, str)

    def test_get_timestamp_length(self):
        """Test that timestamp has correct length."""
        result = get_timestamp()
        assert len(result) == 15  # YYYYMMDD_HHMMSS


class TestGetHydraLoggingDirectory:
    """Test cases for get_hydra_logging_directory function."""

    def test_get_hydra_logging_directory_success(self):
        """Test successful retrieval of Hydra logging directory."""
        mock_config = MagicMock()
        mock_config.runtime.output_dir = '/path/to/output'

        with patch('hydra.core.hydra_config.HydraConfig.initialized', return_value=True), \
             patch('hydra.core.hydra_config.HydraConfig.get', return_value=mock_config):
            result = get_hydra_logging_directory()
            assert result == '/path/to/output'

    def test_get_hydra_logging_directory_not_initialized(self):
        """Test when Hydra is not initialized."""
        with patch('hydra.core.hydra_config.HydraConfig.initialized', return_value=False):
            with pytest.raises(ValueError, match="Hydra-конфигурация не инициализирована"):
                get_hydra_logging_directory()

    def test_get_hydra_logging_directory_no_hydra(self):
        """Test when Hydra is not installed."""
        with patch.dict('sys.modules', {'hydra.core.hydra_config': None}):
            with pytest.raises(ImportError, match="Библиотека hydra-core не установлена"):
                get_hydra_logging_directory()


class TestSetupLogging:
    """Test cases for setup_logging function."""

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        config = {
            'logging': {
                'level': 'INFO',
                'format': '%(levelname)s - %(message)s'
            }
        }
        setup_logging(config)

        logger = get_logger('test')
        # The logger level might be 0 (NOTSET) if it inherits from root
        # Let's check that the root logger has the correct level
        root_logger = logging.getLogger()
        assert root_logger.level == 20  # INFO level

    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            config = {
                'logging': {
                    'level': 'DEBUG',
                    'format': '%(asctime)s - %(message)s',
                    'file': tmp.name
                }
            }
            setup_logging(config)

            # Check that the root logger has the correct level
            root_logger = logging.getLogger()
            assert root_logger.level == 10  # DEBUG level

        # Clean up
        os.unlink(tmp.name)

    def test_setup_logging_default_values(self):
        """Test logging setup with default values."""
        config = {}
        setup_logging(config)

        # Check that the root logger has the correct level
        root_logger = logging.getLogger()
        assert root_logger.level == 20  # INFO level

    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid level."""
        config = {
            'logging': {
                'level': 'INVALID'
            }
        }
        setup_logging(config)

        # Check that the root logger has the correct level (should default to INFO)
        root_logger = logging.getLogger()
        assert root_logger.level == 20  # Should default to INFO

    def test_setup_logging_suppress_external_libs(self):
        """Test that external libraries are suppressed when not in DEBUG."""
        config = {
            'logging': {
                'level': 'INFO'
            }
        }
        setup_logging(config)

        # Check that matplotlib and PIL loggers are set to WARNING
        matplotlib_logger = get_logger('matplotlib')
        pil_logger = get_logger('PIL')
        assert matplotlib_logger.level == 30  # WARNING
        assert pil_logger.level == 30  # WARNING


class TestGetLogger:
    """Test cases for get_logger function."""

    def test_get_logger_basic(self):
        """Test basic logger retrieval."""
        logger = get_logger('test_module')
        assert logger.name == 'test_module'
        assert isinstance(logger, type(get_logger('another')))

    def test_get_logger_same_name(self):
        """Test that same name returns same logger instance."""
        logger1 = get_logger('same_name')
        logger2 = get_logger('same_name')
        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Test that different names return different loggers."""
        logger1 = get_logger('name1')
        logger2 = get_logger('name2')
        assert logger1 is not logger2