# src/utils.py

import logging
import os
import random
import time
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional, Callable

import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ==================================================================================
# Воспроизводимость (Reproducibility)
# ==================================================================================

def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across all random number generators.

    This function ensures deterministic behavior in experiments by seeding
    Python's random module, NumPy, and PyTorch (if available). It also configures
    PyTorch's CUDA settings for deterministic operations when possible.

    Args:
        seed (int): The seed value to use for all random number generators.
            Must be a non-negative integer.

    Raises:
        TypeError: If seed is not an integer.
        ValueError: If seed is negative.

    Example:
        >>> seed_everything(42)
        >>> np.random.rand()
        0.3745401188473625  # Always the same value

    Note:
        - PyTorch seeding is only applied if PyTorch is installed.
        - CUDA deterministic mode may slow down training but ensures reproducibility.
        - Some operations (especially with cuDNN) may still have slight variations.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Set seeds for PyTorch if used in the project
    # (e.g., for transformer embeddings)
    if TORCH_AVAILABLE:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # Some operations in cuDNN may be non-deterministic.
                # These flags attempt to fix this, but may slow down training.
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to set seeds for PyTorch: {e}")
    else:
        logger = get_logger(__name__)
        logger.warning("PyTorch not found. PyTorch seeds not set.")

    logger = get_logger(__name__)
    logger.info(f"All random sources fixed with seed: {seed}")

# ==================================================================================
# Вспомогательные функции (Helpers)
# ==================================================================================

def get_timestamp() -> str:
    """Generate a timestamp string suitable for file naming.

    Returns:
        str: Current timestamp in 'YYYYMMDD_HHMMSS' format.

    Example:
        >>> timestamp = get_timestamp()
        >>> print(timestamp)
        '20231015_143022'
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


# ==================================================================================
# Функции для работы с Hydra (Опционально, но полезно)
# ==================================================================================

def get_hydra_logging_directory() -> str:
    """Get the current Hydra logging directory for the running experiment.

    This function retrieves the output directory path created by Hydra for the
    current run, which is useful for accessing logs, checkpoints, and other
    outputs from anywhere in the codebase.

    Returns:
        str: Absolute path to the Hydra output directory for the current run.

    Raises:
        ImportError: If hydra-core package is not installed.
        ValueError: If called outside of a Hydra application context.

    Example:
        >>> output_dir = get_hydra_logging_directory()
        >>> print(output_dir)
        '/path/to/outputs/2023-10-15/14-30-22'
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        
        if HydraConfig.initialized():
            return HydraConfig.get().runtime.output_dir
        else:
            raise ValueError("Hydra configuration not initialized. Run the script as a Hydra application.")
            
    except ImportError:
        raise ImportError("hydra-core library is not installed.")

# ==================================================================================
# Logging Configuration
# ==================================================================================

def setup_logging(config: Dict[str, Any]) -> None:
    """Configure centralized logging based on the provided configuration.

    This function sets up Python's logging system with the specified level,
    format, and optional file output. It also suppresses verbose logging from
    common libraries when not in DEBUG mode.

    Args:
        config (Dict[str, Any]): Configuration dictionary with logging settings.
            Expected structure:
            {
                'logging': {
                    'level': str,  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
                    'format': str,  # Log format string (optional)
                    'file': str     # Log file path (optional)
                }
            }

    Raises:
        ValueError: If the logging level is invalid.

    Example:
        >>> config = {'logging': {'level': 'INFO', 'file': 'experiment.log'}}
        >>> setup_logging(config)
    """
    logging_config = config.get('logging', {})
    level = getattr(logging, logging_config.get('level', 'INFO').upper(), logging.INFO)
    log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = logging_config.get('file')

    # Configure root logger
    if log_file:
        logging.basicConfig(
            level=level,
            format=log_format,
            filename=log_file,
            filemode='a',
            force=True  # Override any existing configuration
        )
    else:
        logging.basicConfig(
            level=level,
            format=log_format,
            force=True  # Override any existing configuration
        )

    # Suppress verbose logging from external libraries if not in DEBUG mode
    if level > logging.DEBUG:
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        # Add more libraries as needed

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance with the specified name.

    This function returns a logger that inherits the configuration set by
    setup_logging(). Use __name__ as the name for module-specific logging.

    Args:
        name (str): Logger name, typically __name__ for hierarchical logging.
            Use descriptive names for different components.

    Returns:
        logging.Logger: Configured logger instance ready for use.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)

# ==================================================================================
# Performance Monitoring
# ==================================================================================

def time_execution(func: Callable) -> Callable:
    """Decorator to measure and log execution time of a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapped function with timing.

    Example:
        >>> @time_execution
        ... def my_function():
        ...     time.sleep(1)
        >>> my_function()  # Logs: "my_function executed in 1.0000 seconds"
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = get_logger(func.__module__)
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
    return wrapper

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.

    Returns:
        Dict[str, float]: Dictionary containing memory usage information:
            - 'rss': Resident Set Size in MB
            - 'vms': Virtual Memory Size in MB
            - 'percent': Memory usage percentage

    Note:
        Requires psutil package. If not available, returns empty dict.
    """
    if not PSUTIL_AVAILABLE:
        return {}

    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()

    return {
        'rss': memory_info.rss / (1024 * 1024),  # Convert to MB
        'vms': memory_info.vms / (1024 * 1024),  # Convert to MB
        'percent': memory_percent
    }

def log_memory_usage(logger: logging.Logger, prefix: str = "Memory usage") -> None:
    """Log current memory usage.

    Args:
        logger (logging.Logger): Logger instance to use for logging.
        prefix (str): Prefix for the log message.
    """
    memory_stats = get_memory_usage()
    if memory_stats:
        logger.info(f"{prefix}: RSS={memory_stats['rss']:.2f}MB, "
                   f"VMS={memory_stats['vms']:.2f}MB, "
                   f"Percent={memory_stats['percent']:.2f}%")
    else:
        logger.warning("Memory monitoring not available (psutil not installed)")

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor both execution time and memory usage of a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapped function with performance monitoring.

    Example:
        >>> @performance_monitor
        ... def my_function():
        ...     # Some computation
        ...     return result
        >>> my_function()  # Logs timing and memory usage
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)

        # Log memory before execution
        log_memory_usage(logger, f"Memory before {func.__name__}")

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time

            # Log memory after execution
            log_memory_usage(logger, f"Memory after {func.__name__}")

            logger.info(f"{func.__name__} performance: {execution_time:.4f}s execution time")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
            raise
    return wrapper

# ==================================================================================
# Input Validation Decorators
# ==================================================================================

def validate_type(*types):
    """Decorator to validate parameter types.

    Args:
        *types: Expected types for each parameter.

    Raises:
        TypeError: If parameter type does not match expected type.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i, (arg, expected_type) in enumerate(zip(args[1:], types), 1):  # Skip 'self' if method
                if not isinstance(arg, expected_type):
                    raise TypeError(f"Argument {i} of {func.__name__} must be of type {expected_type.__name__}, got {type(arg).__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_range(min_val=None, max_val=None):
    """Decorator to validate numeric parameter ranges.

    Args:
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Raises:
        ValueError: If parameter is out of range.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for arg in args[1:]:  # Skip 'self' if method
                if not isinstance(arg, (int, float)):
                    continue
                if min_val is not None and arg < min_val:
                    raise ValueError(f"Argument in {func.__name__} must be >= {min_val}, got {arg}")
                if max_val is not None and arg > max_val:
                    raise ValueError(f"Argument in {func.__name__} must be <= {max_val}, got {arg}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_non_empty():
    """Decorator to validate that parameters are not empty.

    Checks for empty strings, lists, dicts, etc.

    Raises:
        ValueError: If parameter is empty.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i, arg in enumerate(args[1:], 1):  # Skip 'self' if method
                if hasattr(arg, '__len__') and len(arg) == 0:
                    raise ValueError(f"Argument {i} of {func.__name__} must not be empty")
            return func(*args, **kwargs)
        return wrapper
    return decorator