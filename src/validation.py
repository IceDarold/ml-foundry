from sklearn.model_selection import StratifiedKFold


def get_splitter(validation_config):
    """
    Instantiate a cross-validation splitter based on the provided configuration.

    Args:
        validation_config (dict): Configuration dictionary containing 'name' and other parameters.

    Returns:
        splitter: The instantiated cross-validation splitter.

    Raises:
        ValueError: If the splitter name is not supported.
    """
    name = validation_config.get('name')
    if name == 'StratifiedKFold':
        n_splits = validation_config.get('n_splits', 5)
        return StratifiedKFold(n_splits=n_splits)
    else:
        raise ValueError(f"Unsupported splitter: {name}")