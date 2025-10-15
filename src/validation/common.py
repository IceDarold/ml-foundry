import re
from .base import BaseSplitter
from sklearn.model_selection._split import BaseCrossValidator
from sklearn import model_selection

# Whitelist of allowed sklearn splitter classes for security
ALLOWED_SPLITTERS = {
    'sklearn.model_selection.StratifiedKFold': model_selection.StratifiedKFold,
    # Add more allowed splitters as needed
}

class SklearnSplitterWrapper(BaseSplitter):
    def __init__(self, splitter_class: str, **kwargs):
        # Validate splitter_class string for security
        if not isinstance(splitter_class, str):
            raise ValueError("splitter_class must be a string")
        if not re.match(r'^[a-zA-Z_.]+$', splitter_class):
            raise ValueError("splitter_class contains invalid characters. Only letters, dots, and underscores are allowed.")
        if '..' in splitter_class or splitter_class.startswith('.') or splitter_class.endswith('.'):
            raise ValueError("splitter_class has invalid format")

        # Validate kwargs for security (basic check)
        for key, value in kwargs.items():
            if not isinstance(key, str):
                raise ValueError("Parameter keys must be strings")
            # Prevent potentially dangerous values (basic check)
            if isinstance(value, str) and ('..' in value or value.startswith('/') or value.startswith('\\')):
                raise ValueError(f"Parameter '{key}' contains potentially dangerous path-like value")

        # Validate and get splitter class from whitelist
        if splitter_class not in ALLOWED_SPLITTERS:
            raise ValueError(f"Splitter class '{splitter_class}' is not in the allowed list for security reasons")
        splitter_constructor = ALLOWED_SPLITTERS[splitter_class]

        self.splitter: BaseCrossValidator = splitter_constructor(**kwargs)

    def split(self, data, y, groups=None):
        return self.splitter.split(data, y, groups)

    def get_n_splits(self, data=None, y=None, groups=None):
        return self.splitter.get_n_splits(data, y, groups)