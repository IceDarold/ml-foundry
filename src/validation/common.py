import re
from dataclasses import dataclass, field
from .base import BaseSplitter
from sklearn.model_selection._split import BaseCrossValidator
from sklearn import model_selection

# Whitelist of allowed sklearn splitter classes for security
ALLOWED_SPLITTERS = {
    'sklearn.model_selection.StratifiedKFold': model_selection.StratifiedKFold,
    # Add more allowed splitters as needed
}

@dataclass
class SklearnSplitterWrapper(BaseSplitter):
    splitter_class: str
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    splitter: BaseCrossValidator = field(init=False)

    def __post_init__(self):
        # Validate splitter_class string for security
        if not isinstance(self.splitter_class, str):
            raise ValueError("splitter_class must be a string")
        if not re.match(r'^[a-zA-Z_.]+$', self.splitter_class):
            raise ValueError("splitter_class contains invalid characters. Only letters, dots, and underscores are allowed.")
        if '..' in self.splitter_class or self.splitter_class.startswith('.') or self.splitter_class.endswith('.'):
            raise ValueError("splitter_class has invalid format")

        # Validate kwargs for security (basic check) - but since we don't have kwargs in dataclass, we'll skip this for now
        # In a real implementation, you might want to add a kwargs field

        # Validate and get splitter class from whitelist
        if self.splitter_class not in ALLOWED_SPLITTERS:
            raise ValueError(f"Splitter class '{self.splitter_class}' is not in the allowed list for security reasons")
        splitter_constructor = ALLOWED_SPLITTERS[self.splitter_class]

        try:
            self.splitter = splitter_constructor(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        except TypeError:
            # Fallback for splitters that don't accept the full signature
            self.splitter = splitter_constructor()

    def split(self, data, y, groups=None):
        return self.splitter.split(data, y, groups)

    def get_n_splits(self, data=None, y=None, groups=None):
        return self.splitter.get_n_splits(data, y, groups)
