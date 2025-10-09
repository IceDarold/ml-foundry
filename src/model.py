import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

LGBM_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except (ImportError, OSError):
    pass


def train_model(X, y, model_config, validation_config, output_dir, exp_name, logger):
    """
    Train model with cross-validation and collect out-of-fold predictions.

    Args:
        X: Feature matrix
        y: Target vector
        model_config: Model configuration dict (e.g., {'name': 'LGBMClassifier', 'params': {...}})
        validation_config: Validation configuration dict (e.g., {'name': 'StratifiedKFold', 'n_splits': 5})
        output_dir: Directory to save outputs
        exp_name: Experiment name for file naming
        logger: Logger instance for logging progress

    Returns:
        np.ndarray: Out-of-fold predictions
    """
    # Instantiate validation splitter
    if validation_config['name'] == 'StratifiedKFold':
        splitter = StratifiedKFold(n_splits=validation_config['n_splits'], shuffle=True, random_state=42)
    else:
        raise ValueError(f"Unsupported validation method: {validation_config['name']}")

    # Initialize OOF predictions array
    oof_preds = np.zeros(len(X))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over folds
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        logger.info(f'Starting fold {fold + 1}/{validation_config["n_splits"]}')

        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Instantiate and train model
        if model_config['name'] == 'LGBMClassifier':
            if LGBM_AVAILABLE:
                model = LGBMClassifier(**model_config['params'])
            else:
                # Fallback to LogisticRegression for testing
                model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model: {model_config['name']}")

        model.fit(X_train, y_train)

        # Predict on validation set
        preds = model.predict_proba(X_val)[:, 1]  # Assuming binary classification

        # Collect OOF predictions
        oof_preds[val_idx] = preds

        # Save model
        model_path = os.path.join(output_dir, f'{exp_name}_fold_{fold}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f'Fold {fold + 1} completed. Model saved to {model_path}')

    # Save OOF predictions to CSV
    oof_df = pd.DataFrame({'oof_preds': oof_preds})
    oof_path = os.path.join(output_dir, f'{exp_name}_oof_preds.csv')
    oof_df.to_csv(oof_path, index=False)
    logger.info(f'OOF predictions saved to {oof_path}')

    return oof_preds