import argparse
import os
import pickle
import numpy as np
import pandas as pd
from src.config import load_config
from src.utils import seed_everything, setup_logger
from src.data_loader import load_data
from src.feature_engineering import create_feature
from src.model import train_model


def main():
    parser = argparse.ArgumentParser(description="Train ML model for ml-foundry project")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup seed
    seed_everything(config.general.seed)

    # Setup logger
    log_path = os.path.join(config.output_dir, f"{config.exp_name}.log")
    logger = setup_logger(log_path)

    logger.info("Starting training pipeline")

    # Load data
    X, y, X_test = load_data(config.data.train_path, config.data.test_path, config.data.target_col)

    # Apply feature engineering
    if config.features:
        for feature_name in config.features:
            feature = create_feature(feature_name)
            X = feature.create(X)
            X_test = feature.create(X_test)

    # Train model
    oof_preds = train_model(X, y, config.model, config.validation, config.output_dir, config.exp_name, logger)

    # Save submission predictions if test data is available
    if config.data.test_path:
        test_preds = np.zeros(len(X_test))
        for fold in range(config.validation.n_splits):
            model_path = os.path.join(config.output_dir, f'{config.exp_name}_fold_{fold}.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            preds = model.predict_proba(X_test)[:, 1]
            test_preds += preds / config.validation.n_splits

        # Assuming X_test has 'id' column
        submission_df = pd.DataFrame({'id': X_test['id'], 'target': test_preds})
        submission_path = os.path.join(config.output_dir, f'{config.exp_name}_submission.csv')
        submission_df.to_csv(submission_path, index=False)
        logger.info(f'Submission saved to {submission_path}')

    logger.info("Training pipeline completed")


if __name__ == "__main__":
    main()