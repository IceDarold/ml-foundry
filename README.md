# ML Foundry

A standardized ML project template designed for reproducibility, modularity, and flexibility. This template provides a structured approach to building machine learning projects with clear separation of concerns, configuration management, and best practices for data handling, feature engineering, model training, and validation.

## Features

- Modular codebase with separate components for data loading, feature engineering, modeling, and validation
- Configuration-driven experiments using YAML files
- Reproducible results through structured pipelines
- Support for various ML algorithms and frameworks

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/IceDarold/ml-foundry.git
   cd ml-foundry
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run a training experiment, use the following command:

```bash
python train.py --config configs/lgbm_exp001.yaml
```

Replace `configs/lgbm_exp001.yaml` with the path to your desired configuration file.

## Project Structure

```
ml-foundry/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── train.py
├── configs/
│   ├── base_config.yaml
│   └── lgbm_baseline_exp001.yaml
├── data/
├── notebooks/
│   └── .gitkeep
├── outputs/
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── feature_engineering.py
    ├── model.py
    ├── utils.py
    └── validation.py
```

- `src/`: Source code modules
  - `config.py`: Configuration management
  - `data_loader.py`: Data loading utilities
  - `feature_engineering.py`: Feature engineering functions
  - `model.py`: Model definitions and training
  - `utils.py`: Utility functions
  - `validation.py`: Validation and evaluation
- `configs/`: YAML configuration files for experiments
- `data/`: Directory for datasets
- `notebooks/`: Jupyter notebooks for exploration and prototyping
- `outputs/`: Directory for model outputs and results
- `train.py`: Main training script
- `requirements.txt`: Python dependencies
- `LICENSE`: Project license
- `.gitignore`: Git ignore rules

## Contributing

Contributions are welcome! Please follow the established code structure and add appropriate tests for new features.

## License

This project is licensed under the terms specified in the LICENSE file.