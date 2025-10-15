# Kaggle Datasets Integration

This document provides comprehensive guidance for using Kaggle datasets and competitions in your ML Foundry projects. The integration supports seamless data loading from Kaggle datasets and competitions, with automatic caching and authentication handling.

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Configuration Guide](#configuration-guide)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Migration Guide](#migration-guide)
- [Best Practices](#best-practices)

## Setup Instructions

### Prerequisites

Before using Kaggle datasets, ensure you have:

1. A Kaggle account
2. The Kaggle API package installed: `pip install kaggle`
3. Valid Kaggle API credentials

### API Credentials Setup

#### Local Development

1. **Download API Token:**
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Scroll to "API" section
   - Click "Create New API Token"
   - Download `kaggle.json`

2. **Place Credentials:**
   ```bash
   # Create directory if it doesn't exist
   mkdir -p ~/.kaggle

   # Move the downloaded file
   mv ~/Downloads/kaggle.json ~/.kaggle/

   # Set proper permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

#### Environment Variables (Alternative)

Instead of using `kaggle.json`, you can set environment variables:

```bash
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-api-key"
```

#### Google Colab

For Colab notebooks, upload your `kaggle.json` file:

```python
from google.colab import files
files.upload()  # Upload kaggle.json

# Move to correct location
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

#### Kaggle Kernels

In Kaggle kernels, authentication is automatic - no additional setup required.

### Verification

Test your setup by running:

```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print("Kaggle API authentication successful!")
```

## Configuration Guide

The data source configuration is specified in your YAML configuration files under the `data.data_source` section.

### Configuration Structure

```yaml
data:
  data_source:
    type: "kaggle"  # Required: "local" or "kaggle"
    # For Kaggle datasets:
    dataset_name: "username/dataset-slug"  # Required for datasets
    # For Kaggle competitions:
    competition_name: "competition-slug"  # Required for competitions
    # Optional settings:
    data_dir: "data"  # Base directory for data storage (default: "data")
```

### Data Source Types

#### Kaggle Datasets

Use this for public datasets hosted on Kaggle:

```yaml
data:
  data_source:
    type: "kaggle"
    dataset_name: "username/dataset-name"
    data_dir: "data"
```

**Example:** `dataset_name: "titanic/train"` for the Titanic dataset.

#### Kaggle Competitions

Use this for competition data:

```yaml
data:
  data_source:
    type: "kaggle"
    competition_name: "titanic"
    data_dir: "data"
```

**Note:** You must join the competition and accept rules before downloading data.

#### Local Files

For local development or when data is already available:

```yaml
data:
  data_source:
    type: "local"
    train_path: "data/train.csv"
    test_path: "data/test.csv"
```

#### Future URL Sources

The framework is designed to support URL-based data sources in the future:

```yaml
data:
  data_source:
    type: "url"  # Not yet implemented
    train_url: "https://example.com/train.csv"
    test_url: "https://example.com/test.csv"
```

### File Discovery

The system automatically discovers train/test files using common naming patterns:

- **Training data:** `train.csv`, `training.csv`, `train_data.csv`
- **Test data:** `test.csv`, `testing.csv`, `test_data.csv`

If standard names aren't found, it searches recursively for files containing "train" or "test" in the filename.

## Usage Examples

### Example 1: Titanic Dataset

```yaml
# conf/projects/titanic/titanic.yaml
defaults:
  - _self_
  - /base/model: lgbm
  - /base/metric: accuracy
  - /base/validation: stratified_kfold

data:
  data_source:
    type: "kaggle"
    dataset_name: "heptapod/titanic"
  target_col: "Survived"

globals:
  seed: 42
  id_col: "PassengerId"
```

### Example 2: Competition Data

```yaml
# conf/projects/house_prices/house_prices.yaml
defaults:
  - _self_
  - /base/model: xgb
  - /base/metric: rmse
  - /base/validation: kfold

data:
  data_source:
    type: "kaggle"
    competition_name: "house-prices-advanced-regression-techniques"
  target_col: "SalePrice"

globals:
  seed: 42
  id_col: "Id"
```

### Example 3: Local Development Fallback

```yaml
# conf/projects/my_project/local_dev.yaml
defaults:
  - _self_
  - /base/model: lgbm

data:
  data_source:
    type: "local"
    train_path: "data/train.csv"
    test_path: "data/test.csv"
  target_col: "target"

globals:
  seed: 42
  id_col: "id"
```

### Example 4: Custom Data Directory

```yaml
data:
  data_source:
    type: "kaggle"
    dataset_name: "username/large-dataset"
    data_dir: "/mnt/data"  # Custom data directory
```

## Troubleshooting

### Authentication Failures

**Error:** `RuntimeError: Kaggle API authentication failed`

**Solutions:**

1. **Check kaggle.json location:**
   ```bash
   ls -la ~/.kaggle/kaggle.json
   ```

2. **Verify permissions:**
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Test API connection:**
   ```python
   from kaggle.api.kaggle_api_extended import KaggleApi
   api = KaggleApi()
   api.authenticate()
   ```

4. **Use environment variables:**
   ```bash
   export KAGGLE_USERNAME="your-username"
   export KAGGLE_KEY="your-api-key"
   ```

### Dataset Not Found Errors

**Error:** `Dataset {name} not found`

**Solutions:**

1. **Verify dataset name:** Check the exact name on Kaggle (case-sensitive)
2. **Check dataset visibility:** Ensure the dataset is public
3. **Confirm ownership:** If private, ensure you have access

**Example correct names:**
- `titanic/train` (not `titanic`)
- `uciml/iris` (not `iris-dataset`)

### Permission Issues

**Error:** `Access denied for dataset/competition {name}`

**Solutions:**

1. **For competitions:** Join the competition and accept rules
2. **For private datasets:** Request access from dataset owner
3. **Check account status:** Ensure your Kaggle account is in good standing

### Network Connectivity Problems

**Error:** `Failed to download dataset/competition`

**Solutions:**

1. **Check internet connection**
2. **Retry download:** The system caches downloads, so retrying is safe
3. **Use VPN if needed:** Some networks block Kaggle
4. **Check file sizes:** Large datasets may need more time

### File Not Found Errors

**Error:** `Training/Test data file not found`

**Solutions:**

1. **Check dataset contents:** Verify the dataset contains expected CSV files
2. **Custom file patterns:** If files have unusual names, the system may not find them
3. **Inspect downloaded data:** Check the cache directory manually

### Common Issues in Different Environments

#### Colab Issues

```python
# Ensure kaggle.json is uploaded and moved correctly
!ls -la ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

#### Docker/Container Issues

```dockerfile
# Add to Dockerfile
RUN mkdir -p ~/.kaggle
COPY kaggle.json ~/.kaggle/
RUN chmod 600 ~/.kaggle/kaggle.json
```

## Migration Guide

### From Local Files to Kaggle Datasets

1. **Backup your data:**
   ```bash
   cp -r data data_backup
   ```

2. **Update configuration:**
   ```yaml
   # Before
   data:
     data_source:
       type: "local"
       train_path: "data/train.csv"
       test_path: "data/test.csv"

   # After
   data:
     data_source:
       type: "kaggle"
       dataset_name: "username/my-dataset"
   ```

3. **Test the migration:**
   ```bash
   python -c "from src.data_loaders.factory import create_data_loader; loader = create_data_loader({'data_source': {'type': 'kaggle', 'dataset_name': 'username/my-dataset'}}); loader.ensure_data_available()"
   ```

4. **Update file paths if needed:** The system handles file discovery automatically.

### Preserving Local Development

Keep both configurations for different environments:

```yaml
# conf/projects/my_project/prod.yaml
data:
  data_source:
    type: "kaggle"
    dataset_name: "username/production-data"

# conf/projects/my_project/dev.yaml
data:
  data_source:
    type: "local"
    train_path: "data/train.csv"
    test_path: "data/test.csv"
```

## Best Practices

### Data Organization on Kaggle

1. **Dataset Naming:**
   - Use descriptive, lowercase names with hyphens: `customer-churn-prediction-data`
   - Include version numbers: `titanic-dataset-v2`

2. **File Structure:**
   - Place data files in the root directory
   - Use standard naming: `train.csv`, `test.csv`
   - Include metadata files: `README.md`, `data_description.txt`

3. **Documentation:**
   - Provide clear dataset descriptions
   - Document column meanings and data types
   - Include data collection methodology

### Configuration Management

1. **Environment-Specific Configs:**
   ```yaml
   # base_config.yaml
   data:
     data_source:
       type: "${DATA_SOURCE_TYPE:local}"

   # Override in specific configs
   # prod.yaml: DATA_SOURCE_TYPE=kaggle
   # dev.yaml: DATA_SOURCE_TYPE=local
   ```

2. **Version Control:**
   - Commit configuration files
   - Use different configs for different environments
   - Document configuration changes

### Performance Optimization

1. **Caching Strategy:**
   - Data is automatically cached in `data/cache/kaggle/`
   - Clear cache manually if needed: `rm -rf data/cache/kaggle/*`

2. **Large Datasets:**
   - Consider using external storage for very large datasets
   - Use `data_dir` to specify custom locations

3. **Network Efficiency:**
   - Downloads happen only once per dataset/competition
   - Failed downloads can be resumed

### Security Considerations

1. **API Keys:**
   - Never commit `kaggle.json` to version control
   - Use environment variables in production
   - Rotate keys regularly

2. **Access Control:**
   - Use private datasets when appropriate
   - Limit competition data access to team members

3. **Data Privacy:**
   - Ensure compliance with data usage terms
   - Be aware of dataset licensing restrictions

### Development Workflow

1. **Local Development:**
   - Use local files for rapid iteration
   - Switch to Kaggle data for final testing

2. **CI/CD Integration:**
   - Use environment variables for authentication
   - Mock data loading in unit tests

3. **Team Collaboration:**
   - Share dataset access with team members
   - Document data dependencies clearly

This integration makes it easy to work with Kaggle data while maintaining flexibility for different development and deployment scenarios.