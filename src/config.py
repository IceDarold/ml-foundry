from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str) -> DictConfig:
    config = OmegaConf.load(config_path)
    return config