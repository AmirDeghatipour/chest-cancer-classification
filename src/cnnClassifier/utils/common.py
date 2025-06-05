from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
import yaml
from src.cnnClassifier.logging import logger


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads YAML file and returns its content as ConfigBox."""
    try:
        with path_to_yaml.open("r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


def create_directories(paths: list[Path], verbose: bool = True):
    """Creates a list of directories."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")