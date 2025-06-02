import yaml
import config
import os


def load(path: str) -> dict:
    """Load settings from a YAML file."""
    with open(path, "r") as file:
        settings = yaml.safe_load(file)
    return settings

def get(file = None, key = None) -> dict:
    base_path = config.base_path
    if file is None or file == 'demo':
        path = os.path.join(base_path, 'demo_arguments.yaml')
    else:
        file_path = os.path.join(*file.split('.'))
        path = os.path.join(base_path, f'{file_path}.yaml')

    if not os.path.exists(path):
        raise FileNotFoundError(f"Settings file not found at {path}")

    settings = load(path)

    if key is None:
        return settings

    if key not in settings:
        raise ValueError(f"Settings for '{key}' not found in '{path}'.")
    return settings[key]
