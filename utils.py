import os
import json

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                   'configs/pretrained_models.json')


def load_pretrained_config(model_name, config_path=DEFAULT_CONFIG_PATH):
    with open(config_path, 'r') as f:
        config = json.load(f)
    if model_name not in config:
        raise ValueError(f"{model_name} not found in {config} at {config_path}")
    return config[model_name]
