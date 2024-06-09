import yaml


def load_yaml_config(config_path):
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def load_config(config_path):
    if config_path.endswith(".yaml"):
        return load_yaml_config(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")
    