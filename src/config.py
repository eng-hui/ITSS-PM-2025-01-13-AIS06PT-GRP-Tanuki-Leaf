import yaml
import os

def load_config(config_file="config.yaml"):
    # Assume config.yaml is in the project root.
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(current_dir, config_file)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
