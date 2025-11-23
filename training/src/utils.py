import yaml
import logging
from typing import Dict, Any

def load_config(config_path: str = "training/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

def resolve_target_column_name(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned.startswith("target_"):
        return cleaned
    if cleaned.startswith("t") and cleaned[1:].isdigit():
        return f"target_{cleaned}"
    if cleaned.isdigit():
        return f"target_t{cleaned}"
    if cleaned.startswith("t"):
        cleaned = cleaned[1:]
    return f"target_{cleaned}"