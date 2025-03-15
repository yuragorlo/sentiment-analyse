import os
import logging
import yaml
from logging import config as logging_config

from app.utils.config import get_log_dir


def get_logger(name: str = "app"):
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)

def setup_logging(config_path: str = "./app/utils/log_conf.yaml", default_level: str = "INFO"):
    """Setup logging configuration from a YAML file."""
    log_dir = get_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")

    log_level = os.getenv("LOG_LEVEL", default_level).upper()
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        for logger in config.get("loggers", {}).values():
            logger["level"] = log_level
        for handler in config.get("handlers", {}).values():
            handler["level"] = log_level
        config["root"]["level"] = log_level
        config["handlers"]["file"]["filename"] = log_path
        logging_config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.getLevelName(log_level),
            format="[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.warning(f"Logging config file not found at {config_path}. Using basic configuration.")