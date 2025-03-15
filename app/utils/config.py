import os


def make_cache_dir():
    """Create cache directory."""
    cache_dir = os.getenv("CACHE_DIR", "./runtime/cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.chmod(cache_dir, 0o777)

def get_cache_dir():
    """Get the path to the cache directory."""
    return os.getenv("CACHE_DIR", "./runtime/cache")

def get_log_dir():
    """Get the path to the cache directory."""
    return os.getenv("LOG_DIR", "./runtime/logs")

def get_model_path():
    """Get the path to the ML model."""
    return os.getenv("ML_MODEL_PATH", "./app/models/sentiment_model.pkl")
