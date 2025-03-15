import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

from app.utils.config import get_cache_dir
from app.utils.logger import get_logger

logger = get_logger()

class RequestCache:
    """Simple file-based cache for API requests and responses."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the cache with a directory path."""
        self.cache_dir = Path(cache_dir or get_cache_dir())
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _generate_key(self, endpoint: str, data: Dict[str, Any]) -> str:
        """Generate a unique key for a request based on endpoint and data."""
        data_str = json.dumps(data, sort_keys=True)
        hash_key = hashlib.md5(f"{endpoint}:{data_str}".encode()).hexdigest()
        return hash_key
    
    def get(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get a cached response for a request if it exists."""
        key = self._generate_key(endpoint, data)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading cache file: {e}")
                return None
        return None
    
    def set(self, endpoint: str, data: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Cache a response for a request."""
        key = self._generate_key(endpoint, data)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f)
        except IOError as e:
            logger.error(f"Error writing to cache: {e}")

cache = RequestCache() 