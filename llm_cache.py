"""
LLM Response Caching Module
Uses SQLite to store prompt signatures and responses to save tokens and reduce latency.
"""
import sqlite3
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Constants
CACHE_DB_PATH = Path(__file__).parent / "llm_cache.db"

def init_cache():
    """Initialize the cache database table."""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_responses (
                id TEXT PRIMARY KEY,
                prompt TEXT,
                system_prompt TEXT,
                response TEXT,
                provider TEXT,
                model_name TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"LLM Cache initialized at {CACHE_DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM cache: {e}")

def _generate_cache_key(prompt: str, system_prompt: str, model_name: str) -> str:
    """Generate a unique deterministic hash for the request."""
    # Normalize inputs
    p = (prompt or "").strip()
    sp = (system_prompt or "").strip()
    m = (model_name or "").strip()
    
    # Create combined string
    combined = f"{m}::{sp}::{p}"
    
    # Return SHA256 hash
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()

def get_cached_response(prompt: str, system_prompt: str, model_name: str) -> str | None:
    """Retrieve a response from the cache if it exists."""
    try:
        key = _generate_cache_key(prompt, system_prompt, model_name)
        
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT response FROM llm_responses WHERE id = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            logger.info(f"CACHE HIT for encoded prompt {key[:8]}...")
            return result[0]
        else:
            logger.debug(f"CACHE MISS for encoded prompt {key[:8]}...")
            return None
            
    except Exception as e:
        logger.error(f"Cache retrieval error: {e}")
        return None

def cache_response(prompt: str, system_prompt: str, model_name: str, response: str, provider: str = "gemini"):
    """Store a response in the cache."""
    try:
        if not response or "ERROR" in response:
            return  # Don't cache errors
            
        key = _generate_cache_key(prompt, system_prompt, model_name)
        
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO llm_responses (id, prompt, system_prompt, response, provider, model_name, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (key, prompt, system_prompt, response, provider, model_name, datetime.now())
        )
        conn.commit()
        conn.close()
        logger.debug(f"Cached response for {key[:8]}...")
        
    except Exception as e:
        logger.error(f"Cache write error: {e}")

# Initialize on module load
init_cache()
