"""
VAVE AI Prompt Registry — Feature 8: Prompt Versioning
Loads all prompts from prompts/vave_prompts.yaml at startup.
Provides format() helper, version tracking, and change logging.
"""
import yaml
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

_PROMPTS_FILE = Path(__file__).parent / "prompts" / "vave_prompts.yaml"
_registry: dict = {}
_loaded_at: str = ""


def load_registry(path: Path = _PROMPTS_FILE) -> dict:
    """Load all prompts from the YAML registry file."""
    global _registry, _loaded_at
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Strip the top-level 'meta' key; keep only prompt entries
        _registry = {k: v for k, v in data.items() if k != "meta"}
        _loaded_at = datetime.now().isoformat()
        logger.info(
            f"[PromptRegistry] Loaded {len(_registry)} prompts from {path} "
            f"(registry v{data.get('meta', {}).get('registry_version', '?')})"
        )
        return _registry
    except FileNotFoundError:
        logger.error(f"[PromptRegistry] Prompts file not found: {path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"[PromptRegistry] YAML parse error: {e}")
        return {}


def get(key: str, **kwargs) -> str:
    """
    Retrieve a prompt by key and format it with the given kwargs.

    Example:
        prompt = get("multi_query_reformulation", query="brake cost ideas", n=4)
    """
    if not _registry:
        load_registry()

    entry = _registry.get(key)
    if entry is None:
        logger.warning(f"[PromptRegistry] Prompt key '{key}' not found.")
        return ""

    content: str = entry.get("content", "")
    version: str = entry.get("version", "?")

    try:
        if kwargs:
            content = content.format(**kwargs)
    except KeyError as e:
        logger.warning(f"[PromptRegistry] Missing placeholder {e} in prompt '{key}'")

    return content


def get_with_version(key: str, **kwargs) -> tuple[str, str]:
    """Return (content, version) tuple for audit logging."""
    if not _registry:
        load_registry()
    entry = _registry.get(key, {})
    content = get(key, **kwargs)
    version = entry.get("version", "unknown")
    return content, version


def get_version(key: str) -> str:
    """Return the version string of a prompt."""
    if not _registry:
        load_registry()
    return _registry.get(key, {}).get("version", "unknown")


def list_prompts() -> dict:
    """Return summary of all loaded prompts with versions."""
    if not _registry:
        load_registry()
    return {k: {"version": v.get("version"), "updated": v.get("updated")}
            for k, v in _registry.items()}


# Auto-load on import
load_registry()
