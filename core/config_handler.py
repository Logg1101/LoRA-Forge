"""
Configuration I/O — bridges the Pydantic schema with JSON file persistence.
"""
import json
from pathlib import Path

from core.config_schema import TrainingConfig

CONFIG_DIR = Path("configs")
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.json"
PRESETS_DIR = CONFIG_DIR / "presets"


def save_config(config: TrainingConfig, path: Path | None = None) -> Path:
    """Save config to JSON. Returns the path written."""
    target = path or DEFAULT_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=4)
    return target


def load_config(path: Path | None = None) -> TrainingConfig:
    """Load config from JSON, falling back to defaults for missing keys."""
    target = path or DEFAULT_CONFIG_PATH
    if target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TrainingConfig.from_legacy_dict(data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Config load failed ({e}), using defaults.")
    return TrainingConfig()


def save_preset(config: TrainingConfig, name: str) -> Path:
    """Save a named preset."""
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "_- " else "_" for c in name)
    path = PRESETS_DIR / f"{safe_name}.json"
    return save_config(config, path)


def load_preset(name: str) -> TrainingConfig | None:
    """Load a named preset."""
    safe_name = "".join(c if c.isalnum() or c in "_- " else "_" for c in name)
    path = PRESETS_DIR / f"{safe_name}.json"
    if path.exists():
        return load_config(path)
    return None


def list_user_presets() -> list[str]:
    """List saved user preset names."""
    if not PRESETS_DIR.exists():
        return []
    return [p.stem for p in sorted(PRESETS_DIR.glob("*.json"))]


# Legacy compat
def save_config_to_file(config_dict: dict) -> dict:
    """Legacy wrapper — accepts raw dict, saves, returns it."""
    cfg = TrainingConfig.from_legacy_dict(config_dict)
    save_config(cfg)
    return cfg.to_engine_dict()


def load_config_from_file() -> dict:
    """Legacy wrapper — returns flat dict for backward compat."""
    return load_config().to_engine_dict()