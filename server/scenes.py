"""
Scene management for Nulltracer.

Scenes are JSON files stored in a configurable directory.
Each scene contains all render parameters needed to reproduce a view.
"""

import json
import os
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default scenes directory (relative to server/)
DEFAULT_SCENES_DIR = Path(__file__).parent / "scenes"

# Valid scene name pattern (alphanumeric, hyphens, underscores, spaces)
SCENE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9 _-]{0,62}$')

# All valid scene parameter keys (whitelist for security)
VALID_PARAMS = {
    "spin", "charge", "inclination", "fov", "phi0",
    "steps", "step_size", "obs_dist", "method",
    "bg_mode", "star_layers", "show_disk", "show_grid",
    "disk_temp", "doppler_boost", "quality",
    "srgb_output", "disk_alpha", "disk_max_crossings",
    "bloom_enabled", "bloom_radius",
    # Metadata (not render params, but stored in scene)
    "description",
}


class SceneManager:
    """Manages scene files on disk."""

    def __init__(self, scenes_dir: Optional[Path] = None):
        self.scenes_dir = scenes_dir or DEFAULT_SCENES_DIR
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Scene directory: %s", self.scenes_dir)

    def _scene_path(self, name: str) -> Path:
        """Get the file path for a scene name."""
        # Sanitize: use the name as-is but ensure .json extension
        return self.scenes_dir / f"{name}.json"

    def validate_name(self, name: str) -> bool:
        """Check if a scene name is valid."""
        return bool(SCENE_NAME_PATTERN.match(name))

    def list_scenes(self) -> list[dict]:
        """List all available scenes with metadata.

        Returns list of dicts with 'name' and 'description' keys.
        """
        scenes = []
        for f in sorted(self.scenes_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                scenes.append({
                    "name": f.stem,
                    "description": data.get("description", ""),
                    "builtin": data.get("_builtin", False),
                })
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Skipping invalid scene file %s: %s", f, e)
        return scenes

    def get_scene(self, name: str) -> Optional[dict]:
        """Load a scene by name. Returns None if not found."""
        path = self._scene_path(name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            # Filter to valid params only (security)
            filtered = {k: v for k, v in data.items() if k in VALID_PARAMS}
            return filtered
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Error reading scene %s: %s", name, e)
            return None

    def save_scene(self, name: str, params: dict) -> bool:
        """Save a scene. Returns True on success."""
        if not self.validate_name(name):
            return False
        # Check if it's a builtin scene (don't overwrite)
        path = self._scene_path(name)
        if path.exists():
            try:
                existing = json.loads(path.read_text())
                if existing.get("_builtin", False):
                    return False  # Can't overwrite builtin scenes
            except (json.JSONDecodeError, OSError):
                pass

        # Filter to valid params only
        filtered = {k: v for k, v in params.items() if k in VALID_PARAMS}

        try:
            path.write_text(json.dumps(filtered, indent=2))
            logger.info("Saved scene: %s", name)
            return True
        except OSError as e:
            logger.error("Error saving scene %s: %s", name, e)
            return False

    def delete_scene(self, name: str) -> bool:
        """Delete a scene. Returns True on success. Cannot delete builtins."""
        path = self._scene_path(name)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            if data.get("_builtin", False):
                return False  # Can't delete builtin scenes
        except (json.JSONDecodeError, OSError):
            pass

        try:
            path.unlink()
            logger.info("Deleted scene: %s", name)
            return True
        except OSError as e:
            logger.error("Error deleting scene %s: %s", name, e)
            return False
