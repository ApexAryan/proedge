"""Versioned model registry — local disk with optional Azure ML backend."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib

from proedge.config import get_settings
from proedge.pipeline.models.ensemble import OverUnderEnsemble

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelRegistry:
    """
    Stores and loads versioned OverUnderEnsemble models.
    Each artifact is saved as:
      <registry_path>/<sport>/<version>/model.joblib
      <registry_path>/<sport>/<version>/meta.json
    """

    def __init__(self, registry_path: str | None = None):
        self.root = Path(registry_path or settings.model_registry_path)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ save

    def save(
        self,
        model: OverUnderEnsemble,
        sport: str,
        version: str,
        metrics: dict | None = None,
        feature_names: list[str] | None = None,
        feature_medians: dict[str, float] | None = None,
    ) -> str:
        artifact_dir = self._artifact_dir(sport, version)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifact_dir / "model.joblib"
        joblib.dump(model, model_path, compress=3)

        meta = {
            "version": version,
            "sport": sport,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(model.feature_names),
            "feature_names": model.feature_names,
            "feature_medians": feature_medians or {},
            "xgb_weight": model.xgb_weight,
            "lgb_weight": model.lgb_weight,
            "metrics": metrics or {},
        }
        with open(artifact_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Update the "latest" symlink
        latest_link = self.root / sport / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(artifact_dir.resolve())

        logger.info("Saved model %s/%s → %s", sport, version, model_path)
        return str(model_path)

    # ------------------------------------------------------------------ load

    def load(self, sport: str, version: str = "latest") -> OverUnderEnsemble:
        artifact_dir = self._artifact_dir(sport, version)
        if version == "latest":
            artifact_dir = self.root / sport / "latest"
        model_path = artifact_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No model at {model_path}")
        model = joblib.load(model_path)
        logger.info("Loaded model %s/%s from %s", sport, version, model_path)
        return model

    def load_meta(self, sport: str, version: str = "latest") -> dict:
        artifact_dir = self._artifact_dir(sport, version)
        if version == "latest":
            artifact_dir = self.root / sport / "latest"
        meta_path = artifact_dir / "meta.json"
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------ list

    def list_versions(self, sport: str) -> list[dict]:
        sport_dir = self.root / sport
        if not sport_dir.exists():
            return []
        versions = []
        for d in sorted(sport_dir.iterdir()):
            if d.is_dir() and not d.is_symlink() and (d / "meta.json").exists():
                with open(d / "meta.json") as f:
                    versions.append(json.load(f))
        return sorted(versions, key=lambda x: x.get("trained_at", ""), reverse=True)

    def latest_version(self, sport: str) -> str | None:
        versions = self.list_versions(sport)
        return versions[0]["version"] if versions else None

    # ------------------------------------------------------------------ helpers

    def _artifact_dir(self, sport: str, version: str) -> Path:
        return self.root / sport / version
