"""Persistent file I/O for PyAmpScope.

Analysis code stays independent of file formats; calibration and GUI entry points call these helpers explicitly.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional

from config import config as cfg


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_cal_profile(
            output_dir: Path,
            prefix: str = "cal_send",
            metadata_dict: Optional[dict] = None,
            correction_filter: Optional[np.ndarray] = None,
            ir: Optional[np.ndarray] = None,
            freqs: Optional[np.ndarray] = None,
            response_H: Optional[np.ndarray] = None,
        ) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{prefix}_profile.npz"
    payload = {
        cfg._NPZ_KEY_FREQS: np.asarray(freqs if freqs is not None else [], dtype=float),
        cfg._NPZ_KEY_RESPONSE: np.asarray(response_H if response_H is not None else [], dtype=float),
    }
    if correction_filter is not None:
        payload[cfg._NPZ_KEY_CORRECTION] = np.asarray(correction_filter)
    if ir is not None:
        payload[cfg._NPZ_KEY_IR] = np.asarray(ir)
    if metadata_dict and "chart_png_bytes" in metadata_dict:
        payload[cfg._NPZ_KEY_PNG_BYTES] = np.frombuffer(metadata_dict["chart_png_bytes"], dtype=np.uint8)
    np.savez(str(npz_path), **payload)

    meta = {k: v for k, v in (metadata_dict or {}).items() if k != "chart_png_bytes"}
    (output_dir / f"{prefix}_profile.meta.json").write_text(
        json.dumps(_jsonable(meta), indent=2), encoding="utf-8"
    )
    return npz_path


def save_correction_profile(
            output_dir: Path,
            filename: str,
            frequencies: np.ndarray,
            correction_factors: np.ndarray,
            measured_db: np.ndarray,
            smoothed_db: np.ndarray,
            reference_db: float,
            metadata: Optional[dict] = None,
        ) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    np.savez(
        str(path),
        frequencies=np.asarray(frequencies, dtype=float),
        correction_factor=np.asarray(correction_factors, dtype=float),
        response_H=np.asarray(measured_db, dtype=float),
        smoothed_H_db=np.asarray(smoothed_db, dtype=float),
        reference_db=np.asarray([reference_db], dtype=float),
        metadata_json=np.asarray([json.dumps(_jsonable(metadata or {}))]),
    )
    return path


def load_send_corrections(data_dir: Path) -> np.ndarray:
    """Compatibility helper returning only the send correction factors."""
    from .audio.calibration import load_send_correction
    profile = load_send_correction(Path(data_dir))
    if profile is None:
        raise FileNotFoundError("cal_send_corrections.npz not found; run calibrate_send.py")
    return profile.factors


def normalize_result_basename(path: str | Path) -> Path:
    """Treat either .png or .json input as a common result basename."""
    p = Path(path)
    if p.suffix.lower() in {".png", ".json"}:
        p = p.with_suffix("")
    elif p.suffix:
        # Preserve a user-entered basename with an unexpected extension by
        # stripping only the final extension; outputs are always PNG + JSON.
        p = p.with_suffix("")
    return p


def save_result_bundle(path: str | Path, figure, metrics: dict, chart_data: Optional[dict] = None,
                       parameters: Optional[dict] = None) -> tuple[Path, Path]:
    """Save one GUI result as matching ``.png`` and ``.json`` files."""
    base = normalize_result_basename(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    json_path = base.with_suffix(".json")
    figure.savefig(str(png_path), dpi=150, bbox_inches="tight")
    payload = {
        "parameters": parameters or {},
        "metrics": metrics,
    }
    if chart_data is not None:
        payload["chart"] = chart_data
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return png_path, json_path
