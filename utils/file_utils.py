import json
import numpy as np
from pathlib import Path
import sys
from typing import Optional


# Add repo root to path so we can import config directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from config.config import _NPZ_KEY_FREQS, _NPZ_KEY_RESPONSE, _NPZ_KEY_CORRECTION, _NPZ_KEY_IR, _NPZ_KEY_PNG_BYTES


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_cal_profile(
    output_dir: Path,
    prefix: str = "cal_send",
    metadata_dict: Optional[dict] = None,
    correction_filter: Optional[np.ndarray] = None,
    ir: Optional[np.ndarray] = None,
    freqs: Optional[np.ndarray] = None,
    response_H: Optional[np.ndarray] = None,
) -> Path:
    """Save calibration profile (and optional correction filter/IR) as NPZ.

    If PNG chart data is in `metadata_dict["chart_png_bytes"]`, it is stored under
    ``NPZ_KEY_PNG_BYTES`` for programmatic access.

    Returns the path to the saved NPZ file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{prefix}_profile.npz"

    save_dict: dict = {
        _NPZ_KEY_FREQS: freqs if freqs is not None else np.array([]),
        _NPZ_KEY_RESPONSE: response_H if response_H is not None else np.array([], dtype=np.complex128),
    }

    if correction_filter is not None:
        save_dict[_NPZ_KEY_CORRECTION] = correction_filter
    if ir is not None:
        save_dict[_NPZ_KEY_IR] = ir
    if "chart_png_bytes" in (metadata_dict or {}):
        save_dict[_NPZ_KEY_PNG_BYTES] = metadata_dict["chart_png_bytes"]

    np.savez(str(npz_path), **save_dict)

    # Also write a metadata JSON alongside for human readability
    meta_path = output_dir / f"{prefix}_profile.meta.json"
    meta_for_json = {}
    if metadata_dict:
        clean_meta = {k: v for k, v in metadata_dict.items()
                       if not isinstance(v, (np.ndarray, bytes))}
        meta_for_json.update(clean_meta)
    meta_path.write_text(json.dumps(meta_for_json, indent=2, default=str))

    return npz_path


def load_send_corrections(data_dir: Path) -> np.ndarray:
    """Load per-bin send correction factors from pre-computed calibration output.

    Returns correction_factor array or exits with error if not found.
    """
    search_paths = [
        data_dir / "cal_send_corrections.npz",
        _REPO_ROOT / "data" / "cal_send_corrections.npz",
        Path.cwd() / "data" / "cal_send_corrections.npz",
        _REPO_ROOT / "logs" / "cal_send_corrections.npz",
    ]
    for p in search_paths:
        if p.exists():
            d = np.load(str(p))
            factors = d["correction_factor"]
            print(f"  Loaded send corrections from : {p}")
            return factors
    print("ERROR: cal_send_corrections.npz not found. Run calibrate_send.py first.", file=sys.stderr)
    sys.exit(1)
