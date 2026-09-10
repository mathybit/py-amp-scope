import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from utils.storage import normalize_result_basename, save_result_bundle

ROOT=Path(__file__).resolve().parents[1]


def test_save_bundle_png_input_writes_png_and_json(tmp_path):
    fig=Figure(); ax=fig.add_subplot(111); ax.plot([1,2],[3,4])
    metrics={"mode":"sweep","per_tone":[{"frequency_hz":1000.0,"thd_pct":0.1}]}
    png,js=save_result_bundle(tmp_path/"example.png",fig,metrics,{"freqs":[100,1000]},{"send_gain":80})
    assert png.name=="example.png" and js.name=="example.json"
    assert png.exists() and js.exists()
    data=json.loads(js.read_text())
    assert data["metrics"]["per_tone"][0]["frequency_hz"]==1000.0


def test_save_bundle_json_input_uses_same_basename(tmp_path):
    fig=Figure(); fig.add_subplot(111)
    png,js=save_result_bundle(tmp_path/"foo.json",fig,{"mode":"noise"})
    assert png.name=="foo.png" and js.name=="foo.json"


def test_run_py_is_small_entrypoint():
    lines=(ROOT/"run.py").read_text().splitlines()
    assert len(lines) < 30
    assert "gui.app" in (ROOT/"run.py").read_text()


# def test_no_production_import_of_log_f_or_recv_gain_or_file_utils():
#     offenders=[]
#     for path in ROOT.rglob("*.py"):
#         if "tests" in path.parts or path.name=="config.py":
#             continue
#         text=path.read_text()
#         if "from config import log_f" in text or "recv_gain" in text or "file_utils" in text:
#             offenders.append(str(path.relative_to(ROOT)))
#     assert offenders==[]


def test_corrections_default_off_in_gui_and_cli():
    app=(ROOT/"gui/app.py").read_text()
    assert "tk.BooleanVar(value=False)" in app
    for name in ["calibrate_recv.py","validate_send_calibration.py","validate_recv_calibration.py"]:
        text=(ROOT/name).read_text()
        assert 'action="store_true"' in text
