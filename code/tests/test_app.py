"""The desktop app, driven programmatically (skipped without a display or the model folders)."""
import tkinter as tk

import numpy as np
import pytest

from conftest import CODE_DIR
from mesh_tools import read_stl
from reconstruction_engine import load_guide
from ssm_io import ROOT

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def shared_app():
    """One window for all tests: creating many Tk instances in one process is unreliable on Windows."""
    for cfg in load_guide().values():
        if not (ROOT / cfg["model_dir"]).is_dir():
            pytest.skip(f"{cfg['model_dir']} not found")
    from bone_reconstruction_app import App
    try:
        a = App(background_tasks=False)
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    yield a
    a.destroy()


@pytest.fixture
def app(shared_app, tmp_path):
    shared_app.output_dir = tmp_path
    shared_app.show_choose()
    shared_app.update()
    return shared_app


@pytest.mark.parametrize("bone", ["Femur", "Tibia"])
def test_full_workflow(app, tmp_path, bone):
    app.load_bone(bone)
    app.update()
    page = app.pages["landmarks"]
    assert page.rec.bone == bone and "disabled" in page.go.state()

    coords = np.loadtxt(CODE_DIR / "examples" / f"{bone.lower()}_example.txt")
    page.set_values(coords)
    app.update()
    assert "disabled" not in page.go.state()
    page.select(len(coords) - 1)          # selecting a row updates the description
    assert page.rec.landmarks[-1].name in page.where_title.cget("text")

    page.reconstruct()
    app.update()
    stl = app.last_saved
    assert stl.parent == tmp_path and stl.name == f"001_{bone}_L_reconstructed.stl"
    assert len(read_stl(stl).faces) == len(app.last_result.mesh.faces)
    saved_lm = np.loadtxt(tmp_path / f"001_{bone}_L_landmarks.txt")
    assert np.allclose(saved_lm, coords)

    # second run, with the values coming from a file: its name goes into the output name
    app.show_landmarks()
    page.input_name = "z036"
    page.reconstruct()
    app.update()
    assert app.last_saved.name == f"002_z036_{bone}_L_reconstructed.stl"
    assert "Result" in [lbl.cget("text") for lbl in app.step_labels][2]


def test_invalid_entry_blocks_reconstruction(app):
    app.load_bone("Femur")
    app.update()
    page = app.pages["landmarks"]
    page.set_values(np.loadtxt(CODE_DIR / "examples" / "femur_example.txt"))
    page.vars[2][1].set("12,5.3")          # not a number
    app.update()
    assert page.status[2].cget("text") == "✗" and "disabled" in page.go.state()
    page.vars[2][1].set("12,5")            # decimal comma is accepted
    app.update()
    assert page.status[2].cget("text") == "✓" and "disabled" not in page.go.state()
    page.clear()
    app.update()
    assert "disabled" in page.go.state() and page.hint.cget("text").startswith("0 of 4")


def test_paste_several_rows_into_one_field(app):
    app.load_bone("Tibia")
    app.update()
    page = app.pages["landmarks"]
    app.clipboard_clear()
    app.clipboard_append((CODE_DIR / "examples" / "tibia_example.txt").read_text())
    assert page.paste(None, 0, 0) == "break"
    assert [page.row_values(i) is not None for i in range(5)] == [True] * 5


def test_errors_are_reported_not_raised(app, monkeypatch):
    shown = []
    monkeypatch.setattr("bone_reconstruction_app.messagebox.showerror", lambda *a, **k: shown.append(a))
    app.load_bone("Femur")
    app.update()
    rec = app.current
    app.reconstruct(rec, np.loadtxt(CODE_DIR / "examples" / "femur_example.txt") / 10)  # centimetres
    app.update()
    assert shown and "millimetres" in shown[0][1]
