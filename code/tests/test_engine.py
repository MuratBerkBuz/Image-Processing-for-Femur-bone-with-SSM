"""Reconstruction engine behind the GUI: input parsing, checks and the two example cases."""
import numpy as np
import pytest

from conftest import CODE_DIR
from mesh_tools import SurfaceDistance, read_stl
from reconstruction_engine import (BoneReconstructor, available_bones, handedness, load_guide, output_paths,
                                   parse_landmark_text)
from ssm_io import ROOT

EXAMPLES = CODE_DIR / "examples"


# ------------------------------------------------------------------ parsing (no data needed)
def test_parse_separators_comments_blank_lines():
    text = "# ME\n1 2 3\n\n4,5,6\n7;8;9   # comment\n10\t11\t12\n"
    assert parse_landmark_text(text, 4).tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]


@pytest.mark.parametrize("text, message", [
    ("1 2\n", "line 1: expected 3 numbers"),
    ("1 2 3\n1 x 3\n", "line 2: not a number"),
    ("1 2 3\n", "expected 4 landmarks"),
])
def test_parse_errors(text, message):
    with pytest.raises(ValueError, match=message):
        parse_landmark_text(text, 4)


def test_guide_lists_both_bones():
    guide = load_guide()
    assert available_bones(guide) == ["Femur", "Tibia"]
    for cfg in guide.values():
        assert cfg["landmarks"] and all(v["name"] and v["description"] for v in cfg["landmarks"].values())


def test_output_paths_numbering(tmp_path):
    stl, lm = output_paths(tmp_path, "Tibia", "L")
    assert stl.name == "001_Tibia_L_reconstructed.stl" and lm.name == "001_Tibia_L_landmarks.txt"
    stl.write_text("x")
    stl, _ = output_paths(tmp_path, "Tibia", "L", "z036")
    assert stl.name == "002_z036_Tibia_L_reconstructed.stl"
    (tmp_path / "017_old_Femur_R_reconstructed.stl").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    stl, _ = output_paths(tmp_path, "Femur", "R", "patient 7 (scan)")   # unsafe characters replaced
    assert stl.name == "018_patient_7_scan_Femur_R_reconstructed.stl"
    assert output_paths(tmp_path / "new_folder", "Femur", "L")[0].name == "001_Femur_L_reconstructed.stl"


def test_guide_sides():
    guide = load_guide()
    assert guide["Femur"]["side"] == "L" and guide["Tibia"]["side"] == "R"


def test_handedness_flips_with_mirror(rng):
    lm = rng.normal(size=(5, 3))
    assert np.sign(handedness(lm)) == -np.sign(handedness(lm * [-1, 1, 1]))


# ------------------------------------------------------------------ real models
@pytest.fixture(scope="module", params=["Femur", "Tibia"])
def reconstructor(request):
    cfg = load_guide()[request.param]
    if not (ROOT / cfg["model_dir"]).is_dir():
        pytest.skip(f"{cfg['model_dir']} not found")
    return BoneReconstructor(request.param)


@pytest.fixture
def example(reconstructor):
    return np.loadtxt(EXAMPLES / f"{reconstructor.bone.lower()}_example.txt")




@pytest.mark.data
def test_example_reconstructs(reconstructor, example, tmp_path):
    res = reconstructor.reconstruct(example)
    assert res.mirrored == (reconstructor.bone == "Tibia")   # the tibia example is from the other side
    assert res.side == "L"            # left femur (as the model); left tibia (mirror of the right model)
    assert res.residuals.max() < 8 and not res.warnings
    assert SurfaceDistance(res.mesh)(example).max() < 5      # landmarks lie on the reconstructed surface
    path = res.save_stl(tmp_path / "bone.stl")
    back = read_stl(path)
    assert len(back.faces) == len(res.mesh.faces)
    tri = back.points[back.faces]
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    assert volume > 0                                          # outward-facing triangles, also when mirrored


@pytest.mark.data
def test_mirrored_input_gives_mirrored_bone(reconstructor, example):
    a = reconstructor.reconstruct(example)
    b = reconstructor.reconstruct(example * [-1, 1, 1])
    assert a.mirrored != b.mirrored
    assert np.allclose(a.residuals, b.residuals, atol=1e-6)
    assert np.allclose(a.mesh.points * [-1, 1, 1], b.mesh.points, atol=1e-6)


@pytest.mark.data
def test_pose_does_not_matter(reconstructor, example):
    R = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
    R *= np.sign(np.linalg.det(R))
    moved = example @ R.T + [100, -50, 20]
    a, b = reconstructor.reconstruct(example), reconstructor.reconstruct(moved)
    assert np.allclose(a.residuals, b.residuals, atol=1e-4)
    assert np.allclose(a.mesh.points @ R.T + [100, -50, 20], b.mesh.points, atol=1e-3)


@pytest.mark.data
def test_bad_inputs(reconstructor, example):
    with pytest.raises(ValueError, match="millimetres"):
        reconstructor.reconstruct(example / 10)
    with pytest.raises(ValueError, match="expected"):
        reconstructor.reconstruct(example[:-1])
    dup = example.copy()
    dup[1] = dup[0] + 1
    with pytest.raises(ValueError, match="apart"):
        reconstructor.reconstruct(dup)
    with pytest.raises(ValueError, match="finite"):
        reconstructor.reconstruct(np.where(np.arange(example.size).reshape(example.shape) == 0, np.nan, example))


@pytest.mark.data
def test_swapped_landmarks_are_flagged(reconstructor, example):
    swapped = example[[1, 0] + list(range(2, len(example)))]
    res = reconstructor.reconstruct(swapped)
    if reconstructor.bone == "Femur":
        # Swapping ME and LE flips the handedness: it looks like the other side. The user is told
        # that the bone was mirrored and which landmarks to check.
        assert res.mirrored and any("ME and LE" in n for n in res.notes)
    else:  # IN/MC swap on the tibia: flagged as a wrong order or a bad fit
        assert res.warnings


@pytest.mark.data
def test_side_override(reconstructor, example):
    auto = reconstructor.reconstruct(example)
    other = {"L": "R", "R": "L"}[auto.side]
    forced = reconstructor.reconstruct(example, side=other)
    assert forced.side == other and forced.mirrored != auto.mirrored
    assert forced.residuals.mean() > auto.residuals.mean()   # the detected side fits better
    same = reconstructor.reconstruct(example, side=auto.side)
    assert np.allclose(same.mesh.points, auto.mesh.points)
    with pytest.raises(ValueError, match="side must be"):
        reconstructor.reconstruct(example, side="left")
