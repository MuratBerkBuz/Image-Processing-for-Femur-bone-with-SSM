"""Checks on the real ShapeWorks exports (marked `data`; a missing folder is skipped).

    python -m pytest -m data
"""
import numpy as np
import pytest

from mesh_tools import SurfaceDistance, read_mesh, read_vtk
from reconstruct_new_bones import mesh_signature
from ssm_io import (ROOT, default_align, find_project, load_cohort, read_landmark_coords, read_landmark_indices,
                    read_project, transform_scale)
from ssm_model import ShapeModel

pytestmark = pytest.mark.data

MODELS = {  # folder: (subjects, landmarks, built with scaling)
    "Femur_SSM": (29, 4, True),
    "Femur_SSM_prealigned": (29, 4, False),
    "Tibia_SSM": (30, 5, True),
    "Tibia_SSM_prealigned": (30, 5, False),
}


def model_dir(name):
    d = ROOT / name
    if not d.is_dir():
        pytest.skip(f"{d} not found")
    return d


@pytest.fixture(params=sorted(MODELS), scope="module")
def cohort(request):
    d = model_dir(request.param)
    subjects, proj = read_project(find_project(d))
    return request.param, d, subjects, proj


def test_project_paths_resolve(cohort):
    name, _, subjects, _ = cohort
    assert len(subjects) == MODELS[name][0]
    for s in subjects:
        for p in (s.groomed, s.local_particles, s.world_particles, s.landmarks_file, s.shape):
            assert p is not None and p.exists(), f"{s.name}: {p}"


def test_landmark_files(cohort):
    name, d, subjects, _ = cohort
    n = MODELS[name][1]
    for s in subjects:
        assert read_landmark_coords(s.landmarks_file).shape == (n, 3)
    if (d / "landmark_indices.txt").exists():
        names, idx = read_landmark_indices(d / "landmark_indices.txt")
        assert len(idx) == n and idx.min() >= 0 and idx.max() < 512


def test_exports_match_particles(cohort):
    _, d, _, _ = cohort
    exported = ShapeModel.from_exports(d)
    _, shapes = load_cohort(d)
    rebuilt = ShapeModel.from_particles(shapes)
    assert np.abs(rebuilt.mean - exported.mean).max() < 0.05
    k = min(exported.n_modes, rebuilt.n_modes)
    rel = np.abs(rebuilt.eigenvalues[:k] - exported.eigenvalues[:k]) / exported.eigenvalues[:k]
    assert rel.max() < 1e-3
    cos = np.abs(np.sum(rebuilt.modes[:, :k] * exported.modes[:, :k], axis=0))
    assert cos.min() > 0.999


def test_scaling(cohort):
    name, _, subjects, _ = cohort
    scales = np.array([transform_scale(s.local_to_world) for s in subjects])
    if MODELS[name][2]:
        assert np.ptp(scales) > 0.05 and default_align(subjects) == "similarity"
    else:
        assert np.allclose(scales, 1.0, atol=1e-6) and default_align(subjects) == "rigid"


def test_groomed_frame_matches_particles(cohort):
    _, _, subjects, _ = cohort
    s = subjects[0]
    mesh = read_vtk(s.groomed).transformed(s.local_to_world)
    assert SurfaceDistance(mesh)(s.load_world_particles()).mean() < 0.5


def test_landmarks_lie_on_input_mesh(cohort):
    _, _, subjects, _ = cohort
    for s in subjects[:3]:
        d = SurfaceDistance(read_mesh(s.shape))(read_landmark_coords(s.landmarks_file))
        assert d.max() < 1.0, s.name


def test_new_bone_04_is_training_subject_103559():
    actual = ROOT / "new_bones_to_be_reconstructed" / "actual_bone_stls" / "04_Tibia_R.stl"
    d = model_dir("Tibia_SSM")
    if not actual.exists():
        pytest.skip("new bones not found")
    subjects, _ = read_project(find_project(d))
    sig = mesh_signature(read_mesh(actual))
    hits = [s.name for s in subjects if np.allclose(mesh_signature(read_mesh(s.shape)), sig, rtol=1e-5)]
    assert hits == ["103559-tibia-cortical-remesh"]
