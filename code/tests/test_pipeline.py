"""Landmark helpers, reconstruction and the command-line scripts on synthetic projects."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import ellipsoid, make_project, random_rotation, run_script
from fit_landmarks import landmark_selector, reconstruct, reference_mesh
from landmarks import (cohort_landmarks, correspondence, farthest_point_sampling, indices_from_coords,
                       subject_landmarks)
from mesh_tools import Mesh, read_mesh, read_stl, write_stl
from prealign_stls import centroid_size, medoid, rigid_gpa, robust_flags
from reconstruct_new_bones import mesh_signature
from ssm_io import apply_transform, find_project, load_cohort, read_landmark_indices, read_particles, write_particles
from ssm_model import ShapeModel


# ------------------------------------------------------------------ helpers
def test_farthest_point_sampling_spreads(rng):
    pts = rng.normal(size=(300, 3))
    idx = farthest_point_sampling(pts, 5)
    assert len(set(idx.tolist())) == 5
    d = np.linalg.norm(pts[idx][:, None] - pts[idx][None], axis=2) + np.eye(5) * 1e9
    assert d.min() > 1.0


def test_indices_from_coords(project):
    _, shapes = load_cohort(project["dir"])
    lms = shapes[:, project["lm_idx"]]
    idx, dist = indices_from_coords(shapes, lms)
    assert idx.tolist() == project["lm_idx"].tolist() and dist.max() < 1e-9


def test_subject_landmarks_sources(project, rng):
    subjects, shapes = load_cohort(project["dir"])
    idx = project["lm_idx"]
    from_files = subject_landmarks(subjects[0], idx, "files", shapes[0])
    assert np.allclose(from_files, shapes[0][idx], atol=1e-3)
    noisy = subject_landmarks(subjects[0], idx, "particles", shapes[0], noise=1.0, rng=rng)
    assert 0 < np.abs(noisy - shapes[0][idx]).max() < 6
    with pytest.raises(ValueError):
        subject_landmarks(subjects[0], idx, "nowhere", shapes[0])


def test_cohort_landmarks_and_correspondence(project):
    subjects, shapes = load_cohort(project["dir"])
    have, lms = cohort_landmarks(subjects, "local")
    assert len(have) == len(subjects) and lms.shape == (len(subjects), 6, 3)
    idx = project["lm_idx"]
    assert correspondence("nearest", idx, shapes, lms) is idx
    W = correspondence("interpolated", idx, shapes, lms, n_neighbors=6)
    assert np.allclose(np.einsum("kn,snc->skc", W, shapes), lms, atol=0.5)


# ------------------------------------------------------------------ reconstruction
@pytest.mark.parametrize("scaled", [False, True])
def test_reconstruct_new_bone_in_its_own_frame(tmp_path, rng, scaled):
    info = make_project(tmp_path, scaled=scaled, seed=3)
    subjects, _ = read_project_subjects(info)
    model = ShapeModel.from_exports(info["dir"])
    idx = info["lm_idx"]
    ref = reference_mesh(info["dir"], model, subjects)[:2]
    sel = landmark_selector("nearest", idx, subjects)

    truth = ellipsoid((43, 29, 104))         # an unseen bone inside the shape space
    T = np.eye(4)
    T[:3, :3] = random_rotation(rng)
    T[:3, 3] = [-120, 160, -1100]             # its own CT frame
    bone = truth.transformed(T)
    align = "similarity" if scaled else "rigid"
    fit, _, mesh_in, _ = reconstruct(model, bone.points[idx], sel, 1e-8, ref, align=align)
    assert fit.landmark_rms < 1e-3
    assert np.abs(mesh_in.points - bone.points).max() < 1e-2


def read_project_subjects(info):
    from ssm_io import read_project
    return read_project(find_project(info["dir"]))


def test_reference_mesh_falls_back_to_subject(project):
    subjects, _ = read_project_subjects(project)
    model = ShapeModel.from_exports(project["dir"])
    mesh, pts, name = reference_mesh(project["dir"], model, subjects)
    assert name in project["names"] and len(pts) == model.n_particles


def test_mesh_signature_is_rigid_invariant(rng):
    m = ellipsoid((40, 30, 100))
    T = np.eye(4)
    T[:3, :3] = random_rotation(rng)
    T[:3, 3] = [5, 6, 7]
    assert np.allclose(mesh_signature(m), mesh_signature(m.transformed(T)))
    assert not np.allclose(mesh_signature(m), mesh_signature(ellipsoid((41, 30, 100))))


# ------------------------------------------------------------------ pre-alignment
def test_prealign_helpers(rng):
    base = rng.normal(size=(5, 3)) * 50
    lms = np.stack([apply_transform(np.r_[np.c_[random_rotation(rng), rng.normal(size=3) * 100], [[0, 0, 0, 1]]],
                                    base + rng.normal(size=base.shape) * (0.1 if i else 3))
                    for i in range(8)])
    assert medoid(lms) != 0  # subject 0 is the noisiest
    ref, changes = rigid_gpa(lms, medoid(lms), max_passes=10, tol=1e-6)
    assert changes[-1] < 1e-6
    assert centroid_size(ref) == pytest.approx(centroid_size(base), rel=0.02)
    flags = robust_flags(np.array([1.0, 1.1, 0.9, 1.0, 5.0]))
    assert flags.tolist() == [False, False, False, False, True]


def test_prealign_script(tmp_path):
    info = make_project(tmp_path, scaled=True, seed=5)
    out = tmp_path / "aligned"
    run_script("prealign_stls.py", "--bone-dir", info["dir"], "--out", out)
    log = pd.read_csv(out / "Bone" / "alignment_log.csv")
    assert len(log) == len(info["names"])
    assert np.allclose(log.centroid_size_before, log.centroid_size_after, atol=1e-9)
    assert np.allclose(log.mesh_size_before, log.mesh_size_after, atol=1e-9)
    assert np.allclose(log.det_R, 1.0)
    # aligned landmarks sit on the aligned meshes, and all subjects share one pose
    for name in info["names"][:3]:
        m = read_stl(out / "Bone" / f"{name}.stl")
        lm = read_particles(out / "Bone" / f"{name}_landmarks.particles")
        assert np.min(np.linalg.norm(m.points[:, None] - lm[None], axis=2), axis=0).max() < 1e-3


# ------------------------------------------------------------------ scripts end to end
def test_check_model_script(project):
    res = run_script("check_model.py", "--bone-dir", project["dir"])
    assert "All checks passed." in res.stdout


def test_landmarks_and_loo_scripts(project, tmp_path):
    run_script("landmarks.py", "--bone-dir", project["dir"], "from-coords", "--frame", "local")
    names, idx = read_landmark_indices(project["dir"] / "landmark_indices.txt")
    assert names == ["A", "B", "C", "D", "E", "F"] and idx.tolist() == project["lm_idx"].tolist()
    out = tmp_path / "loo"
    run_script("loo_validation.py", "--bone-dir", project["dir"], "--landmark-source", "files",
               "--correspondence", "nearest", "--pose", "estimate", "--lambdas", "1e-8", "--skip-mesh",
               "--out", out)
    df = pd.read_csv(out / "loo_errors.csv")
    fit = df[df.method == "ssm_fit"]
    assert len(fit) == len(project["names"]) and fit.particle_max.max() < 1e-2
    assert (df[df.method == "mean_shape"].particle_rms.mean() > 10 * fit.particle_rms.mean())


def test_reconstruct_new_bones_script(project, tmp_path, rng):
    run_script("landmarks.py", "--bone-dir", project["dir"], "from-coords", "--frame", "local")
    new, actual = tmp_path / "new", tmp_path / "new" / "actual"
    actual.mkdir(parents=True)
    idx = project["lm_idx"]
    # 01: unseen bone; 02: the raw mesh of training subject s03 (must be detected and left out)
    unseen = ellipsoid((42, 31, 98))
    T = np.eye(4)
    T[:3, :3] = random_rotation(rng)
    T[:3, 3] = [300, -50, 20]
    bones = {"01": unseen.transformed(T), "02": read_mesh(project["root"] / "data_raw" / "s03.stl")}
    for bid, m in bones.items():
        write_stl(actual / f"{bid}_Bone_R.stl", m)
        write_particles(new / f"{bid}_Bone_landmarks.particles", m.points[idx] if bid == "01" else
                        read_particles(project["dir"] / "s03_landmarks.particles"))
    out = tmp_path / "out"
    run_script("reconstruct_new_bones.py", "--bone-dir", project["dir"], "--landmarks-dir", new,
               "--actual-dir", actual, "--lam", "1e-8", "--correspondence", "nearest", "--out", out)
    df = pd.read_csv(out / "new_bone_errors.csv")
    fit = df[df.method == "ssm_fit"].set_index("bone")
    assert "s03" in str(fit.loc[2, "note"])
    assert fit.surf_mean.max() < 0.05
    assert (out / "01_reconstructed.stl").exists()


def test_compare_models_script(project, scaled_project, tmp_path):
    out = tmp_path / "cmp"
    run_script("compare_models.py", "--models", project["dir"], scaled_project["dir"], "--out", out)
    table = pd.read_csv(out / "model_comparison.csv")
    assert table.scale_min.iloc[0] == pytest.approx(1.0) and table.scale_min.iloc[1] < 1.0
