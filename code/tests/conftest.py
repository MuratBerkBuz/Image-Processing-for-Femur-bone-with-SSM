"""Shared fixtures: synthetic ellipsoid cohorts written as ShapeWorks-like projects.

The synthetic shapes are ellipsoids whose three semi-axes vary between
subjects. Ellipsoid points are linear in the axes, so the shape space is
exactly three-dimensional and every vertex is a perfect correspondence: fits
and reconstructions have known, exact answers.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from mesh_tools import Mesh, write_stl, write_vtk  # noqa: E402
from ssm_io import apply_transform, write_particles  # noqa: E402
from ssm_model import ShapeModel  # noqa: E402

LANDMARK_NAMES = ["A", "B", "C", "D", "E", "F"]


def ellipsoid(axes=(1.0, 1.0, 1.0), n_lat: int = 10, n_lon: int = 14) -> Mesh:
    """Closed triangulated ellipsoid (poles + latitude rings)."""
    pts = [[0.0, 0.0, 1.0]]
    for i in range(1, n_lat):
        th = np.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * np.pi * j / n_lon
            pts.append([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    pts.append([0.0, 0.0, -1.0])
    pts = np.asarray(pts) * np.asarray(axes, float)
    bot = len(pts) - 1

    def ring(i, j):
        return 1 + (i - 1) * n_lon + j % n_lon

    faces = [[0, ring(1, j), ring(1, j + 1)] for j in range(n_lon)]
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            a, b, c, d = ring(i, j), ring(i, j + 1), ring(i + 1, j), ring(i + 1, j + 1)
            faces += [[a, c, b], [b, c, d]]
    faces += [[bot, ring(n_lat - 1, j + 1), ring(n_lat - 1, j)] for j in range(n_lon)]
    return Mesh(pts, np.asarray(faces))


def random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def similarity(rng, scale: float = 1.0, shift: float = 200.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = scale * random_rotation(rng)
    T[:3, 3] = rng.uniform(-shift, shift, 3)
    return T


def cohort_axes(n: int, rng) -> np.ndarray:
    z = rng.normal(size=(n, 3))
    return np.c_[40 + 3 * z[:, 0], 30 + 2 * z[:, 1], 100 + 8 * z[:, 0] + 4 * z[:, 2]]


# Spread-out vertices used as landmarks (poles and equator points).
def landmark_vertices(n_lat: int = 10, n_lon: int = 14) -> np.ndarray:
    eq = 1 + (n_lat // 2 - 1) * n_lon
    n = 1 + (n_lat - 1) * n_lon + 1
    return np.array([0, n - 1, eq, eq + n_lon // 4, eq + n_lon // 2, 1 + (n_lat // 4 - 1) * n_lon + 3])


def make_project(root: Path, n_subjects: int = 10, scaled: bool = False, stale_names: bool = False,
                 seed: int = 0) -> dict:
    """Write a ShapeWorks-like project for a synthetic ellipsoid cohort.

    Layout: <root>/data_raw/<name>.stl (raw input, local frame) and
    <root>/Bone_SSM/ with groomed meshes, particles, exports, landmark files and
    a .swproj. world = alignment @ local; with ``scaled`` the alignment also
    scales (like ShapeWorks landmark grooming). With ``stale_names`` the
    .swproj refers to an old particles folder name.
    """
    rng = np.random.default_rng(seed)
    proj = root / "Bone_SSM"
    for d in ("groomed", "Bone_particles", "mean_shape_particles", "eigenvectors", "eigenvalues"):
        (proj / d).mkdir(parents=True, exist_ok=True)
    (root / "data_raw").mkdir(parents=True, exist_ok=True)

    axes = cohort_axes(n_subjects, rng)
    lm_idx = landmark_vertices()
    names = [f"s{i:02d}" for i in range(n_subjects)]
    template = ellipsoid()
    world, transforms, rows = [], [], []
    pdir = "OLD_particles" if stale_names else "Bone_particles"
    for name, ax in zip(names, axes):
        w = ellipsoid(ax)
        A = similarity(rng, scale=rng.uniform(0.85, 1.15) if scaled else 1.0)
        local = w.transformed(np.linalg.inv(A))
        write_vtk(proj / "groomed" / f"{name}_groomed.vtk", local)
        write_stl(root / "data_raw" / f"{name}.stl", local)
        write_particles(proj / "Bone_particles" / f"{name}_groomed_local.particles", local.points)
        write_particles(proj / "Bone_particles" / f"{name}_groomed_world.particles", w.points)
        write_particles(proj / f"{name}_landmarks.particles", local.points[lm_idx])
        world.append(w.points)
        transforms.append(A)
        rows.append({
            "name": name, "notes": "",
            "shape_1": f"..\\data_raw\\{name}.stl",
            "landmarks_file_1": f"{name}_landmarks.particles",
            "groomed_1": f"groomed\\{name}_groomed.vtk",
            "local_particles_1": f"{pdir}\\{name}_groomed_local.particles",
            "world_particles_1": f"{pdir}\\{name}_groomed_world.particles",
            "alignment_1": " ".join(f"{v:.9f}" for v in A.ravel()),
            "procrustes_1": " ".join(f"{v:.1f}" for v in np.eye(4).ravel()),
        })
    world = np.stack(world)
    model = ShapeModel.from_particles(world)
    write_particles(proj / "mean_shape_particles" / "mean.particles", model.mean)
    for k in range(model.n_modes):
        np.savetxt(proj / "eigenvectors" / f"eigenvectors_P_{k}.eval", model.modes[:, k].reshape(-1, 3))
    np.savetxt(proj / "eigenvalues" / "eigenvalues.eval", model.eigenvalues)
    project = {"data": rows, "landmarks": [{"name": n} for n in LANDMARK_NAMES], "groom": {}, "optimize": {}}
    (proj / "Bone.swproj").write_text(json.dumps(project, indent=1))
    return {"dir": proj, "root": root, "names": names, "world": world, "transforms": transforms,
            "lm_idx": lm_idx, "faces": template.faces, "axes": axes}


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "rigid")


@pytest.fixture
def scaled_project(tmp_path):
    return make_project(tmp_path / "scaled", scaled=True, seed=1)


def run_script(script: str, *args, cwd=CODE_DIR) -> subprocess.CompletedProcess:
    """Run one of the command-line scripts; fail the test with its output on error."""
    res = subprocess.run([sys.executable, str(CODE_DIR / script), *map(str, args)], cwd=cwd,
                         capture_output=True, text=True)
    assert res.returncode == 0, f"{script} failed:\n{res.stdout}\n{res.stderr}"
    return res
