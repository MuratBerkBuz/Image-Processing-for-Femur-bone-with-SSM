import numpy as np
import pytest

from conftest import ellipsoid, random_rotation
from mesh_tools import (Mesh, SurfaceDistance, _closest_point_on_triangles, read_mesh, read_stl, read_vtk,
                        surface_distance_metrics, warp_mesh, write_mesh, write_stl, write_vtk)


@pytest.fixture
def mesh():
    return ellipsoid((40, 30, 100))


@pytest.mark.parametrize("suffix", [".vtk", ".stl"])
def test_mesh_roundtrip(tmp_path, mesh, suffix):
    write_mesh(tmp_path / f"m{suffix}", mesh)
    back = read_mesh(tmp_path / f"m{suffix}")
    assert len(back.faces) == len(mesh.faces)
    # same triangles (vertex order may differ after STL vertex merging)
    a = np.sort(np.round(mesh.points[mesh.faces], 3).reshape(len(mesh.faces), -1), axis=0)
    b = np.sort(np.round(back.points[back.faces], 3).reshape(len(back.faces), -1), axis=0)
    assert np.allclose(a, b, atol=1e-3)


def test_ascii_stl(tmp_path):
    (tmp_path / "a.stl").write_text(
        "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
        "endloop\nendfacet\nendsolid t\n")
    m = read_stl(tmp_path / "a.stl")
    assert m.points.shape == (3, 3) and m.faces.shape == (1, 3)


def test_vtk_rejects_binary(tmp_path):
    (tmp_path / "b.vtk").write_text("# vtk DataFile Version 4.2\nx\nBINARY\nDATASET POLYDATA\n")
    with pytest.raises(ValueError):
        read_vtk(tmp_path / "b.vtk")


def test_vtk_quads_are_triangulated(tmp_path):
    (tmp_path / "q.vtk").write_text(
        "# vtk DataFile Version 4.2\nq\nASCII\nDATASET POLYDATA\nPOINTS 4 float\n"
        "0 0 0 1 0 0 1 1 0 0 1 0\nPOLYGONS 1 5\n4 0 1 2 3\n")
    assert read_vtk(tmp_path / "q.vtk").faces.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_closest_point_matches_brute_force(rng):
    a, b, c = rng.normal(size=(3, 500, 3))
    p = rng.normal(size=(500, 3)) * 2
    d = np.linalg.norm(_closest_point_on_triangles(p, a, b, c) - p, axis=1)
    u, v = np.meshgrid(np.linspace(0, 1, 121), np.linspace(0, 1, 121))
    keep = u + v <= 1
    samples = a[:, None] + (b - a)[:, None] * u[keep][None, :, None] + (c - a)[:, None] * v[keep][None, :, None]
    brute = np.linalg.norm(samples - p[:, None], axis=2).min(axis=1)
    assert np.all(d <= brute + 1e-9)          # never worse than any surface point
    assert np.all(brute - d < 0.02)           # and as good as the dense sampling


def test_surface_distance_plane():
    square = Mesh(np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], float),
                  np.array([[0, 1, 2], [0, 2, 3]]))
    q = np.array([[5, 5, 3], [2, 7, -4], [15, 5, 0]], float)
    assert np.allclose(SurfaceDistance(square)(q), [3, 4, 5])


def test_metrics_zero_for_identical_and_known_offset(mesh):
    m0 = surface_distance_metrics(mesh, mesh)
    assert m0["surf_mean"] < 1e-9 and m0["hausdorff"] < 1e-9
    big = Mesh(mesh.points * 1.05, mesh.faces)
    m1 = surface_distance_metrics(big, mesh)
    assert 0 < m1["surf_mean"] < m1["hausdorff"] <= 5.0 + 1e-6  # tip moves 5 mm


def test_warp_is_exact_at_particles_and_rigid_invariant(mesh, rng):
    src = mesh.points[::3]
    T = np.eye(4)
    T[:3, :3] = random_rotation(rng)
    T[:3, 3] = [10, 20, 30]
    moved = warp_mesh(mesh, src, src @ T[:3, :3].T + T[:3, 3])
    assert np.allclose(moved.points, mesh.transformed(T).points, atol=1e-6)  # affine part is exact
    tgt = ellipsoid((45, 28, 110)).points[::3]
    warped = warp_mesh(mesh, src, tgt)
    assert np.allclose(warped.points[::3], tgt, atol=1e-6)


def test_write_stl_unit_normals(tmp_path):
    m = ellipsoid((10, 10, 10))
    write_stl(tmp_path / "s.stl", m)
    raw = (tmp_path / "s.stl").read_bytes()
    dt = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec = np.frombuffer(raw, dt, offset=84)
    assert len(rec) == len(m.faces)
    assert np.allclose(np.linalg.norm(rec["normal"], axis=1), 1, atol=1e-5)
    edges = rec["v"][:, 1] - rec["v"][:, 0]
    assert np.abs(np.einsum("ij,ij->i", rec["normal"], edges)).max() < 1e-4
    write_vtk(tmp_path / "s.vtk", m)  # smoke: both writers accept the same mesh
