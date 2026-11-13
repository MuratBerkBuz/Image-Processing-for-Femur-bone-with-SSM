"""Mesh I/O, particle-driven mesh warping and surface distances (README steps 5-6).

The warp is a thin-plate-spline (biharmonic) deformation defined by the
reference particles -> target particles, applied to the reference mesh. This is
the same idea as ShapeWorks' MeshWarper; it is implemented with SciPy here so
the pipeline runs without the ShapeWorks Python package.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from ssm_io import apply_transform


@dataclass
class Mesh:
    points: np.ndarray  # (V, 3)
    faces: np.ndarray   # (F, 3) triangles

    def transformed(self, T: np.ndarray) -> "Mesh":
        return Mesh(apply_transform(T, self.points), self.faces.copy())


# --------------------------------------------------------------------------
# Legacy ASCII VTK polydata
# --------------------------------------------------------------------------
def read_vtk(path) -> Mesh:
    lines = Path(path).read_text().splitlines()
    if len(lines) < 4 or lines[2].strip().upper() != "ASCII":
        raise ValueError(f"{path}: only legacy ASCII VTK files are supported")
    tok = " ".join(lines[4:]).split()

    i = tok.index("POINTS")
    n = int(tok[i + 1])
    points = np.asarray(tok[i + 3:i + 3 + 3 * n], dtype=float).reshape(n, 3)

    j = tok.index("POLYGONS", i + 3 + 3 * n)
    n_poly, size = int(tok[j + 1]), int(tok[j + 2])
    raw = np.asarray(tok[j + 3:j + 3 + size], dtype=np.int64)
    if size == 4 * n_poly and np.all(raw[::4] == 3):
        faces = raw.reshape(n_poly, 4)[:, 1:]
    else:  # mixed polygons: fan-triangulate
        faces, k = [], 0
        for _ in range(n_poly):
            c = raw[k]
            ids = raw[k + 1:k + 1 + c]
            faces += [(ids[0], ids[m], ids[m + 1]) for m in range(1, c - 1)]
            k += c + 1
        faces = np.asarray(faces, dtype=np.int64)
    return Mesh(points, faces)


def write_vtk(path, mesh: Mesh) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nf = len(mesh.faces)
    with path.open("w") as f:
        f.write("# vtk DataFile Version 4.2\nvtk output\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(mesh.points)} float\n")
        np.savetxt(f, mesh.points, fmt="%.6g")
        f.write(f"POLYGONS {nf} {4 * nf}\n")
        np.savetxt(f, np.c_[np.full(nf, 3), mesh.faces], fmt="%d")


# --------------------------------------------------------------------------
# STL
# --------------------------------------------------------------------------
def read_stl(path) -> Mesh:
    """Binary or ASCII STL; duplicate corner vertices are merged."""
    data = Path(path).read_bytes()
    n = int.from_bytes(data[80:84], "little") if len(data) >= 84 else -1
    if len(data) == 84 + 50 * n:
        dt = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        corners = np.frombuffer(data, dt, count=n, offset=84)["v"].reshape(-1, 3)
    else:
        tok = data.decode("ascii", errors="ignore").split()
        corners = np.asarray([tok[i + 1:i + 4] for i, t in enumerate(tok) if t == "vertex"], dtype=float)
    points, inverse = np.unique(corners.astype(float), axis=0, return_inverse=True)
    return Mesh(points, inverse.reshape(-1, 3))


def write_stl(path, mesh: Mesh) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tri = mesh.points[mesh.faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    dt = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec = np.zeros(len(tri), dt)
    rec["normal"], rec["v"] = normal, tri
    with path.open("wb") as f:
        f.write(b"binary STL written by WP5 mesh_tools".ljust(80, b" "))
        f.write(np.uint32(len(tri)).tobytes())
        f.write(rec.tobytes())


def read_mesh(path) -> Mesh:
    return read_stl(path) if str(path).lower().endswith(".stl") else read_vtk(path)


def write_mesh(path, mesh: Mesh) -> None:
    (write_stl if str(path).lower().endswith(".stl") else write_vtk)(path, mesh)


# --------------------------------------------------------------------------
# Warping
# --------------------------------------------------------------------------
def warp_mesh(reference: Mesh, ref_particles: np.ndarray, target_particles: np.ndarray,
              smoothing: float = 0.0) -> Mesh:
    """Deform ``reference`` so that ref_particles move onto target_particles."""
    rbf = RBFInterpolator(ref_particles, target_particles - ref_particles,
                          kernel="thin_plate_spline", degree=1, smoothing=smoothing)
    return Mesh(reference.points + rbf(reference.points), reference.faces.copy())


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------
def _closest_point_on_triangles(p, a, b, c):
    """Vectorised closest point on triangle (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    dot = lambda u, v: np.einsum("...i,...i->...", u, v)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        res = a + ab * (vb / denom)[..., None] + ac * (vc / denom)[..., None]
        # Regions in reverse priority so that the first matching test in
        # Ericson's ordering (A, B, AB, C, AC, BC, interior) wins.
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        res = np.where(m[..., None], b + (c - b) * w[..., None], res)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        res = np.where(m[..., None], a + ac * (d2 / (d2 - d6))[..., None], res)
        res = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, res)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        res = np.where(m[..., None], a + ab * (d1 / (d1 - d3))[..., None], res)
        res = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, res)
        res = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, res)
    return res


class SurfaceDistance:
    """Point-to-surface distance to a triangle mesh (exact over k candidate triangles)."""

    def __init__(self, mesh: Mesh, k: int = 16):
        self.mesh = mesh
        self.tri = mesh.points[mesh.faces]                 # (F, 3, 3)
        self.tree = cKDTree(self.tri.mean(axis=1))
        self.vtree = cKDTree(mesh.points)
        self.k = min(k, len(mesh.faces))

    def __call__(self, query: np.ndarray, chunk: int = 20000) -> np.ndarray:
        query = np.asarray(query, float)
        out = np.empty(len(query))
        for s in range(0, len(query), chunk):
            q = query[s:s + chunk]
            _, idx = self.tree.query(q, k=self.k)
            idx = idx.reshape(len(q), -1)
            t = self.tri[idx]                                  # (Q, k, 3, 3)
            cp = _closest_point_on_triangles(q[:, None, :], t[..., 0, :], t[..., 1, :], t[..., 2, :])
            d = np.linalg.norm(cp - q[:, None, :], axis=-1)
            d = np.where(np.isfinite(d), d, np.inf).min(axis=1)
            out[s:s + chunk] = np.minimum(d, self.vtree.query(q)[0])
        return out


def surface_distance_metrics(pred: Mesh, true: Mesh) -> dict:
    """Symmetric surface distances (mm) between two meshes.

    mean: mean of both directed vertex-to-surface distances; p95: 95th
    percentile; hausdorff: max of both directed maxima.
    """
    d_pt = SurfaceDistance(true)(pred.points)   # pred -> true
    d_tp = SurfaceDistance(pred)(true.points)   # true -> pred
    both = np.concatenate([d_pt, d_tp])
    return {
        "surf_mean": float(both.mean()),
        "surf_rms": float(np.sqrt((both ** 2).mean())),
        "surf_p95": float(np.percentile(both, 95)),
        "hausdorff": float(max(d_pt.max(), d_tp.max())),
        "surf_mean_pred_to_true": float(d_pt.mean()),
        "surf_mean_true_to_pred": float(d_tp.mean()),
    }
