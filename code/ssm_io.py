"""I/O for the ShapeWorks SSM exports (WP5): Femur_SSM, Tibia_SSM, *_SSM_prealigned.

Frames
------
* "local": the frame of the groomed meshes (``groomed/*.vtk``) and of the
  ``*_local.particles`` files. ShapeWorks does not move the groomed meshes;
  the groom alignment is stored as a transform in the project file instead.
* "world": the model frame. Mean shape, eigenvectors and ``*_world.particles``
  live here. world = procrustes @ alignment @ local  (4x4, row-major,
  translation in the last column).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEMUR_DIR = ROOT / "Femur_SSM_prealigned"
DEFAULT_RESULTS_DIR = ROOT / "results"
DATA_ALIGNED_DIR = ROOT / "data_aligned"   # output of prealign_stls.py, one folder per bone


def bone_name(bone_dir) -> str:
    """'Femur' for Femur_SSM, Femur_SSM_prealigned, Femur, ..."""
    return Path(bone_dir).resolve().name.split("_")[0]


# --------------------------------------------------------------------------
# Plain-text exports
# --------------------------------------------------------------------------
def require_file(path, what: str, hint: str = "") -> Path:
    """Return ``path`` if it is an existing file, else raise a FileNotFoundError naming it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}" + (f"\n{hint}" if hint else ""))
    return path


def read_particles(path) -> np.ndarray:
    """Read a ShapeWorks .particles file (one 'x y z' row per particle)."""
    require_file(path, "particles file")
    if not Path(path).read_text(errors="replace").strip():
        raise ValueError(f"{path}: file is empty")
    try:
        pts = np.loadtxt(path, dtype=float, ndmin=2)
    except ValueError as e:
        raise ValueError(f"{path}: not a list of 'x y z' numbers ({e})") from None
    if pts.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 columns, got {pts.shape[1]}")
    return pts


def write_particles(path, pts: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(pts).reshape(-1, 3), fmt="%.6f", delimiter="    ")


def read_eigenvalues(path) -> np.ndarray:
    require_file(path, "eigenvalues file", "Export it in ShapeWorks Studio: File > Export > Export Eigenvalues.")
    return np.loadtxt(path, dtype=float).ravel()


def read_eigenvectors(directory) -> np.ndarray:
    """Read ``eigenvectors_P_<k>.eval`` files into a (3N, M) matrix.

    Each file holds one mode as N rows of 'dx dy dz'; the column is the mode
    flattened in the same (x0, y0, z0, x1, ...) order as the particles.
    """
    files = list(Path(directory).glob("eigenvectors_P_*.eval"))
    if not files:
        raise FileNotFoundError(f"no eigenvectors_P_*.eval in {directory}\n"
                                "Export them in ShapeWorks Studio: File > Export > Export Eigenvectors.")
    files.sort(key=lambda f: int(re.search(r"_P_(\d+)\.eval$", f.name).group(1)))
    return np.stack([np.loadtxt(f, dtype=float).ravel() for f in files], axis=1)


def read_pca_scores(path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("name")


def read_landmark_indices(path) -> tuple[list[str], np.ndarray]:
    """Read 'name index' rows (comments with '#'). A bare index is also accepted."""
    require_file(path, "landmark index file",
                 "Create it with: python landmarks.py --bone-dir <model folder> from-coords --names ...")
    names, idx = [], []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 1:
            names.append(f"L{len(names)}")
            idx.append(int(parts[0]))
        else:
            names.append(parts[0])
            idx.append(int(parts[1]))
    return names, np.asarray(idx, dtype=int)


def write_landmark_indices(path, names, indices, header: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {h}" for h in header.splitlines()] if header else []
    lines.append("# name  particle_index")
    lines += [f"{n} {int(i)}" for n, i in zip(names, indices)]
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
def parse_matrix(text: str) -> np.ndarray:
    return np.asarray(text.split(), dtype=float).reshape(4, 4)


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


def transform_scale(T: np.ndarray) -> float:
    """Isotropic scale of the linear part (1.0 for a rigid transform)."""
    return float(np.cbrt(abs(np.linalg.det(T[:3, :3]))))


# --------------------------------------------------------------------------
# ShapeWorks project
# --------------------------------------------------------------------------
@dataclass
class Subject:
    name: str
    groomed: Path
    local_particles: Path
    world_particles: Path
    landmarks_file: Path | None
    alignment: np.ndarray
    procrustes: np.ndarray
    shape: Path | None = None  # raw input mesh (shape_1)

    @property
    def local_to_world(self) -> np.ndarray:
        return self.procrustes @ self.alignment

    @property
    def world_to_local(self) -> np.ndarray:
        return np.linalg.inv(self.local_to_world)

    def load_world_particles(self) -> np.ndarray:
        return read_particles(self.world_particles)


def _resolve(base: Path, rel: str | None) -> Path | None:
    """Path stored in the project (relative to the project folder, Windows or POSIX separators)."""
    if not rel:
        return None
    return (base / rel.replace("\\", "/")).resolve()


REQUIRED_KEYS = ("groomed_1", "local_particles_1", "world_particles_1", "shape_1")


def read_project(path) -> tuple[list[Subject], dict]:
    """Parse a .swproj (JSON) file. Returns the subjects and the raw project dict.

    Raises FileNotFoundError listing every groomed mesh or particles file the
    project refers to that does not exist.
    """
    path = require_file(path, "ShapeWorks project")
    try:
        proj = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not a valid .swproj (JSON) file ({e})") from None
    if not proj.get("data"):
        raise ValueError(f"{path}: project has no subjects")
    base = path.parent
    eye = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    subjects = []
    for i, d in enumerate(proj["data"]):
        absent = [k for k in REQUIRED_KEYS if not d.get(k)]
        if absent:
            raise ValueError(f"{path}: subject {i + 1} has no {', '.join(absent)}")
        shape = d["shape_1"]
        subjects.append(Subject(
            # Studio may leave "name" empty; the input file name is the subject id then.
            name=d.get("name") or Path(shape.replace("\\", "/")).stem,
            groomed=_resolve(base, d["groomed_1"]),
            local_particles=_resolve(base, d["local_particles_1"]),
            world_particles=_resolve(base, d["world_particles_1"]),
            landmarks_file=_resolve(base, d.get("landmarks_file_1")),
            alignment=parse_matrix(d.get("alignment_1", eye)),
            procrustes=parse_matrix(d.get("procrustes_1", eye)),
            shape=_resolve(base, shape),
        ))
        s = subjects[-1]
        # Projects built from pre-aligned STLs reference no landmarks; prealign_stls.py
        # writes them next to the aligned STL, in the same (= groomed) frame.
        if s.landmarks_file is None and s.shape is not None:
            f = s.shape.parent / f"{s.name}_landmarks.particles"
            s.landmarks_file = f if f.exists() else None

    missing = [(s.name, p) for s in subjects for p in (s.groomed, s.local_particles, s.world_particles)
               if not p.is_file()]
    if missing:
        listing = "\n".join(f"  {name}: {p}" for name, p in missing[:10])
        more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(f"{path.name}: {len(missing)} file(s) referenced by the project are missing:\n"
                                f"{listing}{more}\nCheck the folder names or re-save the project in ShapeWorks Studio.")
    return subjects, proj


def model_is_scaled(subjects: list[Subject], tol: float = 1e-3) -> bool:
    """True if the groom/Procrustes transforms scale the subjects (model frame not in real mm)."""
    return any(abs(transform_scale(s.local_to_world) - 1) > tol for s in subjects)


def default_align(subjects: list[Subject]) -> str:
    """Pose estimation matching how the model was built: similarity for a
    scale-normalised model, rigid when size is kept."""
    return "similarity" if model_is_scaled(subjects) else "rigid"


def find_project(bone_dir) -> Path:
    bone_dir = Path(bone_dir)
    if not bone_dir.is_dir():
        raise FileNotFoundError(f"model folder not found: {bone_dir}")
    files = sorted(bone_dir.glob("*.swproj"))
    if len(files) != 1:
        found = ", ".join(f.name for f in files) or "none"
        raise FileNotFoundError(f"expected exactly one .swproj in {bone_dir}, found {len(files)} ({found})")
    return files[0]


def select_subjects(subjects: list[Subject], exclude=()) -> list[Subject]:
    """Drop subjects whose name starts with any entry of ``exclude`` (e.g. 'z063')."""
    exclude = tuple(exclude or ())
    return [s for s in subjects if not s.name.startswith(exclude)] if exclude else list(subjects)


def load_cohort(bone_dir=DEFAULT_FEMUR_DIR, exclude=()) -> tuple[list[Subject], np.ndarray]:
    """Subjects of the project (minus ``exclude``) and their world particles (S, N, 3)."""
    subjects, _ = read_project(find_project(bone_dir))
    subjects = select_subjects(subjects, exclude)
    if not subjects:
        raise ValueError(f"{bone_dir}: no subjects left after excluding {list(exclude)}")
    shapes = [s.load_world_particles() for s in subjects]
    counts = {len(x) for x in shapes}
    if len(counts) != 1:
        raise ValueError(f"{bone_dir}: subjects have different particle counts {sorted(counts)}")
    return subjects, np.stack(shapes)


def read_landmark_coords(path, n_expected: int | None = None) -> np.ndarray:
    if path is None:
        raise FileNotFoundError("no landmark file for this subject")
    lm = read_particles(path)
    if n_expected is not None and len(lm) != n_expected:
        raise ValueError(f"{path}: {len(lm)} landmarks, expected {n_expected}")
    return lm


def landmarks_to_world(lm: np.ndarray, subject: Subject, frame: str) -> np.ndarray:
    """Bring landmark coordinates into the model (world) frame."""
    if frame == "world":
        return lm
    if frame == "local":
        return apply_transform(subject.local_to_world, lm)
    raise ValueError(f"unknown frame {frame!r} (use 'local' or 'world')")
