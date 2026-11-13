"""Landmark definitions: which particles the landmarks correspond to.

Writes a landmark index file ('name particle_index' per line) used by
fit_landmarks.py and loo_validation.py.

  # From digitized landmarks (ShapeWorks *_landmarks.particles, one row per
  # landmark in the project order ME, LE, PN, GT). For each landmark, picks the
  # particle closest to it on average over subjects:
  python landmarks.py from-coords [--frame local] [--landmarks-dir ../Femur_SSM]

  # Development stand-in until digitized landmarks exist: K well-spread
  # particles on the mean shape (farthest-point sampling):
  python landmarks.py proxy --n 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ssm_io import (DEFAULT_FEMUR_DIR, Subject, find_project, landmarks_to_world, load_cohort,
                    read_landmark_coords, read_project, write_landmark_indices)
from ssm_model import ShapeModel, landmark_weights

INDEX_FILE_NAME = "landmark_indices.txt"  # stored in the bone folder


def landmark_file(subject: Subject, landmarks_dir: Path | None) -> Path:
    """Per-subject landmark file: the project's landmarks_file_1, optionally in another folder."""
    name = subject.landmarks_file.name if subject.landmarks_file else f"{subject.name}_landmarks.particles"
    return (Path(landmarks_dir) / name) if landmarks_dir else subject.landmarks_file


def subject_landmarks(subject: Subject, indices: np.ndarray, source: str, world_particles: np.ndarray,
                      frame: str = "local", landmarks_dir=None, noise: float = 0.0,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Landmarks of one subject in the world frame.

    source='particles': the subject's own particles at ``indices`` (+ Gaussian
    noise of ``noise`` mm per axis) - a stand-in for digitized landmarks.
    source='files': digitized landmarks read from file and mapped to world.
    """
    if source == "particles":
        lm = world_particles[indices].copy()
        if noise > 0:
            lm += (rng or np.random.default_rng()).normal(scale=noise, size=lm.shape)
        return lm
    if source == "files":
        lm = read_landmark_coords(landmark_file(subject, landmarks_dir), n_expected=len(indices))
        return landmarks_to_world(lm, subject, frame)
    raise ValueError(f"unknown landmark source {source!r}")


def cohort_landmarks(subjects: list[Subject], frame: str = "local", landmarks_dir=None,
                     n_expected: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """World-frame digitized landmarks of every subject that has a file.
    Returns (positions of those subjects in ``subjects``, landmarks (S', K, 3))."""
    have, lms = [], []
    for i, s in enumerate(subjects):
        f = landmark_file(s, landmarks_dir)
        if f is not None and f.exists():
            have.append(i)
            lms.append(landmarks_to_world(read_landmark_coords(f, n_expected), s, frame))
    return np.asarray(have, dtype=int), (np.stack(lms) if lms else np.empty((0, 0, 3)))


def correspondence(mode: str, indices: np.ndarray, shapes: np.ndarray, lms: np.ndarray,
                   n_neighbors: int = 12) -> np.ndarray:
    """Landmark selector for ShapeModel.fit_landmarks: nearest-particle indices,
    or a weight matrix learned from these subjects' shapes (S, N, 3) and landmarks (S, K, 3)."""
    if mode == "nearest":
        return indices
    if mode == "interpolated":
        return landmark_weights(shapes, lms, n_neighbors)
    raise ValueError(f"unknown correspondence {mode!r}")


def farthest_point_sampling(pts: np.ndarray, k: int, seed_index: int | None = None) -> np.ndarray:
    pts = np.asarray(pts, float)
    first = int(np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1))) if seed_index is None else seed_index
    chosen = [first]
    d = np.linalg.norm(pts - pts[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(pts - pts[nxt], axis=1))
    return np.asarray(chosen)


def indices_from_coords(shapes: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """shapes (S, N, 3), landmarks (S, K, 3), both world frame.
    Returns per-landmark particle index and the per-subject distance (S, K) to it."""
    d = np.linalg.norm(shapes[:, :, None, :] - landmarks[:, None, :, :], axis=-1)  # (S, N, K)
    idx = d.mean(axis=0).argmin(axis=0)
    return idx, d[:, idx, np.arange(idx.size)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bone-dir", "--femur-dir", dest="bone_dir", type=Path, default=DEFAULT_FEMUR_DIR,
                    help="ShapeWorks model folder, e.g. ../Femur_SSM or ../Tibia_SSM")
    ap.add_argument("--out", type=Path, default=None, help="default: <bone-dir>/landmark_indices.txt")
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("from-coords", help="match digitized landmarks to particles")
    c.add_argument("--landmarks-dir", type=Path, default=None,
                   help="folder with <subject>_landmarks.particles (default: paths in the .swproj)")
    c.add_argument("--frame", choices=["local", "world"], default="local",
                   help="frame of the landmark files: 'local' = groomed mesh / CT frame (ShapeWorks default)")
    c.add_argument("--exclude", nargs="*", default=[])
    c.add_argument("--neighbors", type=int, default=12, help="particles per interpolated landmark")
    c.add_argument("--names", nargs="*", default=None,
                   help="landmark names, if the project defines none (e.g. ME LE PN GT)")
    p = sub.add_parser("proxy", help="farthest-point sampling on the mean shape")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed-index", type=int, default=None)
    args = ap.parse_args()
    args.out = args.out or args.bone_dir / INDEX_FILE_NAME

    if args.mode == "proxy":
        model = ShapeModel.from_exports(args.bone_dir)
        idx = farthest_point_sampling(model.mean, args.n, args.seed_index)
        names = [f"P{i}" for i in range(args.n)]
        write_landmark_indices(args.out, names, idx,
                               header=f"PROXY landmarks: farthest-point sampling, n={args.n} (not anatomical)")
        print(f"wrote {args.out}: {dict(zip(names, idx.tolist()))}")
        return

    _, proj = read_project(find_project(args.bone_dir))
    names = args.names or [lm["name"] for lm in proj.get("landmarks", [])]
    subjects, shapes = load_cohort(args.bone_dir, args.exclude)
    have, lms = cohort_landmarks(subjects, args.frame, args.landmarks_dir, len(names) or None)
    for i in sorted(set(range(len(subjects))) - set(have.tolist())):
        print(f"  missing landmarks for {subjects[i].name}: {landmark_file(subjects[i], args.landmarks_dir)}")
    if not have.size:
        raise SystemExit("no landmark files found")
    names = names or [f"L{i}" for i in range(lms.shape[1])]
    shapes = shapes[have]
    idx, dist = indices_from_coords(shapes, lms)

    # Leave-one-out error of the interpolated correspondence (weights learned without the subject).
    interp = np.zeros_like(dist)
    for s in range(len(have)):
        rest = np.arange(len(have)) != s
        W = landmark_weights(shapes[rest], lms[rest], args.neighbors)
        interp[s] = np.linalg.norm(W @ shapes[s] - lms[s], axis=1)
    print(f"{len(have)} subjects with landmarks; landmark position error from particles (mm):")
    print(f"  {'':4s} {'particle':>8s}   {'nearest mean/max':>16s}   {'interpolated (LOO) mean/max':>27s}")
    for k, n in enumerate(names):
        print(f"  {n:4s} {idx[k]:8d}   {dist[:, k].mean():7.2f} {dist[:, k].max():7.2f}"
              f"   {interp[:, k].mean():14.2f} {interp[:, k].max():7.2f}")
    write_landmark_indices(args.out, names, idx,
                           header=f"from digitized landmarks ({len(lms)} subjects, frame={args.frame})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
