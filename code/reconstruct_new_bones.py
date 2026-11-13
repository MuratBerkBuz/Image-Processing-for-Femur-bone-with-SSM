"""Reconstruct new bones from landmark files and compare them with their true meshes.

  python reconstruct_new_bones.py --bone-dir ../Tibia_SSM \
      --landmarks-dir ../new_bones_to_be_reconstructed \
      --actual-dir ../new_bones_to_be_reconstructed/actual_bone_stls

Each <id>_..._landmarks.particles file (rows in the order of
<bone-dir>/landmark_indices.txt, any CT frame) is fitted with an estimated
pose: similarity for a model built with scaling, rigid for a size-preserving
model (--align auto). The mesh is written back in the landmarks' frame. If the
actual-dir holds a mesh whose name starts with the same <id>_, the
reconstruction and a posed mean-shape baseline are compared with it.

A new bone that is a training subject is left out of the model, the landmark
interpolation and the reference, so the test stays out-of-sample. It is
detected by its true mesh being identical to a training input mesh (rotation-
invariant signature, so pre-aligned inputs are found too), or, without a true
mesh, by landmarks within --duplicate-tol mm in the same CT frame.

For a project without landmark files (e.g. built from pre-aligned STLs), pass
the training landmarks with --cohort-landmarks-dir and, if the shape_1 input
meshes are not where the project says, --training-meshes-dir.

Writes <out>/<id>_reconstructed.stl/.vtk and <out>/new_bone_errors.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fit_landmarks import landmark_selector, reconstruct, reference_mesh
from landmarks import INDEX_FILE_NAME, landmark_file
from mesh_tools import Mesh, read_mesh, surface_distance_metrics, write_mesh
from ssm_io import (DEFAULT_RESULTS_DIR, default_align, find_project, read_landmark_coords,
                    read_landmark_indices, read_project)
from ssm_model import ShapeModel


def mesh_signature(mesh: Mesh) -> np.ndarray:
    """Rigid-motion invariant: triangle count, surface area, vertex centroid size."""
    tri = mesh.points[mesh.faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    size = np.sqrt(((mesh.points - mesh.points.mean(0)) ** 2).sum(1).mean())
    return np.array([len(mesh.faces), area, size])


def training_mesh(subject, meshes_dir: Path | None) -> Path | None:
    if subject.shape is not None and subject.shape.exists():
        return subject.shape
    if meshes_dir is not None and subject.shape is not None:
        f = meshes_dir / subject.shape.name
        return f if f.exists() else None
    return None


def matching_subject(lm: np.ndarray, actual: Path | None, subjects, signatures: dict,
                     landmarks_dir, tol: float):
    """Training subject that is the same bone, and how it was recognised."""
    if actual is not None and signatures:
        sig = mesh_signature(read_mesh(actual))
        for s in subjects:
            ref = signatures.get(s.name)
            if ref is not None and np.allclose(sig, ref, rtol=1e-5):
                return s, "true mesh identical to its input mesh"
    best, dist = None, np.inf
    for s in subjects:
        f = landmark_file(s, landmarks_dir)
        if f and f.exists():
            d = np.abs(read_landmark_coords(f) - lm).max()
            if d < dist:
                best, dist = s, d
    return (best, f"landmarks within {dist:.1f} mm") if dist < tol else (None, "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bone-dir", type=Path, required=True, help="ShapeWorks model folder, e.g. ../Tibia_SSM")
    ap.add_argument("--landmarks-dir", type=Path, required=True)
    ap.add_argument("--actual-dir", type=Path, default=None, help="true meshes (.stl/.vtk) for comparison")
    ap.add_argument("--pattern", default="*_landmarks.particles")
    ap.add_argument("--lam", type=float, default=10.0, help="regularisation weight (LOO optimum: 10)")
    ap.add_argument("--align", choices=["auto", "similarity", "rigid"], default="auto")
    ap.add_argument("--cohort-landmarks-dir", type=Path, default=None,
                    help="training landmarks (default: the files the .swproj references)")
    ap.add_argument("--training-meshes-dir", type=Path, default=None,
                    help="training input meshes, if not at the .swproj shape_1 paths")
    ap.add_argument("--correspondence", choices=["interpolated", "nearest"], default="interpolated")
    ap.add_argument("--duplicate-tol", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR / "new_bones")
    args = ap.parse_args()

    subjects, _ = read_project(find_project(args.bone_dir))
    names, idx = read_landmark_indices(args.bone_dir / INDEX_FILE_NAME)
    full_model = ShapeModel.from_exports(args.bone_dir)
    files = sorted(args.landmarks_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"no {args.pattern} in {args.landmarks_dir}")
    align = args.align if args.align != "auto" else default_align(subjects)
    print(f"{args.bone_dir.name} model: {len(subjects)} subjects, landmarks {names}; {len(files)} new bones; "
          f"pose: {align}")
    signatures = {}
    if args.actual_dir:
        for s in subjects:
            f = training_mesh(s, args.training_meshes_dir)
            if f is not None:
                signatures[s.name] = mesh_signature(read_mesh(f))
        if len(signatures) < len(subjects):
            print(f"  note: {len(subjects) - len(signatures)} training input meshes not found; "
                  "duplicate check falls back to landmarks for those")

    rows = []
    for f in files:
        bone_id = f.name.split("_")[0]
        lm = read_landmark_coords(f, n_expected=len(idx))
        actual = sorted(args.actual_dir.glob(f"{bone_id}_*")) if args.actual_dir else []
        actual = [a for a in actual if a.suffix.lower() in (".stl", ".vtk")]
        dup, how = matching_subject(lm, actual[0] if actual else None, subjects, signatures,
                                    args.cohort_landmarks_dir, args.duplicate_tol)
        if dup is not None:
            pool = [s for s in subjects if s is not dup]
            model = ShapeModel.from_particles(np.stack([s.load_world_particles() for s in pool]))
            note = f"same bone as training subject {dup.name} ({how}): left out"
            # The Studio mean mesh contains the left-out subject; warp a training subject instead.
            nearest = min(pool, key=lambda s: np.linalg.norm(s.load_world_particles() - model.mean))
            ref_mesh, ref_pts, ref_name = reference_mesh(args.bone_dir, model, pool, nearest.name)
        else:
            pool, model, note = subjects, full_model, ""
            ref_mesh, ref_pts, ref_name = reference_mesh(args.bone_dir, model, pool)
        sel = landmark_selector(args.correspondence, idx, pool, landmarks_dir=args.cohort_landmarks_dir)

        fit, _, mesh, to_model = reconstruct(model, lm, sel, args.lam, (ref_mesh, ref_pts), align=align)
        _, _, mean_mesh, _ = reconstruct(model, lm, sel, 1e12, (ref_mesh, ref_pts), align=align)
        scale = float(np.cbrt(np.linalg.det(to_model[:3, :3])))
        write_mesh(args.out / f"{bone_id}_reconstructed.stl", mesh)
        write_mesh(args.out / f"{bone_id}_reconstructed.vtk", mesh)
        write_mesh(args.out / f"{bone_id}_mean_shape_posed.stl", mean_mesh)

        print(f"\n{f.name}" + (f"\n  NOTE: {note}" if note else ""))
        print(f"  model {model.n_modes} modes, reference {ref_name}, scale to model {scale:.3f}, "
              f"landmark residual RMS {fit.landmark_rms / scale:.2f} mm")
        base = {"bone": bone_id, "landmarks": f.name, "lam": args.lam, "align": align,
                "scale_to_model": scale, "landmark_rms": fit.landmark_rms / scale, "note": note}

        if not actual:
            rows.append({**base, "method": "ssm_fit"})
            continue
        true_mesh = read_mesh(actual[0])
        for method, m in (("ssm_fit", mesh), ("mean_shape", mean_mesh)):
            err = surface_distance_metrics(m, true_mesh)
            rows.append({**base, "method": method, "actual": actual[0].name, **err})
            print(f"  {method:10s} vs {actual[0].name}: mean {err['surf_mean']:.2f}  rms {err['surf_rms']:.2f}  "
                  f"p95 {err['surf_p95']:.2f}  hausdorff {err['hausdorff']:.2f} mm")

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "new_bone_errors.csv", index=False)
    print(f"\nwrote meshes and {args.out / 'new_bone_errors.csv'}")


if __name__ == "__main__":
    main()
