"""Rigid (no scaling) landmark pre-alignment of the raw STLs before ShapeWorks grooming.

Implements raw-stl-alignment-plan.md:
  1. Kabsch per subject: rotation + translation only, reflections excluded.
  2. Reference: start from the medoid subject, align everyone to it, replace the
     reference by the mean of the aligned landmarks, repeat until it stops
     changing (rigid generalized Procrustes). The mean is re-posed onto the
     medoid each pass so the common frame stays close to the medoid's CT frame.
  3. Every vertex v -> R v + t; facet normals are recomputed from the rotated
     triangles (equivalent to rotating them with R, since det R = +1).

  python prealign_stls.py --bone-dir ../Femur_SSM
  python prealign_stls.py --bone-dir ../Tibia_SSM
  python prealign_stls.py --bone-dir ../Tibia_SSM --reference ../data_aligned/Tibia/reference_landmarks.particles

Inputs come from the ShapeWorks project: shape_1 (raw STL) and landmarks_file_1
per subject. Writes to <out>/<bone>/:
  <same name>.stl                     aligned mesh
  <same name>_landmarks.particles     aligned landmarks (for Studio)
  reference_landmarks.particles       final reference configuration
  alignment_log.csv                   per-subject residuals, centroid sizes, det(R), transform

Residuals: the rigid residual includes the real size differences between bones.
Outlier flags therefore use a size-free residual (after an extra scale fit that
is only computed for QC, never applied): robust median + 3 MAD per landmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mesh_tools import SurfaceDistance, read_stl, write_stl
from ssm_io import (DATA_ALIGNED_DIR, DEFAULT_FEMUR_DIR, apply_transform, bone_name, find_project,
                    read_landmark_coords, read_particles, read_project, select_subjects, write_particles)
from ssm_model import kabsch

DEFAULT_OUT = DATA_ALIGNED_DIR


def centroid_size(pts: np.ndarray) -> float:
    """RMS distance of the points from their centroid."""
    return float(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean()))


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def medoid(lms: np.ndarray) -> int:
    """Subject whose landmarks need the least rigid residual to fit all others."""
    cost = [sum(rms(apply_transform(kabsch(lms[j], lms[i]), lms[j]), lms[i]) for j in range(len(lms)))
            for i in range(len(lms))]
    return int(np.argmin(cost))


def rigid_gpa(lms: np.ndarray, start: int, max_passes: int, tol: float):
    """Iterate align-to-reference / reference = mean. Returns reference and per-pass change."""
    anchor = lms[start]
    ref, changes = anchor.copy(), []
    for _ in range(max_passes):
        aligned = np.stack([apply_transform(kabsch(l, ref), l) for l in lms])
        mean = aligned.mean(0)
        mean = apply_transform(kabsch(mean, anchor), mean)  # keep the frame at the medoid's pose
        changes.append(float(np.abs(mean - ref).max()))
        ref = mean
        if changes[-1] < tol:
            break
    return ref, changes


def robust_flags(values: np.ndarray, k: float = 3.0) -> np.ndarray:
    med = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - med), axis=0)
    return values > med + k * np.maximum(mad, 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bone-dir", type=Path, default=DEFAULT_FEMUR_DIR, help="ShapeWorks project folder")
    ap.add_argument("--exclude", nargs="*", default=[], help="subject name prefixes to skip")
    ap.add_argument("--reference", type=Path, default=None,
                    help="fixed reference landmarks (skip the iteration), e.g. from an earlier run")
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1e-3, help="stop when the reference moves less (mm)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    project = find_project(args.bone_dir)
    subjects, proj = read_project(project)
    subjects = select_subjects(subjects, args.exclude)
    lm_names = [lm["name"] for lm in proj.get("landmarks", [])]
    lms = np.stack([read_landmark_coords(s.landmarks_file, len(lm_names) or None) for s in subjects])
    lm_names = lm_names or [f"L{k}" for k in range(lms.shape[1])]
    out = args.out / bone_name(args.bone_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{project.name}: {len(subjects)} subjects, landmarks {lm_names}")

    if args.reference:
        ref, changes, start = read_particles(args.reference), [], None
        print(f"fixed reference: {args.reference}")
    else:
        start = medoid(lms)
        ref, changes = rigid_gpa(lms, start, args.passes, args.tol)
        print(f"rough reference (medoid): {subjects[start].name}")
        print("reference change per pass (mm):", " ".join(f"{c:.4f}" for c in changes))
        if changes[-1] >= args.tol:
            print(f"WARNING: reference still moving after {len(changes)} passes")
    write_particles(out / "reference_landmarks.particles", ref)

    rows, per_lm = [], []
    for s, lm in zip(subjects, lms):
        T = kabsch(lm, ref)
        R = T[:3, :3]
        lm_al = apply_transform(T, lm)
        res = np.linalg.norm(lm_al - ref, axis=1)
        # QC only: with size preserved, the rigid residual is dominated by bone
        # size. The residual after an additional (not applied) scale fit isolates
        # landmark-configuration outliers, i.e. possible digitization errors.
        lm_sim = apply_transform(kabsch(lm, ref, scale=True), lm)
        per_lm.append(np.linalg.norm(lm_sim - ref, axis=1))

        mesh = read_stl(s.shape)
        on_surface = SurfaceDistance(mesh)(lm).max()        # plan check 1: landmarks on the raw STL
        aligned = mesh.transformed(T)
        stl_out = out / s.shape.name
        write_stl(stl_out, aligned)
        write_particles(out / f"{s.name}_landmarks.particles", lm_al)

        rows.append({
            "subject": s.name, "input_stl": str(s.shape), "aligned_stl": str(stl_out),
            "landmark_rms": rms(lm_al, ref),
            **{f"res_{n}": r for n, r in zip(lm_names, res)},
            "sizefree_rms": rms(lm_sim, ref),
            **{f"sizefree_res_{n}": r for n, r in zip(lm_names, per_lm[-1])},
            "centroid_size_before": centroid_size(lm), "centroid_size_after": centroid_size(lm_al),
            "mesh_size_before": centroid_size(mesh.points), "mesh_size_after": centroid_size(aligned.points),
            "det_R": float(np.linalg.det(R)),
            "orthonormality_error": float(np.abs(R.T @ R - np.eye(3)).max()),
            "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
            "translation_mm": float(np.linalg.norm(T[:3, 3])),
            "landmark_to_raw_surface_max": float(on_surface),
            "transform_row_major": " ".join(f"{v:.6f}" for v in T.ravel()),
        })

    df = pd.DataFrame(rows)
    per_lm = np.asarray(per_lm)
    flags = robust_flags(per_lm)
    df["flagged_landmarks"] = [" ".join(n for n, f in zip(lm_names, fl) if f) for fl in flags]
    df["flag_rms"] = robust_flags(df["sizefree_rms"].to_numpy())
    df["reference_start"] = subjects[start].name if start is not None else str(args.reference)
    df["passes"] = len(changes)
    df.to_csv(out / "alignment_log.csv", index=False)

    size_change = (df.centroid_size_after - df.centroid_size_before).abs().max()
    mesh_change = (df.mesh_size_after - df.mesh_size_before).abs().max()
    print(f"\nrigid landmark RMS to reference: median {df.landmark_rms.median():.2f}, max {df.landmark_rms.max():.2f} mm"
          f" (includes bone size: corr with centroid size difference "
          f"{np.corrcoef(df.landmark_rms, (df.centroid_size_before - centroid_size(ref)).abs())[0, 1]:.2f})")
    print(f"size-free RMS (QC, flags use this): median {df.sizefree_rms.median():.2f}, max {df.sizefree_rms.max():.2f} mm")
    print("size-free per-landmark residual median (mm):",
          {n: round(float(v), 2) for n, v in zip(lm_names, np.median(per_lm, axis=0))})
    print(f"max |centroid size change|: landmarks {size_change:.2e} mm, meshes {mesh_change:.2e} mm")
    print(f"det(R): min {df.det_R.min():.6f}, max {df.det_R.max():.6f}")
    print(f"max landmark distance to raw STL surface: {df.landmark_to_raw_surface_max.max():.2f} mm")
    flagged = df[(df.flagged_landmarks != "") | df.flag_rms]
    for _, r in flagged.iterrows():
        print(f"  FLAG {r.subject}: size-free RMS {r.sizefree_rms:.2f} mm, landmarks [{r.flagged_landmarks}]")
    print(f"\nwrote {len(df)} aligned STLs and alignment_log.csv to {out}")


if __name__ == "__main__":
    main()
