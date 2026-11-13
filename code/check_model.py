"""README steps 1-2: load the exports, check shapes and consistency, check the mean shape.

    python check_model.py [--bone-dir ../Femur_SSM] [--exclude z063]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mesh_tools import SurfaceDistance, read_vtk
from ssm_io import DEFAULT_FEMUR_DIR, bone_name, load_cohort, read_pca_scores, transform_scale
from ssm_model import ShapeModel

README_PARTICLES = 512
# Subjects under review, per bone folder (README "Model conventions").
FLAGGED = {"Femur": {"z063": "possible artifact", "015": "two render sets", "016": "two render sets",
                     "023": "two render sets", "z050": "two render sets"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bone-dir", "--femur-dir", dest="bone_dir", type=Path, default=DEFAULT_FEMUR_DIR,
                    help="ShapeWorks model folder, e.g. ../Femur_SSM or ../Tibia_SSM")
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'FAIL'}] {msg}")

    # ---------------------------------------------------------------- step 1
    print("Step 1: exported model")
    model = ShapeModel.from_exports(args.bone_dir)
    n, m = model.n_particles, model.n_modes
    print(f"  mean {model.mean.shape}, eigenvectors {model.modes.shape}, eigenvalues {model.eigenvalues.shape}")
    check(model.mean.shape == (n, 3), "mean is particles x 3")
    check(model.modes.shape == (3 * n, m), "eigenvectors match particle count")
    check(model.eigenvalues.size == model.modes.shape[1], "eigenvalue count == mode count")
    check(np.abs(model.modes.T @ model.modes - np.eye(m)).max() < 1e-4, "eigenvectors orthonormal")
    check(np.all(np.diff(model.eigenvalues) <= 0) and np.all(model.eigenvalues > 0),
          "eigenvalues positive, descending")
    if n != README_PARTICLES:
        print(f"  NOTE: model has {n} particles; README says {README_PARTICLES}.")
    cum = np.cumsum(model.eigenvalues) / model.eigenvalues.sum()
    print("  modes for 90/95/99% variance:",
          [int(np.searchsorted(cum, f) + 1) for f in (0.90, 0.95, 0.99)])

    subjects, shapes = load_cohort(args.bone_dir, args.exclude)
    names = [s.name for s in subjects]
    print(f"\n  cohort: {len(subjects)} subjects in project")
    for key, why in FLAGGED.get(bone_name(args.bone_dir), {}).items():
        hit = [nm for nm in names if nm.startswith(key)]
        print(f"    {key:5s} ({why}): {'present' if hit else 'not in project'}")

    if not args.exclude:
        rebuilt = ShapeModel.from_particles(shapes)
        check(np.abs(rebuilt.mean - model.mean).max() < 1e-2,
              f"mean == average of world particles (max diff {np.abs(rebuilt.mean - model.mean).max():.2e} mm)")
        k = min(m, rebuilt.n_modes)
        rel = np.abs(rebuilt.eigenvalues[:k] - model.eigenvalues[:k]) / model.eigenvalues[:k]
        check(rel.max() < 1e-3, f"eigenvalues reproduced from particles (max rel diff {rel.max():.1e})")
        cos = np.abs(np.sum(rebuilt.modes[:, :k] * model.modes[:, :k], axis=0))
        check(cos.min() > 0.999, f"modes reproduced up to sign (min |cos| {cos.min():.4f})")
        scores_path = args.bone_dir / "pca_scores.csv"
        if scores_path.exists():
            sc = read_pca_scores(scores_path)
            b = np.stack([model.project(x) for x in shapes])
            exp = sc.loc[names].to_numpy()[:, :m]
            check(np.abs(b - exp).max() < 1e-2, f"pca_scores.csv reproduced (max diff {np.abs(b - exp).max():.1e})")

    # ---------------------------------------------------------------- step 2
    print("\nStep 2: mean shape")
    check(np.allclose(model.reconstruct(np.zeros(m)), model.mean), "mu + P*0 == mu")
    mean_mesh_files = sorted(args.bone_dir.glob("mean_shape_mesh*.vtk"))
    if mean_mesh_files:
        mesh = read_vtk(mean_mesh_files[0])
        d = SurfaceDistance(mesh)(model.mean)
        print(f"  {mean_mesh_files[0].name}: {len(mesh.points)} vertices, {len(mesh.faces)} faces")
        print(f"  mean particles -> Studio mean mesh: mean {d.mean():.3f} mm, max {d.max():.3f} mm")
        check(d.mean() < 1.0, "mean particles lie on the Studio mean mesh")

    # Frame check: groomed meshes (local) mapped to world must carry the world particles.
    print("\n  frame check (world particles -> groomed mesh mapped to world):")
    print(f"    {'subject':14s} {'scale':>7s} {'mean':>7s} {'max':>7s}")
    worst = 0.0
    for s, x in zip(subjects, shapes):
        mesh = read_vtk(s.groomed).transformed(s.local_to_world)
        d = SurfaceDistance(mesh)(x)
        worst = max(worst, d.mean())
        sc = transform_scale(s.local_to_world)
        flag = "  <- alignment has scaling" if abs(sc - 1) > 1e-3 else ""
        print(f"    {s.name:14s} {sc:7.4f} {d.mean():7.3f} {d.max():7.3f}{flag}")
    check(worst < 1.0, "world = alignment @ local holds for all groomed meshes")

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
