"""README steps 6-7: leave-one-out validation of landmark fitting + mesh reconstruction.

For each held-out subject the PCA is rebuilt from the remaining subjects'
world particles, the held-out subject is fitted from its landmarks, the dense
mesh is warped from a training reference (the training subject closest to the
training mean), and the result is compared to the held-out groomed mesh.
All errors are measured in the subject's own groomed/CT frame, i.e. in real mm
(the model frame is scaled by the groom alignment).

--pose known     landmarks are mapped to the model frame with the subject's
                 stored groom alignment (what the model "knows" about the subject)
--pose estimate  landmarks stay in the subject's CT frame and the similarity
                 transform to the model is estimated from them - the situation
                 of a new bone (see fit_landmarks.py --frame raw)

Methods in the output table:
  ssm_fit           landmark fit for each lambda
  mean_shape        b = 0 (baseline: no landmarks used)
  oracle_particles  warp onto the subject's true particles (floor set by the warp/particle density)

  # Development run with particle landmarks + 1 mm digitisation noise, all lambdas:
  python loo_validation.py --landmark-source particles --noise 1.0

  # Digitized landmarks, lambda chosen by nested LOO inside each fold:
  python loo_validation.py --landmark-source files --frame local --lambda-select nested

  # New-bone situation for the tibia model:
  python loo_validation.py --bone-dir ../Tibia_SSM --landmark-source files --pose estimate

Writes <out>/loo_errors.csv (one row per subject x method x lambda) and
<out>/loo_summary.csv (mean and SD over subjects).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from landmarks import INDEX_FILE_NAME, correspondence, subject_landmarks
from mesh_tools import read_vtk, surface_distance_metrics, warp_mesh, write_vtk
from ssm_io import (DEFAULT_FEMUR_DIR, DEFAULT_RESULTS_DIR, apply_transform, default_align, load_cohort,
                    read_landmark_indices)
from ssm_model import ShapeModel, particle_errors

DEFAULT_LAMBDAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def nested_lambda(shapes, train, lambdas, lms, selector_for, n_modes, align) -> float:
    """Pick lambda by inner leave-one-out on the training subjects (particle RMS)."""
    err = np.zeros(len(lambdas))
    for j in train:
        inner = [k for k in train if k != j]
        model = ShapeModel.from_particles(shapes[inner], n_modes=n_modes)
        sel = selector_for(inner)
        for a, lam in enumerate(lambdas):
            fit = model.fit_landmarks(lms[j], sel, lam, align=align)
            err[a] += particle_errors(fit.particles, shapes[j])["particle_rms"]
    return float(lambdas[int(np.argmin(err))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bone-dir", "--femur-dir", dest="bone_dir", type=Path, default=DEFAULT_FEMUR_DIR,
                    help="ShapeWorks model folder, e.g. ../Femur_SSM or ../Tibia_SSM")
    ap.add_argument("--indices", type=Path, default=None, help="default: <bone-dir>/landmark_indices.txt")
    ap.add_argument("--landmark-source", choices=["particles", "files"], default="particles")
    ap.add_argument("--landmarks-dir", type=Path, default=None)
    ap.add_argument("--frame", choices=["local", "world"], default="local",
                    help="frame of landmark files (source=files)")
    ap.add_argument("--correspondence", choices=["auto", "nearest", "interpolated"], default="auto",
                    help="landmark -> particles: nearest particle, or interpolated from nearby particles "
                         "(weights learned on the training fold). auto: interpolated for source=files")
    ap.add_argument("--neighbors", type=int, default=12, help="particles per interpolated landmark")
    ap.add_argument("--noise", type=float, default=0.0, help="landmark noise SD per axis, mm (source=particles)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    ap.add_argument("--lambda-select", choices=["all", "nested"], default="all")
    ap.add_argument("--modes", type=int, default=None, help="max number of modes")
    ap.add_argument("--pose", choices=["known", "estimate"], default="known")
    ap.add_argument("--align", choices=["auto", "rigid", "similarity"], default="auto",
                    help="transform estimated with --pose estimate (auto: similarity if the model was built "
                         "with scaling, else rigid)")
    ap.add_argument("--exclude", nargs="*", default=[], help="subject name prefixes, e.g. z063 015")
    ap.add_argument("--only", nargs="*", default=None, help="hold out only these subjects (quick runs)")
    ap.add_argument("--skip-mesh", action="store_true", help="particle errors only (fast)")
    ap.add_argument("--save-meshes", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR / "loo")
    args = ap.parse_args()

    subjects, shapes = load_cohort(args.bone_dir, args.exclude)
    lm_names, idx = read_landmark_indices(args.indices or args.bone_dir / INDEX_FILE_NAME)
    rng = np.random.default_rng(args.seed)
    estimate = args.pose == "estimate"
    align = (args.align if args.align != "auto" else default_align(subjects)) if estimate else "none"
    print(f"{len(subjects)} subjects, {shapes.shape[1]} particles, landmarks {lm_names} -> {idx.tolist()}")
    corr = args.correspondence
    if corr == "auto":
        corr = "interpolated" if args.landmark_source == "files" else "nearest"
    print(f"landmark source: {args.landmark_source}"
          + (f", noise {args.noise} mm" if args.landmark_source == "particles" else f", frame {args.frame}")
          + f", correspondence: {corr}, pose: {args.pose}" + (f" ({align})" if estimate else ""))

    # One landmark realisation per subject, shared by all lambdas and folds.
    lms = np.stack([subject_landmarks(s, idx, args.landmark_source, x, frame=args.frame,
                                      landmarks_dir=args.landmarks_dir, noise=args.noise, rng=rng)
                    for s, x in zip(subjects, shapes)])

    # What the fit sees: model-frame landmarks (pose known) or CT-frame landmarks (pose estimated).
    lms_fit = np.stack([apply_transform(s.world_to_local, l) for s, l in zip(subjects, lms)]) if estimate else lms

    def selector_for(ids):
        return correspondence(corr, idx, shapes[ids], lms[ids], args.neighbors)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    held_out = [i for i, s in enumerate(subjects) if not args.only or s.name.startswith(tuple(args.only))]
    for i in held_out:
        t0 = time.time()
        held = subjects[i]
        train = [k for k in range(len(subjects)) if k != i]
        model = ShapeModel.from_particles(shapes[train], n_modes=args.modes)
        sel = selector_for(train)
        if args.lambda_select == "nested":
            lambdas = [nested_lambda(shapes, train, args.lambdas, lms_fit, selector_for, args.modes, align)]
        else:
            lambdas = args.lambdas

        # (method, lambda, model-frame particles, fit, model frame -> subject CT frame)
        known = held.world_to_local
        if estimate:  # mean shape posed from the landmarks (b ~ 0)
            fit0 = model.fit_landmarks(lms_fit[i], sel, 1e12, align=align)
            candidates = [("mean_shape", np.nan, model.mean, None, np.linalg.inv(fit0.transform))]
        else:
            candidates = [("mean_shape", np.nan, model.mean, None, known)]
        candidates.append(("oracle_particles", np.nan, shapes[i], None, known))
        for lam in lambdas:
            fit = model.fit_landmarks(lms_fit[i], sel, lam, align=align)
            candidates.append(("ssm_fit", lam, fit.particles, fit,
                               np.linalg.inv(fit.transform) if estimate else known))

        true_local = apply_transform(known, shapes[i])
        if not args.skip_mesh:
            ref = train[int(np.argmin([np.linalg.norm(shapes[k] - model.mean) for k in train]))]
            ref_mesh = read_vtk(subjects[ref].groomed).transformed(subjects[ref].local_to_world)
            true_mesh = read_vtk(held.groomed)

        for method, lam, particles, fit, to_local in candidates:
            row = {"subject": held.name, "method": method, "lam": lam, "correspondence": corr,
                   "pose": args.pose, "n_modes": model.n_modes, "n_landmarks": idx.size,
                   "scale_error": float(np.cbrt(np.linalg.det(to_local[:3, :3]) / np.linalg.det(known[:3, :3])) - 1),
                   "landmark_rms": fit.landmark_rms if fit else np.nan,
                   **particle_errors(apply_transform(to_local, particles), true_local)}
            if not args.skip_mesh:
                mesh = warp_mesh(ref_mesh, shapes[ref], particles).transformed(to_local)
                row["reference"] = subjects[ref].name
                row.update(surface_distance_metrics(mesh, true_mesh))
                if args.save_meshes and method == "ssm_fit":
                    write_vtk(args.out / "meshes" / f"{held.name}_lam{lam:g}.vtk", mesh)
            rows.append(row)

        best = min((r for r in rows[-len(candidates):] if r["method"] == "ssm_fit"),
                   key=lambda r: r.get("surf_mean", r["particle_rms"]))
        metric = "surf_mean" if not args.skip_mesh else "particle_rms"
        print(f"  {held.name:14s} best lambda {best['lam']:g}: {metric} {best[metric]:.2f} mm "
              f"({time.time() - t0:.1f} s)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "loo_errors.csv", index=False)
    metrics = [c for c in df.columns if c.startswith(("particle_", "surf_", "hausdorff", "landmark_rms"))]
    summary = df.groupby(["method", "lam"], dropna=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(args.out / "loo_summary.csv", index=False)

    show = [c for c in ("particle_rms", "surf_mean", "surf_p95", "hausdorff") if c in metrics]
    with pd.option_context("display.width", 160, "display.float_format", "{:.2f}".format):
        print("\nmean over held-out subjects (mm):")
        print(df.groupby(["method", "lam"], dropna=False)[show].mean())
    print(f"\nwrote {args.out / 'loo_errors.csv'} and {args.out / 'loo_summary.csv'}")


if __name__ == "__main__":
    main()
