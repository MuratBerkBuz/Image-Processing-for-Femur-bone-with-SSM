"""Compare SSMs of the same cohort built with different alignments.

  python compare_models.py --models ../Femur_SSM ../Femur_SSM_prealigned
  python compare_models.py --models ../Tibia_SSM ../Tibia_SSM_prealigned

Per model:
  scaling     isotropic scale of each subject's groom/Procrustes transform
  size        bone size = centroid size of the particles in the subject's own
              (input) frame, in mm, vs. size in the model frame
  compactness variance per mode, modes for 90/95/99 %
  size modes  correlation of each PC score with the real bone size
  generalization  leave-one-out: rebuild the PCA without a subject, project it
              with k modes, particle error converted to real mm (divided by the
              subject's groom scale)

Writes <out>/model_comparison.csv (one row per model) and
<out>/generalization.csv (error per model and k).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ssm_io import DEFAULT_RESULTS_DIR, apply_transform, load_cohort, transform_scale
from ssm_model import ShapeModel

K_VALUES = (1, 3, 5, 10, 15, 20)


def centroid_size(pts: np.ndarray) -> float:
    return float(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean()))


def analyse(bone_dir: Path, label: str) -> tuple[dict, pd.DataFrame, pd.Series]:
    subjects, shapes = load_cohort(bone_dir)
    names = [s.name for s in subjects]
    scales = np.array([transform_scale(s.local_to_world) for s in subjects])
    real = np.array([centroid_size(apply_transform(s.world_to_local, x)) for s, x in zip(subjects, shapes)])
    model_size = np.array([centroid_size(x) for x in shapes])

    model = ShapeModel.from_particles(shapes)
    ev = model.eigenvalues
    cum = np.cumsum(ev) / ev.sum()
    scores = np.stack([model.project(x) for x in shapes])
    r = np.array([np.corrcoef(scores[:, k], real)[0, 1] for k in range(min(10, model.n_modes))])
    best = int(np.argmax(np.abs(r)))

    gen = {k: [] for k in K_VALUES}
    for i in range(len(subjects)):
        train = [j for j in range(len(subjects)) if j != i]
        m = ShapeModel.from_particles(shapes[train])
        b = m.project(shapes[i])
        for k in K_VALUES:
            kk = min(k, m.n_modes)
            d = np.linalg.norm(m.reconstruct(b[:kk]) - shapes[i], axis=1) / scales[i]
            gen[k].append(np.sqrt((d ** 2).mean()))
    gen_df = pd.DataFrame({"model": label, "k": list(gen), "rms_mm": [np.mean(v) for v in gen.values()],
                           "sd_mm": [np.std(v) for v in gen.values()]})

    row = {
        "model": label, "subjects": len(subjects), "particles": shapes.shape[1], "modes": model.n_modes,
        "scale_min": scales.min(), "scale_max": scales.max(),
        "real_size_mean_mm": real.mean(), "real_size_cv_pct": 100 * real.std() / real.mean(),
        "model_size_cv_pct": 100 * model_size.std() / model_size.mean(),
        "corr_model_size_real_size": np.corrcoef(model_size, real)[0, 1],
        "total_variance": ev.sum(),
        "pc1_pct": 100 * ev[0] / ev.sum(),
        **{f"modes_{int(f * 100)}pct": int(np.searchsorted(cum, f) + 1) for f in (0.9, 0.95, 0.99)},
        "corr_pc1_size": r[0], "size_pc": best + 1, "corr_size_pc": r[best],
        "pc_size_pct_variance": 100 * ev[best] / ev.sum(),
    }
    return row, gen_df, pd.Series(real, index=names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: ../results/compare_<first model>")
    args = ap.parse_args()
    out = args.out or DEFAULT_RESULTS_DIR / f"compare_{args.models[0].resolve().name}"
    out.mkdir(parents=True, exist_ok=True)

    rows, gens, sizes = [], [], []
    labels = [d.resolve().name for d in args.models]
    if len(set(labels)) < len(labels):  # same folder name in different places
        labels = [f"{d.resolve().parent.name}/{d.resolve().name}" for d in args.models]
    for d, label in zip(args.models, labels):
        row, gen, real = analyse(d, label)
        rows.append(row), gens.append(gen), sizes.append(real)
    table = pd.DataFrame(rows).set_index("model")
    gen = pd.concat(gens)
    table.to_csv(out / "model_comparison.csv")
    gen.to_csv(out / "generalization.csv", index=False)

    common = sizes[0].index.intersection(sizes[-1].index)
    if len(sizes) > 1 and len(common):
        diff = (sizes[-1][common] - sizes[0][common]).abs()
        print(f"real bone size agreement between first and last model ({len(common)} subjects): "
              f"max |difference| {diff.max():.2f} mm")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(table.T)
        print("\ngeneralization (LOO particle RMS, real mm) by number of modes k:")
        print(gen.pivot(index="k", columns="model", values="rms_mm"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
