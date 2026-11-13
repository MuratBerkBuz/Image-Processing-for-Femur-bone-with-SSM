"""README steps 3-5: fit the SSM to one set of landmarks and build the dense mesh.

Works for any ShapeWorks model folder (--bone-dir ../Femur_SSM, ../Tibia_SSM).

  # new bone, landmarks in its own CT frame: the pose and scale relative to
  # the model are estimated (similarity transform, as in ShapeWorks' landmark
  # grooming); the mesh is written back in the landmarks' frame:
  python fit_landmarks.py --bone-dir ../Tibia_SSM --frame raw --lam 10 \
      --landmarks ../new_bones_to_be_reconstructed/01_Tibia_landmarks.particles \
      [--compare ../new_bones_to_be_reconstructed/actual_bone_stls/01_Tibia_R.stl]

  # project subject, using its stored groom alignment:
  python fit_landmarks.py --landmarks ../Femur_SSM/002_Femur_L_landmarks.particles \
      --subject 002_Femur_L --frame local --lam 10

  # landmarks already in the model (world) frame:
  python fit_landmarks.py --landmarks lm.txt --frame world --lam 10

Landmark rows must follow the order of <bone-dir>/landmark_indices.txt.
By default each landmark is interpolated from nearby particles, with weights
learned from the cohort's digitized landmarks (leaving out --subject).

Outputs in <out>/<prefix>_*: particles + mesh in the model frame, and the
mesh in the frame of the input landmarks (.vtk and .stl).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from landmarks import INDEX_FILE_NAME, cohort_landmarks, correspondence
from mesh_tools import Mesh, read_mesh, read_vtk, surface_distance_metrics, warp_mesh, write_mesh
from ssm_io import (DEFAULT_FEMUR_DIR, DEFAULT_RESULTS_DIR, Subject, default_align, find_project, read_landmark_coords,
                    read_landmark_indices, read_project, write_particles)
from ssm_model import FitResult, ShapeModel


def reference_mesh(bone_dir: Path, model: ShapeModel, subjects: list[Subject], ref_subject: str | None = None,
                   exclude: Subject | None = None) -> tuple[Mesh, np.ndarray, str]:
    """Reference mesh + its particles (model frame) for the warp.

    Default: the Studio mean mesh if it was exported, else the groomed mesh of
    the subject closest to the mean shape.
    """
    if ref_subject is None:
        mean_meshes = sorted(Path(bone_dir).glob("mean_shape_mesh*.vtk"))
        if mean_meshes:
            return read_vtk(mean_meshes[0]), model.mean, mean_meshes[0].name
        pool = [s for s in subjects if s is not exclude]
        s = min(pool, key=lambda s: np.linalg.norm(s.load_world_particles() - model.mean))
    else:
        s = next(s for s in subjects if s.name.startswith(ref_subject))
    return read_vtk(s.groomed).transformed(s.local_to_world), s.load_world_particles(), s.name


def landmark_selector(mode: str, idx: np.ndarray, subjects: list[Subject], exclude: Subject | None = None,
                      cohort_frame: str = "local", n_neighbors: int = 12, landmarks_dir=None) -> np.ndarray:
    """Nearest-particle indices, or interpolation weights learned from the cohort's landmark files."""
    if mode == "nearest":
        return idx
    have, lms = cohort_landmarks(subjects, cohort_frame, landmarks_dir, n_expected=len(idx))
    keep = np.array([subjects[i] is not exclude for i in have], dtype=bool)
    if keep.sum() < 3:
        print("too few cohort landmark files to learn the interpolation; using nearest particles")
        return idx
    shapes = np.stack([subjects[i].load_world_particles() for i in have[keep]])
    return correspondence("interpolated", idx, shapes, lms[keep], n_neighbors)


def reconstruct(model: ShapeModel, landmarks: np.ndarray, selector: np.ndarray, lam: float,
                ref: tuple[Mesh, np.ndarray], pre: np.ndarray = np.eye(4),
                align: str = "none") -> tuple[FitResult, Mesh, Mesh, np.ndarray]:
    """Fit + warp. ``pre`` maps the landmark frame to the model frame when it is
    known (stored groom alignment); ``align`` estimates (the rest of) it.
    Returns the fit, the mesh in the model frame, the mesh in the landmark
    frame, and the landmark-frame -> model-frame transform."""
    lm_model = landmarks @ pre[:3, :3].T + pre[:3, 3]
    fit = model.fit_landmarks(lm_model, selector, lam, align=align)
    mesh = warp_mesh(ref[0], ref[1], fit.particles)
    to_model = fit.transform @ pre
    return fit, mesh, mesh.transformed(np.linalg.inv(to_model)), to_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks", type=Path, required=True, help="K rows 'x y z', same order as the index file")
    ap.add_argument("--bone-dir", "--femur-dir", dest="bone_dir", type=Path, default=DEFAULT_FEMUR_DIR,
                    help="ShapeWorks model folder, e.g. ../Femur_SSM or ../Tibia_SSM")
    ap.add_argument("--indices", type=Path, default=None, help="default: <bone-dir>/landmark_indices.txt")
    ap.add_argument("--frame", choices=["raw", "local", "world"], default="raw",
                    help="raw: any frame, pose estimated (new bones); local: groomed/CT frame of --subject; "
                         "world: model frame")
    ap.add_argument("--align", choices=["auto", "none", "rigid", "similarity"], default="auto",
                    help="transform estimated during the fit (auto, for --frame raw: similarity if the model "
                         "was built with scaling, rigid if it keeps size; otherwise none)")
    ap.add_argument("--subject", help="project subject name (needed for --frame local)")
    ap.add_argument("--lam", type=float, default=10.0, help="regularisation weight lambda")
    ap.add_argument("--modes", type=int, default=None, help="use only the first K modes")
    ap.add_argument("--correspondence", choices=["interpolated", "nearest"], default="interpolated")
    ap.add_argument("--neighbors", type=int, default=12, help="particles per interpolated landmark")
    ap.add_argument("--cohort-frame", choices=["local", "world"], default="local",
                    help="frame of the cohort landmark files used to learn the interpolation")
    ap.add_argument("--cohort-landmarks-dir", type=Path, default=None,
                    help="folder with <subject>_landmarks.particles (default: the files the .swproj references)")
    ap.add_argument("--reference", default=None,
                    help="subject whose groomed mesh is warped (default: Studio mean mesh, else subject nearest the mean)")
    ap.add_argument("--compare", type=Path, default=None, help="true mesh (.stl/.vtk) in the landmarks' frame")
    ap.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR / "fits")
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()

    model = ShapeModel.from_exports(args.bone_dir)
    if args.modes:
        model = model.truncated(args.modes)
    subjects, _ = read_project(find_project(args.bone_dir))
    subject = next((s for s in subjects if s.name.startswith(args.subject)), None) if args.subject else None
    if args.frame == "local" and subject is None:
        ap.error("--frame local needs a valid --subject")
    align = args.align if args.align != "auto" else (default_align(subjects) if args.frame == "raw" else "none")
    pre = subject.local_to_world if args.frame == "local" else np.eye(4)

    names, idx = read_landmark_indices(args.indices or args.bone_dir / INDEX_FILE_NAME)
    lm = read_landmark_coords(args.landmarks, n_expected=len(idx))
    sel = landmark_selector(args.correspondence, idx, subjects, subject, args.cohort_frame, args.neighbors,
                            args.cohort_landmarks_dir)
    ref_mesh, ref_pts, ref_name = reference_mesh(args.bone_dir, model, subjects, args.reference, subject)

    fit, mesh, mesh_in, to_model = reconstruct(model, lm, sel, args.lam, (ref_mesh, ref_pts), pre, align)
    scale = float(np.cbrt(np.linalg.det(to_model[:3, :3])))
    print(f"fit with {len(idx)} landmarks, {model.n_modes} modes, lambda={args.lam}, align={align}, "
          f"reference={ref_name}")
    print(f"  landmark frame -> model scale {scale:.3f}")
    for n, r in zip(names, fit.landmark_residuals / scale):
        print(f"  {n:6s} residual {r:6.2f} mm")
    print("  b / sqrt(eigenvalue), first 5 modes:", np.round(fit.b[:5] / np.sqrt(model.eigenvalues[:5]), 2))

    prefix = args.prefix or (subject.name if subject else args.landmarks.stem.replace("_landmarks", ""))
    write_particles(args.out / f"{prefix}_particles_model.particles", fit.particles)
    write_mesh(args.out / f"{prefix}_mesh_model.vtk", mesh)
    write_mesh(args.out / f"{prefix}_reconstructed.vtk", mesh_in)
    write_mesh(args.out / f"{prefix}_reconstructed.stl", mesh_in)
    print(f"wrote {args.out / prefix}_*")

    if args.compare:
        m = surface_distance_metrics(mesh_in, read_mesh(args.compare))
        print(f"vs {args.compare.name}: mean {m['surf_mean']:.2f}  rms {m['surf_rms']:.2f}  "
              f"p95 {m['surf_p95']:.2f}  hausdorff {m['hausdorff']:.2f} mm")


if __name__ == "__main__":
    main()
