# WP5 code

Requires numpy, scipy and pandas (`pip install -r requirements.txt`). ShapeWorks and VTK are not needed. Run all scripts from this folder. They work for any ShapeWorks model folder, chosen with `--bone-dir ../Femur_SSM` (the default) or `--bone-dir ../Tibia_SSM`. Output goes to `../results`.

| Script | README step | What it does |
|---|---|---|
| `check_model.py` | 1–2 | Checks shapes, orthonormality, and that the PCA, eigenvalues and `pca_scores.csv` (if exported) are reproduced from the world particles. Checks the mean shape against the Studio mean mesh (if exported) and the groomed-frame → model-frame transform for every subject. |
| `landmarks.py` | — | Writes `<bone-dir>/landmark_indices.txt`: `from-coords` matches digitized landmarks to particles; `proxy --n K` picks K spread-out particles as a stand-in. |
| `fit_landmarks.py` | 3–5 | Fits one landmark set and writes the reconstructed mesh (.stl/.vtk) in the landmarks' own frame. `--frame raw` (the default) is for new bones: pose and scale are estimated. `--compare` measures the result against a true mesh. |
| `reconstruct_new_bones.py` | 3–5 | Runs the same for a folder of new bones and compares each with its true mesh. A new bone that is really a training subject is detected and left out of the model. |
| `prealign_stls.py` | — | Rigid (Kabsch, no scaling) landmark pre-alignment of the raw STLs before grooming, with a QC log. |
| `compare_models.py` | — | Compares models of the same cohort: scaling, real bone size, compactness, size modes, and generalization in real mm. |
| `loo_validation.py` | 6–7 | Runs leave-one-out with the PCA rebuilt in every fold. Errors are in mm in each subject's CT frame. `--pose estimate` validates the new-bone case (pose from landmarks). Methods: `ssm_fit` (each λ), `mean_shape` (baseline), `oracle_particles` (the warp floor). |

Library modules: `ssm_io.py` (file formats, project, frames), `ssm_model.py` (PCA, regularized fit, rigid/similarity alignment, landmark interpolation), `mesh_tools.py` (VTK/STL I/O, thin-plate-spline warp, point-to-surface distance).

## Typical run

```bash
python check_model.py --bone-dir ../Tibia_SSM
python landmarks.py --bone-dir ../Tibia_SSM from-coords --frame local
python loo_validation.py --bone-dir ../Tibia_SSM --landmark-source files --pose estimate
python reconstruct_new_bones.py --bone-dir ../Tibia_SSM --landmarks-dir ../new_bones_to_be_reconstructed --actual-dir ../new_bones_to_be_reconstructed/actual_bone_stls
```

Training landmarks are the files the .swproj references (`<bone-dir>/<subject>_landmarks.particles`). Each has one `x y z` row per landmark in the CT frame, in the project's landmark order (femur ME, LE, PN, GT; tibia IN, MC, LC, TT, MM). New-bone landmark files use the same order.

Models built from pre-aligned STLs (`../Femur_SSM_prealigned`, `../Tibia_SSM_prealigned`) reference no landmark files. Their landmarks are picked up automatically from next to the aligned input STLs (`../data_aligned/<Bone>/<subject>_landmarks.particles`). Only the landmark names must be given once:

```bash
python landmarks.py --bone-dir ../Tibia_SSM_prealigned from-coords --names IN MC LC TT MM
python loo_validation.py --bone-dir ../Tibia_SSM_prealigned --landmark-source files --pose estimate
python reconstruct_new_bones.py --bone-dir ../Tibia_SSM_prealigned --landmarks-dir ../new_bones_to_be_reconstructed --actual-dir ../new_bones_to_be_reconstructed/actual_bone_stls
```

Pose estimation (`--align auto`) is similarity for a model groomed with scaling and rigid for a size-preserving one.

## Folder layout

| Folder | Content |
|---|---|
| `Femur_SSM`, `Tibia_SSM` | ShapeWorks models groomed with landmark alignment (scaled) |
| `Femur_SSM_prealigned`, `Tibia_SSM_prealigned` | ShapeWorks models built from `data_aligned` (no scaling) |
| `data_femur_raw`, `data_tibia_raw` | raw input STLs |
| `data_aligned/<Bone>` | output of `prealign_stls.py`: aligned STLs + landmarks |

The .swproj files may still contain folder names from before the renaming. `ssm_io.FOLDER_ALIASES` maps them to the current names, so the projects work whether or not they are re-saved in Studio.

## Tests

```bash
python -m pytest            # all tests (data tests skip if a folder is missing)
python -m pytest -m "not data"   # unit tests only, synthetic data, a few seconds
```

## Notes

- Frames: groomed meshes and `*_local.particles` are in the CT frame. The model frame is `world = procrustes @ alignment_1 @ local`, using the transforms from the .swproj. `Femur_SSM` and `Tibia_SSM` were groomed with landmark alignment, which includes scaling, so their model frame is not in real mm. The `*_prealigned` models keep size (identity transforms). For a new bone the code estimates the matching transform from its landmarks and maps the result back.
- Landmark correspondence: `--correspondence interpolated` (the default for digitized landmarks) models each landmark as an affine combination of its 12 nearest particles. The weights are learned from the training subjects' digitized landmarks, without the held-out subject in LOO. `nearest` snaps each landmark to one particle instead (3–5 mm off with 512 particles).
- λ acts like the landmark noise variance (mm²). λ = 10 was best in LOO for both bones.
- Mesh warp: the reference is the Studio mean mesh if exported, otherwise the (training) subject closest to the mean. Groomed meshes are smoothed: they differ from the raw STLs by about 1.2 mm on average, which sets a floor when comparing with raw STLs.
- Particle correspondences come from the joint ShapeWorks optimization, which included every subject. That leakage is inherent to the exported model.
