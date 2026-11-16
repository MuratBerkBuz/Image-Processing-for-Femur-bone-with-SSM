# WP5 code

Requires numpy, scipy and pandas (`pip install -r requirements.txt`). ShapeWorks and VTK are not needed. Run all scripts from this folder. They work for any ShapeWorks model folder, chosen with `--bone-dir ../Femur_SSM_prealigned` (the default) or `--bone-dir ../Tibia_SSM_prealigned`. Output goes to `../results`.

## Reconstruction app

```bash
python bone_reconstruction_app.py
```

1. Choose the femur or the tibia.
2. Enter the x y z coordinates (mm) of each landmark, all measured in the same coordinate system (e.g. one CT scan). Selecting a row shows where the landmark lies on the 3D bone and how to find it. Values can be typed, pasted (several rows at once), or loaded from a file.
3. The reconstructed bone is shown in the coordinate system of the entered landmarks and saved to `../results/reconstructions/` as `NNN_[<input file>_]<Bone>_<L|R>_reconstructed.stl`, e.g. `001_Tibia_L_reconstructed.stl` or, with landmarks loaded from `z036.txt`, `002_z036_Tibia_L_reconstructed.stl`. NNN counts up from the highest number already in the folder. The entered landmarks are saved next to it as `..._landmarks.txt`. "Save STL as…" stores a copy elsewhere.

The app uses the size-preserving models (`*_SSM_prealigned`): a left femur model and a right tibia model (`side` in `landmark_guide.json`). It estimates the bone's pose with a rigid fit and detects whether the bone is left or right from the handedness of its landmarks. A bone from the other side is reconstructed by mirroring the model, and the app says so. The side can also be set by hand (Body side: Left / Right). It warns about a landmark far from the fitted bone (> 8 mm), a likely wrong landmark order, a shape outside the training range, and a size that suggests wrong units.

Two example inputs are in `examples/`. Open the app on one directly with:

```bash
python bone_reconstruction_app.py --bone Femur --landmarks examples/femur_example.txt --auto
```

The bones offered, their model folders and the landmark descriptions are in `landmark_guide.json`, which can be edited without changing code. Modules: `bone_reconstruction_app.py` (window), `reconstruction_engine.py` (fit, checks, STL), `mesh_view.py` (3D view).

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
| `Femur_SSM`, `Tibia_SSM` | first ShapeWorks models (landmark-groomed, scaled); kept for comparison only, their exports no longer match the re-saved projects |
| `Femur_SSM_prealigned`, `Tibia_SSM_prealigned` | ShapeWorks models built from `data_aligned` (no scaling) |
| `data_femur_raw`, `data_tibia_raw` | raw input STLs |
| `data_aligned/<Bone>` | output of `prealign_stls.py`: aligned STLs + landmarks |
| `files_for_gui` | app icon |

Paths inside the .swproj files must match these folder names. A missing project, export or referenced file stops the scripts with an error that names the file.

## Tests

```bash
python -m pytest            # all tests (data tests skip if a folder is missing)
python -m pytest -m "not data"   # unit tests only, synthetic data, a few seconds
```

`test_exports_match_particles` fails when a model's Studio exports (mean, eigenvectors, eigenvalues) are older than its particles. Re-export them in Studio to fix it. The app and the reconstruction do not depend on these exports: they build the PCA from the particles.

## Notes

- Frames: groomed meshes and `*_local.particles` are in the CT frame. The model frame is `world = procrustes @ alignment_1 @ local`, using the transforms from the .swproj. `Femur_SSM` and `Tibia_SSM` were groomed with landmark alignment, which includes scaling, so their model frame is not in real mm. The `*_prealigned` models keep size (identity transforms). For a new bone the code estimates the matching transform from its landmarks and maps the result back.
- Landmark correspondence: `--correspondence interpolated` (the default for digitized landmarks) models each landmark as an affine combination of its 12 nearest particles. The weights are learned from the training subjects' digitized landmarks, without the held-out subject in LOO. `nearest` snaps each landmark to one particle instead (3–5 mm off with 512 particles).
- λ acts like the landmark noise variance (mm²). λ = 10 was best in LOO for both bones.
- Mesh warp: the reference is the Studio mean mesh if exported, otherwise the (training) subject closest to the mean. Groomed meshes are smoothed: they differ from the raw STLs by about 1.2 mm on average, which sets a floor when comparing with raw STLs.
- Particle correspondences come from the joint ShapeWorks optimization, which included every subject. That leakage is inherent to the exported model.
