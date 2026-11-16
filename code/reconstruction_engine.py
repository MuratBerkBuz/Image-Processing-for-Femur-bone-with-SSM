"""Bone reconstruction from a few landmarks: the non-GUI core of bone_reconstruction_app.py.

    rec = BoneReconstructor("Femur")
    result = rec.reconstruct(parse_landmark_text(text, rec.n_landmarks))
    result.save_stl("femur.stl")

The model is the size-preserving SSM (``*_SSM_prealigned``). Its PCA is built
from the project's current particles, so it always matches the project even if
the Studio exports are older than the last optimization run. The landmarks can
be in any scanner/CT frame (mm): the pose is estimated with a rigid fit and the
mesh is returned in the frame of the entered landmarks.

A bone from the other body side is recognised by the handedness of its
landmarks (sign of the volume spanned by the first four). It is fitted with the
mirrored landmarks and the result is mirrored back.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fit_landmarks import reconstruct as fit_and_warp
from landmarks import INDEX_FILE_NAME, cohort_landmarks
from mesh_tools import Mesh, read_vtk, warp_mesh, write_stl
from ssm_io import ROOT, apply_transform, default_align, load_cohort, read_landmark_indices, write_particles
from ssm_model import ShapeModel, kabsch, landmark_weights

GUIDE_FILE = Path(__file__).resolve().parent / "landmark_guide.json"
DEFAULT_LAMBDA = 10.0          # LOO optimum for both bones
RESIDUAL_WARN_MM = 8.0         # fitted bone further than this from an entered landmark (typical: 1-7 mm)
OUTLIER_SD = 3.0               # shape parameter beyond this many SD of the training bones
MIN_SEPARATION_MM = 5.0        # two landmarks closer than this are almost certainly a typo
MIRROR = np.diag([-1.0, 1.0, 1.0])


def handedness(lm: np.ndarray) -> float:
    """Signed volume spanned by the first four landmarks; its sign flips for the other body side."""
    return float(np.linalg.det(np.stack([lm[1] - lm[0], lm[2] - lm[0], lm[3] - lm[0]])))


def load_guide(path: Path = GUIDE_FILE) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"landmark guide not found: {path}")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))["bones"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"{path}: invalid landmark guide ({e})") from None


def available_bones(guide: dict | None = None) -> list[str]:
    return list((guide or load_guide()).keys())


def parse_landmark_text(text: str, n_expected: int | None = None) -> np.ndarray:
    """Coordinates from text: one landmark per line, 'x y z' separated by spaces,
    tabs, commas or semicolons. Blank lines and '#' comments are ignored."""
    rows = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].replace(",", " ").replace(";", " ").strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"line {lineno}: expected 3 numbers (x y z), found {len(parts)}: {line!r}")
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            raise ValueError(f"line {lineno}: not a number in {line!r}") from None
    if n_expected is not None and len(rows) != n_expected:
        raise ValueError(f"expected {n_expected} landmarks (one per line), found {len(rows)}")
    return np.asarray(rows, dtype=float).reshape(-1, 3)


@dataclass
class Landmark:
    code: str
    name: str
    description: str
    position: np.ndarray        # on the mean bone, model frame


@dataclass
class Reconstruction:
    bone: str
    mesh: Mesh                  # reconstructed bone, frame of the entered landmarks
    landmarks: np.ndarray       # entered (K, 3)
    fitted_landmarks: np.ndarray  # where the reconstructed bone has them (K, 3)
    residuals: np.ndarray       # |fitted - entered| (K,) mm
    shape_sd: np.ndarray        # shape parameters in SD of the training bones
    to_model: np.ndarray        # 4x4 (mirrored, if `mirrored`) entered frame -> model frame
    codes: list[str]
    mirrored: bool = False      # bone from the other side: fitted mirrored, result mirrored back
    warnings: list[str] = field(default_factory=list)   # something looks wrong
    notes: list[str] = field(default_factory=list)      # information the user must be aware of

    def save_stl(self, path) -> Path:
        path = Path(path)
        write_stl(path, self.mesh)
        return path

    def save_landmarks(self, path) -> Path:
        path = Path(path)
        write_particles(path, self.landmarks)
        return path


class BoneReconstructor:
    def __init__(self, bone: str, model_dir=None, guide: dict | None = None, lam: float = DEFAULT_LAMBDA):
        guide = guide or load_guide()
        if bone not in guide:
            raise ValueError(f"unknown bone {bone!r}; available: {', '.join(guide)}")
        cfg = guide[bone]
        self.bone, self.lam, self.view = bone, lam, cfg.get("view", {})
        self.description = cfg.get("description", "")
        self.model_dir = Path(model_dir) if model_dir else ROOT / cfg["model_dir"]

        subjects, shapes = load_cohort(self.model_dir)
        self.n_subjects = len(subjects)
        self.model = ShapeModel.from_particles(shapes)
        self.align = default_align(subjects)
        codes, idx = read_landmark_indices(self.model_dir / INDEX_FILE_NAME)
        unknown = [c for c in codes if c not in cfg["landmarks"]]
        if unknown:
            raise ValueError(f"{GUIDE_FILE.name}: no description for landmark(s) {unknown} of the {bone} model")

        have, lms = cohort_landmarks(subjects, "local", n_expected=len(idx))
        if len(have) >= 3:   # landmark = learned combination of nearby particles
            self.selector = landmark_weights(shapes[have], lms)
        else:
            self.selector = idx
        W = self.model.landmark_operator(self.selector)

        # Warp reference: the training bone closest to the mean (its groomed mesh carries its particles).
        ref = int(np.argmin(np.linalg.norm(shapes - self.model.mean, axis=(1, 2))))
        s = subjects[ref]
        self.reference = (read_vtk(s.groomed).transformed(s.local_to_world), shapes[ref])
        self.mean_mesh = warp_mesh(*self.reference, self.model.mean)
        self.landmarks = [Landmark(c, cfg["landmarks"][c]["name"], cfg["landmarks"][c]["description"], p)
                          for c, p in zip(codes, W @ self.model.mean)]

    # ------------------------------------------------------------------ helpers
    @property
    def n_landmarks(self) -> int:
        return len(self.landmarks)

    @property
    def codes(self) -> list[str]:
        return [lm.code for lm in self.landmarks]

    def landmark_positions(self) -> np.ndarray:
        return np.stack([lm.position for lm in self.landmarks])

    def view_axes(self, positions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(up, front) unit vectors for displaying the bone, from the guide's view hint."""
        P = dict(zip(self.codes, self.landmark_positions() if positions is None else positions))
        try:
            up = P[self.view["up_to"]] - np.mean([P[c] for c in self.view["up_from"]], axis=0)
            front = P[self.view["front"]] - np.mean(list(P.values()), axis=0)
        except KeyError:
            return np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])
        up /= np.linalg.norm(up)
        front -= front.dot(up) * up
        return up, front / np.linalg.norm(front)

    def is_other_side(self, lm: np.ndarray) -> bool:
        """True if the landmarks have the opposite handedness to the model's."""
        own, model = handedness(lm), handedness(self.landmark_positions())
        if abs(own) > 0.05 * abs(model):
            return np.sign(own) != np.sign(model)
        # nearly flat configuration: decide by which version fits the model better
        P = self.landmark_positions()

        def rms(x):
            return np.sqrt(np.mean(np.sum((apply_transform(kabsch(x, P), x) - P) ** 2, axis=1)))
        return rms(lm @ MIRROR) < rms(lm)

    def validate(self, lm) -> np.ndarray:
        """Count, finite values, not all on one line. Raises ValueError with a readable message."""
        lm = np.asarray(lm, dtype=float)
        if lm.ndim != 2 or lm.shape != (self.n_landmarks, 3):
            raise ValueError(f"{self.bone}: expected {self.n_landmarks} landmarks ({', '.join(self.codes)}), "
                             f"got {lm.shape[0] if lm.ndim == 2 else lm.size}")
        if not np.all(np.isfinite(lm)):
            raise ValueError("all coordinates must be finite numbers")
        sv = np.linalg.svd(lm - lm.mean(0), compute_uv=False)
        if sv[1] < 1e-3 * sv[0]:
            raise ValueError("the landmarks lie on one line; the bone's orientation cannot be determined")
        return lm

    def check_landmarks(self, lm: np.ndarray) -> list[str]:
        """Raise ValueError for unusable input; return warnings for suspicious input.
        ``lm`` must already be on the model's side (see is_other_side)."""
        lm = self.validate(lm)
        P = self.landmark_positions()
        scale = np.cbrt(np.linalg.det(kabsch(lm, P, scale=True)[:3, :3]))   # entered -> model
        if not 0.25 < scale < 4:
            raise ValueError(f"the distances between the landmarks are {1 / scale:.2g} times those of an average "
                             f"{self.bone.lower()}; the coordinates must be in millimetres")
        for i, j in itertools.combinations(range(len(lm)), 2):
            if np.linalg.norm(lm[i] - lm[j]) < MIN_SEPARATION_MM:
                raise ValueError(f"{self.codes[i]} and {self.codes[j]} are only "
                                 f"{np.linalg.norm(lm[i] - lm[j]):.1f} mm apart; check the coordinates")
        warnings = []
        if not 0.7 < scale < 1.4:
            warnings.append(f"The bone is {1 / scale:.2f} times the size of an average {self.bone.lower()} in the "
                            "model. Check that the coordinates are in millimetres.")

        def rigid_rms(x):
            return np.sqrt(np.mean(np.sum((apply_transform(kabsch(x, P), x) - P) ** 2, axis=1)))

        given = rigid_rms(lm)
        if given > 8:
            best = min(itertools.permutations(range(len(lm))), key=lambda p: rigid_rms(lm[list(p)]))
            if list(best) != list(range(len(lm))) and rigid_rms(lm[list(best)]) < 0.5 * given:
                order = ", ".join(self.codes[best.index(k)] for k in range(len(lm)))
                warnings.append(f"The landmarks may be in the wrong order: they fit best as {order} "
                                f"(expected {', '.join(self.codes)}).")
        return warnings

    # ------------------------------------------------------------------ reconstruction
    def reconstruct(self, landmarks, side: str = "auto") -> Reconstruction:
        """side: 'auto' (detect from handedness), 'model' (as entered) or 'other' (mirror)."""
        lm = self.validate(landmarks)
        if side not in ("auto", "model", "other"):
            raise ValueError(f"side must be 'auto', 'model' or 'other', not {side!r}")
        mirrored = {"auto": None, "model": False, "other": True}[side]
        if mirrored is None:
            mirrored = self.is_other_side(lm)
        lm_fit = lm @ MIRROR if mirrored else lm
        warnings = self.check_landmarks(lm_fit)
        fit, _, mesh, to_model = fit_and_warp(self.model, lm_fit, self.selector, self.lam, self.reference,
                                              align=self.align)
        fitted = apply_transform(np.linalg.inv(to_model), self.model.landmark_operator(self.selector) @ fit.particles)
        if mirrored:  # back to the entered side; reflection reverses the triangle winding
            mesh = Mesh(mesh.points @ MIRROR, mesh.faces[:, ::-1].copy())
            fitted = fitted @ MIRROR
        residuals = np.linalg.norm(fitted - lm, axis=1)
        shape_sd = fit.b / np.sqrt(self.model.eigenvalues)

        for c, r in zip(self.codes, residuals):
            if r > RESIDUAL_WARN_MM:
                warnings.append(f"{c}: the reconstructed bone passes {r:.1f} mm from this landmark. "
                                "Check its coordinates.")
        k = int(np.argmax(np.abs(shape_sd)))
        if abs(shape_sd[k]) > OUTLIER_SD:
            warnings.append(f"The shape is outside the range of the training bones (mode {k + 1} at "
                            f"{shape_sd[k]:+.1f} SD). The result is an extrapolation.")
        notes = []
        if mirrored:
            how = "detected from the landmark positions" if side == "auto" else "as selected"
            notes.append(f"Reconstructed as the mirror image of the model ({self.description}): the bone is "
                         f"from the other body side ({how}). If it is not, two landmarks on opposite sides "
                         f"may be swapped (e.g. {self.side_pair_hint()}).")
        return Reconstruction(self.bone, mesh, lm, fitted, residuals, shape_sd, to_model, self.codes,
                              mirrored, warnings, notes)

    def side_pair_hint(self) -> str:
        """Medial/lateral landmark pair whose swap looks like the other side."""
        pairs = {"Femur": ("ME", "LE"), "Tibia": ("MC", "LC")}
        a, b = pairs.get(self.bone, (self.codes[0], self.codes[1]))
        return f"{a} and {b}"
