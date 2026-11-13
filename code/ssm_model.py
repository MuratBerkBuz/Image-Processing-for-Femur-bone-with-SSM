"""Point distribution model: x = mu + P b, with landmark fitting (README steps 3-4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ssm_io import read_eigenvalues, read_eigenvectors, read_particles


@dataclass
class FitResult:
    b: np.ndarray                 # (M,) shape parameters
    particles: np.ndarray         # (N, 3) reconstructed particles, world frame
    landmark_residuals: np.ndarray  # (K,) |model landmark - target| in mm
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))  # input frame -> world

    @property
    def particles_input_frame(self) -> np.ndarray:
        """Reconstructed particles in the frame of the input landmarks."""
        Ti = np.linalg.inv(self.transform)
        return self.particles @ Ti[:3, :3].T + Ti[:3, 3]

    @property
    def landmark_rms(self) -> float:
        return float(np.sqrt(np.mean(self.landmark_residuals ** 2)))


class ShapeModel:
    def __init__(self, mean: np.ndarray, modes: np.ndarray, eigenvalues: np.ndarray):
        self.mean = np.asarray(mean, float).reshape(-1, 3)   # (N, 3)
        self.modes = np.asarray(modes, float)                # (3N, M)
        self.eigenvalues = np.asarray(eigenvalues, float)    # (M,)
        n, m = self.mean.shape[0], self.eigenvalues.size
        if self.modes.shape != (3 * n, m):
            raise ValueError(f"modes {self.modes.shape} do not match {n} particles x {m} eigenvalues")

    # ---------------------------------------------------------------- build
    @classmethod
    def from_exports(cls, bone_dir) -> "ShapeModel":
        bone_dir = Path(bone_dir)
        mean_files = sorted((bone_dir / "mean_shape_particles").glob("*.particles"))
        if len(mean_files) != 1:
            raise FileNotFoundError(f"expected one mean .particles file, found {len(mean_files)}")
        return cls(read_particles(mean_files[0]),
                   read_eigenvectors(bone_dir / "eigenvectors"),
                   read_eigenvalues(bone_dir / "eigenvalues" / "eigenvalues.eval"))

    @classmethod
    def from_particles(cls, shapes: np.ndarray, n_modes: int | None = None,
                       var_fraction: float | None = None, rel_tol: float = 1e-8) -> "ShapeModel":
        """PCA of corresponding shapes (S, N, 3), same convention as ShapeWorks
        (eigenvalues = sample variance with S-1)."""
        shapes = np.asarray(shapes, float)
        s = shapes.shape[0]
        X = shapes.reshape(s, -1)
        mu = X.mean(axis=0)
        _, sv, vt = np.linalg.svd(X - mu, full_matrices=False)
        ev = sv ** 2 / (s - 1)
        keep = ev > rel_tol * ev[0]
        if var_fraction is not None:
            cum = np.cumsum(ev) / ev.sum()
            keep &= np.arange(ev.size) <= np.searchsorted(cum, var_fraction)
        if n_modes is not None:
            keep &= np.arange(ev.size) < n_modes
        return cls(mu, vt[keep].T, ev[keep])

    def truncated(self, n_modes: int) -> "ShapeModel":
        return ShapeModel(self.mean, self.modes[:, :n_modes], self.eigenvalues[:n_modes])

    # ---------------------------------------------------------------- basics
    @property
    def n_particles(self) -> int:
        return self.mean.shape[0]

    @property
    def n_modes(self) -> int:
        return self.eigenvalues.size

    def reconstruct(self, b: np.ndarray | None = None) -> np.ndarray:
        if b is None:
            return self.mean.copy()
        b = np.asarray(b, float)
        return self.mean + (self.modes[:, :b.size] @ b).reshape(-1, 3)

    def project(self, shape: np.ndarray) -> np.ndarray:
        """Shape parameters of a full particle set (least squares, modes are orthonormal)."""
        return self.modes.T @ (np.asarray(shape, float).ravel() - self.mean.ravel())

    # ---------------------------------------------------------------- fitting
    def landmark_operator(self, selector: np.ndarray) -> np.ndarray:
        """L as a (K, N) weight matrix: landmarks = L @ particles.

        ``selector`` is either particle indices (K,) - nearest-particle
        correspondence - or a (K, N) weight matrix from landmark_weights().
        """
        selector = np.asarray(selector)
        if selector.ndim == 2:
            return selector.astype(float)
        W = np.zeros((selector.size, self.n_particles))
        W[np.arange(selector.size), selector.astype(int)] = 1.0
        return W

    def _solve_b(self, targets: np.ndarray, W: np.ndarray, lam: float) -> np.ndarray:
        """argmin_b ||L(mu + P b) - l||^2 + lam * b^T Lambda^-1 b."""
        P3 = self.modes.reshape(self.n_particles, 3, self.n_modes)
        A = np.einsum("kn,ncm->kcm", W, P3).reshape(-1, self.n_modes)  # (3K, M)
        r = targets.ravel() - (W @ self.mean).ravel()                   # (3K,)
        if lam > 0:
            reg = np.diag(np.sqrt(lam / self.eigenvalues))
            A = np.vstack([A, reg])
            r = np.concatenate([r, np.zeros(self.n_modes)])
        # lam == 0 with 3K < M is under-determined: lstsq returns the minimum-norm b.
        return np.linalg.lstsq(A, r, rcond=None)[0]

    def fit_landmarks(self, landmarks: np.ndarray, selector: np.ndarray, lam: float,
                      align: str = "none", n_iter: int = 100, tol: float = 1e-7) -> FitResult:
        """Fit b to landmark coordinates (K, 3).

        ``selector``: particle indices (K,) or a (K, N) weight matrix, see
        landmark_operator(). With align='none' the landmarks must already be in
        the model (world) frame. 'rigid' or 'similarity' (rigid + isotropic
        scale, like ShapeWorks' landmark grooming) estimates the transform
        landmarks -> model alternately with b; needs >= 3 non-collinear landmarks.
        """
        if align not in ("none", "rigid", "similarity"):
            raise ValueError(f"unknown align {align!r}")
        landmarks = np.asarray(landmarks, float).reshape(-1, 3)
        W = self.landmark_operator(selector)
        if landmarks.shape[0] != W.shape[0]:
            raise ValueError(f"{landmarks.shape[0]} landmarks but {W.shape[0]} in the selector")
        T = np.eye(4)
        if align != "none":
            if W.shape[0] < 3:
                raise ValueError(f"{align} alignment needs at least 3 landmarks")
            scale = align == "similarity"
            T = kabsch(landmarks, W @ self.mean, scale=scale)
            for _ in range(n_iter):
                tgt = landmarks @ T[:3, :3].T + T[:3, 3]
                b = self._solve_b(tgt, W, lam)
                T_new = kabsch(landmarks, W @ self.reconstruct(b), scale=scale)
                done = np.abs(T_new - T).max() < tol
                T = T_new
                if done:
                    break
        tgt = landmarks @ T[:3, :3].T + T[:3, 3]
        b = self._solve_b(tgt, W, lam)
        x = self.reconstruct(b)
        res = np.linalg.norm(W @ x - tgt, axis=1)
        return FitResult(b=b, particles=x, landmark_residuals=res, transform=T)


def landmark_weights(shapes: np.ndarray, landmarks: np.ndarray, n_neighbors: int = 12,
                     ridge: float = 1e-3) -> np.ndarray:
    """Learn landmark positions as affine combinations of nearby particles.

    shapes (S, N, 3) and landmarks (S, K, 3), world frame, same subjects. For
    each landmark, weights w over its ``n_neighbors`` nearest particles (on the
    mean shape) minimise sum_s ||sum_j w_j x_sj - l_s||^2 + rho ||w - 1/n||^2
    subject to sum(w) = 1, so the landmark follows the surface between
    particles instead of snapping to the nearest one. Returns W (K, N).
    """
    shapes = np.asarray(shapes, float)
    landmarks = np.asarray(landmarks, float)
    mean_shape, mean_lm = shapes.mean(0), landmarks.mean(0)
    n = min(n_neighbors, shapes.shape[1])
    W = np.zeros((landmarks.shape[1], shapes.shape[1]))
    for k in range(landmarks.shape[1]):
        J = np.argsort(np.linalg.norm(mean_shape - mean_lm[k], axis=1))[:n]
        # Centre on the per-subject neighbourhood mean: with sum(w) = 1 this is
        # equivalent and keeps the system well conditioned.
        c = shapes[:, J].mean(axis=1, keepdims=True)                  # (S, 1, 3)
        A = (shapes[:, J] - c).transpose(0, 2, 1).reshape(-1, n)      # (3S, n)
        b = (landmarks[:, k] - c[:, 0]).ravel()                       # (3S,)
        w0 = np.full(n, 1.0 / n)
        rho = ridge * np.trace(A.T @ A) / n
        kkt = np.block([[A.T @ A + rho * np.eye(n), np.ones((n, 1))],
                        [np.ones((1, n)), np.zeros((1, 1))]])
        rhs = np.concatenate([A.T @ b + rho * w0, [1.0]])
        W[k, J] = np.linalg.solve(kkt, rhs)[:n]
    return W


def kabsch(src: np.ndarray, dst: np.ndarray, scale: bool = False) -> np.ndarray:
    """4x4 transform minimising ||s R src + t - dst||^2 (Kabsch; Umeyama when scale=True)."""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, sv, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = float(np.sum(sv * np.diag(D)) / np.sum((src - cs) ** 2)) if scale else 1.0
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = cd - s * R @ cs
    return T


def particle_errors(pred: np.ndarray, true: np.ndarray) -> dict:
    d = np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=1)
    return {"particle_mean": float(d.mean()), "particle_rms": float(np.sqrt((d ** 2).mean())),
            "particle_max": float(d.max())}
