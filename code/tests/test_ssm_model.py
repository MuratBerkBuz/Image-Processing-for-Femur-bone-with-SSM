import numpy as np
import pytest

from conftest import cohort_axes, ellipsoid, landmark_vertices, random_rotation
from ssm_io import apply_transform
from ssm_model import ShapeModel, kabsch, landmark_weights, particle_errors


@pytest.fixture
def shapes(rng):
    return np.stack([ellipsoid(ax).points for ax in cohort_axes(15, rng)])


@pytest.fixture
def model(shapes):
    return ShapeModel.from_particles(shapes)


def test_pca_of_linear_cohort(model, shapes):
    assert model.n_modes == 3  # axes vary in 3 directions
    assert np.allclose(model.modes.T @ model.modes, np.eye(3), atol=1e-10)
    assert np.all(np.diff(model.eigenvalues) <= 0)
    X = shapes.reshape(len(shapes), -1)
    assert np.isclose(model.eigenvalues.sum(), X.var(axis=0, ddof=1).sum())


def test_project_reconstruct_roundtrip(model, shapes):
    for x in shapes:
        assert np.allclose(model.reconstruct(model.project(x)), x, atol=1e-8)
    assert np.allclose(model.reconstruct(), model.mean)


def test_truncation_and_var_fraction(shapes):
    m = ShapeModel.from_particles(shapes, n_modes=2)
    assert m.n_modes == 2
    assert ShapeModel.from_particles(shapes).truncated(1).n_modes == 1
    assert ShapeModel.from_particles(shapes, var_fraction=0.5).n_modes <= 2


def test_mismatched_modes_rejected():
    with pytest.raises(ValueError):
        ShapeModel(np.zeros((4, 3)), np.zeros((10, 2)), np.ones(2))


def test_fit_exact_from_all_particles(model, shapes):
    idx = np.arange(model.n_particles)
    fit = model.fit_landmarks(shapes[3], idx, lam=0.0)
    assert np.allclose(fit.particles, shapes[3], atol=1e-8)
    assert fit.landmark_rms < 1e-8


def test_fit_from_few_landmarks(model, shapes):
    idx = landmark_vertices()
    fit = model.fit_landmarks(shapes[5][idx], idx, lam=1e-8)
    assert particle_errors(fit.particles, shapes[5])["particle_max"] < 1e-4


def test_regularisation_shrinks_towards_mean(model, shapes):
    idx = landmark_vertices()
    b_small = model.fit_landmarks(shapes[5][idx], idx, lam=1e-6).b
    b_large = model.fit_landmarks(shapes[5][idx], idx, lam=1e6).b
    assert np.linalg.norm(b_large) < 0.1 * np.linalg.norm(b_small)


@pytest.mark.parametrize("align, scale", [("rigid", 1.0), ("similarity", 1.3)])
def test_fit_recovers_pose(model, shapes, rng, align, scale):
    idx = landmark_vertices()
    T = np.eye(4)
    T[:3, :3] = scale * random_rotation(rng)
    T[:3, 3] = [50, -20, 300]
    lm = apply_transform(T, shapes[2][idx])  # landmarks in an unknown frame
    fit = model.fit_landmarks(lm, idx, lam=1e-8, align=align)
    if align == "rigid":
        assert np.allclose(fit.transform @ T, np.eye(4), atol=1e-4)
    # With similarity, a uniform scale is also a shape change in this cohort (all three
    # axes vary), so scale and b trade off: only the geometry is unique.
    assert np.allclose(fit.particles_input_frame, apply_transform(T, shapes[2]), atol=1e-3)


def test_fit_argument_checks(model):
    with pytest.raises(ValueError):
        model.fit_landmarks(np.zeros((3, 3)), np.arange(4), lam=1)
    with pytest.raises(ValueError):
        model.fit_landmarks(np.zeros((2, 3)), np.arange(2), lam=1, align="rigid")
    with pytest.raises(ValueError):
        model.fit_landmarks(np.zeros((4, 3)), np.arange(4), lam=1, align="affine")


def test_landmark_operator(model):
    W = model.landmark_operator(np.array([2, 0]))
    assert W.shape == (2, model.n_particles) and W[0, 2] == 1 and W[1, 0] == 1 and W.sum() == 2
    assert model.landmark_operator(W) is not None


@pytest.mark.parametrize("scale", [False, True])
def test_kabsch_recovers_transform(rng, scale):
    src = rng.normal(size=(10, 3))
    T = np.eye(4)
    T[:3, :3] = (1.7 if scale else 1.0) * random_rotation(rng)
    T[:3, 3] = rng.normal(size=3)
    est = kabsch(src, apply_transform(T, src), scale=scale)
    assert np.allclose(est, T, atol=1e-10)


def test_kabsch_never_reflects(rng):
    src = rng.normal(size=(8, 3))
    dst = src * [-1, 1, 1]  # mirror image
    assert np.linalg.det(kabsch(src, dst)[:3, :3]) > 0


def test_landmark_weights_between_particles(shapes):
    # landmark = midpoint of particles 20 and 21 on every subject
    lms = 0.5 * (shapes[:, 20] + shapes[:, 21])[:, None, :]
    W = landmark_weights(shapes[:-1], lms[:-1], n_neighbors=8, ridge=1e-9)
    assert W.sum() == pytest.approx(1.0)
    assert np.linalg.norm(W @ shapes[-1] - lms[-1]) < 1e-3  # generalises to an unseen subject


def test_particle_errors():
    e = particle_errors(np.zeros((2, 3)), np.array([[3, 4, 0], [0, 0, 0]]))
    assert e["particle_mean"] == 2.5 and e["particle_max"] == 5
