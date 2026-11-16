import json

import numpy as np
import pytest

from conftest import make_project
from ssm_io import (apply_transform, bone_name, default_align, find_project,
                    landmarks_to_world, load_cohort, model_is_scaled, parse_matrix, read_eigenvectors,
                    read_landmark_coords, read_landmark_indices, read_particles, read_project, select_subjects,
                    transform_scale, write_landmark_indices, write_particles)


def test_particles_roundtrip(tmp_path, rng):
    pts = rng.normal(size=(50, 3)) * 100
    write_particles(tmp_path / "a.particles", pts)
    assert np.allclose(read_particles(tmp_path / "a.particles"), pts, atol=1e-5)


def test_read_particles_rejects_wrong_columns(tmp_path):
    (tmp_path / "bad.particles").write_text("1 2\n3 4\n")
    with pytest.raises(ValueError):
        read_particles(tmp_path / "bad.particles")


def test_landmark_indices_roundtrip(tmp_path):
    write_landmark_indices(tmp_path / "idx.txt", ["ME", "LE"], [3, 17], header="test\nheader")
    names, idx = read_landmark_indices(tmp_path / "idx.txt")
    assert names == ["ME", "LE"] and idx.tolist() == [3, 17]


def test_landmark_indices_bare_numbers(tmp_path):
    (tmp_path / "idx.txt").write_text("# comment\n5\n9  # trailing\n")
    names, idx = read_landmark_indices(tmp_path / "idx.txt")
    assert names == ["L0", "L1"] and idx.tolist() == [5, 9]


def test_landmark_coords_count_check(tmp_path):
    write_particles(tmp_path / "lm.particles", np.zeros((3, 3)))
    with pytest.raises(ValueError):
        read_landmark_coords(tmp_path / "lm.particles", n_expected=4)


def test_eigenvectors_sorted_numerically(tmp_path):
    for k in (0, 1, 2, 10):
        np.savetxt(tmp_path / f"eigenvectors_P_{k}.eval", np.full((2, 3), k))
    P = read_eigenvectors(tmp_path)
    assert P.shape == (6, 4) and P[0].tolist() == [0, 1, 2, 10]


def test_transform_helpers(rng):
    T = parse_matrix("2 0 0 1  0 2 0 2  0 0 2 3  0 0 0 1")
    assert transform_scale(T) == pytest.approx(2.0)
    assert np.allclose(apply_transform(T, [[1, 1, 1]]), [[3, 4, 5]])


@pytest.mark.parametrize("folder, expected", [
    ("Femur_SSM", "Femur"), ("Femur_SSM_prealigned", "Femur"), ("Tibia_SSM", "Tibia"), ("Tibia", "Tibia")])
def test_bone_name(tmp_path, folder, expected):
    assert bone_name(tmp_path / folder) == expected


def test_read_project_resolves_paths(project):
    subjects, proj = read_project(find_project(project["dir"]))
    assert [s.name for s in subjects] == project["names"]
    for s in subjects:
        for p in (s.groomed, s.local_particles, s.world_particles, s.landmarks_file, s.shape):
            assert p is not None and p.exists(), p
    assert [lm["name"] for lm in proj["landmarks"]][:2] == ["A", "B"]


def test_missing_project_files_are_listed(tmp_path):
    info = make_project(tmp_path)
    (info["dir"] / "Bone_particles" / "s03_groomed_world.particles").unlink()
    (info["dir"] / "groomed" / "s05_groomed.vtk").unlink()
    with pytest.raises(FileNotFoundError, match=r"(?s)2 file\(s\).*s03_groomed_world.*s05_groomed"):
        read_project(find_project(info["dir"]))


def test_missing_model_folder_and_project(tmp_path):
    with pytest.raises(FileNotFoundError, match="model folder not found"):
        find_project(tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match="found 0"):
        find_project(tmp_path)


def test_invalid_project_file(tmp_path):
    (tmp_path / "bad.swproj").write_text("{not json")
    with pytest.raises(ValueError, match="not a valid .swproj"):
        read_project(tmp_path / "bad.swproj")
    (tmp_path / "empty.swproj").write_text('{"data": []}')
    with pytest.raises(ValueError, match="no subjects"):
        read_project(tmp_path / "empty.swproj")


def test_missing_and_bad_input_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="particles file not found"):
        read_particles(tmp_path / "none.particles")
    with pytest.raises(FileNotFoundError, match="landmark index file not found"):
        read_landmark_indices(tmp_path / "none.txt")
    (tmp_path / "text.particles").write_text("a b c\n")
    with pytest.raises(ValueError, match="not a list of"):
        read_particles(tmp_path / "text.particles")
    (tmp_path / "empty.particles").write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_particles(tmp_path / "empty.particles")


def test_inconsistent_exports(project):
    from ssm_model import ShapeModel
    np.savetxt(project["dir"] / "eigenvalues" / "eigenvalues.eval", [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="do not fit together"):
        ShapeModel.from_exports(project["dir"])


def test_read_project_empty_name_and_landmarks_next_to_shape(tmp_path):
    info = make_project(tmp_path)
    swproj = find_project(info["dir"])
    proj = json.loads(swproj.read_text())
    for row in proj["data"]:
        row["name"] = ""
        name = row.pop("landmarks_file_1").replace("_landmarks.particles", "")
        (info["dir"] / f"{name}_landmarks.particles").rename(tmp_path / "data_raw" / f"{name}_landmarks.particles")
    swproj.write_text(json.dumps(proj))
    subjects, _ = read_project(swproj)
    assert [s.name for s in subjects] == info["names"]
    assert all(s.landmarks_file.parent.name == "data_raw" for s in subjects)


def test_load_cohort_and_frames(project):
    subjects, shapes = load_cohort(project["dir"])
    assert np.allclose(shapes, project["world"], atol=1e-4)
    for s, A in zip(subjects, project["transforms"]):
        assert np.allclose(s.local_to_world, A, atol=1e-8)
        lm_local = read_landmark_coords(s.landmarks_file)
        lm_world = landmarks_to_world(lm_local, s, "local")
        assert np.allclose(lm_world, s.load_world_particles()[project["lm_idx"]], atol=1e-3)
    with pytest.raises(ValueError):
        landmarks_to_world(lm_local, subjects[0], "nowhere")


def test_select_subjects(project):
    subjects, _ = read_project(find_project(project["dir"]))
    kept = select_subjects(subjects, ["s00", "s01"])
    assert len(kept) == len(subjects) - 2 and all(not s.name.startswith(("s00", "s01")) for s in kept)


def test_default_align(project, scaled_project):
    rigid, _ = read_project(find_project(project["dir"]))
    scaled, _ = read_project(find_project(scaled_project["dir"]))
    assert not model_is_scaled(rigid) and default_align(rigid) == "rigid"
    assert model_is_scaled(scaled) and default_align(scaled) == "similarity"
