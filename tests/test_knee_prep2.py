import os, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from test_knee_prep import _write
import knee_prep2 as kp2


def _series(d, n, shape=(64, 64), spacing=None):
    perm = np.random.RandomState(0).permutation(n)            # scrambled file names / instance numbers
    for fi, k in enumerate(perm):
        p = os.path.join(d, f"{fi}.dcm"); _write(p, float(k), int((k * 7) % n), np.full(shape, 100 * (k + 1)))
        if spacing:
            import pydicom; ds = pydicom.dcmread(p); ds.PixelSpacing = [spacing, spacing]; ds.save_as(p)


def test_triplets_follow_physical_order_and_have_expected_shape():
    with tempfile.TemporaryDirectory() as d:
        _series(d, 10)
        v = kp2.load_series_v2(d, size=16, n_groups=4)
        assert v.shape == (4, 3, 16, 16) and v.dtype == np.uint8
        flat = v.reshape(4, 3, -1).mean(-1)
        assert np.all(np.diff(flat[:, 1]) > 0), flat           # group centres increase along z
        assert np.all(flat[:, 0] <= flat[:, 1]) and np.all(flat[:, 1] <= flat[:, 2]), flat  # neighbours ordered


def test_fov_crop_uses_physical_size():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "a"))
        rng = np.random.RandomState(1)
        # 100x100 image, 3 mm pixels -> 300 mm field of view; a 150 mm crop is the central 50x50 px.
        # Only that central square is textured; the surround is exactly 0, so a wrong crop would be mostly zeros.
        for k in range(3):
            img = np.zeros((100, 100)); img[25:75, 25:75] = 500 + rng.randint(0, 500, (50, 50))
            _write(os.path.join(d, "a", f"{k}.dcm"), float(k), k, img)
        import pydicom
        for f in os.listdir(os.path.join(d, "a")):
            p = os.path.join(d, "a", f); ds = pydicom.dcmread(p); ds.PixelSpacing = [3.0, 3.0]; ds.save_as(p)
        v = kp2.load_series_v2(os.path.join(d, "a"), size=16, fov_mm=150.0, n_groups=1)
        assert (v == 0).mean() < 0.05, f"crop leaked the zero surround: {(v == 0).mean():.2f} zeros"
        v_wide = kp2.load_series_v2(os.path.join(d, "a"), size=16, fov_mm=300.0, n_groups=1)
        assert (v_wide == 0).mean() > 0.5, "a 300 mm field of view should include the zero surround"


def test_study_marks_missing_and_corrupt_planes():
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "train_series", "S1", "g"); bad = os.path.join(d, "train_series", "S1", "b")
        os.makedirs(good); os.makedirs(bad); _series(good, 5); open(os.path.join(bad, "0.dcm"), "wb").write(b"junk")
        df = pd.DataFrame({"SeriesInstanceUID": ["g", "b"], "Fluid_Sensitive": [1, 1], "Fat_Suppression": [1, 1], "Anatomical_Plane": ["Axial", "Sagittal"]})
        vols, present = kp2.load_study_v2(d, "train", "S1", df, size=16, n_groups=3)
        assert vols.shape == (3, 3, 3, 16, 16) and present.tolist() == [False, False, True]
