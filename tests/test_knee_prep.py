import os, sys, tempfile
import numpy as np, pandas as pd, pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import knee_prep as kp


def _write(path, pos_z, inst, img):
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage; meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = img.shape; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16; ds.BitsStored = 12; ds.HighBit = 11; ds.PixelRepresentation = 0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]; ds.ImagePositionPatient = [0, 0, pos_z]
    ds.InstanceNumber = inst; ds.PixelData = img.astype(np.uint16).tobytes()
    ds.save_as(path)


def test_slices_are_ordered_by_position_not_filename_or_instance_number():
    with tempfile.TemporaryDirectory() as d:
        # slice k has constant value 100*(k+1); file names and InstanceNumbers are scrambled on purpose
        perm = [3, 0, 4, 1, 2]
        for fname_i, k in enumerate(perm):
            _write(os.path.join(d, f"{fname_i}.dcm"), pos_z=float(k), inst=(7 - k) % 5, img=np.full((40, 60), 100 * (k + 1)))
        vol = kp.load_series(d, size=16, depth=5)
        assert vol.shape == (5, 16, 16) and vol.dtype == np.uint8
        means = vol.reshape(5, -1).mean(1)
        assert np.all(np.diff(means) > 0), means  # monotonic increasing along z


def test_choose_series_prefers_fluid_fat_sat_and_skips_missing_planes():
    df = pd.DataFrame({
        "SeriesInstanceUID": ["a", "b", "c", "d"],
        "Fluid_Sensitive": [0, 1, 1, 1], "Fat_Suppression": [0, 0, 1, 1],
        "Anatomical_Plane": ["Sagittal", "Sagittal", "Sagittal", "Coronal"]})
    assert kp.choose_series(df) == {"Sagittal": "c", "Coronal": "d"}


def test_load_study_marks_missing_planes():
    with tempfile.TemporaryDirectory() as d:
        sd = os.path.join(d, "train_series", "S1", "ser1"); os.makedirs(sd)
        for k in range(4):
            _write(os.path.join(sd, f"{k}.dcm"), float(k), k, np.full((32, 32), 50 * (k + 1)))
        df = pd.DataFrame({"SeriesInstanceUID": ["ser1"], "Fluid_Sensitive": [1], "Fat_Suppression": [1], "Anatomical_Plane": ["Axial"]})
        vols, present = kp.load_study(d, "train", "S1", df, size=16, depth=4)
        assert vols.shape == (3, 4, 16, 16) and present.tolist() == [False, False, True]
        assert vols[0].max() == 0 and vols[2].max() > 0


def test_load_study_survives_a_corrupt_series():
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "train_series", "S1", "good"); bad = os.path.join(d, "train_series", "S1", "bad")
        os.makedirs(good); os.makedirs(bad)
        for k in range(4):
            _write(os.path.join(good, f"{k}.dcm"), float(k), k, np.full((32, 32), 50 * (k + 1)))
        open(os.path.join(bad, "0.dcm"), "wb").write(b"not a dicom")
        df = pd.DataFrame({"SeriesInstanceUID": ["good", "bad"], "Fluid_Sensitive": [1, 1], "Fat_Suppression": [1, 1],
                           "Anatomical_Plane": ["Axial", "Sagittal"]})
        vols, present = kp.load_study(d, "train", "S1", df, size=16, depth=4)
        assert present.tolist() == [False, False, True]
