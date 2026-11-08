from pathlib import Path
import numpy as np
import pytest
from pysrfit.io.base import DetectorInfo, UnsupportedFormatError, HistogramRun
from pysrfit.io.loaders import LOADERS, load_data
from pysrfit.io.asymmetry import (
    asymmetry_from_components,
    detector_group,
    make_asymmetry,
    make_histogram,
)
from pysrfit.models import RunConfig


def test_csv_3col_and_uncertainty_guard(tmp_path):
    file = tmp_path / "data.csv"
    np.savetxt(file, [[0, .2, .01], [1, .1, .01]], delimiter=",")
    s = load_data(file)
    assert np.allclose(s.asymmetry, [.2, .1])
    np.savetxt(file, [[0, .2], [1, .1]], delimiter=",")
    with pytest.raises(UnsupportedFormatError, match="no uncertainties"):
        load_data(file)


def test_histogram_asymmetry_and_alignment():
    t = np.arange(50) * 0.01
    counts = np.vstack([np.zeros_like(t), np.zeros_like(t),
                        np.full_like(t, 110.), np.full_like(t, 90.)])
    data = HistogramRun(t, counts)
    run = RunConfig(options={"forward": ["3"], "backward": ["4"],
                             "t0": ["10", "12"], "background": ["0", "5", "0", "5"],
                             "data": ["14", "30", "16", "32"], "packing": ["2"]})
    # Here all counts are constant; background subtraction is zero, invalid.
    with pytest.raises(ValueError, match="Nonpositive denominator"):
        make_asymmetry(data, run)
    counts[2, :6] = 10
    counts[3, :6] = 10
    s = make_asymmetry(HistogramRun(t, counts), run)
    assert len(s.time_us) == 8
    assert np.allclose(s.asymmetry, (100 - 80)/(100 + 80))
    assert np.all(s.error > 0)
    changed, changed_error = asymmetry_from_components(s, 2.0)
    assert np.allclose(changed, (2 * 100 - 80) / (2 * 100 + 80))
    assert np.all(changed_error > 0)
    assert not np.allclose(changed, s.asymmetry)
    assert np.all(changed > s.asymmetry)


def test_detector_groups_align_each_histogram_by_its_t0():
    time = np.arange(30) * 0.01
    counts = np.zeros((4, 30))
    # The same signal begins at a different raw channel in each detector.
    for detector, t0 in enumerate((5, 7, 6, 8)):
        counts[detector, :] = 2
        counts[detector, t0:t0 + 10] = 12 + detector
    detectors = [DetectorInfo(i + 1, name, t0) for i, (name, t0) in enumerate(
        zip(("F1", "F2", "B1", "B2"), (5, 7, 6, 8)))]
    raw = HistogramRun(time, counts, detectors=detectors)
    run = RunConfig(fittype=2, options={
        "forward": ["1-2"], "backward": ["3", "4"],
        "background": ["0", "3", "0", "3"], "data": ["5", "14", "6", "15"],
        "packing": ["1"],
    })
    spectrum = make_asymmetry(raw, run)
    assert detector_group(raw, ["F1", "2"]) == [0, 1]
    assert len(spectrum.time_us) == 10
    assert spectrum.metadata["t0"] == [5.0, 6.0, 7.0, 8.0]
    assert np.all(np.isfinite(spectrum.asymmetry))


def test_single_histogram_preparation_keeps_raw_counts():
    time = np.arange(20) * 0.1
    counts = np.full((1, 20), 4.0)
    counts[0, 5:] = np.arange(15) + 20
    raw = HistogramRun(time, counts, detectors=[DetectorInfo(1, "Forward", 5, 6, 15)])
    run = RunConfig(fittype=0, options={
        "forward": ["1"], "background": ["0", "3"],
        "data": ["6", "15"], "packing": ["2"],
    })
    spectrum = make_histogram(raw, run)
    assert spectrum.kind == "histogram"
    assert np.allclose(spectrum.counts, [43, 47, 51, 55, 59])
    assert spectrum.metadata["background_per_packed_bin"] == 8
    assert np.allclose(spectrum.error, np.sqrt(spectrum.counts))


def test_unrecognized_binary_does_not_get_misread(tmp_path):
    p = tmp_path / "raw.bin"
    p.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(UnsupportedFormatError, match="No reader"):
        load_data(p)


def test_hdf5_nxdata_subset(tmp_path):
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        pytest.skip(f"h5py is unavailable or binary-incompatible: {exc}")
    path = tmp_path / "run.nxs"
    with h5py.File(path, "w") as f:
        g = f.create_group("entry/data")
        g.attrs["NX_class"] = "NXdata"
        g.attrs["signal"] = "counts"
        time = g.create_dataset("time", data=np.arange(8))
        time.attrs["units"] = "ns"
        g.create_dataset("counts", data=np.ones((4, 8)))
    r = load_data(path)
    assert r.counts.shape == (4, 8)
    assert np.isclose(r.time_us[1], .001)


def test_psi_gps_musrroot_fixture():
    pytest.importorskip("uproot")
    path = Path(__file__).with_name("deltatbc_tdc_gps_2025_3093.root")
    run = load_data(path)
    assert [loader.name for loader in LOADERS][:2] == ["MusrRoot", "NeXus NXdata"]
    assert run.counts.shape == (15, 102401)
    assert run.metadata["facility"] == "PSI"
    assert run.metadata["run_number"] == 3093
    assert run.metadata["sample"] == "KCoAsO4-R3"
    assert run.channel_width_us == pytest.approx(0.00009765625)
    assert [(detector.number, detector.name) for detector in run.detectors[:2]] == [
        (1, "Forw"), (2, "Back")]
    assert [detector.t0_bin for detector in run.detectors[:2]] == [1610.0, 1602.0]

    config = RunConfig(fittype=2, options={
        "forward": ["1"], "backward": ["2"],
        "background": ["0", "1400", "0", "1400"],
        "data": ["1621", "50000", "1613", "49992"], "packing": ["16"],
    })
    spectrum = make_asymmetry(run, config)
    assert len(spectrum.time_us) > 1000
    assert np.all(np.isfinite(spectrum.asymmetry))
    assert np.all(spectrum.error > 0)
