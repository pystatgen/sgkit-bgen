import numpy.testing as npt
import pytest
from sgkit_bgen import read_bgen


@pytest.mark.parametrize("chunks", [(100, 200), (100, 500), (199, 500), "auto"])
def test_read_bgen(shared_datadir, chunks):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=chunks)

    # check some of the data (in different chunks)
    assert ds["call_dosage"].shape == (199, 500)
    npt.assert_almost_equal(ds["call_dosage"].values[1][0], 1.987, decimal=3)
    npt.assert_almost_equal(ds["call_dosage"].values[100][0], 0.160, decimal=3)


def test_read_bgen_with_sample_file(shared_datadir):
    # The example file was generated using
    # qctool -g sgkit_bgen/tests/data/example.bgen -og sgkit_bgen/tests/data/example-separate-samples.bgen -os sgkit_bgen/tests/data/example-separate-samples.sample -incl-samples sgkit_bgen/tests/data/samples
    # Then editing example-separate-samples.sample to change the sample IDs
    path = shared_datadir / "example-separate-samples.bgen"
    ds = read_bgen(path)
    # Check the sample IDs are the ones from the .sample file
    assert ds["sample_id"].values.tolist() == ["s1", "s2", "s3", "s4", "s5"]


def test_read_bgen_with_no_samples(shared_datadir):
    # The example file was generated using
    # qctool -g sgkit_bgen/tests/data/example.bgen -og sgkit_bgen/tests/data/example-no-samples.bgen -os sgkit_bgen/tests/data/example-no-samples.sample -bgen-omit-sample-identifier-block -incl-samples sgkit_bgen/tests/data/samples
    # Then deleting example-no-samples.sample
    path = shared_datadir / "example-no-samples.bgen"
    ds = read_bgen(path)
    # Check the sample IDs are generated
    assert ds["sample_id"].values.tolist() == [
        "sample_0",
        "sample_1",
        "sample_2",
        "sample_3",
        "sample_4",
    ]
