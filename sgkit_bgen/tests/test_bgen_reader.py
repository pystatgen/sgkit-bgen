import numpy.testing as npt
from sgkit_bgen import read_bgen


def test_read_bgen(shared_datadir):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=(100, 500))

    # check some of the data (in different chunks)
    npt.assert_almost_equal(ds["call/dosage"].values[1][0], 1.987, decimal=3)
    npt.assert_almost_equal(ds["call/dosage"].values[100][0], 0.160, decimal=3)


def test_read_bgen_with_sample_file(shared_datadir):
    path = shared_datadir / "complex.bgen"
    ds = read_bgen(path)
    # Check the sample IDs are the ones from the .sample file
    assert ds["sample/id"].values.tolist() == ["s0", "s1", "s2", "s3"]
