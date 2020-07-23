import numpy.testing as npt
import pytest
from sgkit_bgen import read_bgen


@pytest.mark.parametrize("chunks", [(100, 200), (100, 500), (199, 500), "auto"])
def test_read_bgen(shared_datadir, chunks):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=chunks)

    # check some of the data (in different chunks)
    assert ds["call/dosage"].shape == (199, 500)
    npt.assert_almost_equal(ds["call/dosage"].values[1][0], 1.987, decimal=3)
    npt.assert_almost_equal(ds["call/dosage"].values[100][0], 0.160, decimal=3)


# def test_read_bgen_with_sample_file(shared_datadir):
#     path = shared_datadir / "complex.bgen"
#     ds = read_bgen(path)
#     # Check the sample IDs are the ones from the .sample file
#     assert ds["sample/id"].values.tolist() == ["s0", "s1", "s2", "s3"]


# def test_read_bgen_with_no_samples(shared_datadir):
#     path = shared_datadir / "complex.23bits.no.samples.bgen"
#     ds = read_bgen(path)
#     # Check the sample IDs are generated
#     assert ds["sample/id"].values.tolist() == [
#         "sample_0",
#         "sample_1",
#         "sample_2",
#         "sample_3",
#     ]
