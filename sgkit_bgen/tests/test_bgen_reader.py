from pathlib import Path
from typing import Any, Tuple

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from sgkit_bgen import read_bgen
from sgkit_bgen.bgen_reader import (
    GT_DATA_VARS,
    BgenReader,
    rechunk_from_zarr,
    rechunk_to_zarr,
    unpack_variables,
)

CHUNKS = [
    (100, 200, 3),
    (100, 200, 1),
    (100, 500, 3),
    (199, 500, 3),
    ((100, 99), 500, 2),
    "auto",
]
INDEXES = [0, 10, 20, 100, -1]

# Expectations below generated using bgen-reader directly, ex:
# > from bgen_reader import open_bgen
# > bgen = open_bgen('sgkit_bgen/tests/data/example.bgen', verbose=False)
# > bgen.read(-1)[0] # Probabilities for last variant, first sample
# array([[0.0133972 , 0.98135378, 0.00524902]]
# > bgen.allele_expectation(-1)[0, 0, -1] # Dosage for last variant, first sample
# 0.9918518217727197
EXPECTED_PROBABILITIES = np.array(
    [  # Generated using bgen-reader directly
        [np.nan, np.nan, np.nan],
        [0.007, 0.966, 0.0259],
        [0.993, 0.002, 0.003],
        [0.916, 0.007, 0.0765],
        [0.013, 0.981, 0.0052],
    ]
)
EXPECTED_DOSAGES = np.array(
    [np.nan, 1.018, 0.010, 0.160, 0.991]  # Generated using bgen-reader directly
)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_read_bgen(shared_datadir, chunks):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=chunks)

    # check some of the data (in different chunks)
    assert ds["call_dosage"].shape == (199, 500)
    npt.assert_almost_equal(ds["call_dosage"].values[1][0], 1.987, decimal=3)
    npt.assert_almost_equal(ds["call_dosage"].values[100][0], 0.160, decimal=3)
    npt.assert_array_equal(ds["call_dosage_mask"].values[0, 0], [True])
    npt.assert_array_equal(ds["call_dosage_mask"].values[0, 1], [False])
    assert ds["call_genotype_probability"].shape == (199, 500, 3)
    npt.assert_almost_equal(
        ds["call_genotype_probability"].values[1][0], [0.005, 0.002, 0.992], decimal=3
    )
    npt.assert_almost_equal(
        ds["call_genotype_probability"].values[100][0], [0.916, 0.007, 0.076], decimal=3
    )
    npt.assert_array_equal(
        ds["call_genotype_probability_mask"].values[0, 0], [True] * 3
    )
    npt.assert_array_equal(
        ds["call_genotype_probability_mask"].values[0, 1], [False] * 3
    )


def test_read_bgen__with_sample_file(shared_datadir):
    # The example file was generated using
    # qctool -g sgkit_bgen/tests/data/example.bgen -og sgkit_bgen/tests/data/example-separate-samples.bgen -os sgkit_bgen/tests/data/example-separate-samples.sample -incl-samples sgkit_bgen/tests/data/samples
    # Then editing example-separate-samples.sample to change the sample IDs
    path = shared_datadir / "example-separate-samples.bgen"
    ds = read_bgen(path)
    # Check the sample IDs are the ones from the .sample file
    assert ds["sample_id"].values.tolist() == ["s1", "s2", "s3", "s4", "s5"]


def test_read_bgen__with_no_samples(shared_datadir):
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


@pytest.mark.parametrize("chunks", CHUNKS)
def test_read_bgen__fancy_index(shared_datadir, chunks):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=chunks)
    npt.assert_almost_equal(
        ds["call_genotype_probability"][INDEXES, 0], EXPECTED_PROBABILITIES, decimal=3
    )
    npt.assert_almost_equal(ds["call_dosage"][INDEXES, 0], EXPECTED_DOSAGES, decimal=3)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_read_bgen__scalar_index(shared_datadir, chunks):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=chunks)
    for i, ix in enumerate(INDEXES):
        npt.assert_almost_equal(
            ds["call_genotype_probability"][ix, 0], EXPECTED_PROBABILITIES[i], decimal=3
        )
        npt.assert_almost_equal(
            ds["call_dosage"][ix, 0], EXPECTED_DOSAGES[i], decimal=3
        )
        for j in range(3):
            npt.assert_almost_equal(
                ds["call_genotype_probability"][ix, 0, j],
                EXPECTED_PROBABILITIES[i, j],
                decimal=3,
            )


def test_read_bgen__raise_on_invalid_indexers(shared_datadir):
    path = shared_datadir / "example.bgen"
    reader = BgenReader(path)
    with pytest.raises(IndexError, match="Indexer must be tuple"):
        reader[[0]]
    with pytest.raises(IndexError, match="Indexer must have 3 items"):
        reader[(slice(None),)]
    with pytest.raises(IndexError, match="Indexer must contain only slices or ints"):
        reader[([0], [0], [0])]


def _rechunk_to_zarr(
    shared_datadir: Path, tmp_path: Path, **kwargs: Any
) -> Tuple[xr.Dataset, str]:
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=(10, -1, -1))
    store = tmp_path / "example.zarr"
    rechunk_to_zarr(ds, store, **kwargs)
    return ds, str(store)


def _open_zarr(store: str, **kwargs: Any) -> xr.Dataset:
    # Force concat_characters False to avoid to avoid https://github.com/pydata/xarray/issues/4405
    return xr.open_zarr(store, concat_characters=False, **kwargs)  # type: ignore[no-any-return,no-untyped-call]


@pytest.mark.parametrize("chunk_width", [10, 50, 500])
def test_rechunk_to_zarr__chunk_size(shared_datadir, tmp_path, chunk_width):
    _, store = _rechunk_to_zarr(
        shared_datadir, tmp_path, chunk_width=chunk_width, pack=False
    )
    dsr = _open_zarr(store)
    for v in GT_DATA_VARS:
        # Chunks shape should equal (
        #   length of chunks on read,
        #   width of chunks on rechunk
        # )
        assert dsr[v].data.chunksize[0] == 10
        assert dsr[v].data.chunksize[1] == chunk_width


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_rechunk_to_zarr__probability_encoding(shared_datadir, tmp_path, dtype):
    ds, store = _rechunk_to_zarr(
        shared_datadir, tmp_path, probability_dtype=dtype, pack=False
    )
    dsr = _open_zarr(store, mask_and_scale=False)
    v = "call_genotype_probability"
    assert dsr[v].shape == ds[v].shape
    assert dsr[v].dtype == dtype
    dsr = _open_zarr(store, mask_and_scale=True)
    # There are two missing calls which equates to
    # 6 total nan values across 3 possible genotypes
    assert np.isnan(dsr[v].values).sum() == 6
    tolerance = 1.0 / (np.iinfo(dtype).max - 1)
    np.testing.assert_allclose(ds[v], dsr[v], atol=tolerance)


def test_rechunk_to_zarr__variable_packing(shared_datadir, tmp_path):
    ds, store = _rechunk_to_zarr(
        shared_datadir, tmp_path, probability_dtype=None, pack=True
    )
    dsr = _open_zarr(store, mask_and_scale=True)
    dsr = unpack_variables(dsr)
    # A minor tolerance is necessary here when packing is enabled
    # because one of the genotype probabilities is constructed from the others
    xr.testing.assert_allclose(ds.compute(), dsr.compute(), atol=1e-6)  # type: ignore[no-untyped-call]


def test_rechunk_to_zarr__raise_on_invalid_chunk_length(shared_datadir, tmp_path):
    with pytest.raises(
        ValueError,
        match="Chunk size in variant dimension for variable .* must evenly divide target chunk size",
    ):
        _rechunk_to_zarr(shared_datadir, tmp_path, chunk_length=11)


@pytest.mark.parametrize("chunks", [(10, 10), (50, 50), (100, 50), (50, 100)])
def test_rechunk_from_zarr__target_chunks(shared_datadir, tmp_path, chunks):
    ds, store = _rechunk_to_zarr(
        shared_datadir,
        tmp_path,
        chunk_length=chunks[0],
        chunk_width=chunks[1],
        pack=False,
    )
    ds = rechunk_from_zarr(store, chunk_length=chunks[0], chunk_width=chunks[1])
    for v in GT_DATA_VARS:
        assert ds[v].data.chunksize[:2] == chunks


@pytest.mark.parametrize("dtype", ["uint32", "int8", "float32"])
def test_rechunk_from_zarr__invalid_probability_type(shared_datadir, tmp_path, dtype):
    with pytest.raises(ValueError, match="Probability integer dtype invalid"):
        _rechunk_to_zarr(shared_datadir, tmp_path, probability_dtype=dtype)


def test_unpack_variables__invalid_gp_dims(shared_datadir, tmp_path):
    # Validate that an error is thrown when variables are
    # unpacked without being packed in the first place
    _, store = _rechunk_to_zarr(shared_datadir, tmp_path, pack=False)
    dsr = _open_zarr(store, mask_and_scale=True)
    with pytest.raises(
        ValueError,
        match="Expecting variable 'call_genotype_probability' to have genotypes dimension of size 2",
    ):
        unpack_variables(dsr)


def test_rechunk_from_zarr__self_consistent(shared_datadir, tmp_path):
    # With no probability dtype or packing, rechunk_{to,from}_zarr is a noop
    ds, store = _rechunk_to_zarr(
        shared_datadir, tmp_path, probability_dtype=None, pack=False
    )
    dsr = rechunk_from_zarr(store)
    xr.testing.assert_allclose(ds.compute(), dsr.compute())  # type: ignore[no-untyped-call]
