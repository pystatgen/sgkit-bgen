"""BGEN reader implementation (using bgen_reader)"""
import tempfile
from pathlib import Path
from typing import Any, Dict, Hashable, Mapping, MutableMapping, Optional, Tuple, Union

import dask.array as da
import dask.dataframe as dd
import numpy as np
import xarray as xr
import zarr
from bgen_reader._bgen_file import bgen_file
from bgen_reader._bgen_metafile import bgen_metafile
from bgen_reader._metafile import create_metafile
from bgen_reader._reader import infer_metafile_filepath
from bgen_reader._samples import generate_samples, read_samples_file
from rechunker import api as rechunker_api
from xarray import Dataset

from sgkit import create_genotype_dosage_dataset
from sgkit.typing import ArrayLike, DType
from sgkit.utils import encode_array

PathType = Union[str, Path]


def _to_dict(df: dd.DataFrame, dtype: Any = None) -> Dict[str, da.Array]:
    return {
        c: df[c].to_dask_array(lengths=True).astype(dtype[c] if dtype else df[c].dtype)
        for c in df
    }


VARIANT_FIELDS = [
    ("id", str, "U"),
    ("rsid", str, "U"),
    ("chrom", str, "U"),
    ("pos", str, "int32"),
    ("nalleles", str, "int8"),
    ("allele_ids", str, "U"),
    ("vaddr", str, "int64"),
]
VARIANT_DF_DTYPE = dict([(f[0], f[1]) for f in VARIANT_FIELDS])
VARIANT_ARRAY_DTYPE = dict([(f[0], f[2]) for f in VARIANT_FIELDS])

GT_DATA_VARS = [
    "call_genotype_probability",
    "call_genotype_probability_mask",
    "call_dosage",
    "call_dosage_mask",
]


class BgenReader:

    name = "bgen_reader"

    def __init__(
        self, path: PathType, persist: bool = True, dtype: Any = np.float32
    ) -> None:
        self.path = Path(path)

        self.metafile_filepath = infer_metafile_filepath(Path(self.path))
        if not self.metafile_filepath.exists():
            create_metafile(path, self.metafile_filepath, verbose=False)

        with bgen_metafile(self.metafile_filepath) as mf:
            self.n_variants = mf.nvariants
            self.npartitions = mf.npartitions
            self.partition_size = mf.partition_size

            df = mf.create_variants()
            if persist:
                df = df.persist()
            variant_arrs = _to_dict(df, dtype=VARIANT_ARRAY_DTYPE)

            self.variant_id = variant_arrs["id"]
            self.contig = variant_arrs["chrom"]
            self.pos = variant_arrs["pos"]

            def split_alleles(
                alleles: np.ndarray, block_info: Any = None
            ) -> np.ndarray:
                if block_info is None or len(block_info) == 0:
                    return alleles

                def split(allele_row: np.ndarray) -> np.ndarray:
                    alleles_list = allele_row[0].split(",")
                    assert len(alleles_list) == 2  # bi-allelic
                    return np.array(alleles_list)

                return np.apply_along_axis(split, 1, alleles[:, np.newaxis])

            self.variant_alleles = variant_arrs["allele_ids"].map_blocks(split_alleles)

        with bgen_file(self.path) as bgen:
            sample_path = self.path.with_suffix(".sample")
            if sample_path.exists():
                self.sample_id = read_samples_file(sample_path, verbose=False)
            else:
                if bgen.contain_samples:
                    self.sample_id = bgen.read_samples()
                else:
                    self.sample_id = generate_samples(bgen.nsamples)

        self.shape = (self.n_variants, len(self.sample_id), 3)
        self.dtype = dtype
        self.ndim = 3

    def __getitem__(self, idx: Any) -> np.ndarray:
        if not isinstance(idx, tuple):
            raise IndexError(f"Indexer must be tuple (received {type(idx)})")
        if len(idx) != self.ndim:
            raise IndexError(
                f"Indexer must have {self.ndim} items (received {len(idx)} slices)"
            )
        if not all(isinstance(i, slice) or isinstance(i, int) for i in idx):
            raise IndexError(
                f"Indexer must contain only slices or ints (received types {[type(i) for i in idx]})"
            )
        # Determine which dims should have unit size in result
        squeeze_dims = tuple(i for i in range(len(idx)) if isinstance(idx[i], int))
        # Convert all indexers to slices
        idx = tuple(slice(i, i + 1) if isinstance(i, int) else i for i in idx)

        if idx[0].start == idx[0].stop:
            return np.empty((0,) * self.ndim, dtype=self.dtype)

        # Determine start and end partitions that correspond to the
        # given variant dimension indexer
        start_partition = idx[0].start // self.partition_size
        start_partition_offset = idx[0].start % self.partition_size
        end_partition = (idx[0].stop - 1) // self.partition_size
        end_partition_offset = (idx[0].stop - 1) % self.partition_size

        # Create a list of all offsets into the underlying file at which
        # data for each variant begins
        all_vaddr = []
        with bgen_metafile(self.metafile_filepath) as mf:
            for i in range(start_partition, end_partition + 1):
                partition = mf.read_partition(i)
                start_offset = start_partition_offset if i == start_partition else 0
                end_offset = (
                    end_partition_offset + 1
                    if i == end_partition
                    else self.partition_size
                )
                vaddr = partition["vaddr"].tolist()
                all_vaddr.extend(vaddr[start_offset:end_offset])

        # Read the probabilities for each variant, apply indexer for
        # samples dimension to give probabilities for all genotypes,
        # and then apply final genotype dimension indexer
        with bgen_file(self.path) as bgen:
            res = None
            for i, vaddr in enumerate(all_vaddr):
                probs = bgen.read_genotype(vaddr)["probs"][idx[1]]
                assert len(probs.shape) == 2 and probs.shape[1] == 3
                if res is None:
                    res = np.zeros((len(all_vaddr), len(probs), 3), dtype=self.dtype)
                res[i] = probs
            res = res[..., idx[2]]  # type: ignore[index]
            return np.squeeze(res, axis=squeeze_dims)


def _to_dosage(probs: ArrayLike) -> ArrayLike:
    """Calculate the dosage from genotype likelihoods (probabilities)"""
    assert (
        probs.shape[-1] == 3
    ), f"Expecting genotype (trailing) dimension of size 3, got array of shape {probs.shape}"
    return probs[..., 1] + 2 * probs[..., 2]


def read_bgen(
    path: PathType,
    chunks: Union[str, int, Tuple[int, ...]] = "auto",
    lock: bool = False,
    persist: bool = True,
    dtype: Any = "float32",
) -> Dataset:
    """Read BGEN dataset.

    Loads a single BGEN dataset as dask arrays within a Dataset
    from a bgen file.

    Parameters
    ----------
    path : PathType
        Path to BGEN file.
    chunks : Union[str, int, tuple], optional
        Chunk size for genotype probability data (3 dimensions),
        by default "auto".
    lock : bool, optional
        Whether or not to synchronize concurrent reads of
        file blocks, by default False. This is passed through to
        [dask.array.from_array](https://docs.dask.org/en/latest/array-api.html#dask.array.from_array).
    persist : bool, optional
        Whether or not to persist variant information in
        memory, by default True.  This is an important performance
        consideration as the metadata file for this data will
        be read multiple times when False.
    dtype : Any
        Genotype probability array data type, by default float32.

    Warnings
    --------
    Only bi-allelic, diploid BGEN files are currently supported.
    """

    bgen_reader = BgenReader(path, persist, dtype=dtype)

    variant_contig, variant_contig_names = encode_array(bgen_reader.contig.compute())
    variant_contig_names = list(variant_contig_names)
    variant_contig = variant_contig.astype("int16")
    variant_position = np.asarray(bgen_reader.pos, dtype=int)
    variant_alleles = np.asarray(bgen_reader.variant_alleles, dtype="S")
    variant_id = np.asarray(bgen_reader.variant_id, dtype=str)
    sample_id = np.asarray(bgen_reader.sample_id, dtype=str)

    call_genotype_probability = da.from_array(
        bgen_reader,
        chunks=chunks,
        lock=lock,
        fancy=False,
        asarray=False,
        name=f"{bgen_reader.name}:read_bgen:{path}",
    )
    call_dosage = _to_dosage(call_genotype_probability)

    ds: Dataset = create_genotype_dosage_dataset(
        variant_contig_names=variant_contig_names,
        variant_contig=variant_contig,
        variant_position=variant_position,
        variant_alleles=variant_alleles,
        sample_id=sample_id,
        call_dosage=call_dosage,
        call_genotype_probability=call_genotype_probability,
        variant_id=variant_id,
    )

    return ds


def encode_variables(
    ds: Dataset,
    chunk_length: int,
    chunk_width: int,
    compressor: Optional[Any] = zarr.Blosc(cname="zstd", clevel=7, shuffle=2),
    probability_dtype: Optional[Any] = "uint8",
) -> Dict[Hashable, Dict[str, Any]]:
    encoding = {}
    for v in ds:
        e = {}
        if compressor is not None:
            e.update({"compressor": compressor})
        if v in GT_DATA_VARS:
            e.update({"chunks": (chunk_length, chunk_width) + ds[v].shape[2:]})
        if probability_dtype is not None and v == "call_genotype_probability":
            dtype = np.dtype(probability_dtype)
            # Xarray will decode into float32 so any int greater than
            # 16 bits will cause overflow/underflow
            # See https://en.wikipedia.org/wiki/Floating-point_arithmetic#Internal_representation
            # *bits precision column for single precision floats
            if dtype not in [np.uint8, np.uint16]:
                raise ValueError(
                    "Probability integer dtype invalid, must "
                    f"be uint8 or uint16 not {probability_dtype}"
                )
            divisor = np.iinfo(dtype).max - 1
            e.update(
                {
                    "dtype": probability_dtype,
                    "add_offset": -1.0 / divisor,
                    "scale_factor": 1.0 / divisor,
                    "_FillValue": 0,
                }
            )
        if e:
            encoding[v] = e
    return encoding


def pack_variables(ds: Dataset) -> Dataset:
    # Remove dosage as it is unnecessary and should be redefined
    # based on encoded probabilities later (w/ reduced precision)
    ds = ds.drop_vars(["call_dosage", "call_dosage_mask"], errors="ignore")

    # Remove homozygous reference GP and redefine mask
    gp = ds["call_genotype_probability"][..., 1:]
    gp_mask = ds["call_genotype_probability_mask"].any(dim="genotypes")
    ds = ds.drop_vars(["call_genotype_probability", "call_genotype_probability_mask"])
    ds = ds.assign(call_genotype_probability=gp, call_genotype_probability_mask=gp_mask)
    return ds


def unpack_variables(ds: Dataset, dtype: DType = "float32") -> Dataset:
    # Restore homozygous reference GP
    gp = ds["call_genotype_probability"].astype(dtype)  # type: ignore[no-untyped-call]
    if gp.sizes["genotypes"] != 2:
        raise ValueError(
            "Expecting variable 'call_genotype_probability' to have genotypes "
            f"dimension of size 2 (received sizes = {dict(gp.sizes)})"
        )
    ds = ds.drop_vars("call_genotype_probability")
    ds["call_genotype_probability"] = xr.concat(
        [1 - gp.sum(dim="genotypes", skipna=False), gp], dim="genotypes"
    )

    # Restore dosage
    ds["call_dosage"] = gp[..., 0] + 2 * gp[..., 1]
    ds["call_dosage_mask"] = ds["call_genotype_probability_mask"]
    ds["call_genotype_probability_mask"] = ds[
        "call_genotype_probability_mask"
    ].broadcast_like(ds["call_genotype_probability"])
    return ds


def rechunk_bgen(
    ds: Dataset,
    output: Union[PathType, MutableMapping[str, bytes]],
    *,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    compressor: Optional[Any] = zarr.Blosc(cname="zstd", clevel=7, shuffle=2),
    probability_dtype: Optional[DType] = "uint8",
    max_mem: str = "4GB",
    pack: bool = True,
    tempdir: Optional[PathType] = None,
) -> Dataset:
    """Rechunk BGEN dataset as Zarr.

    This function will use the algorithm https://rechunker.readthedocs.io/en/latest/
    to rechunk certain fields in a provided Dataset for better downstream performance.
    Depending on the system memory available (and the `max_mem` setting) this
    rechunking may occur without the need of any intermediate data store. Otherwise,
    approximately as much disk space is required as was needed to store the original
    bgen data. Experiments show that this Zarr representation is ~20% larger even
    with all available optimizations and fairly aggressive compression (i.e. the
    default `clevel` 7).

    Note that this function is not evaluated lazily. The rechunking algorithm
    will run inline so calls to it may be slow. The resulting Dataset is
    generated based on the final, serialized Zarr data.

    Parameters
    ----------
    ds : Dataset
        Dataset to rechunk, typically the result from `read_bgen`.
    output : Union[PathType, MutableMapping[str, bytes]]
        Zarr store or path to directory in file system.
    chunk_length : int
        Length (number of variants) of chunks in which data are stored, by default 10_000.
    chunk_width : int
        Width (number of samples) to use when storing chunks in output, by default 1_000.
    compressor : Optional[Any]
        Zarr compressor, no compression is used when set as None.
    probability_dtype : DType
        Data type used to encode genotype probabilities, must be either uint8 or uint16.
        Setting this parameter results in a loss of precision. If None, probabilities
        will not be altered when stored.
    max_mem : str
        The amount of memory (in bytes) that workers are allowed to use. A string
        (e.g. 100MB) can also be used.
    pack : bool
        Whether or not to optimize variable representations by removing unnecessary
        dimensions and elements. This includes storing 2 genotypes instead of 3, omitting
        dosage and collapsing the genotype probability mask to 2 dimensions. All of
        the above are restored in the resulting Dataset at the expense of extra
        computations on read.
    tempdir : Optional[PathType]
        Temporary directory where intermediate files are stored. The default None means
        use the system default temporary directory.

    Warnings
    --------
    This functional is only applicable to diploid, bi-allelic bgen datasets.

    Returns
    -------
    Dataset
        The rechunked dataset.
    """
    if isinstance(output, Path):
        output = str(output)

    chunk_length = min(chunk_length, ds.dims["variants"])
    chunk_width = min(chunk_width, ds.dims["samples"])

    if pack:
        ds = pack_variables(ds)

    encoding = encode_variables(
        ds,
        chunk_length=chunk_length,
        chunk_width=chunk_width,
        compressor=compressor,
        probability_dtype=probability_dtype,
    )
    with tempfile.TemporaryDirectory(
        prefix="bgen_to_zarr_", suffix=".zarr", dir=tempdir
    ) as tmpdir:
        rechunked = rechunker_api.rechunk_dataset(
            ds,
            encoding=encoding,
            max_mem=max_mem,
            target_store=output,
            temp_store=tmpdir,
            executor="dask",
        )
        rechunked.execute()

    ds: Dataset = xr.open_zarr(output, concat_characters=False)  # type: ignore[no-untyped-call]
    if pack:
        ds = unpack_variables(ds)

    return ds


def bgen_to_zarr(
    input: PathType,
    output: Union[PathType, MutableMapping[str, bytes]],
    region: Optional[Mapping[Hashable, Any]] = None,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    temp_chunk_length: int = 100,
    compressor: Optional[Any] = zarr.Blosc(cname="zstd", clevel=7, shuffle=2),
    probability_dtype: Optional[DType] = "uint8",
    max_mem: str = "4GB",
    pack: bool = True,
    tempdir: Optional[PathType] = None,
) -> Dataset:
    """Rechunk BGEN dataset as Zarr.

    This function will use the algorithm https://rechunker.readthedocs.io/en/latest/
    to rechunk certain fields in a provided Dataset for better downstream performance.
    Depending on the system memory available (and the `max_mem` setting) this
    rechunking may occur without the need of any intermediate data store. Otherwise,
    approximately as much disk space is required as was needed to store the original
    bgen data. Experiments show that this Zarr representation is ~20% larger even
    with all available optimizations and fairly aggressive compression (i.e. the
    default `clevel` 7).

    Note that this function is not evaluated lazily. The rechunking algorithm
    will run inline so calls to it may be slow. The resulting Dataset is
    generated based on the final, serialized Zarr data.

    Parameters
    ----------
    input : PathType
        Path to local bgen dataset.
    output : Union[PathType, MutableMapping[str, bytes]]
        Zarr store or path to directory in file system.
    region : Optional[Mapping[Hashable, Any]]
        Indexers on dataset dimensions used to define a subset of data to convert.
        Must be None or a dict with keys matching dimension names and values
        equal to integers or slice objects. This is passed directly to `Dataset.isel`
        so it has the same semantics.
    chunk_length : int
        Length (number of variants) of chunks in which data are stored, by default 10_000.
    chunk_width : int
        Width (number of samples) to use when storing chunks in output, by default 1_000.
    temp_chunk_length : int
        Length of chunks used in raw bgen read, by default 100. This defines the vertical
        chunking (i.e. in the variants dimension) used when reading the raw data and because
        there is no horizontal chunking at this phase (i.e. in the samples dimension), this
        value should be much smaller than the target `chunk_length`.
    compressor : Optional[Any]
        Zarr compressor, by default Blosc + zstd with compression level 7. No compression
        is used when set as None.
    probability_dtype : DType
        Data type used to encode genotype probabilities, must be either uint8 or uint16.
        Setting this parameter results in a loss of precision. If None, probabilities
        will not be altered when stored.
    max_mem : str
        The amount of memory (in bytes) that workers are allowed to use. A string
        (e.g. 100MB) can also be used.
    pack : bool
        Whether or not to optimize variable representations by removing unnecessary
        dimensions and elements. This includes storing 2 genotypes instead of 3, omitting
        dosage and collapsing the genotype probability mask to 2 dimensions. All of
        the above are restored in the resulting Dataset at the expense of extra
        computations on read.
    tempdir : Optional[PathType]
        Temporary directory where intermediate files are stored. The default None means
        use the system default temporary directory.

    Warnings
    --------
    This functional is only applicable to diploid, bi-allelic bgen datasets.

    Returns
    -------
    Dataset
        The rechunked dataset.
    """
    ds = read_bgen(input, chunks=(temp_chunk_length, -1, -1))
    if region is not None:
        ds = ds.isel(indexers=region)
    return rechunk_bgen(
        ds,
        output,
        chunk_length=chunk_length,
        chunk_width=chunk_width,
        compressor=compressor,
        probability_dtype=probability_dtype,
        max_mem=max_mem,
        pack=pack,
        tempdir=tempdir,
    )
