"""BGEN reader implementation (using bgen_reader)"""
from pathlib import Path
from typing import Any, Union

import dask.array as da
import numpy as np
from bgen_reader._bgen_file import bgen_file
from bgen_reader._bgen_metafile import bgen_metafile
from bgen_reader._metafile import create_metafile
from bgen_reader._reader import _infer_metafile_filepath
from bgen_reader._samples import generate_samples, read_samples_file
from xarray import Dataset

from sgkit import create_genotype_dosage_dataset
from sgkit.typing import ArrayLike
from sgkit.utils import encode_array

PathType = Union[str, Path]


def _to_dict(df, dtype=None):
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


class BgenReader:

    name = "bgen_reader"

    def __init__(self, path, persist=True, dtype=np.float32):
        self.path = Path(path)

        self.metafile_filepath = _infer_metafile_filepath(Path(self.path))
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

            def split_alleles(alleles, block_info=None):
                if block_info is None or len(block_info) == 0:
                    return alleles

                def split(allele_row):
                    alleles_list = allele_row[0].split(",")
                    assert len(alleles_list) == 2  # bi-allelic
                    return np.array(alleles_list)

                return np.apply_along_axis(split, 1, alleles[:, np.newaxis])

            variant_alleles = variant_arrs["allele_ids"].map_blocks(split_alleles)

            def max_str_len(arr: ArrayLike) -> Any:
                return arr.map_blocks(
                    lambda s: np.char.str_len(s.astype(str)), dtype=np.int8
                ).max()

            max_allele_length = max(max_str_len(variant_alleles).compute())
            self.variant_alleles = variant_alleles.astype(f"S{max_allele_length}")

        with bgen_file(self.path) as bgen:
            sample_path = self.path.with_suffix(".sample")
            if sample_path.exists():
                self.sample_id = read_samples_file(sample_path, verbose=False)
            else:
                if bgen.contain_samples:
                    self.sample_id = bgen.read_samples()
                else:
                    self.sample_id = generate_samples(bgen.nsamples)

        self.shape = (self.n_variants, len(self.sample_id))
        self.dtype = dtype
        self.ndim = 2

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            raise IndexError(  # pragma: no cover
                f"Indexer must be tuple (received {type(idx)})"
            )
        if len(idx) != self.ndim:
            raise IndexError(  # pragma: no cover
                f"Indexer must be two-item tuple (received {len(idx)} slices)"
            )

        if idx[0].start == idx[0].stop:
            return np.empty((0, 0), dtype=self.dtype)

        start_partition = idx[0].start // self.partition_size
        start_partition_offset = idx[0].start % self.partition_size
        end_partition = (idx[0].stop - 1) // self.partition_size
        end_partition_offset = (idx[0].stop - 1) % self.partition_size

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

        with bgen_file(self.path) as bgen:
            genotypes = [bgen.read_genotype(vaddr) for vaddr in all_vaddr]
            all_probs = [genotype["probs"] for genotype in genotypes]
            d = [_to_dosage(probs) for probs in all_probs]
            return np.stack(d)[:, idx[1]]


def _to_dosage(probs: ArrayLike):
    """Calculate the dosage from genotype likelihoods (probabilities)"""
    assert len(probs.shape) == 2 and probs.shape[1] == 3
    return 2 * probs[:, -1] + probs[:, 1]


def read_bgen(
    path: PathType,
    chunks: Union[str, int, tuple] = "auto",
    lock: bool = False,
    persist: bool = True,
) -> Dataset:
    """Read BGEN dataset.

    Loads a single BGEN dataset as dask arrays within a Dataset
    from a bgen file.

    Parameters
    ----------
    path : PathType
        Path to BGEN file.
    chunks : Union[str, int, tuple], optional
        Chunk size for genotype data, by default "auto"
    lock : bool, optional
        Whether or not to synchronize concurrent reads of
        file blocks, by default False. This is passed through to
        [dask.array.from_array](https://docs.dask.org/en/latest/array-api.html#dask.array.from_array).
    persist : bool, optional
        Whether or not to persist variant information in
        memory, by default True.  This is an important performance
        consideration as the metadata file for this data will
        be read multiple times when False.

    Warnings
    --------
    Only bi-allelic, diploid BGEN files are currently supported.
    """

    bgen_reader = BgenReader(path, persist)

    variant_contig, variant_contig_names = encode_array(bgen_reader.contig.compute())
    variant_contig_names = list(variant_contig_names)
    variant_contig = variant_contig.astype("int16")

    variant_position = np.array(bgen_reader.pos, dtype=int)
    variant_alleles = np.array(bgen_reader.variant_alleles, dtype="S1")
    variant_id = np.array(bgen_reader.variant_id, dtype=str)

    sample_id = np.array(bgen_reader.sample_id, dtype=str)

    call_dosage = da.from_array(
        bgen_reader,
        chunks=chunks,
        lock=lock,
        asarray=False,
        name=f"{bgen_reader.name}:read_bgen:{path}",
    )

    ds = create_genotype_dosage_dataset(
        variant_contig_names=variant_contig_names,
        variant_contig=variant_contig,
        variant_position=variant_position,
        variant_alleles=variant_alleles,
        sample_id=sample_id,
        call_dosage=call_dosage,
        variant_id=variant_id,
    )

    return ds
