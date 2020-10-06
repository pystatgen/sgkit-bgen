# Temporary workaround until https://github.com/pangeo-data/rechunker/pull/52 is in and released
# type: ignore

import tempfile
from typing import Mapping, Union

import dask
import xarray
import zarr
from rechunker.api import Rechunked, _get_executor, _setup_array_rechunk
from rechunker.types import Executor
from xarray.backends.zarr import (
    DIMENSION_KEY,
    encode_zarr_attr_value,
    encode_zarr_variable,
    extract_zarr_variable_encoding,
)


def rechunk_dataset(
    source: xarray.Dataset,
    encoding: Mapping,
    max_mem,
    target_store,
    temp_store=None,
    executor: Union[str, Executor] = "dask",
):
    def _encode_zarr_attributes(attrs):
        return {k: encode_zarr_attr_value(v) for k, v in attrs.items()}

    if isinstance(executor, str):
        executor = _get_executor(executor)
    if temp_store:
        temp_group = zarr.group(temp_store)
    else:
        temp_group = zarr.group(
            tempfile.mkdtemp(".zarr", "temp_store_")
        )  # pragma: no cover
    target_group = zarr.group(target_store)
    target_group.attrs.update(_encode_zarr_attributes(source.attrs))

    copy_specs = []
    for variable in source:
        array = source[variable].copy()

        # Update the array encoding with provided parameters and apply it
        has_chunk_encoding = "chunks" in array.encoding
        array.encoding.update(encoding.get(variable, {}))
        array = encode_zarr_variable(array)

        # Determine target chunking for array and remove it prior to
        # validation/extraction ONLY if the array isn't also coming
        # from a Zarr store (otherwise blocks need to be checked for overlap)
        target_chunks = array.encoding.get("chunks")
        if not has_chunk_encoding:
            array.encoding.pop("chunks", None)
        array_encoding = extract_zarr_variable_encoding(
            array, raise_on_invalid=True, name=variable
        )

        # Default to chunking based on array shape if not explicitly provided
        default_chunks = array_encoding.pop("chunks")
        target_chunks = target_chunks or default_chunks

        # Extract array attributes along with reserved property for
        # xarray dimension names
        array_attrs = _encode_zarr_attributes(array.attrs)
        array_attrs[DIMENSION_KEY] = encode_zarr_attr_value(array.dims)

        copy_spec = _setup_array_rechunk(
            dask.array.asarray(array),
            target_chunks,
            max_mem,
            target_group,
            target_options=array_encoding,
            temp_store_or_group=temp_group,
            temp_options=array_encoding,
            name=variable,
        )
        copy_spec.write.array.attrs.update(array_attrs)  # type: ignore
        copy_specs.append(copy_spec)
    plan = executor.prepare_plan(copy_specs)
    return Rechunked(executor, plan, source, temp_group, target_group)
