"""Compare density datasets, scientific metadata, and storage policies in bounded reads."""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np


def scientific_metadata(value):
    if isinstance(value, dict):
        return {
            key: scientific_metadata(item)
            for key, item in value.items()
            if key not in {"created_utc", "profile_uid"}
        }
    if isinstance(value, list):
        return [scientific_metadata(item) for item in value]
    return value


def compare(before, after):
    count = 0
    with h5py.File(before, "r") as left, h5py.File(after, "r") as right:
        left_names: list[str] = []
        right_names: list[str] = []
        left.visit(left_names.append)
        right.visit(right_names.append)
        assert left_names == right_names

        def check(name, source):
            nonlocal count
            target = right[name]
            assert set(source.attrs) == set(target.attrs), name
            for key in source.attrs:
                if key in {"created_utc", "profile_uid", "linak_version"}:
                    continue
                if key == "metadata_json":
                    assert scientific_metadata(
                        json.loads(source.attrs[key])
                    ) == scientific_metadata(json.loads(target.attrs[key])), name
                else:
                    np.testing.assert_equal(
                        source.attrs[key], target.attrs[key], err_msg=f"{name}:{key}"
                    )
            if not isinstance(source, h5py.Dataset):
                return
            count += 1
            assert (
                source.shape,
                source.dtype,
                source.chunks,
                source.compression,
                source.compression_opts,
                source.shuffle,
            ) == (
                target.shape,
                target.dtype,
                target.chunks,
                target.compression,
                target.compression_opts,
                target.shuffle,
            ), name
            for start in range(0, len(source), 65536) if source.ndim else [0]:
                a = source[start : start + 65536] if source.ndim else source[()]
                b = target[start : start + 65536] if source.ndim else target[()]
                if source.dtype.kind == "f":
                    np.testing.assert_allclose(
                        a,
                        b,
                        rtol=2e-6 if source.dtype.itemsize == 4 else 5e-12,
                        atol=1e-30,
                        equal_nan=True,
                        err_msg=name,
                    )
                else:
                    np.testing.assert_array_equal(a, b, err_msg=name)

        check("/", left)
        left.visititems(check)
    print(
        f"{count} datasets are equivalent; integer data, scientific metadata and storage policies match."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    compare(args.before, args.after)
