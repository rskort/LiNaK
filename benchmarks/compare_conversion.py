"""Compare trajectory files, excluding only their creation timestamps."""

from __future__ import annotations
import argparse
import h5py
import numpy as np


def compare(before, after):
    datasets = 0
    with h5py.File(before, "r") as left, h5py.File(after, "r") as right:

        def check(name, obj):
            nonlocal datasets
            other = right[name]
            assert set(obj.attrs) == set(other.attrs), name
            for key in obj.attrs:
                if key != "created_utc":
                    np.testing.assert_equal(
                        obj.attrs[key], other.attrs[key], err_msg=f"{name}:{key}"
                    )
            if isinstance(obj, h5py.Dataset):
                datasets += 1
                assert obj.shape == other.shape and obj.dtype == other.dtype, name
                assert (obj.chunks, obj.compression, obj.compression_opts, obj.shuffle) == (
                    other.chunks,
                    other.compression,
                    other.compression_opts,
                    other.shuffle,
                ), name
                if obj.ndim:
                    for start in range(0, len(obj), 256):
                        np.testing.assert_equal(
                            obj[start : start + 256], other[start : start + 256], err_msg=name
                        )
                else:
                    np.testing.assert_equal(obj[()], other[()], err_msg=name)

        left_names: list[str] = []
        right_names: list[str] = []
        left.visit(left_names.append)
        right.visit(right_names.append)
        assert left_names == right_names
        check("/", left)
        left.visititems(check)
    print(
        f"{datasets} datasets and all non-timestamp attributes match exactly; storage policies match."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    compare(args.before, args.after)
