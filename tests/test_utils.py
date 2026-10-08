from __future__ import annotations

import numpy as np

from linak.utils import scale_series_with_si_prefix


def test_scale_series_with_si_prefix_handles_multiple_nonfinite_series():
    scaled, unit, factor = scale_series_with_si_prefix(
        [np.asarray([np.nan, 2.0e-6]), np.asarray([np.inf, -4.0e-6])],
        "m",
    )

    assert unit == "um"
    assert factor == 1.0e6
    np.testing.assert_allclose(scaled[0][1], 2.0)
    np.testing.assert_allclose(scaled[1][1], -4.0)
