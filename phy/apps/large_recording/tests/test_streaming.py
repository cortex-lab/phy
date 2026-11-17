"""Memory bounds and exact row mapping for streaming accelerators."""

import numpy as np
import pytest

from phy.apps.large_recording.streaming import UnitRows, bounded_map


@pytest.mark.parametrize('units', [[0, 2, 4], [0, 2, 2147483647]])
def test_unit_rows_compact_and_sparse(units):
    lookup = UnitRows(np.array(units, dtype=np.int64))
    assert (lookup.table is None) == (units[-1] == 2147483647)
    np.testing.assert_array_equal(lookup(np.array(units[::-1])), [2, 1, 0])
    for values in ([-1], [1], [2147483648]):
        with pytest.raises(ValueError, match='catalog mismatch'):
            lookup(np.array(values))
    assert lookup(np.array([], dtype=np.int64)).size == 0


@pytest.mark.parametrize('units', [[2, 1], [1, 1], [-1, 2], [1.5]])
def test_unit_rows_reject_invalid_catalog(units):
    with pytest.raises(ValueError, match='Unit catalog'):
        UnitRows(units)


def test_bounded_ordered_map():
    assert list(bounded_map(lambda x: x * 2, range(10), workers=3)) == list(range(0, 20, 2))
    for workers in (0, 33, 1.5, True):
        with pytest.raises(ValueError, match='workers'):
            list(bounded_map(str, [], workers))
