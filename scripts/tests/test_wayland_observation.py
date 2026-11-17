"""Do not infer observation success from missing or skipped measurements."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'ordinary_application',
    Path(__file__).parent / 'wayland' / 'application.py',
)
application = importlib.util.module_from_spec(spec)
spec.loader.exec_module(application)


@pytest.mark.parametrize(
    'checks,errors,reason,expected',
    [
        (['PASS'] * 4, [], 'completed', 'PASS'),
        (['PASS', 'PASS', 'PASS', 'SKIP'], [], 'completed', 'INCOMPLETE'),
        (['SKIP', 'PASS', 'PASS', 'PASS'], [], 'completed', 'INCOMPLETE'),
        (['PASS', 'PASS', 'PASS', 'FAIL'], [], 'completed', 'FAIL'),
        (['PASS'] * 4, ['Worker error'], 'completed', 'FAIL'),
        (['PASS'] * 3, [], 'completed', 'INCOMPLETE'),
        (['PASS'] * 4, [], 'hard-timeout', 'INCOMPLETE'),
        ([], [], 'startup', 'INCOMPLETE'),
    ],
)
def test_all_phases_need_observations(checks, errors, reason, expected):
    phases = [{'views': [{'check': check}]} for check in checks]
    assert application._status(phases, errors, reason) == expected
