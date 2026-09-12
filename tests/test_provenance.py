"""Provenance checks for measured_constants.yaml.

Every recorded number must have backing raw data, a script that produced it,
and a parseable date. This test fails loudly if anyone records a number without
full provenance.
"""

import datetime
import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONSTANTS_PATH = REPO_ROOT / "config" / "measured_constants.yaml"

REQUIRED_KEYS = {"value", "units", "source", "script", "date"}


def _walk_leaves(d, prefix=""):
    """Yield (dotted_key_path, leaf_dict) for every leaf mapping in the YAML."""
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and REQUIRED_KEYS & v.keys():
            yield path, v
        elif isinstance(v, dict):
            yield from _walk_leaves(v, path)


@pytest.fixture(scope="module")
def constants():
    with open(CONSTANTS_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def leaves(constants):
    return list(_walk_leaves(constants))


def test_yaml_loads(constants):
    assert constants is not None, "measured_constants.yaml failed to parse"


def test_all_leaves_have_required_keys(leaves):
    errors = []
    for path, leaf in leaves:
        missing = REQUIRED_KEYS - leaf.keys()
        if missing:
            errors.append(f"{path}: missing keys {missing}")
    assert not errors, "Leaves with missing keys:\n" + "\n".join(errors)


def test_units_always_present(leaves):
    errors = []
    for path, leaf in leaves:
        units = leaf.get("units")
        if not isinstance(units, str) or not units.strip():
            errors.append(f"{path}: units is missing or empty (got {units!r})")
    assert not errors, "Leaves with bad units:\n" + "\n".join(errors)


def test_measured_entries_have_provenance(leaves):
    errors = []
    for path, leaf in leaves:
        if leaf.get("value") is None:
            continue

        if leaf.get("source") == "PENDING" or not leaf.get("source"):
            errors.append(f"{path}: value is set but source is PENDING or empty")

        source_path = REPO_ROOT / leaf["source"]
        if not source_path.exists():
            errors.append(f"{path}: source path does not exist: {leaf['source']}")

        script = leaf.get("script")
        if script:
            script_path = REPO_ROOT / script
            if not script_path.exists():
                errors.append(f"{path}: script path does not exist: {script}")

        date = leaf.get("date")
        if date is None:
            errors.append(f"{path}: value is set but date is null")
        else:
            try:
                if isinstance(date, str):
                    datetime.date.fromisoformat(date)
                elif not isinstance(date, datetime.date):
                    raise ValueError(f"unexpected type {type(date)}")
            except ValueError as e:
                errors.append(f"{path}: date does not parse as ISO-8601: {date!r} ({e})")

    assert not errors, "Provenance failures:\n" + "\n".join(errors)
