#!/usr/bin/env python3

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / 'pyproject.toml'
INIT_PATH = REPO_ROOT / 'phy' / '__init__.py'
STATE_DIR = REPO_ROOT / '.release-smoke'
LATEST_TESTPYPI_VERSION_PATH = STATE_DIR / 'latest-testpypi-version.txt'

PROJECT_NAME = 'phy'
TESTPYPI_JSON_URL = f'https://test.pypi.org/pypi/{PROJECT_NAME}/json'
TESTPYPI_PUBLISH_URL = 'https://test.pypi.org/legacy/'
VERSION_RE = re.compile(r'(?m)^version = "([^"]+)"')
INIT_VERSION_RE = re.compile(r"(?m)^__version__ = '([^']+)'")
PYPIRC_PATH = Path.home() / '.pypirc'


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print('+', ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def read_pypirc_section(section: str) -> dict[str, str]:
    if not PYPIRC_PATH.exists():
        return {}

    parser = configparser.RawConfigParser()
    parser.read(PYPIRC_PATH, encoding='utf-8')
    if not parser.has_section(section):
        return {}

    return {key: value.strip() for key, value in parser.items(section)}


def get_publish_token(*env_vars: str) -> str:
    for env_var in env_vars:
        value = os.environ.get(env_var, '').strip()
        if value:
            return value
    return ''


def get_token_from_pypirc(section: str) -> str:
    config = read_pypirc_section(section)
    username = config.get('username', '')
    password = config.get('password', '')
    if username == '__token__' and password:
        return password
    return ''


def get_testpypi_publish_args() -> list[str]:
    token = get_publish_token('TESTPYPI_TOKEN', 'TEST_PYPI_TOKEN', 'UV_PUBLISH_TOKEN')
    if not token:
        token = get_token_from_pypirc('testpypi')
    if not token:
        raise RuntimeError(
            'Missing TestPyPI publish token. Set TESTPYPI_TOKEN, TEST_PYPI_TOKEN, or '
            'UV_PUBLISH_TOKEN, or add username=__token__ and password=<token> under '
            f'[testpypi] in {PYPIRC_PATH}. Username/password uploads are no longer '
            'supported by PyPI/TestPyPI.'
        )
    return [
        '--publish-url',
        TESTPYPI_PUBLISH_URL,
        '--token',
        token,
    ]


def get_pypi_publish_args() -> list[str]:
    token = get_publish_token('PYPI_TOKEN', 'UV_PUBLISH_TOKEN')
    if not token:
        token = get_token_from_pypirc('pypi')
    if not token:
        raise RuntimeError(
            'Missing PyPI publish token. Set PYPI_TOKEN or UV_PUBLISH_TOKEN, or add '
            f'username=__token__ and password=<token> under [pypi] in {PYPIRC_PATH}. '
            'Username/password uploads are no longer supported by PyPI/TestPyPI.'
        )
    return [
        '--token',
        token,
    ]


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def write_text(path: Path, contents: str) -> None:
    path.write_text(contents, encoding='utf-8')


def get_current_version() -> str:
    match = VERSION_RE.search(read_text(PYPROJECT_PATH))
    if not match:
        raise RuntimeError(f'Unable to find project.version in {PYPROJECT_PATH}')
    return match.group(1)


def set_version_in_tree(root: Path, version: str) -> None:
    pyproject_path = root / 'pyproject.toml'
    init_path = root / 'phy' / '__init__.py'

    pyproject = read_text(pyproject_path)
    pyproject_new, pyproject_count = VERSION_RE.subn(f'version = "{version}"', pyproject, count=1)
    if pyproject_count != 1:
        raise RuntimeError(f'Unable to update version in {pyproject_path}')

    init_py = read_text(init_path)
    init_new, init_count = INIT_VERSION_RE.subn(f"__version__ = '{version}'", init_py, count=1)
    if init_count != 1:
        raise RuntimeError(f'Unable to update version in {init_path}')

    write_text(pyproject_path, pyproject_new)
    write_text(init_path, init_new)


def fetch_testpypi_versions() -> set[str]:
    try:
        with urlopen(TESTPYPI_JSON_URL) as response:  # noqa: S310
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return set()
        raise RuntimeError(f'Unable to query TestPyPI: HTTP {exc.code}') from exc
    except URLError as exc:
        raise RuntimeError(f'Unable to query TestPyPI: {exc.reason}') from exc

    releases = payload.get('releases', {})
    return set(releases)


def get_next_dev_version(base_version: str, existing_versions: set[str]) -> str:
    if '.dev' in base_version:
        raise RuntimeError(
            f'Base version {base_version!r} already contains .dev; keep pyproject.toml on the final '
            'candidate version and let this helper derive disposable .devN versions.'
        )

    pattern = re.compile(rf'^{re.escape(base_version)}\.dev(\d+)$')
    max_dev = 0
    for version in existing_versions:
        match = pattern.match(version)
        if match:
            max_dev = max(max_dev, int(match.group(1)))
    return f'{base_version}.dev{max_dev + 1}'


def make_stage_dir() -> Path:
    tmp_root = Path(tempfile.mkdtemp(prefix='phy-release-'))
    stage_dir = tmp_root / 'repo'

    def ignore(path: str, names: list[str]) -> set[str]:
        ignored = set()
        for name in names:
            if name in {'.git', '.venv', '.tox', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.release-smoke',
                        'build', 'dist', '__pycache__'}:
                ignored.add(name)
            elif name.endswith('.egg-info'):
                ignored.add(name)
        return ignored

    shutil.copytree(REPO_ROOT, stage_dir, ignore=ignore)
    return stage_dir


def record_latest_testpypi_version(version: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    write_text(LATEST_TESTPYPI_VERSION_PATH, version + '\n')


def publish_testpypi_dev() -> int:
    base_version = get_current_version()
    existing_versions = fetch_testpypi_versions()
    dev_version = get_next_dev_version(base_version, existing_versions)
    publish_args = get_testpypi_publish_args()

    print(f'Base version:      {base_version}')
    print(f'Chosen dev build:  {dev_version}')

    stage_dir = make_stage_dir()
    try:
        set_version_in_tree(stage_dir, dev_version)
        dist_dir = stage_dir / 'dist'
        run(['uv', 'build', '--out-dir', str(dist_dir)], cwd=stage_dir)
        dist_glob = str(dist_dir / '*')
        run(
            [
                'uv',
                'publish',
                *publish_args,
                dist_glob,
            ],
            cwd=stage_dir,
        )
    finally:
        shutil.rmtree(stage_dir.parent, ignore_errors=True)

    record_latest_testpypi_version(dev_version)
    print()
    print('Published to TestPyPI.')
    print(f'Version: {dev_version}')
    print(f'Recorded in: {LATEST_TESTPYPI_VERSION_PATH}')
    print()
    print('Next step:')
    print(f'  make release-smoke-testpypi RELEASE_SMOKE_VERSION={dev_version}')
    return 0


def publish_pypi() -> int:
    version = get_current_version()
    if '.dev' in version:
        raise RuntimeError(
            f'Refusing to publish dev version {version!r} to PyPI. Set pyproject.toml to the final version first.'
        )
    publish_args = get_pypi_publish_args()

    print(f'Publishing final version to PyPI: {version}')
    run(['uv', 'build'], cwd=REPO_ROOT)
    run(['uv', 'publish', *publish_args], cwd=REPO_ROOT)
    return 0


def print_latest_testpypi_version() -> int:
    if not LATEST_TESTPYPI_VERSION_PATH.exists():
        raise RuntimeError(
            f'No recorded TestPyPI version found at {LATEST_TESTPYPI_VERSION_PATH}. '
            'Run make release-publish-testpypi-dev first.'
        )

    sys.stdout.write(read_text(LATEST_TESTPYPI_VERSION_PATH).strip())
    sys.stdout.write('\n')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('publish-testpypi-dev')
    subparsers.add_parser('publish-pypi')
    subparsers.add_parser('print-latest-testpypi-version')

    args = parser.parse_args(argv)

    if args.command == 'publish-testpypi-dev':
        return publish_testpypi_dev()
    if args.command == 'publish-pypi':
        return publish_pypi()
    if args.command == 'print-latest-testpypi-version':
        return print_latest_testpypi_version()
    raise AssertionError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    raise SystemExit(main())
