"""Package identity and curation provenance (without importing the GUI)."""

import json
import logging
import subprocess
import time
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


def _git_info(package_file):
    """Identify the checkout containing this package, never the caller's checkout."""
    root = Path(package_file).resolve().parent.parent
    if not (root / '.git').exists():
        return {'commit': None, 'dirty': None}

    def git(*args):
        return (
            subprocess.check_output(
                ['git', '-C', str(root), *args], stderr=subprocess.DEVNULL, timeout=2
            )
            .decode('utf8')
            .strip()
        )

    try:
        # A package installed inside another repository is not that repository's source.
        if Path(git('rev-parse', '--show-toplevel')).resolve() != root:
            return {'commit': None, 'dirty': None}
        git('ls-files', '--error-unmatch', str(Path(package_file).resolve().relative_to(root)))
        return {
            'commit': git('rev-parse', 'HEAD'),
            'dirty': bool(git('status', '--porcelain', '--untracked-files=no')),
        }
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}


def _git_version(package_file):
    info = _git_info(package_file)
    if info['commit'] is None:
        return ''
    return '+git.' + info['commit'][:8] + ('.dirty' if info['dirty'] else '')


def _software_info():
    import phylib

    import phy

    return {
        name: {'version': module.__version__, **_git_info(module.__file__)}
        for name, module in (('phy', phy), ('phylib', phylib))
    }


def _timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def _new_session():
    return {
        'schema_version': 1,
        'session_id': str(uuid4()),
        'session_started_at': _timestamp(),
        'software': _software_info(),
    }


def _append_curation_save(dir_path, session):
    """Append only after all curation writes have succeeded."""
    record = {**session, 'event': 'curation_saved', 'saved_at': _timestamp()}
    path = Path(dir_path) / 'phy_curation_history.jsonl'
    try:
        with path.open('a', encoding='utf8') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
    except OSError:
        # The curation has already been saved: distinguish a provenance failure from
        # a failed data save, and leave existing assignments and metadata alone.
        logger.warning(
            'Curation saved, but could not append provenance to %s.', path, exc_info=True
        )


class _FileFormatter(logging.Formatter):
    converter = time.gmtime

    def format(self, record):
        # phylib's console formatter abbreviates levelname on the shared record.
        record = copy(record)
        record.levelname = logging.getLevelName(record.levelno)
        return super().format(record)


def _add_log_file(filename):
    """Create a plain-text file log with full UTC dates."""
    handler = logging.FileHandler(str(filename), encoding='utf8')
    handler.setLevel(logging.DEBUG)
    formatter = _FileFormatter(
        '%(asctime)s.%(msecs)03dZ [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
    )
    handler.setFormatter(formatter)
    logging.getLogger('phy').addHandler(handler)
    return handler
