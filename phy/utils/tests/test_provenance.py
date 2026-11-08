"""Regression coverage for checkout identity and durable save provenance."""

import json
import logging
import subprocess
from datetime import datetime

from phy._provenance import (
    _add_log_file,
    _append_curation_save,
    _git_info,
    _git_version,
    _new_session,
)


def test_checkout_identity_and_worktree(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args]).decode().strip()

    git('init', '-q')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.org')
    package = tmp_path / 'phy'
    package.mkdir()
    source = package / '__init__.py'
    source.write_text('version = 1\n')
    git('add', 'phy')
    git('commit', '-qm', 'Initial package')
    revision = git('rev-parse', 'HEAD')
    monkeypatch.chdir(tmp_path.parent)
    assert _git_info(source) == {'commit': revision, 'dirty': False}
    assert _git_version(source) == '+git.' + revision[:8]

    # Wheels installed under somebody else's checkout must not inherit its revision.
    wheel_source = tmp_path / 'site-packages' / 'phy' / '__init__.py'
    wheel_source.parent.mkdir(parents=True)
    wheel_source.write_text('version = 1\n')
    assert _git_info(wheel_source) == {'commit': None, 'dirty': None}
    assert _git_version(wheel_source) == ''

    worktree = tmp_path / 'worktree'
    git('worktree', 'add', '--detach', str(worktree), 'HEAD')
    assert (worktree / '.git').is_file()
    assert _git_info(worktree / 'phy' / '__init__.py')['commit'] == revision

    source.write_text('version = 2\n')
    assert _git_version(source).endswith('.dirty')

    # Source archives remain usable when Git is unavailable.
    monkeypatch.setenv('PATH', '')
    assert _git_version(source) == ''


def test_save_history_and_log(tmp_path):
    handler = _add_log_file(tmp_path / 'phy.log')
    try:
        session = _new_session()
        logging.getLogger('phy').info('Session: %s', session)
        _append_curation_save(tmp_path, session)
        _append_curation_save(tmp_path, session)
    finally:
        logging.getLogger('phy').removeHandler(handler)
        handler.close()
    records = [
        json.loads(line)
        for line in (tmp_path / 'phy_curation_history.jsonl').read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[0]['session_id'] == records[1]['session_id'] == session['session_id']
    assert records[0]['event'] == 'curation_saved'
    assert records[0]['software']['phy']['version']
    assert records[0]['software']['phylib']['version']
    assert datetime.fromisoformat(records[0]['saved_at']).utcoffset().total_seconds() == 0
    log = (tmp_path / 'phy.log').read_text()
    assert log.startswith(session['session_started_at'][:10] + 'T')
    assert 'Z [INFO]' in log
    assert '\x1b' not in log


def test_history_write_failure_warns(tmp_path, caplog):
    (tmp_path / 'phy_curation_history.jsonl').mkdir()
    _append_curation_save(tmp_path, _new_session())
    assert 'Curation saved, but could not append provenance' in caplog.text
