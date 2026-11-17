"""Throttled progress reporting for bounded-recording command-line work."""

import signal
import sys
import threading
import time
from concurrent.futures import CancelledError
from numbers import Integral


def _duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f'{hours:d}:{minutes:02d}:{seconds:02d}'
    if minutes:
        return f'{minutes:d}:{seconds:02d}'
    return f'{seconds:d}s'


class ProgressReporter:
    """Render throttled stage progress to stderr and expose cooperative cancel."""

    def __init__(self, stream=None, interval=2.0, clock=time.monotonic):
        if interval < 0:
            raise ValueError('Progress reporting interval must be nonnegative.')
        self.stream = stream if stream is not None else sys.stderr
        self.interval = float(interval)
        self.clock = clock
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._stage = None
        self._stage_started = None
        self._stage_initial_done = 0
        self._last_report = None
        self._last_values = None
        self._previous_sigint = None

    def __enter__(self):
        try:
            self._previous_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_sigint)
        except ValueError:
            # Signal handlers can only be installed by the main thread.
            self._previous_sigint = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._previous_sigint is not None:
            signal.signal(signal.SIGINT, self._previous_sigint)
        self._previous_sigint = None

    def _handle_sigint(self, signum, frame):
        self.cancel()
        raise CancelledError('Operation cancelled.')

    def cancel(self):
        """Request cancellation; the next worker callback raises CancelledError."""
        self._cancelled.set()

    def check_cancelled(self):
        if self._cancelled.is_set():
            raise CancelledError('Operation cancelled.')

    @staticmethod
    def _validate(stage, done, total):
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError('Progress stage must be a nonempty string.')
        for name, value in (('done', done), ('total', total)):
            if value is None and name == 'total':
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f'Progress {name} must be a nonnegative integer or None.')
            if value < 0:
                raise ValueError(f'Progress {name} must be nonnegative.')

    @staticmethod
    def _unit(stage):
        name = stage.lower()
        if any(word in name for word in ('finalizing', 'attaching')):
            return None
        if 'window' in name:
            return 'windows'
        if 'row' in name:
            return 'rows'
        if any(word in name for word in ('byte', 'hash', 'copy', 'authenticating')):
            return 'bytes'
        return 'spikes'

    @staticmethod
    def _number(value):
        return f'{int(value):,}'

    @staticmethod
    def _rate(value):
        return f'{value:,.0f}' if value.is_integer() else f'{value:,.1f}'

    def _message(self, stage, done, total, now):
        unit = self._unit(stage)
        if unit is None:
            return stage
        elapsed = max(now - self._stage_started, 0.0)
        processed = max(done - self._stage_initial_done, 0)
        rate = processed / elapsed if elapsed > 0 else 0.0
        if total:
            percent = min(100.0, 100.0 * done / total)
            message = (
                f'{stage}: {self._number(done)} / {self._number(total)} {unit} ({percent:.1f}%)'
            )
            remaining = max(total - done, 0)
            eta = (
                _duration(remaining / rate)
                if rate > 0
                else ('0s' if not remaining else 'calculating')
            )
            message += f', elapsed {_duration(elapsed)}, {self._rate(rate)} {unit}/s, ETA {eta}'
        else:
            message = f'{stage}: {self._number(done)} {unit}, elapsed {_duration(elapsed)}'
            if rate > 0:
                message += f', {self._rate(rate)} {unit}/s'
        return message

    def __call__(self, stage, done, total):
        self._validate(stage, done, total)
        with self._lock:
            self.check_cancelled()
            now = self.clock()
            stage_changed = stage != self._stage
            if stage_changed:
                self._stage = stage
                self._stage_started = now
                self._stage_initial_done = done
                self._last_report = None
            complete = total is not None and total > 0 and done >= total
            should_report = (
                stage_changed
                or complete
                or self._last_report is None
                or now - self._last_report >= self.interval
            )
            self._last_values = (stage, done, total)
            if should_report:
                self.stream.write(self._message(stage, done, total, now) + '\n')
                self.stream.flush()
                self._last_report = now
