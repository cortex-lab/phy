"""Compatibility for phylib's explicit trusted configuration loading option."""

from inspect import signature


def _config_execution_kwargs(loader, allow_python_execution):
    if not allow_python_execution:
        # Preserve compatibility with released phylib; newer loaders default to safe evaluation.
        return {}
    if 'allow_python_execution' not in signature(loader).parameters:
        raise RuntimeError(
            '--allow-python-execution requires phylib with the trusted configuration API. '
            'Install a phylib version containing cortex-lab/phylib#71; '
            'see the source installation instructions. No fallback was attempted.'
        )
    return {'allow_python_execution': True}
