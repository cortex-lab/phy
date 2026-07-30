"""Check local Markdown links and MkDocs navigation targets."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
MKDOCS = ROOT / 'mkdocs.yml'

LINK_RE = re.compile(r'(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+["\'][^"\']*["\'])?\)')
HEADING_RE = re.compile(r'^#{1,6}\s+(.+?)\s*#*\s*$', re.MULTILINE)
HTML_ANCHOR_RE = re.compile(r'<a\s+(?:name|id)=["\']([^"\']+)["\']', re.IGNORECASE)
NAV_TARGET_RE = re.compile(r':\s*([A-Za-z0-9_./-]+\.md)\s*$')
SPIKE_LIMIT_NAMES = (
    'n_spikes_waveforms',
    'batch_size_waveforms',
    'n_spikes_features',
    'n_spikes_features_background',
    'n_spikes_amplitudes',
    'n_spikes_correlograms',
)


def slugify(value: str) -> str:
    """Approximate Python-Markdown's heading slug generation."""
    value = re.sub(r'<[^>]+>', '', value)
    value = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', value)
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-')


def anchors(path: Path) -> set[str]:
    text = path.read_text(encoding='utf8')
    out = {slugify(heading) for heading in HEADING_RE.findall(text)}
    out.update(anchor.lower() for anchor in HTML_ANCHOR_RE.findall(text))
    return out


def check_markdown_links() -> list[str]:
    errors = []
    markdown_files = sorted(DOCS.rglob('*.md'))
    anchor_cache = {path.resolve(): anchors(path) for path in markdown_files}

    for source in markdown_files:
        text = source.read_text(encoding='utf8')
        for raw_target in LINK_RE.findall(text):
            target = unquote(raw_target.strip('<>'))
            if target.startswith(('http://', 'https://', 'mailto:', 'ftp://')):
                continue

            path_part, separator, fragment = target.partition('#')
            destination = (source.parent / path_part).resolve() if path_part else source.resolve()
            if not destination.exists():
                errors.append(f'{source.relative_to(ROOT)}: missing link target {target!r}')
                continue
            if destination.suffix.lower() != '.md' or not separator or not fragment:
                continue

            known_anchors = anchor_cache.get(destination)
            if known_anchors is None:
                known_anchors = anchors(destination)
                anchor_cache[destination] = known_anchors
            if fragment.lower() not in known_anchors:
                errors.append(
                    f'{source.relative_to(ROOT)}: missing anchor #{fragment} '
                    f'in {destination.relative_to(ROOT)}'
                )
    return errors


def check_navigation() -> list[str]:
    errors = []
    for line in MKDOCS.read_text(encoding='utf8').splitlines():
        match = NAV_TARGET_RE.search(line)
        if match and not (DOCS / match.group(1)).exists():
            errors.append(f'mkdocs.yml: missing navigation target {match.group(1)!r}')
    return errors


def check_documented_defaults() -> list[str]:
    """Ensure the performance table agrees with controller class defaults."""
    tree = ast.parse((ROOT / 'phy/apps/base.py').read_text(encoding='utf8'))
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.Module)):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in SPIKE_LIMIT_NAMES:
                try:
                    defaults[target.id] = ast.literal_eval(statement.value)
                except (ValueError, TypeError):
                    pass

    errors = []
    performance = (DOCS / 'performance.md').read_text(encoding='utf8')
    for name in SPIKE_LIMIT_NAMES:
        value = defaults.get(name)
        if value is None:
            errors.append(f'phy/apps/base.py: could not find numeric default for {name}')
            continue
        expected = f'{value:,}'
        row = re.compile(rf'\|\s*`{re.escape(name)}`\s*\|\s*{re.escape(expected)}\s*\|')
        if not row.search(performance):
            errors.append(
                f'docs/performance.md: expected default {name}={expected} from phy/apps/base.py'
            )
    return errors


def check_development_version() -> list[str]:
    """Keep the package and unreleased changelog versions synchronized."""
    project_text = (ROOT / 'pyproject.toml').read_text(encoding='utf8')
    package_text = (ROOT / 'phy/__init__.py').read_text(encoding='utf8')
    changelog_text = (DOCS / 'changelog.md').read_text(encoding='utf8')
    project_match = re.search(r'^version\s*=\s*"([^"]+)"', project_text, re.MULTILINE)
    package_match = re.search(r"^__version__\s*=\s*'([^']+)'", package_text, re.MULTILINE)
    changelog_match = re.search(r'^## \[Unreleased\] — ([^\s]+)', changelog_text, re.MULTILINE)
    versions = {
        'pyproject.toml': project_match.group(1) if project_match else None,
        'phy/__init__.py': package_match.group(1) if package_match else None,
        'docs/changelog.md': changelog_match.group(1) if changelog_match else None,
    }
    if None in versions.values() or len(set(versions.values())) != 1:
        return [f'Development versions do not match: {versions}']
    return []


def main() -> int:
    errors = (
        check_navigation()
        + check_markdown_links()
        + check_documented_defaults()
        + check_development_version()
    )
    if errors:
        print('Documentation validation failed:', file=sys.stderr)
        for error in errors:
            print(f'- {error}', file=sys.stderr)
        return 1
    print('Documentation links and navigation targets are valid.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
