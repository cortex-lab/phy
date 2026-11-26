"""Small, immutable pending-proposition annotations for displayed unit IDs.

These colors describe source proposition membership, independently of scientific
trace colors, selection order, and ancestry aliases. No spike arrays are read.
"""

from collections.abc import Mapping
from colorsys import hsv_to_rgb
from dataclasses import dataclass
from types import MappingProxyType

WHITE = (1.0, 1.0, 1.0, 1.0)


def proposition_label_color(slot):
    """Return a separate comparison palette whose first two colors are blue/red."""
    if slot == 0:
        return (0.3, 0.65, 1.0, 1.0)
    if slot == 1:
        return (1.0, 0.4, 0.4, 1.0)
    return (*hsv_to_rgb(((slot - 2) * 0.61803398875 + 0.12) % 1, 0.55, 1.0), 1.0)


@dataclass(frozen=True)
class PropositionLabelGroup:
    key: str
    display_id: str
    color: tuple[float, float, float, float]
    active: bool


@dataclass(frozen=True)
class PropositionLabelAnnotation:
    unit_id: int
    groups: tuple[PropositionLabelGroup, ...]

    @property
    def text(self):
        # One ASCII marker fits the existing font atlas and flags all overlaps.
        return f'{self.unit_id}{"*" if len(self.groups) > 1 else ""}'

    @property
    def color(self):
        return self.groups[0].color if self.groups else WHITE


@dataclass(frozen=True)
class PropositionLabelProjection:
    annotations: Mapping[int, PropositionLabelAnnotation]
    groups: tuple[PropositionLabelGroup, ...]
    color_keys: tuple[str, ...]


def project_proposition_labels(catalog, unit_ids, active_keys=(), color_keys=()):
    """Project exact live pending memberships in O(source memberships + labels).

    ``color_keys`` reserves slots for temporarily undisplayed groups in the same
    workspace. The caller resets it when the active proposition selection changes.
    Active groups take priority on overlapping labels; every membership remains
    explicit in ``groups`` and the overlap marker. Catalog order breaks ties,
    independently of filtered/sorted tables and scientific palette slots.
    """
    unit_ids = tuple(map(int, unit_ids))
    wanted = set(unit_ids)
    memberships = {unit: [] for unit in unit_ids}
    pending, intersecting = {}, set()
    if catalog is not None:
        for entry in catalog.pending_entries():
            pending[entry.key] = entry
            for unit in entry.proposition.unit_ids:
                if unit in wanted:
                    intersecting.add(entry.key)
    active_keys = tuple(key for key in active_keys if key in pending)
    active = set(active_keys)
    ordered = tuple(
        dict.fromkeys(
            (
                *active_keys,
                *color_keys,
                *(key for key in pending if key in intersecting),
            )
        )
    )
    groups = {
        key: PropositionLabelGroup(
            key, f'P{pending[key].index + 1}', proposition_label_color(slot), key in active
        )
        for slot, key in enumerate(ordered)
        if key in pending
    }
    for key, group in groups.items():
        for unit in pending[key].proposition.unit_ids:
            if unit in wanted:
                memberships[unit].append(group)
    annotations = {
        unit: PropositionLabelAnnotation(unit, tuple(memberships[unit])) for unit in unit_ids
    }
    return PropositionLabelProjection(
        MappingProxyType(annotations),
        tuple(groups[key] for key in ordered if key in intersecting),
        ordered,
    )
