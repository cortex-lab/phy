"""Compile semantic membership graphs into finite rank intervals per segment."""

from .curation import validate_predicate
from .source import hash_json


class RangeUnavailableError(ValueError):
    """The verified row path remains available for this unsupported query."""


def normalize(parts):
    result = []
    for lo, hi in sorted(parts):
        if lo >= hi:
            continue
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


class IntervalProgram:
    """Keep rejection obligations even if later operations select no finite rows."""

    def __init__(self, expressions):
        self.nodes, self.roots, self.cuts = [], {}, {}
        known, visiting, objects = {}, set(), {}

        def compile_one(expression):
            identity = id(expression)
            if identity in objects:
                return objects[identity]
            if identity in visiting:
                raise RangeUnavailableError('Cyclic membership expression.')
            visiting.add(identity)
            kind = expression['kind']
            if kind == 'source':
                node = ('source', int(expression['unit']))
            elif kind == 'union':
                node = ('union', tuple(compile_one(p) for p in expression['parts']))
            elif kind == 'split':
                predicate = validate_predicate(expression['predicate'])
                if predicate['attribute'] != 'amplitudes.npy':
                    raise RangeUnavailableError('Only amplitude predicates use the range index.')
                parent = compile_one(expression['parent'])
                operator = predicate['operator']
                bound = '<' if operator in ('<', '>=') else '<='
                cut = hash_json({'operator': bound, 'threshold': predicate['threshold']})
                self.cuts[cut] = dict(predicate, operator=bound, nonfinite='keep')
                node = (
                    'split',
                    parent,
                    cut,
                    operator,
                    bool(expression['selected']),
                    predicate['nonfinite'],
                )
            else:
                raise RangeUnavailableError('Unsupported membership expression.')
            visiting.remove(identity)
            if node not in known:
                known[node] = len(self.nodes)
                self.nodes.append(node)
            objects[identity] = known[node]
            if len(self.nodes) > 8192:
                raise RangeUnavailableError('Membership graph exceeds range query budget.')
            return known[node]

        try:
            for unit, expression in expressions.items():
                self.roots[unit] = compile_one(expression)
        except RecursionError as error:
            raise RangeUnavailableError(
                'Membership history exceeds range compilation budget.'
            ) from error
        self.originals = sorted({node[1] for node in self.nodes if node[0] == 'source'})

    def evaluate(self, original, finite_count, nonfinite_count, cuts, interval_limit=100000):
        states = []
        slots = 0
        for node in self.nodes:
            kind = node[0]
            if kind == 'source':
                match = node[1] == original
                intervals = [(0, finite_count)] if match and finite_count else []
                nonfinite = match
            elif kind == 'union':
                intervals = normalize([p for parent in node[1] for p in states[parent][0]])
                nonfinite = any(states[parent][1] for parent in node[1])
            else:
                _, parent, cut, operator, selected, policy = node
                parts, parent_nonfinite = states[parent]
                if policy == 'reject' and parent_nonfinite and nonfinite_count:
                    raise ValueError('Nonfinite source values violate threshold policy.')
                rank = cuts[cut]
                lower = operator in ('<', '<=')
                if not selected:
                    lower = not lower
                intervals = [
                    (lo, min(hi, rank)) if lower else (max(lo, rank), hi) for lo, hi in parts
                ]
                intervals = [(lo, hi) for lo, hi in intervals if lo < hi]
                nonfinite = parent_nonfinite and not selected
            slots += len(intervals)
            if slots > interval_limit:
                raise RangeUnavailableError('Membership intervals exceed query memory budget.')
            states.append((intervals, nonfinite))
        return {unit: states[root] for unit, root in self.roots.items()}
