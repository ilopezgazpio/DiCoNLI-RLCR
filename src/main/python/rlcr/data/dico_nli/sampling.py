"""Order-independent, seeded sampling of complete source-pair groups."""
from collections import defaultdict
import hashlib

from .pairing import validate_pairs


def select_source_pairs(records, maximum=None, *, seed=42):
    """Keep proportional label-signature strata, with at least one group per stratum.

    Allocation starts with one group per observed label signature, then minimizes
    each stratum's deficit to its proportional target. SHA-256 ranks source IDs
    within strata, making selection independent of CSV ordering and Python RNG.
    """
    validate_pairs(records)
    groups = defaultdict(list)
    for record in records:
        groups[record.pair_id].append(record)
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    if maximum is None:
        return sorted(records, key=lambda row: row.instance_id)
    if type(maximum) is not int or not 1 <= maximum <= len(groups):
        raise ValueError(f"max_source_pairs must be in [1, {len(groups)}].")
    strata = defaultdict(list)
    for pair_id, rows in groups.items():
        signature = tuple(sorted({row.label or "UNLABELED" for row in rows}))
        strata[signature].append(pair_id)
    if maximum < len(strata):
        raise ValueError(
            f"max_source_pairs={maximum} cannot preserve {len(strata)} label strata; increase it."
        )
    ordered = sorted(strata)
    allocations = {key: 1 for key in ordered}
    while sum(allocations.values()) < maximum:
        available = [key for key in ordered if allocations[key] < len(strata[key])]
        chosen = max(
            available, key=lambda key: maximum * len(strata[key]) / len(groups) - allocations[key]
        )
        allocations[chosen] += 1
    selected = set()
    for key in ordered:
        ranked = sorted(
            strata[key],
            key=lambda identifier: (
                hashlib.sha256(f"{seed}\0{identifier}".encode("utf-8")).hexdigest(),
                identifier,
            ),
        )
        selected.update(ranked[: allocations[key]])
    result = sorted(
        (row for row in records if row.pair_id in selected), key=lambda row: row.instance_id
    )
    validate_pairs(result)
    return result
