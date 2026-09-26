"""Portable, pre-scored candidate pools for the allocation algorithms."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .core.validation import PessimismValidationError, require_positive_int


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PessimismValidationError(f"{name} must be a JSON number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise PessimismValidationError(f"{name} must be finite") from exc
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise PessimismValidationError(f"{name} must be finite and {qualifier}")
    return value


def _object(value, required, optional, name):
    if not isinstance(value, dict):
        raise PessimismValidationError(f"{name} must be an object")
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing or extra:
        raise PessimismValidationError(
            f"{name}: missing fields {sorted(missing)}, unknown fields {sorted(extra)}"
        )


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise PessimismValidationError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class GroupPool:
    id: str
    error_bound: float
    candidate_ids: tuple[str, ...]
    proxy_scores: tuple[float, ...]
    judge_scores: tuple[float, ...] | None


@dataclass(frozen=True)
class CandidatePool:
    r_max: float
    groups: tuple[GroupPool, ...]

    @property
    def capacity(self):
        return {group.id: len(group.candidate_ids) for group in self.groups}


def parse_pool(document) -> CandidatePool:
    """Validate without clipping, normalizing, reordering, or inferring errors.

    Judge scores must be present for every candidate or absent everywhere.
    Group error bounds are supplied independently by the caller, never fitted
    from evaluation scores by this runner.
    """
    _object(document, {"r_max", "groups"}, set(), "pool")
    r_max = _number(document["r_max"], "r_max", positive=True)
    raw_groups = document["groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        raise PessimismValidationError("groups must be a nonempty array")
    groups, seen_groups, judge_presence = [], set(), set()
    for raw_group in raw_groups:
        _object(raw_group, {"id", "error_bound", "candidates"}, set(), "group")
        group_id = _identifier(raw_group["id"], "group.id")
        if group_id in seen_groups:
            raise PessimismValidationError(f"duplicate group id: {group_id}")
        seen_groups.add(group_id)
        error = _number(raw_group["error_bound"], "error_bound")
        candidates = raw_group["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise PessimismValidationError("candidates must be a nonempty array")
        ids, proxies, judges, seen_ids = [], [], [], set()
        for candidate in candidates:
            _object(candidate, {"id", "proxy_score"}, {"judge_score"}, "candidate")
            candidate_id = _identifier(candidate["id"], "candidate.id")
            if candidate_id in seen_ids:
                raise PessimismValidationError(f"duplicate candidate id in {group_id}")
            seen_ids.add(candidate_id)
            ids.append(candidate_id)
            proxy = _number(candidate["proxy_score"], "proxy_score")
            if proxy > r_max:
                raise PessimismValidationError("proxy_score exceeds r_max")
            proxies.append(proxy)
            has_judge = "judge_score" in candidate
            judge_presence.add(has_judge)
            if has_judge:
                judge = _number(candidate["judge_score"], "judge_score")
                if judge > r_max:
                    raise PessimismValidationError("judge_score exceeds r_max")
                judges.append(judge)
        groups.append(GroupPool(group_id, error, tuple(ids), tuple(proxies),
                                tuple(judges) if judges else None))
    if len(judge_presence) != 1:
        raise PessimismValidationError(
            "judge_score must be present for every candidate or absent everywhere"
        )
    return CandidatePool(r_max, tuple(groups))


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PessimismValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_pool(path: str | Path) -> CandidatePool:
    with Path(path).open(encoding="utf-8") as handle:
        return parse_pool(json.load(handle, object_pairs_hook=_unique_keys))


class ProxyStream:
    """A fresh ordered stream per method, with no Judge data in its state."""

    def __init__(self, pool: CandidatePool):
        self.rows = {group.id: (group.candidate_ids, group.proxy_scores)
                     for group in pool.groups}
        self.counts = {group.id: 0 for group in pool.groups}

    def __call__(self, group, requested):
        requested = require_positive_int(requested, "requested")
        ids, scores = self.rows[group]
        start = self.counts[group]
        end = start + requested
        if end > len(ids):
            raise PessimismValidationError(
                f"group {group!r}: requested prefix {end} exceeds pool capacity {len(ids)}"
            )
        self.counts[group] = end
        return list(ids[start:end]), list(scores[start:end])
