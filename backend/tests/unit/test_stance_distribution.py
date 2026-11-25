"""Tests for the seeded stance draw (focus_group.persona.stance_distribution)."""

import math

import pytest
from pydantic import ValidationError

from focus_group.models.config import GroupDynamic, SessionConfig
from focus_group.models.persona import DiscussionStance
from focus_group.persona.stance_distribution import (
    CHALLENGER_STANCES,
    STANCE_PRESETS,
    draw_stances,
)

_MAX_GROUP = 15
_SEEDS = range(200)
_FREQUENCY_SAMPLES = 10_000
_FREQUENCY_TOLERANCE = 0.03


def _cap(dynamic: GroupDynamic, n: int) -> int:
    preset = STANCE_PRESETS[dynamic]
    cap = max(1, math.floor(preset.max_challenger_fraction * n + 0.5))
    return min(cap, n - 1) if preset.require_non_challenger else cap


def _challengers(stances: list[DiscussionStance]) -> int:
    return sum(s in CHALLENGER_STANCES for s in stances)


def test_same_seed_same_mix() -> None:
    first = draw_stances(5, GroupDynamic.NATURAL, seed=42)
    assert first == draw_stances(5, GroupDynamic.NATURAL, seed=42)
    assert len(draw_stances(5, GroupDynamic.NATURAL, seed=43)) == 5


@pytest.mark.parametrize("dynamic", list(GroupDynamic))
@pytest.mark.parametrize("n", range(3, _MAX_GROUP + 1))
def test_guardrails_hold(dynamic: GroupDynamic, n: int) -> None:
    for seed in _SEEDS:
        stances = draw_stances(n, dynamic, seed=seed)
        challengers = _challengers(stances)
        assert len(stances) == n
        assert challengers >= 1, (dynamic, n, seed)
        assert challengers <= _cap(dynamic, n), (dynamic, n, seed, stances)


@pytest.mark.parametrize("dynamic", list(GroupDynamic))
@pytest.mark.parametrize("n", [1, 2])
def test_small_groups_unconstrained(dynamic: GroupDynamic, n: int) -> None:
    for seed in (0, 1, None):
        stances = draw_stances(n, dynamic, seed=seed)
        assert len(stances) == n
        assert all(isinstance(s, DiscussionStance) for s in stances)


def test_cooperative_never_contrarian() -> None:
    for seed in range(500):
        assert DiscussionStance.CONTRARIAN not in draw_stances(8, GroupDynamic.COOPERATIVE, seed)


def test_natural_frequencies() -> None:
    weights = STANCE_PRESETS[GroupDynamic.NATURAL].weights
    counts = {stance: 0 for stance in DiscussionStance}
    for seed in range(_FREQUENCY_SAMPLES):
        counts[draw_stances(1, GroupDynamic.NATURAL, seed)[0]] += 1
    for stance, weight in weights.items():
        assert abs(counts[stance] / _FREQUENCY_SAMPLES - weight) < _FREQUENCY_TOLERANCE, stance


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, GroupDynamic.NATURAL, id="omitted"),
        pytest.param("heated", GroupDynamic.HEATED, id="heated"),
        pytest.param("cooperative", GroupDynamic.COOPERATIVE, id="cooperative"),
    ],
)
def test_session_config_group_dynamic(value: str | None, expected: GroupDynamic) -> None:
    data: dict = {"demographics": ["Female, 30"], "questions": ["Why?"]}
    if value is not None:
        data["group_dynamic"] = value
    assert SessionConfig(**data).group_dynamic == expected


def test_session_config_rejects_unknown_dynamic() -> None:
    with pytest.raises(ValidationError):
        SessionConfig(demographics=["Female, 30"], questions=["Why?"], group_dynamic="chaotic")


@pytest.mark.parametrize(
    "names",
    [
        pytest.param(["Ana Ruiz", "Ana Ruiz"], id="exact"),
        pytest.param(["Ana Ruiz", " ana ruiz "], id="case-and-space"),
    ],
)
def test_session_config_rejects_duplicate_participant_names(names: list[str]) -> None:
    """The session tracks speakers by name; duplicates would stall every question."""
    with pytest.raises(ValidationError):
        SessionConfig(demographics=["Female, 30", "Female, 40"], questions=["Why?"], participant_names=names)


def test_session_config_allows_unset_name_slots() -> None:
    config = SessionConfig(
        demographics=["Female, 30", "Female, 40", "Male, 50"],
        questions=["Why?"],
        participant_names=["Ana Ruiz", None, None],
    )
    assert config.participant_names == ["Ana Ruiz", None, None]
