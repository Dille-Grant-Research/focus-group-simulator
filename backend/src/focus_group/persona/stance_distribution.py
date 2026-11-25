"""Seeded draw of discussion stances for a focus group.

Instead of letting the LLM pick each persona's stance in isolation (which in
live runs produced 11 of 12 "challenging" personas), the session draws one
stance per seat from a preset distribution, then applies guardrails. The draw
is seeded by the job seed, so a run is reproducible.

The weights are a starting point chosen by the developers. The research team
owns them; tune them here.
"""

import math
import random
from dataclasses import dataclass

from focus_group.models.config import GroupDynamic
from focus_group.models.persona import DiscussionStance

CHALLENGER_STANCES: frozenset[DiscussionStance] = frozenset(
    {DiscussionStance.CHALLENGING, DiscussionStance.CONTRARIAN}
)

# Guardrails only apply to groups this size or larger; smaller groups get a pure draw.
MIN_GROUP_FOR_GUARDRAILS = 3
# Every guarded group has at least this many challengers (the anti-sycophancy floor).
MIN_CHALLENGERS = 1


@dataclass(frozen=True)
class StancePreset:
    """Stance weights plus the guardrails applied to a group of 3 or more."""

    weights: dict[DiscussionStance, float]
    max_challenger_fraction: float
    require_non_challenger: bool = False


STANCE_PRESETS: dict[GroupDynamic, StancePreset] = {
    GroupDynamic.NATURAL: StancePreset(
        weights={
            DiscussionStance.AGREEABLE: 0.25,
            DiscussionStance.BALANCED: 0.40,
            DiscussionStance.CHALLENGING: 0.25,
            DiscussionStance.CONTRARIAN: 0.10,
        },
        max_challenger_fraction=0.40,
    ),
    GroupDynamic.COOPERATIVE: StancePreset(
        weights={
            DiscussionStance.AGREEABLE: 0.40,
            DiscussionStance.BALANCED: 0.45,
            DiscussionStance.CHALLENGING: 0.15,
            DiscussionStance.CONTRARIAN: 0.00,
        },
        max_challenger_fraction=0.25,
    ),
    GroupDynamic.HEATED: StancePreset(
        weights={
            DiscussionStance.AGREEABLE: 0.10,
            DiscussionStance.BALANCED: 0.30,
            DiscussionStance.CHALLENGING: 0.40,
            DiscussionStance.CONTRARIAN: 0.20,
        },
        max_challenger_fraction=0.60,
        require_non_challenger=True,
    ),
}


def _challenger_cap(preset: StancePreset, group_size: int) -> int:
    """Largest allowed number of challengers (round half up, never below the floor)."""
    cap = max(MIN_CHALLENGERS, math.floor(preset.max_challenger_fraction * group_size + 0.5))
    return min(cap, group_size - 1) if preset.require_non_challenger else cap


def draw_stances(
    group_size: int,
    dynamic: GroupDynamic = GroupDynamic.NATURAL,
    seed: int | None = None,
) -> list[DiscussionStance]:
    """Draw one discussion stance per participant.

    Args:
        group_size: Number of participants.
        dynamic: Preset whose weights and guardrails apply.
        seed: Seed for the draw. The same (group_size, dynamic, seed) always gives
            the same list. None draws unseeded.

    Returns:
        One stance per participant, in seat order.
    """
    preset = STANCE_PRESETS[dynamic]
    rng = random.Random(seed)
    population = list(preset.weights)
    stances = rng.choices(population, weights=[preset.weights[s] for s in population], k=group_size)

    if group_size < MIN_GROUP_FOR_GUARDRAILS:
        return stances

    cap = _challenger_cap(preset, group_size)
    while (challengers := [i for i, s in enumerate(stances) if s in CHALLENGER_STANCES]) and len(
        challengers
    ) > cap:
        stances[rng.choice(challengers)] = DiscussionStance.BALANCED
    while sum(s in CHALLENGER_STANCES for s in stances) < MIN_CHALLENGERS:
        others = [i for i, s in enumerate(stances) if s not in CHALLENGER_STANCES]
        stances[rng.choice(others)] = DiscussionStance.CHALLENGING
    return stances
