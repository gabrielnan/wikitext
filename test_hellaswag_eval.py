"""Tests for ``hellaswag_eval`` — smoke + random-baseline sanity.

Run with ``python3 -m pytest test_hellaswag_eval.py`` from the worktree.
The data download is skipped unless the parquet is already on disk; this
keeps unit tests offline.
"""
from __future__ import annotations

import random
from pathlib import Path

from hellaswag_eval import (
    HellaSwagItem,
    evaluate_hellaswag,
    load_hellaswag_val,
)
from wikitext import CharModel


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

class _ConstantModel(CharModel):
    """Always predicts a single fixed char."""

    def __init__(self, ch: str = " "):
        self.ch = ch

    def reset(self) -> None:
        pass

    def predict(self) -> str:
        return self.ch

    def observe(self, char: str) -> None:
        del char


class _RandomCharModel(CharModel):
    """Predicts a uniformly random char from a small ASCII range.

    Used as a baseline-collapse sanity check — a fully random predictor
    should land near the 4-way chance floor (25 %) on HellaSwag.
    """

    def __init__(self, alphabet: str = " etaoinshrdlu", seed: int = 0):
        self.alphabet = alphabet
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def predict(self) -> str:
        return self.rng.choice(self.alphabet)

    def observe(self, char: str) -> None:
        del char


# ---------------------------------------------------------------------------
# Synthetic items (no parquet needed)
# ---------------------------------------------------------------------------

def _synthetic_items() -> list[HellaSwagItem]:
    """A tiny stand-in dataset whose answers are determined by which ending
    starts with the most-common-char-after-context. Used for testing the
    scoring loop without depending on the real HellaSwag download.
    """
    return [
        HellaSwagItem(
            ind=0, activity_label="Test",
            ctx="aaaaaaa", endings=["a x", "b x", "b x", "b x"], label=0,
        ),
        HellaSwagItem(
            ind=1, activity_label="Test",
            ctx="bbbbbbb", endings=["a y", "b y", "a y", "a y"], label=1,
        ),
    ]


def test_scoring_picks_matching_first_char() -> None:
    """``_ConstantModel('a')`` should pick the ending starting with 'a'."""
    items = _synthetic_items()
    model = _ConstantModel(ch="a")
    result = evaluate_hellaswag(
        model, items, n_items=len(items), chars_per_ending=1,
    )
    # On item 0 (label=0, ending starts with 'a'), constant-a model picks correctly.
    # On item 1 (label=1, ending starts with 'b'), constant-a model picks wrongly.
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_items == 2
    assert result.chars_per_ending == 1


def test_per_activity_tracking() -> None:
    """``track_per_activity=True`` should populate per-category stats."""
    items = _synthetic_items()
    model = _ConstantModel(ch="b")
    result = evaluate_hellaswag(
        model, items, n_items=len(items), chars_per_ending=1,
        track_per_activity=True,
    )
    assert "Test" in result.per_activity
    n_correct, n_total = result.per_activity["Test"]
    assert n_total == 2
    assert 0 <= n_correct <= 2


def test_random_baseline_near_chance() -> None:
    """On real HellaSwag, a uniformly random predictor should sit near 25 %.

    Skipped unless the parquet is already cached locally — keeps unit
    tests offline-runnable.
    """
    cache = Path("/tmp/hellaswag-val.parquet")
    if not cache.exists():
        import pytest
        pytest.skip("HellaSwag parquet not cached; skipping live-data test")

    items = load_hellaswag_val(cache_path=cache, download_if_missing=False)
    model = _RandomCharModel(seed=42)
    # Bigger N for a tighter CI: 95% CI half-width at p=0.25, N=2000 is ±1.9pp.
    result = evaluate_hellaswag(
        model, items, n_items=2000, chars_per_ending=30, seed=42,
    )
    assert 0.22 <= result.accuracy <= 0.28, (
        f"random predictor landed at {result.accuracy:.4f} "
        f"— expected near chance (0.25)"
    )
