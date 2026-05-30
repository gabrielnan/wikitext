"""HellaSwag auxiliary evaluator for the wikitext char-LM benchmark.

HellaSwag (Zellers et al. 2019) is a commonsense sentence-completion
multiple-choice task. Each item gives a context + 4 candidate endings;
the model has to pick the correct ending. Wrong endings are
adversarially generated to look locally plausible, so methods that
fit only local character statistics should land near the 25 % chance
floor while methods that learned structure should pull away.

Why an auxiliary eval here: our primary eval (greedy next-char acc on
WikiText-103 val at the 0.70 gate) sits below where order-14 chained-KN
saturates (0.7184). Count-based methods can hit the gate without
learning anything compositional. HellaSwag is designed to defeat that.

Scoring under the post-#7 ``CharModel.predict() -> str`` contract:
the standard HellaSwag metric (length-normalised log-likelihood of each
ending under the model) requires per-char probabilities, which the new
contract does not expose. We use **teacher-forced char-match scoring**
instead: for each ending, feed it into the model char by char; the
ending's score is the fraction of positions where the model's committed
prediction matched the ending's actual char. The model picks the ending
with the highest match rate. See ``HELLASWAG_INTEGRATION.md`` for the
analysis behind this choice and the expected baselines.

Public surface:
- ``HellaSwagItem`` — a parsed dataset item.
- ``HellaSwagResult`` — aggregate eval output.
- ``load_hellaswag_val(...)`` — pull the val parquet (cached).
- ``evaluate_hellaswag(model, items, ...)`` — run the scoring loop.
"""
from __future__ import annotations

import hashlib
import os
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from wikitext import CharModel


HELLASWAG_VAL_URL = (
    "https://huggingface.co/datasets/Rowan/hellaswag/"
    "resolve/main/data/validation-00000-of-00001.parquet"
)
HELLASWAG_VAL_SHA256 = (
    # Recorded 2026-05-29 from a fresh download. Pin so a server-side
    # rewrite doesn't silently change the eval distribution under us.
    "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd"
)
DEFAULT_CACHE_PATH = Path("/tmp/hellaswag-val.parquet")


@dataclass(frozen=True)
class HellaSwagItem:
    """A single HellaSwag val item, with only the fields the eval needs.

    ``ctx`` is the concatenated context the model sees before scoring
    each ending. ``endings`` is always length 4. ``label`` is the index
    (0-3) of the correct ending.
    """
    ind: int
    activity_label: str
    ctx: str
    endings: list[str]
    label: int


@dataclass
class HellaSwagResult:
    """Aggregate output of one HellaSwag eval pass.

    ``per_activity`` is populated only when ``track_per_activity=True``
    is passed to ``evaluate_hellaswag`` — useful as a diagnostic to see
    which activity categories a method handles well.
    """
    accuracy: float
    n_items: int
    n_correct: int
    chars_per_ending: int
    per_activity: dict[str, tuple[int, int]] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"HellaSwag: acc={self.accuracy:.4f}  "
            f"({self.n_correct}/{self.n_items} items, "
            f"chars_per_ending={self.chars_per_ending})"
        )


def load_hellaswag_val(
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    download_if_missing: bool = True,
    verify_sha: bool = True,
) -> list[HellaSwagItem]:
    """Load the HellaSwag val split from a cached parquet file.

    Downloads from HuggingFace the first time. The cached file is
    pinned by SHA-256 so a server-side rewrite can't silently change
    the eval distribution; set ``verify_sha=False`` for local dev
    where the pin doesn't matter.
    """
    if not cache_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"HellaSwag parquet not cached at {cache_path} and "
                f"download_if_missing=False."
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _download_with_redirects(HELLASWAG_VAL_URL, cache_path)

    if verify_sha:
        actual = _sha256(cache_path)
        if actual != HELLASWAG_VAL_SHA256:
            import warnings
            warnings.warn(
                f"HellaSwag parquet SHA-256 mismatch: "
                f"got {actual}, expected {HELLASWAG_VAL_SHA256}. "
                f"Data may have been updated upstream; re-pin if intentional.",
                stacklevel=2,
            )

    import pyarrow.parquet as pq
    t = pq.read_table(cache_path)
    return [
        HellaSwagItem(
            ind=t.column("ind")[i].as_py(),
            activity_label=t.column("activity_label")[i].as_py(),
            ctx=t.column("ctx")[i].as_py(),
            endings=list(t.column("endings")[i].as_py()),
            label=int(t.column("label")[i].as_py()),
        )
        for i in range(t.num_rows)
    ]


def evaluate_hellaswag(
    model: CharModel,
    items: Iterable[HellaSwagItem],
    *,
    n_items: int = 1000,
    chars_per_ending: int = 50,
    seed: int = 0,
    track_per_activity: bool = False,
    progress_every: int = 0,
) -> HellaSwagResult:
    """Score ``model`` on HellaSwag by teacher-forced char-match.

    For each item we replay the context once per ending (cost ≈
    ``4 × |ctx|`` model ops per item) because there's no save/restore
    interface on ``CharModel``. The picked ending is the one with the
    highest fraction of matches over the first ``chars_per_ending``
    characters. The item is correct iff the pick matches the gold label.

    ``n_items`` are sampled deterministically with ``seed`` so repeated
    calls produce the same subset. Pass ``n_items=None`` (or larger
    than the dataset) to use all items.
    """
    pool = list(items)
    if n_items is None or n_items >= len(pool):
        chosen = pool
    else:
        rng = random.Random(seed)
        chosen = rng.sample(pool, n_items)

    n_correct = 0
    per_activity: dict[str, list[int]] = {}  # category -> [n_correct, n_total]

    for idx, item in enumerate(chosen):
        pick = _score_item(model, item, chars_per_ending=chars_per_ending)
        is_correct = int(pick == item.label)
        n_correct += is_correct

        if track_per_activity:
            counts = per_activity.setdefault(item.activity_label, [0, 0])
            counts[0] += is_correct
            counts[1] += 1

        if progress_every and (idx + 1) % progress_every == 0:
            running = n_correct / (idx + 1)
            print(
                f"  hellaswag {idx + 1:>5,}/{len(chosen):,}  "
                f"acc={running:.4f}",
                flush=True,
            )

    n_total = len(chosen)
    return HellaSwagResult(
        accuracy=n_correct / max(1, n_total),
        n_items=n_total,
        n_correct=n_correct,
        chars_per_ending=chars_per_ending,
        per_activity={k: tuple(v) for k, v in per_activity.items()},
    )


def _score_item(
    model: CharModel,
    item: HellaSwagItem,
    *,
    chars_per_ending: int,
) -> int:
    """Return the index (0-3) of the ending the model picks for ``item``."""
    best_idx = 0
    best_score = -1.0
    for j, ending in enumerate(item.endings):
        model.reset()
        for c in item.ctx:
            model.observe(c)

        truncated = ending[:chars_per_ending]
        if not truncated:
            score = 0.0
        else:
            matches = 0
            for c in truncated:
                pred = model.predict()
                if pred == c:
                    matches += 1
                model.observe(c)
            score = matches / len(truncated)

        if score > best_score:
            best_score = score
            best_idx = j

    return best_idx


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_redirects(url: str, out: Path) -> None:
    print(f"[hellaswag] downloading {url}", flush=True)
    req = urllib.request.Request(
        url, headers={"User-Agent": "wikitext-hellaswag-eval/1.0"}
    )
    with urllib.request.urlopen(req) as resp, out.open("wb") as f:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    size = os.path.getsize(out)
    print(f"[hellaswag] wrote {out} ({size:,} bytes)", flush=True)
