# HellaSwag as an auxiliary eval for `wikitext`

## Why

The current eval — greedy next-char prediction on WikiText-103 val[:60K] gated at 0.70 — sits **below** where count-based methods saturate (~0.72 for chained-KN at order 14). Result: the leaderboard rewards methods that fit local character statistics, not methods that learn structure.

HellaSwag (Zellers 2019) was built specifically to defeat that failure mode in NLI: pick the correct sentence continuation from 4 options where the 3 wrong options were **adversarially generated** to look plausible at the surface level but fail commonsense. Random = 25 %; humans ≈ 95 %; statistical models at release ≈ 48 %.

If we add HellaSwag as an auxiliary metric, methods that learned only local statistics should land near the 25 % chance floor, while methods that learned structure should pull away.

## Data shape (val split, what we'd actually use)

Pulled from `Rowan/hellaswag` on HuggingFace (`validation-00000-of-00001.parquet`, 6.3 MB, 10 042 items).

| Field | Type | Notes |
|---|---|---|
| `ctx_a` | str | Context up through the last complete clause |
| `ctx_b` | str | Trailing incomplete noun phrase, often "he" / "the lady" / etc. Empty for 68 % of items. |
| `ctx` | str | `ctx_a + " " + ctx_b` (the actual context the model sees) |
| `endings` | list[str] of length 4 | The four candidate continuations |
| `label` | str (one of "0"…"3") | Index of correct ending |
| `activity_label` | str | Category (192 distinct: "Personal Care and Style", "Family Life", "Canoeing", …) |

Length stats (chars):
- `ctx`: min 30 / median 240 / mean 223 / max 498
- `endings[*]`: min 6 / median 146 / mean 140 / max 428
- Label distribution: 25.0 % / 24.7 % / 25.7 % / 24.5 % — effectively uniform (random baseline 25.0 %).

**Critical for us**: the 4 endings diverge at character position **0** in most items (median = 0, mean = 0). The first character of the ending is usually enough to distinguish at least one wrong option. This means we can **truncate scoring to the first 20–50 chars per ending** and keep most of the discriminative signal — important for the runtime budget.

## API match

Our post-#7 contract is `CharModel.predict() -> str` (one committed char) + `observe(char)`. No probability distribution exposed.

That kills the *standard* HellaSwag scoring (length-normalised log-likelihood of each ending under the model), which needs per-char probabilities. We can't compute log-likelihood from a single-char prediction.

What we **can** do: **teacher-forced char-match scoring** — feed each ending into the model character by character, and at each position compare `model.predict()` to the ground-truth char from that ending. Whichever ending has the highest match rate is the model's pick.

```
score(item, ending_j) = (1/len(ending_j)) × Σ_t [model.predict() == ending_j[t]]
pick = argmax_j score(item, ending_j)
correct = (pick == int(item.label))
```

This is genuinely different from log-likelihood scoring — it asks "how often would the model, character by character, GUESS this exact ending?" rather than "how confident is the model in this ending?" — but the rank between endings should correlate strongly with log-likelihood rank, because the greedy-argmax pick is exactly the mode of the distribution.

**Caveat**: a model whose top-1 char prediction is wrong but whose ending-likelihood is highest will score badly under our metric. Empirically this is rare for the strong endings — Zellers et al. constructed the wrong endings to look LOCALLY plausible too, so a strong language model often predicts a chunk of every ending correctly. The discriminative signal is in the cumulative match rate, not any single char.

## Recommended approach

### Eval procedure (per item)

For each item:
1. `model.reset()`.
2. For c in `ctx`: `model.observe(c)` (feed context, no scoring).
3. For each of the 4 `endings[j]`:
   - **Save** the model's streaming state.
   - For c in `endings[j][:K]`: emit `pred = model.predict()`; record match `(pred == c)`; `model.observe(c)`.
   - **Restore** the model's streaming state.
   - `score[j] = matches / K`
4. `pick = argmax(score)`. `correct = (pick == int(label))`.

Step "save/restore state" is a problem for arbitrary `CharModel`s — there's no save/restore API. **Workaround**: instead of saving state, do `model.reset()` and **replay** the context for each ending. Cost: 4× context feeding per item.

Total work per item ≈ `(4 × |ctx|) + (4 × K) = 4 × (223 + K)` `observe()`+`predict()` calls. For `K = 40`: `4 × 263 ≈ 1052` ops/item.

### Subset size & truncation

| Subset | K = 40 | K = 100 | K = full (~140) |
|---|---|---|---|
| 200 items | 200K ops | 320K ops | 416K ops |
| 1000 items | 1.05M ops | 1.6M ops | 2.1M ops |
| 10042 items (all) | 10.6M ops | 16M ops | 21M ops |

At a typical GPU n-gram `predict()` cost of ~5 µs/call, all 10K items at K = 40 is ~50 s — fits comfortably in our eval budget. At K = full, ~100 s.

**Recommend default: K = 50, N = 1000 items** (≈ 1 M ops, ~5 s). Stable confidence interval (`σ ≈ √(0.25 × 0.75 / 1000) = 0.014`, so 95 % CI is ±2.8 %). Larger N if needed for resolving close methods.

### Expected baselines

| Method | Expected HellaSwag acc | Note |
|---|---|---|
| Random | 25.0 % | 4-way MC |
| WikiText n-gram (order 11+) | 26–32 % | Can predict "he is" → " " or "is" but can't distinguish four plausible verbs |
| Modded nanogpt (300s cap) | 30–40 % | Captures some structure; far from saturated |
| Frontier LLMs | 90 %+ | For reference |

The expected separation between count-based and NN methods is ~5–10 percentage points — well above the ~2.8 % CI at N = 1000. Should be visible.

## Integration plan

### 1. Data plumbing

- **Don't bake HellaSwag into the Docker image** (we want it to be swappable for other eval sets).
- **Download once at eval time** in `run_eval.py` via `huggingface_hub` (already in our deps via codecarbon → no new install) or a direct `curl` if the deps tree gets messy.
- Cache to `/tmp/hellaswag-val.parquet` so re-runs don't re-download. 6.3 MB, ~1 s download.
- Pin the file SHA in `hellaswag_eval.py` for reproducibility.

### 2. New module `hellaswag_eval.py`

```python
def evaluate_hellaswag(
    model: CharModel,
    n_items: int = 1000,
    chars_per_ending: int = 50,
    seed: int = 0,
) -> HellaSwagResult:
    ...
```

Returns `acc` (fraction correct) + `per_activity` breakdown (diagnostic — does the model do better on "Computers and Electronics" than "Canoeing"?) + `random_baseline` (computed once, for sanity).

### 3. Wire into `run_eval.py`

- New CLI flag: `--hellaswag-subset N` (0 = skip, default 0 to preserve current behaviour; 1000 = enable). Maintains backward compat for existing CI.
- New `result.json` fields: `hellaswag_acc`, `hellaswag_n_items`, `hellaswag_chars_per_ending`. All `None` when skipped.
- Energy: HellaSwag eval is NOT energy-accounted (consistent with the existing "eval is unmetered" rule from `wikitext.py`). It's a diagnostic alongside the energy-scored training.

### 4. Backward compat & non-breaking

- `acc_min` gate (rule 5) keeps using the WikiText val char-acc as the leaderboard floor. HellaSwag is **diagnostic only** — does not gate.
- Existing submissions continue to score WikiText val char-acc as before.
- `README.md` Record History adds a `HellaSwag acc` column for runs that opted in (else blank).

### 5. (Stretch) Future: smarter scoring via optional API

If the project wants more rigorous HellaSwag scoring, add an optional `CharModel.score_char(c: str) -> float` method that returns `log P(c | context)`. Models that implement it get log-likelihood-based ending scoring (matches the standard HellaSwag metric and most published numbers). Models without it fall back to teacher-forced char-match.

## Open questions for review

1. **Default `chars_per_ending`**: 40 (fast, ~5 s of eval) or full ending (~100 s, more discrimination)? Suggest 50 as compromise.
2. **Default subset size**: 1000 fits comfortably in budget. 10 042 (full) takes ~50 s — also tolerable. Suggest 1000 for headline, 10 042 for definitive comparisons.
3. **Where does the data live in the Modal image** — re-download per run, bake into image at fixed digest, or load from a Modal Volume? Suggest re-download (no image rebuild needed, 1 s overhead).
4. **HellaSwag as gate vs diagnostic**: this PR proposes diagnostic-only. A future PR could promote it to a co-gate ("must pass both WikiText 0.70 and HellaSwag 0.30") if maintainers find the separation reliable.

## Sources

- [HellaSwag: Can a Machine Really Finish Your Sentence? (Zellers et al. 2019)](https://arxiv.org/abs/1905.07830)
- [Rowan/hellaswag on HuggingFace](https://huggingface.co/datasets/Rowan/hellaswag)
- [rowanz/hellaswag GitHub (data fields)](https://github.com/rowanz/hellaswag)
