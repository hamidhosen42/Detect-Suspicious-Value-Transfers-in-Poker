# Detect Suspicious Value Transfers in Poker — full competition record

Kaggle · host **Slash** · Sep 5 – Sep 20 2026 · 2,000,000 synthetic six-max NLHE hands, 12,000 players, 112,540 evaluation pairs
Team **hosen42 · esfersami50 · foysalemonshanto** · final public score **0.90025** (rank ≈ 30 / 350)

| document | what it is |
|---|---|
| [docs/SOLUTION_WRITEUP.md](docs/SOLUTION_WRITEUP.md) | the ≤ 1,500-word solution write-up required for verification |
| [docs/METHOD_SELECTED.md](docs/METHOD_SELECTED.md) | full technical description of the two selected submissions (features, models, training, assembly) |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | how to rebuild the selected submission from the raw files (`reproduce/run_all.sh`) |
| [docs/CASE_REVIEWS.md](docs/CASE_REVIEWS.md) | five case reviews from the submitted evidence |
| [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | every step, its leaderboard score and what it taught us — including the dead ends |

---

## 1. The task

Each evaluation pair (two players who shared ≥ 38 hands in the last 2,000 hands of their 30-player table) gets a `risk_score`, a `predicted_behavior` (`directed_transfer`, `soft_play`, `coordinated_isolation`, `other_coordination`, `none`) and up to five `evidence_hand_*` IDs. Score = **0.70 · pair AP + 0.20 · evidence MAP@5 + 0.10 · behaviour MAP**. The development period (first 3,000 hands per table) has 1,860 trusted labels (372 positives with ≤ 5 evidence hands each, 1,488 confirmed non-targets); every other development pair is *unknown*.

## 2. Result

| | public LB | notes |
|---|---|---|
| public baselines (chip-flow features) | 0.66 – 0.69 | |
| our Step 3 (card-conditional policy surprise) | 0.801 | first card-aware model |
| Step 8 (action-level policy surprise + hard negatives) | 0.894 | the pair model that survived to the end |
| Step 16 (+ template / family evidence re-ranker) | 0.896 | |
| **final_J** (20-seed bags + Step 18 evidence) | **0.900** | selected |
| **final_F** (bags + fold-CV models + Step 18 evidence) | 0.900 | selected (hedge) |
| leader | 0.938 | |

Leaderboard probes decomposed our score: pair AP ≈ 0.97, evidence MAP@5 ≈ 0.62, behaviour MAP ≈ 0.96. The gap to the leader is almost entirely evidence (they need ≥ 0.70 there); the rule that selects the 3–5 labelled evidence hands out of a pair's ~10 collusion-mode hands is the one thing we did not find.

## 3. Method in one paragraph

The simulator's bots are strictly card-driven (voluntary-entry rate 2 % → 96 % from worst to best hole cards), so coordination shows up as **two partners acting wrongly for their cards in the same hand**. We fit a 6-class behaviour clone of the population policy on all 18.6 M actions (cards, exact made-hand category, board texture, position, pot/stack ratios, raises faced, plus the player's *other-phase* style — collusion is phase-specific, so the other phase is a clean baseline) and turn every action into a surprise `−log P(action)`. Pair-hand rows (≈ 240 features: joint entries with junk, surprise when facing the partner's bet, exact-strength fold/check-down/dump flags, squeeze patterns, chip flow) are aggregated per pair (sums, top-3, with-partner vs without-partner contrasts, other-phase pair baselines, coincidence z-scores) and ranked by LightGBM trained positive–unlabelled on 2,000-hand development windows that match the evaluation exposure. Evidence is a two-stage ranker over the coordinated pairs' hands (stage 1: pair-hand features + made-hand combinations, street timing, per-player policy surprise; stage 2: top-20 pool re-ranked with family, chronology and action-sequence templates). Families come from three one-vs-rest heads gated to the top 5 % by risk. The final risk is the rank-average of two 20-seed full-data bags (Step-8 and Step-9 feature sets).

## 4. Repository map

```
step1_pipeline.py … step18_pipeline.py   one self-contained pipeline per iteration (# %% cells; jupytext renders the notebooks in deliverables_step*/)
step9_pipeline.py, step18_pipeline.py    the two pipelines the final submission is built from
hyp/                                     evidence feature families added in Step 18 (category_combo, street_timing, player_policy)
reproduce/                               bag20.py (20-seed bags), make_final.py (assembly), run_all.sh (end-to-end)
deliverables_step*/, deliverables_final/ submission CSVs and notebooks of every iteration; final_J / final_F are the selected ones
work_step3/scripts/                      analysis and ablation scripts (dev views, MIL variants, template/outsider tests, leak checks, bagging)
docs/                                    write-up, method, reproduction guide, case reviews, experiment log
public_notebook/, 18-09-26/              audits of public notebooks and repositories (nothing from a top team was public)
```
Competition data is not included (Competition Use Only); place the files in `data/`. No credentials are tracked (`token.txt` is git-ignored and was never committed).

## 5. Rules compliance

All features are computed from gameplay: hole cards, board, seats/positions, stacks, the ordered action log and chronology from `started_at`. No player metadata, ID formats, file/row ordering or generator internals are used. Unlabelled development pairs are treated as unknown.

## 6. What we learned (short)

* Card-conditional policy deviation is the signal; chip flow alone plateaus at 0.69.
* Hidden colluders in the unlabelled development set are few (~150 of 120 k); the "ambiguous" band of loose-but-honest pairs must stay in training as hard negatives.
* Matching training exposure to evaluation exposure (2,000-hand windows) is worth +0.02.
* Development metrics were unreliable for stacked hand models (a multiple-instance hand model gained on every dev view and lost 0.02 on the leaderboard: its within-pair percentile features scale with phase length); only the public leaderboard, used sparingly, settled such questions.
* Seed bagging of full-data models was the only lever that moved the leaderboard *reliably* at the end (+0.0003 to +0.0007 per step).
