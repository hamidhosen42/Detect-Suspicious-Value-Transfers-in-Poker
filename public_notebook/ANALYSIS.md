# Competition analysis — public code snapshot of 16 Sep 2026

Scope: the 8 public notebooks pulled into this folder (score-prefixed), the live leaderboard, and our own submission history. Complements `../COMPETITION_ANALYSIS.md` (14 Sep, data + metric) and `../PUBLIC_RESOURCE_REVIEW.md` (14 Sep, older notebooks). Only new material is covered here.

## 1. Where we stand

| Fact | Value (16 Sep 2026, public LB) |
|---|---|
| Teams | 277 |
| Leader | 0.93827; 14 teams ≥ 0.90, 95 teams ≥ 0.80, 139 teams ≥ 0.70 |
| Median team | 0.701 |
| Our team (esfersami50 / foysalemonshanto / hosen42) | rank 143, best 0.69243 — that is a plain re-run of the public 0.69 notebook, not our own code |
| Our own best | Step 1 `step1_evidence_first_baseline` = 0.67117 |
| Best public notebook | 0.68796 (Hanh Tran); author reports 0.69536 for the same code |

Two consequences:

1. **Public notebooks sit at the median.** Copying or lightly tuning them cannot move us out of the middle of the table. 95 teams are ≥ 0.80, so a signal worth ≥ 0.10 exists that none of the public code uses.
2. **Run-to-run noise ≈ ±0.004.** The identical notebook scored 0.68796 (Kaggle version), 0.69243 (our re-run) and 0.69536 (author). Any change smaller than ~0.005 cannot be judged from one LB submission.

## 2. Public notebook ranking (Code tab, sorted by score)

| Score | Notebook | Author | Approach in one line | Reviewed before? |
|---:|---|---|---|---|
| 0.68796 | Suspicious Value Transfers Detection LB 0.69 | Hanh Tran | Polars pair×hand features → triple-GBDT pair risk, family detectors, dual-engine evidence ranker | **New** — analysed below |
| 0.68387 | Topological Collusion Dynamics Value Transfer EDA | Lâm Huy | Specialist evidence retrieval, graph/topological pair features | Yes (14 Sep) |
| 0.66223 | Suspicious Detection | FOYSAL (teammate) | Broad feature set, pair-random CV | Yes |
| 0.64736 | poker_E23_clean_label_PN | hosen42 (us) | Clean-label P/N branch | Yes |
| 0.55758 | PU-Aware Evidence Ranker | Nomannic | PU pair model + evidence ranker | Yes |
| 0.21459 | Coordinated Collusion & Value-Transfer | Avik Das | 6-max features, broken final combiner | Yes |
| — | Getting Started: Pair Ranking & Evidence Retrieval | Florian DEROO (host side) | Logistic regression on 8 chip-flow features, evidence for top-100 pairs only | Starter; useful only for the host's hint list (§5) |
| — | Slash poker competition metric | Florian DEROO | Reference scorer | Yes |

## 3. The 0.69 notebook — what it actually does

Architecture (verified from the code, not just the write-up):

- **Training population.** 1,860 labelled pairs + up to 60 unlabelled pairs per table (`UNKNOWN_PER_TABLE = 60`, ≈ 24k PU pairs, ≈ 25.9k rows total). Unlabelled pairs exclude any player that is positive in dev labels and require `shared_hands ≥ min(eval shared_hands)`.
- **Fit weights.** positive 2.0, confirmed negative 1.0, PU 0.35. Model selection uses "PU-stress AP": treat PU as negative and reweight each PU row by `n_eval_pairs / n_PU` so the OOF prevalence matches the eval prevalence.
- **Pair risk.** `0.40·XGB(1600 trees, lr 0.022, depth 6) + 0.35·LGB(1200, 0.025) + 0.25·CatBoost(depth 6)`, 5-fold `StratifiedGroupKFold` by `table_id`.
- **Family.** Three one-vs-rest LightGBM detectors trained on positives only; isolation class weight 10. Positive-rate grid {0.5% … 1.4%}: pick the smallest rate whose PU-stress Behavior MAP is within 0.005 of the best; everything outside the top-ρ gets `none`. `other_coordination` is never predicted.
- **Evidence.** Trained on hands of labelled positive pairs only (target = hand in `development_evidence`). Global HGB + per-family "quad" (HGB + LGB + XGB + XGBRanker `rank:pairwise`, `0.80·mean(trees) + 0.20·ranker`). Final `0.60·family-specific + 0.40·global`, family chosen by the pair's *predicted* family. Every one of the 112,540 pairs is scored, in chunks of 8,000 pairs (≈ 9.6M pair-hands).
- **Channel split (their headline idea).** Some hand-level columns are *evidence-only* (`spr_min`, `spr_med`, `dump_spr`, `max_amount_to_bb`, `max_aggr_amount_to_bb`, board texture) and are excluded from the pair-level `mean/max/p95/top5` aggregation. Putting them into pair X cost them score.

Author's own ablation ladder (public LB):

| Public | Change |
|---:|---|
| 0.65682 | baseline |
| 0.6837 | rate/partner-vs-field features + triple GBDT |
| 0.68507 | + `iso_preflop`, `late_vs_blind`, `dump_from_oop`, `partner_calls_dump`; evidence z-blend |
| 0.69536 | SPR/`amount_to`/board as evidence-only; drop z-blend; score evidence for **all** pairs; rolling-20 + `isolation_purity_rate` in pair X |

Things the author says **hurt** (do not repeat): HU check-check / `ip_checks_river_hu` / `donor_stability` as pair rates; SPR/board in pair aggregates; z-scored evidence blend; evidence only for top-50k pairs; `dump_allin_weak` ("the dumper is not weak"); squeeze requiring a post-flop check-down; `other_coordination` as a 4th class; positive-rate cap > 1.4%; extra GBDT/TabPFN on the same pair X; clique / cross-phase / table-level XGBRanker on risk.

### 3.1 Bug found: board texture is dead code

`board_card_columns()` looks for hand columns whose name starts with `board` and treats each as **one card**. Our `hands.parquet` has a single string column `board_cards` = `"5d Jd 9d"`. `card_rank()` strips one trailing suit letter and casts to Int8 → null for every row; `board_paired` / `board_monotone` / `board_twotone` are therefore always 0 and `n_board_cards` is always 1. The "board evidence-only" feature group contributes nothing. This matters for us: **no public notebook uses real board texture or post-flop hand strength**, which is the whole premise of our Step 2.

### 3.2 Other observations

- Hole cards enter only as `hole_strength = max(rank) + 7·pair + 1.5·suited` — a pre-flop proxy. No 5/7-card evaluation, no "who had the best hand at showdown / at the fold".
- Positions come from the first pre-flop actor (`utg_seat`) rather than `button_seat` from `hands.parquet`; with 6 dealt players this is equivalent unless the UTG player has no recorded action.
- `is_pure_dump` uses the raw threshold `transfer ≥ 18 BB`; `partner_calls_dump` uses `transfer > 3 BB`. These are hand-tuned to the generator and may be why the notebook is stuck: chip-flow features saturate around 0.69.
- The write-up's claim that "hidden chip flow alone is never evidence; evidence is behaviour visible in `actions.parquet`" is consistent with our 14 Sep finding that 69% of labelled evidence hands have no showdown.

## 4. Where the 0.69 → 0.93 gap most likely is

All public notebooks share the same information diet: chip transfer, contribution gaps, fold/check counts, pre-flop hole strength, positions. They all plateau at 0.66–0.69. Since 95 teams beat 0.80, the extra signal is common enough to be found independently. Ranked by plausibility given what we verified locally:

1. **Card-aware features** (exact hand strength per street, best-hand-at-fold, best-hand-at-showdown, board texture). Hole cards for *all six* players are visible in `seats.parquet`, including folded hands — a synthetic-data artefact that a generator-planted "fold the best hand to the partner" or "check down the nuts" episode would expose directly. No public notebook does this. This is Step 2.
2. **Evidence MAP@5 quality.** Weight 0.20; public OOF evidence MAP values are ~0.4. Moving 0.4 → 0.8 alone is +0.08 overall. Card-aware hand features should raise this the most because evidence hands are the planted episodes.
3. **Pair AP on the full population.** The 0.69 notebook trains on 25.9k pairs; Step 2 trains on all ≈ 137k eligible pairs. Which is better is untested on the LB (Step 2 has not been submitted yet).

## 5. Host hints in the starter notebook

The official starter lists what a "stronger solution" should add. Ours already covers 2–5; 1 and 6 are not yet done:

1. chronological development validation — *not done; we use grouped folds by table*
2. compare each player with their behaviour against other opponents — partner-vs-field (Step 2)
3. partner-facing folds, calls, raises, heads-up checks — Step 1
4. temporal bursts and repeated interactions — partially (phase bins); rolling-20 window not yet
5. train an evidence ranker with `development_evidence.csv` — Step 1
6. report hard-negative performance and benign alternatives — *not done; needed for the write-up*

## 6. What to take from the 0.69 notebook into Step 2/3

Cheap, verified-to-help-them, not yet in our pipeline:

| Item | Cost | Why |
|---|---|---|
| Evidence-only channel split (`EVIDENCE_ONLY_COLS`) | low | Keeps SPR / bet-size / board (and our exact-strength features) out of pair aggregates where they add noise; author's +0.010 step |
| Four sequence templates: `iso_preflop`, `late_vs_blind`, `dump_from_oop`, `partner_calls_dump` | low | author's +0.0014 step; `iso_preflop` overlaps our Step 2 street-level isolation features — merge, don't duplicate |
| `transfer_roll20_max`, `isolation_roll20_max` (causal 20-hand window by `phase_progress`) | low | catches clustered episodes; host hint 4 |
| `isolation_purity_rate` = rate of hands with outsider fold and no partner fold | low | simple, family-specific |
| Table percentile of key pair metrics (`_table_pct`) | low | normalises stakes/table style |
| Behaviour positive-rate grid selected on PU-stress Behavior MAP | low | we currently pick the family for every pair; their `none` outside top ~1% is what the metric rewards for the 0.10 component |
| Score evidence for **all** pairs, chunked | already done in Step 1 | keep |
| Triple-GBDT blend 0.40/0.35/0.25 | medium | small, stable gain; do after features |

Do **not** copy: their board-texture code (dead), z-blend, `other_coordination`, top-50k evidence.

## 7. Recommended next actions

1. **Run Step 2 locally and submit it.** It is the only pipeline on our side (or public) with card-aware evidence. Compare against 0.67117 (our code) and 0.69243 (public re-run), remembering ±0.004 noise.
2. If Step 2 evidence MAP@5 rises but pair AP does not: add the evidence-only channel split and the four templates above to the pair model (Step 3 scope).
3. Add the `none`-outside-top-ρ rule with the rate grid; it is a 10-line change and touches only the 0.10 component.
4. Keep every experiment on one validation frame: grouped folds by table, PU-stress AP for pair selection, official MAP@5 for evidence, and log all three components separately.
5. Before the deadline (≈ 21 Sep) write the hard-negative / benign-alternative section (host hint 6); it is a judging requirement regardless of LB.

---

# Part B — What are the top teams doing? (added 16 Sep, evening)

## B.1 Public resources from top-ranked teams: none exist yet

Checked on 16 Sep 2026:

| Source | Result |
|---|---|
| Kaggle Code tab | 8 public notebooks, best 0.688; none by a top-100 team member |
| Public notebooks of all 26 top-25 usernames (`kaggle kernels list --user`) | none related to this competition |
| Competition discussion tab | only 5 threads, none by top teams; host answers summarised in B.2 |
| GitHub (repo + code search) | only `phucthaiv02/poker-suspicious-detection` (already audited 14 Sep, last push 7 Sep) |
| Web search | nothing beyond the Kaggle page and that repo |

The competition closes ≈ 21 Sep; winner write-ups will only appear after that. Everything below is therefore **inferred from the data**, not copied from a top team.

## B.2 Host clarifications in the discussion threads

- Unlisted development pairs are **unlabelled, not clean**; treating them as negatives is "an assumption made by your method".
- `players.parquet` pairwise features (same region, same client, account-age gap …) are **allowed**. Tested locally: every such feature has AUC ≈ 0.50 on the 1,860 labels (same_region 0.278 vs 0.304, age-gap AUC 0.500). **Dead end — do not spend time on it.**
- External data and pretrained models are allowed.
- `board_cards` holds flop/turn/river as one string; hands ending early have fewer cards.

## B.3 Reverse-engineering the 0.93 scores

`0.93 = 0.70·AP + 0.20·MAP@5 + 0.10·BMAP` is only reachable with **pair AP ≈ 0.95 and evidence MAP@5 ≈ 0.8**. Our Step 1 has PU-stress AP ≈ 0.65 and MAP@5 0.39; every public notebook is in the same range. So the top teams have a hand-level detector that finds the planted hands almost perfectly. Three hypotheses were tested on the labelled evidence (1,817 hands inside 45,129 hands of the 372 positive pairs):

| Hypothesis | Test | Result |
|---|---|---|
| Temporal clustering ("episodes") | span of the 5 evidence hands within the pair's shared hands | median 0.59 of the pair's history vs 0.67 for 5 random hands → **not clustered**; only mild skew to early hands. Rolling windows will not be the key. |
| Post-flop "folded the better hand" (all hole cards are visible, including folded ones) | donor folded while holding a hand that beats the receiver at the final board | directed_transfer evidence 54% vs 20% baseline; soft_play 33% vs 21%; isolation no signal. Real but secondary; pair-level AP 0.28. |
| **Pre-flop action inconsistent with hole cards** | both partners voluntarily enter the pot (VPIP) and with what cards | **Strong.** See table below. |

Reading raw evidence hands makes the template obvious: the pair members enter pots together with junk (8♦3♠ raises and 6♦3♣ calls; 7♣2♥ calls; T♣3♠ raises; 6♣4♦ calls a raise; 7♥T♦ opens). The simulator's normal policy is strongly card-dependent — VPIP is 2% for Chen ≤ −1.5, 8% for Chen 0, 35% at Chen 5, 83% at Chen 8, 96% at Chen ≥ 12 — so a junk-hand entry is a large policy deviation, and two simultaneous deviations by the same two players are the planted signal.

Hand level (is this hand a labelled evidence hand, within positive pairs):

| Feature | non-evidence rate | evidence rate | AUC all | AUC directed | AUC soft | AUC isolation |
|---|---:|---:|---:|---:|---:|---:|
| both partners VPIP | 0.168 | **0.794** | — | — | — | — |
| both VPIP, one with Chen ≤ 4 | 0.102 | 0.455 | — | — | — | — |
| `junk_vpip_score` = Σ over partners VPIP·max(0, 8 − Chen) | | | 0.789 | 0.851 | 0.693 | **0.809** |
| `junk_pfr_score` (same, pre-flop raise only) | | | 0.717 | 0.739 | 0.594 | **0.864** |
| chip transfer (what public notebooks use) | | | 0.833 | 0.934 | 0.864 | 0.623 |

Pair level (372 positives vs 1,488 confirmed negatives, single feature, no model):

| Pair feature | AP | AUC |
|---|---:|---:|
| mean chip transfer | 0.586 | 0.807 |
| rate of both-VPIP hands | 0.551 | **0.882** |
| rate of both-VPIP-with-junk hands | **0.591** | **0.892** |

One crude card-conditional feature matches the best chip-flow feature on AP and beats it clearly on AUC, and it is nearly orthogonal (isolation family: 0.86 vs 0.62). Nobody public uses it.

## B.4 What this implies the top teams built

Most likely a **policy-deviation model**: fit the population's action policy given private information (hole-card class × position × street × to_call/pot × stack), score every player-action by its surprise under that policy, and aggregate surprise for the pair (a) in hands where both are involved and (b) relative to each player's surprise when the partner is absent. The planted hands are exactly the ones where both partners act "wrongly" for their cards at the same time; with that, both the hand ranker (MAP@5) and the pair ranker (AP) become near-perfect on synthetic data. Chip transfer then only tells you the *direction* (which family).

## B.5 Concrete plan (replaces §7 above)

1. **Step 2 stays** (exact strength, partner-vs-field), but its most valuable component is now clear: card-conditional action features. Add, per player-hand: Chen/169-class of the hole cards, `vpip`, `pfr`, `3bet`, amount/pot, and the **surprise** = −log P(action | hole class, position, to_call bucket) estimated from all development seats (12M rows; a simple grouped frequency table is enough for a first version). Pair-hand features: both-VPIP, sum/min/max surprise of the two partners, junk-entry indicators, plus "partner was already in the pot when the junk entry happened".
2. Pair features: rates of the above over shared hands, and the same rates for each player when the partner is absent (contrast). Expect pair AP to move first; check PU-stress AP on the full eligible population.
3. Evidence ranker: add the same hand-level columns; evidence MAP@5 should rise well above 0.39.
4. Only after that: the cheap items from the 0.69 notebook (evidence-only channel split, `none` outside top-ρ, rolling-20).
5. Submit; compare to 0.67117 / 0.69243 remembering ±0.004 noise.
