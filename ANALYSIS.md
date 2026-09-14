# Detect Suspicious Value Transfers in Poker — Full Competition Analysis

*Kaggle community competition hosted by Slash · analysed 2026-09-14 (≈7 days before close) · based on the official pages, the host's reference metric code, the downloaded data, 8 public notebooks, and 105 adversarially-verified findings from a 7-lens analysis.*

---

## 1. Executive summary

1. **The task is three problems in one**: rank 112,540 evaluation-period player pairs by collusion risk (70% of score), retrieve up to 5 "evidence" hands per pair (20%), and name the collusion family (10%). Score = 0.70·PairAP + 0.20·EvidenceMAP@5 + 0.10·BehaviorMAP.
2. **The metric has dominant strategies, verified by running the host's code**: always submit 5 shared evaluation hands for *every* pair (evidence for negatives is never read, and appending a hand can never lower a score); never predict `none` — it is metric-identical to `other_coordination` and both zero the pair in all three family rankings; never create ties in `risk_score` (ties are broken by lexicographic `pair_id`).
3. **The labels are positive-unlabelled and the evaluation population is structurally different**: 372 trusted positives + 1,488 confirmed hard negatives in the development period; eval pairs contain *no* publicly-positive player and *no* labelled pair, and include an undisclosed 4th family. Every eval positive is a pair whose coordination was hidden, started late, or is of the 4th kind.
4. **"CV 0.97, LB 0.6" is the trap**: on the real data, seats-only pair features give AP 0.80 on positives-vs-confirmed-negatives but only **0.19–0.22** against the true eligible population (positive prevalence ≈0.3%). Public notebooks report the same gap (0.97 vs 0.52–0.73).
5. **Positives and negatives share the same exposure distribution by design** (min 57 dev shared hands, mean ≈121 for both); exposure will not discriminate, but it must be used to normalise everything else.
6. **Evidence hands have crisp, family-specific signatures** (this analysis, on the real data): directed_transfer = same loser in 100% of a pair's evidence hands, partner wins 100%, big pots (79 BB), loser calls partner's bets then folds; soft_play = both hold strong cards, heads-up, call/check down to showdown (35% vs 3.6%), raise-vs-partner rate 0.017; coordinated_isolation = both raise, third parties fold to them (2.98/hand vs 0.88), then one folds to the other. Two of the three families are invisible without action-level "vs-partner" features (family PU-AP 0.05 and 0.004 with seats-only features).
7. **Every public notebook scores 0.56–0.66 on the public LB** (their "PU-stress" CV tracks it; their 0.97 "labelled" CV does not), while the leader is at 0.938 and twelve teams are ≥0.90. Provable bounds for the leader: PairAP ≥ 0.911, EvidenceMAP ≥ 0.69. Positions 2–12 (0.900–0.917) are within public-LB noise. The public recipe is fully documented in §13.
8. **Recommended architecture**: an *evidence-first* pipeline — a pair-conditional, hole-card-aware hand scorer trained on planted evidence hands, aggregated (top-k, counts, burstiness, directedness) into pair risk with GroupKFold-by-table PU-aware validation; family heads on top; the same hand scorer produces the evidence slots.
9. **Rules that bite**: competition data is *Competition Use Only* (do not push it to GitHub — this happened and was reverted); no exploitation of ID formats/row ordering/generator internals; winners must publish a ≤1,500-word write-up, reproducible code (OSI licence) and five case reviews with benign alternatives within 7 days.
10. **7-day plan**: day 1 pipeline + validity-checked baseline; days 2–3 evidence model; days 4–5 pair model + family heads; day 6 ensembling/selection; day 7 freeze and write-up skeleton. Details in §9.

---

## 2. Competition at a glance

| Item | Value |
|---|---|
| Host / sponsor | Slash (Slash Financial, Inc., San Francisco) — community prediction competition, id 163730 |
| Prizes | $5,000 total: $3,000 / $1,250 / $750. No Kaggle points or medals |
| Timeline | Launched ≈ 2026-09-05; closes ≈ 2026-09-21 ("7 days to go" on 2026-09-14). Exact times on Overview › Timeline (login) |
| Participation (2026-09-13) | 374 entrants · 216 participants · 211 teams · 2,152 submissions; leaderboard on 09-14: #1 0.93798, #2 0.91728, 12 teams ≥ 0.90 |
| Submissions | 5 per day; select 2 final; team size ≤ 10; mergers allowed subject to the submission-count rule |
| Public / private | ≈30% / 70% of evaluation pairs, split stratified by behaviour family |
| Data licence | *Competition Use Only* — no redistribution, no publication |
| External data / models | Allowed if public and free (host confirmed in the discussion) |
| Winner licence | Open Source (OSI-approved, no commercial restriction) |
| Winner obligations (7 days after private LB) | Solution write-up ≤ 1,500 words; public reproducible code; **five case reviews** (pair id, hand ids, observable behaviour, plausible benign alternative) |
| Prohibited | "Exploiting ID formats, row/file ordering, generator internals, or any artifact unrelated to gameplay" |
| Links | [Competition](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker) · [Reference metric notebook](https://www.kaggle.com/code/florianderoofr/slash-poker-competition-metric) · [Public baseline README](https://github.com/phucthaiv02/poker-suspicious-detection) |

Discussion has 4 threads (§13.4): rules clarification (external data OK; metric published), *"CV-LB difference"* (unanswered — §7 explains it), a question about the ~156k unlabelled pairs, and a host confirmation that `players.parquet` pairwise features are allowed.

---

## 3. The task

**Goal.** Detect coordinated player relationships in 2,000,000 synthetic 6-max NLHE hands: rank every evaluation pair, identify the behaviour family, and submit the hands that best support each alert.

**Target behaviours**

| Family | Definition (host) | Dev positives |
|---|---|---|
| `directed_transfer` | one player deliberately loses value to another | 148 |
| `soft_play` | partners avoid normal aggression against each other | 132 |
| `coordinated_isolation` | partners pressure other players while limiting conflict between themselves | 92 |
| `other_coordination` | a fourth, undisclosed mechanism that never appears in public positive labels | 0 (exists only in eval) |

**Complications stated by the host**
- Coordination is *episodic*: manipulated hands are mixed with ordinary play; a relationship is not necessarily active across the whole timeline.
- Deliberate benign confounders: tilt, weak play, similar strategies, repeated opponent selection, streaks, strategy changes. Players differ in how much they play, so exposure varies.
- Labels are PU: `development_labels.csv` lists trusted positives and confirmed non-targets; **every unlisted development pair is unknown, not negative**.
- Evaluation pairs exclude publicly-labelled pair ids *and any pair containing a publicly-labelled positive player*.
- Every planted evidence hand "contains a behavior-specific action visible in the public action log; latent scenario activation alone is never evidence."

---

## 4. Data

### 4.1 Entity model

`players (12,000)` → each belongs to exactly one `table_id` (400 pools × 30 players) → each pool plays **exactly 5,000 hands** (first 3,000 = `development`, last 2,000 = `evaluation`) → `hands (2,000,000)` → `seats (12,000,000: 6 per hand)` → `actions (18,609,028, ≈9.3 per hand)`. Join gameplay on `hand_id`, players on `player_id`. Blind level is constant per table (BB = 2: 218 tables, 4: 143, 10: 39). `started_at` spans 2026-01-01 → 2026-02-02 (simulated).

### 4.2 Schemas

**players.parquet** (PK `player_id`): `account_age_days` int · `experience_hands_bucket` {new, developing, experienced, veteran} · `preferred_stake` {micro, low, mid} · `region_bucket` {americas, apac, europe, other} · `client_family` {desktop, mobile, web}.

**hands.parquet** (PK `hand_id`): `table_id` · `started_at` (UTC) · `phase` {development, evaluation} · `button_seat` 0–5 · `small_blind` · `big_blind` (1/2, 2/4, 5/10) · `board_cards` (space-separated; empty when the hand ends preflop — 46.7% of hands) · `final_pot` · `players_dealt` (always 6) · `players_at_showdown` (0 = no showdown; 84.4% of hands).

**seats.parquet** (PK `hand_id, player_id`): `seat_no` · `starting_stack` · `hole_card_1/2` (visible! "operator-side investigation setting") · `total_contribution` · `net_chips` (sums to 0 per hand) · `folded` · `went_to_showdown` · `won_share` ∈ [0,1].

**actions.parquet** (PK `hand_id, action_no`): `street` {preflop, flop, turn, river} · `player_id` · `action` {fold, check, call, bet, raise, all_in} · `amount` (additional chips; 0 for check/fold) · `amount_to` (street total after action) · `pot_before` · `stack_before` · `to_call` (0 when checking is available) · `players_active`. `all_in` can be a call or a raise — compare `amount` with `to_call`.

**development_labels.csv** (1,860 rows): `pair_id, player_1, player_2, label, label_status {confirmed_target 372 / confirmed_non_target 1,488}, behavior_family`.
**development_evidence.csv** (1,817 rows): `pair_id, evidence_rank (1–5), hand_id, behavior_family` — 340 pairs have 5 hands, 21 have 4, 11 have 3.
**evaluation_pairs.csv** (112,540 rows): `pair_id, player_1, player_2, shared_hands` (= number of evaluation hands with both players dealt; verified exact).
**sample_submission.csv**: `risk_score=0.0, predicted_behavior=none, evidence_hand_1..5=NO_EVIDENCE`.

### 4.3 Exposure and the pair-selection rule (reverse-engineered, exact)

- Eval-period pairs with ≥1 shared hand: 172,878. `evaluation_pairs.csv` = exactly the pairs with **`shared_hands ≥ 38`** (1.9% of 2,000) that are not labelled and contain no positive player. Excluded: 19,127 pairs containing a positive player, 1,291 labelled non-target pairs, 39,891 pairs with <38 shared hands, 29 unexplained.
- `shared_hands` distribution: min 38 · p25 56 · median 76 · mean 85.8 · p75 106 · p90 140 · max 419. **No eval pair has fewer than 5 shared hands → `NO_EVIDENCE` is never needed.**
- Labelled dev pairs obey the equivalent rule: dev shared hands ≥ 57 (1.9% of 3,000) for both positives (mean 121.3, median 112) and negatives (mean 120.0, median 108.5). **Exposure is deliberately non-discriminative.**
- 693 positive players (none in eval pairs); 2,599 confirmed-negative players (2,433 in eval pairs); 11,302 distinct players in eval pairs; 245 of 400 tables have ≥1 positive (155×1, 63×2, 18×3, 8×4, 1×5); mean 281 eval pairs per table.
- Hands per player: dev mean 600 (126–1,243), eval mean 400 (44–906).

### 4.4 Evidence-hand anatomy (computed on the real data)

All 1,817 evidence hands are development-phase and contain both players; 69% end without showdown; pots are larger than typical (overall median 9.5 BB; evidence median 34.5 BB for directed_transfer, ≈17 BB for the other two).

Per-family comparison of **A** = evidence hands vs **B** = other shared hands of the same positive pairs vs **C** = shared hands of confirmed negatives (means per hand; `AP(A|B)` = univariate average precision for picking evidence hands among a positive pair's shared hands, base rate 0.04):

| Family | Feature | A | B | C | AP(A\|B) |
|---|---|---|---|---|---|
| directed_transfer | partner won pot & other partner lost money | **1.00** | 0.20 | 0.16 | 0.18 |
| | calls of partner's bet by a partner | 1.60 | 0.24 | 0.07 | 0.20 |
| | both partners voluntarily in pot | 0.90 | 0.14 | 0.07 | 0.19 |
| | pot (BB) | 79 | 44 | 31 | 0.07 |
| | loser's contribution (BB) | 27 | 9.6 | 4.5 | 0.10 |
| | same player is the loser across the pair's evidence hands | **100%** of pairs | | | |
| soft_play | calls of partner's bet | 0.93 | 0.06 | 0.07 | **0.28** |
| | both voluntarily in | 0.84 | 0.12 | 0.07 | 0.19 |
| | both to showdown | 0.35 | 0.04 | 0.01 | 0.13 |
| | heads-up between partners | 0.25 | 0.02 | 0.01 | 0.11 |
| | preflop strength (Chen) p1 / p2 | 6.5 / 6.3 | 3.8 / 3.8 | 3.9 | 0.09 |
| | raises against partner | 0.017 | 0.007 | 0.028 | — |
| coordinated_isolation | third-party folds to a partner's aggression | **2.98** | 0.88 | 0.71 | **0.23** |
| | partners who took an aggressive action | 1.43 | 0.60 | 0.46 | 0.11 |
| | re-raise of partner | 0.47 | 0.10 | 0.03 | 0.09 |
| | fold to partner | 0.81 | 0.27 | 0.18 | 0.10 |

Canonical evidence hands:
- **directed_transfer** — both partners voluntarily play a big pot; the *designated loser* calls the partner's bets (often with the weaker holding) and either folds to a later bet (72% no showdown) or loses at showdown; the same player loses in every evidence hand of the pair (perfectly directed).
- **soft_play** — both partners hold above-average cards, end up heads-up, and neither raises: call/check-down to showdown at 10× the normal rate. The signature is *absence* of aggression given the cards.
- **coordinated_isolation** — one partner raises, the other re-raises (squeeze) while third parties are still in; third parties fold; the first raiser then folds to the partner or the hand ends. Only this family's evidence hands cluster in time (median span 960 hands vs 1,640 for random draws; the other two families spread across the window).

### 4.5 Compute plan

The whole dataset fits in memory on a 24 GB laptop with pandas + pyarrow (column selection!): hands 1s, seats 2s, actions 4s to load. Building all 137,739 eligible dev pair × hand rows with seats-only features takes ≈20s via per-table self-joins (`seats ⋈ seats on hand_id`, 15 pairs per hand). Action-derived pair-conditional features (last aggressor, fold-to-partner, etc.) cost ≈10s per 200k pair-hands with a Python loop; vectorise with `groupby.ffill` of the last aggressor for the full 9.6M eval pair-hands (≈5–10 min). Polars/duckdb are not installed locally; the public notebooks use them on Kaggle.

---

## 5. Metric deep-dive (verified by running the reference code)

```python
def _average_precision(y_true, scores):
    order = np.argsort(-scores, kind="mergesort")   # stable → ties by pair_id order
    ranked = y_true[order]; tp = np.cumsum(ranked); ranks = np.arange(1, len(ranked)+1)
    return float(np.sum((tp / ranks) * ranked) / positives)
# behaviour: behavior_risk = np.where(predicted_behavior == family, risk, 0.0), macro over 3 families
# evidence: for each y_true==1 pair: AP@5 of cleaned submitted ids vs planted ids, / min(|relevant|,5)
final = 0.70*pair_ap + 0.20*evidence_map + 0.10*behavior_map
```

### 5.1 Pair AP (70%)
- Non-interpolated AP = mean over positives of precision at the positive's rank; identical to sklearn when there are no ties. Every well-ranked positive is worth ≈0.7/P of final score.
- **Ties are broken by lexicographic `pair_id`** (truth is `sort_index()`-ed, then stable mergesort). A constant score gives AP ≈ prevalence; 100 positives + 100 negatives tied at the top scored 1.00 or 0.31 depending on ids. → Never quantise below 4 decimals, never clip/floor to a shared value, never zero-out low scores. Monotone calibration (Platt, temperature) is bit-neutral.
- **AP depends strongly on prevalence**: at fixed separation (d′=2.5) AP is 0.27 at 0.18% prevalence, 0.45 at 0.9%, 0.70 at 4.4%; the same scores evaluated on "positives + confirmed negatives" (20% prevalence) read 0.87–0.98. This is the mechanism behind the CV–LB thread.
- Public-LB standard error for PairAP with ~90–150 public positives is ≈0.015–0.03; positions 2–10 are not statistically separated.

### 5.2 Evidence MAP@5 (20%)
- Computed **only over true positive pairs**, from the evidence columns alone; `risk_score`, `predicted_behavior` and the pair's rank are irrelevant; evidence for negative pairs is never read. A "missed target" means *no correct hand*, not *ranked low*.
- `NO_EVIDENCE`, NaN, empty and whitespace cells are stripped **before** ranking; appending any hand can never lower the per-pair score (exhaustive check, 1.9M cases, 0 violations). → **Submit 5 distinct shared evaluation hands for all 112,540 pairs.**
- Order matters: two correct hands at ranks 1–2 = 0.40 (|rel| = 5); ranks 4–5 = 0.13; with |rel| = 2, ranks 1–2 = 1.0 vs ranks 4–5 = 0.325. Denominator = min(|relevant|, 5).
- Random ranking floor ≈ 2.3/N: 0.035 at 70 shared hands, 0.024 at 100. Each +0.05 MAP@5 = +0.01 final.
- Hand ids are matched as exact strings (case-sensitive, whitespace-stripped): write them verbatim as strings; float-formatted ids (`123.0`) score 0. A repeated real id in a row invalidates the submission; repeated `NO_EVIDENCE` is fine.
- A positive pair whose private evidence set is empty scores 0 regardless; other_coordination positives are (presumably) included in the loop — a generic "anomalous-vs-partner" hand scorer is needed to get their credit.

### 5.3 Behaviour MAP (10%)
- Macro over the 3 disclosed families; for each family, score = `risk_score` if that family was predicted else 0, ranked over all 112,540 pairs. Absent family = 0. `other_coordination` is excluded from the component.
- **`none` and `other_coordination` are metric-identical** (both put the pair in the zero block of all three rankings). Predicting a disclosed family for every pair is near-optimal: `none` for the bottom 50–80% changes nothing (±0.0003), `none` for the bottom 95% costs −0.001 to −0.011, for the bottom 99% −0.04 to −0.06. A wrong family at the very top costs ≈4× a `none`, but in aggregate thresholding loses.
- Each family's AP is normalised by its own positive count → labels are worth more in rarer families; choose by ≈ argmax q_F / n_F rather than raw argmax.
- Combining family heads into one `risk_score` needs common calibration: raw max/mean of miscalibrated heads scored 0.18–0.36 vs 0.68 for isotonic-calibrated noisy-OR or max-of-ranks; a single pair-level risk model beat any max-combination by ≈0.05 AP in simulation.
- Realistic attainable BehaviorMAP ≈ 0.85–0.95 (public notebooks: 0.91 OOF).

### 5.4 Score decomposition vs the leaderboard

| Public score | Provable | Plausible (PairAP, Ev, Beh) |
|---|---|---|
| 0.938 (#1) | PairAP ≥ 0.911, EvMAP ≥ 0.69, BehMAP ≥ 0.38 | (0.95, 0.85, 0.95) |
| 0.916 (#2) | PairAP ≥ 0.880 | (0.93, 0.80, 0.93) |
| 0.899 (#10) | PairAP ≥ 0.856 | (0.91, 0.75, 0.92) |
| 0.879 (#25) | PairAP ≥ 0.827 | (0.89, 0.70, 0.90) |
| 0.837 (#49) | 0.2·Ev + 0.1·Beh ≥ 0.137 | (0.85, 0.60, 0.85) |

A working evidence retriever is already table stakes at rank 49. The best public notebook projects ≈0.84 on its own CV; the leader's ≥0.911 public PairAP on the real (PU, positive-player-free) population is the qualitatively different thing.

### 5.5 Submission validity
Invalid: missing/extra `pair_id`s, duplicated `pair_id`, non-numeric or out-of-[0,1] `risk_score`, any behaviour string outside the 5 allowed, a repeated real hand id within a row. Empty cells are tolerated by the metric but Kaggle's CSV upload is not — use `NO_EVIDENCE`. Always run the reference `score()` on a fake solution before uploading.

---

## 6. Poker-domain signatures and confounders

| Family | Mechanism | Hand-level detector (hole cards visible) | Pair-level detector | Benign look-alike & discriminator |
|---|---|---|---|---|
| directed_transfer | B "dumps" chips to A | A wins pot, B loses ≥ x BB; B calls A's bets with the weaker hand then folds/loses; big pot; both VPIP | directed transfer sum A→B per shared hand, its asymmetry, top-3 transfer, residual of B's win-rate vs own baseline, consistency of loser identity | *Weak play / tilt*: B loses to **everyone**, not only to A; tilt is temporally clustered but undirected. Use B's loss share to A vs to the other 28 |
| soft_play | partners refuse to bet into each other | both strong, heads-up, no raise, check/call down, showdown | check-down rate when heads-up, raise-vs-partner rate ÷ raise-vs-others rate (given hand strength), showdown rate vs partner | *Similar passive strategies*: passivity is symmetric vs everyone; soft play is passive only vs the partner |
| coordinated_isolation | squeeze third parties out | raise → partner re-raise while others in → others fold → raiser folds to partner | both-raise-same-hand rate, third-party folds following partner aggression, re-raise-of-partner rate, subsequent fold-to-partner | *Aggressive similar strategies*: both raise everyone; isolation shows re-raise + fold pattern specifically vs each other |
| other_coordination (unknown) | candidates: hole-card sharing (folding hands dominated by partner's cards; hyper-accurate play when partner still in), coordinated all-in shoves, chip-passing via check-downs/chops, seat/timing coordination | fold-with-dominated-hand rate, equity-conditional accuracy vs partner, chop frequency | anomaly score vs confirmed-negative distribution, low on all three family heads | Detect as open-set: high generic risk, low family confidence |

**Hole cards change everything.** Preflop: a 169-class strength table (Chen formula is adequate; a precomputed equity table is better). Postflop: a 7-card evaluator (pure-Python is fine at 10M hands with vectorisation, or `treys`/`phevaluator` on Kaggle) gives "folded the best hand", "called with a hopeless hand", "checked down the nuts". These are the features an investigator would cite in the required case reviews.

**Case-review template** (winner verification): pair id · 5 hand ids · what is observable (e.g., "B called 3 streets with 7-high vs A's top pair and folded river; net −68 BB to A; A never lost to B in 112 shared hands") · plausible benign alternative (e.g., "B is a losing player overall") · why rejected (e.g., "B's losses vs the other 28 players are within baseline").

---

## 7. PU learning, validation protocol, and the CV–LB gap

### 7.1 What the labels really are
- Positives: 372 "trusted" pairs — likely the clearest cases.
- Confirmed negatives: 1,488 pairs chosen to be *hard* (the named confounders). They are excellent for teaching "what collusion is not", but they are not a random sample of negatives.
- Unlabelled dev pairs (≈136k eligible): mostly negative, with an unknown number of hidden positives.
- Eval positives: hidden-in-dev, late-starting, or 4th-family pairs — **by construction not the trusted-positive type**. Their dev-period hands are available and can be used as a within-pair baseline.

### 7.2 Measured on the real data (this analysis, seats-only features, GroupKFold by table)

| Training negatives | AP on positives vs confirmed negatives | AP on positives vs all eligible unlabelled pairs (eval-mirrored) |
|---|---|---|
| confirmed negatives only | 0.804 | 0.191 |
| all unlabelled (PU-naive) | 0.801 | 0.223 |
| unlabelled + confirmed | 0.800 | 0.215 |

Per family (eval-mirrored): directed_transfer **0.343**, soft_play **0.050**, coordinated_isolation **0.004**. Univariate: the best seats-only feature (`transfer_rate`) has AP 0.59 labelled vs 0.08 eval-mirrored. Conclusion: (a) report *only* eval-mirrored AP; (b) action-level pair-conditional features are mandatory for two of the three families; (c) unlabelled pairs are usable as negatives (slightly better than confirmed-only).

### 7.3 Validation recipe
1. Folds = `GroupKFold(table_id)`, 5 folds (leave-pools-out ⇒ player-disjoint).
2. Training set per fold: positives + confirmed negatives + all unlabelled eligible pairs (≥57 dev shared hands) of the training pools, excluding pairs containing a positive player (mirrors the host's rule).
3. Score **all** eligible pairs of the held-out pools; PairAP = positives vs everything else (unlabelled treated as negative — pessimistic if hidden positives are recovered, optimistic if not; report both "vs unlabelled" and "vs confirmed").
4. Evidence: for held-out positive pairs rank their shared dev hands with the hand model; MAP@5 against `development_evidence`. Optionally thin candidates to 2/3 to match the eval candidate-pool size.
5. Behaviour: argmax family head; compute BehaviorMAP with the reference code.
6. Compose with the reference `score()` on the held-out pools; report per-family PairAP and the public-split noise (bootstrap 30% of pairs stratified by family).
7. Add a **temporal check**: train on the first half of the dev window, test on the second half, to see the drift the pool split cannot.

### 7.4 Why CV > LB (ranked by likelihood)
1. CV computed on labelled pairs only (20% prevalence) — AP inflates mechanically (§5.1).
2. Eval positives are a different kind (hidden/late/4th family) — trusted-positive features may not transfer.
3. Player leakage (non-grouped folds, per-player baselines fit on test players).
4. Family heads or thresholds tuned on 92–148 positives.
5. Public LB noise (~100–150 positives).

---

## 8. Evidence retrieval design

- **Unit**: (pair, shared evaluation hand). Candidates per pair: 38–419 (median 76).
- **Training**: positives = 1,817 planted hands; negatives = other shared hands of the same positive pairs (the leader bound ≥0.69 implies the listed 5 are nearly the complete set of visible-action hands, so this adds little label noise) + shared hands of confirmed negatives. Group folds by table.
- **Features (pair-conditional)**: which partner won/lost and how much (BB), both-VPIP, heads-up flag, calls/raises/folds *to the partner* vs *to others* (derive "last aggressor" per action), third-party folds after partner aggression, re-raise-of-partner, pot size, streets reached, both-showdown, each partner's preflop strength and gap, postflop made-hand ranks, "folded the better hand to partner", position vs button, hand's chip flow relative to the pair's own distribution (z-score within pair), position in time within the pair's activity.
- **Model**: GBDT binary (family-agnostic) + three family-specific heads; blend by the pair's predicted family posterior. Public notebooks gain +0.04–0.05 MAP@5 from family-specialised heads (0.45 → 0.50). Consider LambdaRank grouped by pair.
- **Retrieval**: top-5 distinct shared evaluation hands by score for every pair; ties broken by pot size; write ids as strings; never `NO_EVIDENCE` (no pair needs it).
- **Feeds Pair AP**: aggregate hand scores per pair — max, top-3/top-5 mean, count above the confirmed-negative p99 (z-scored against an N-matched null), directedness of the top-k hands' loser, burstiness of top-k hand indices — these are the strongest pair features available.
- **Expected MAP@5**: public notebooks 0.35–0.51 with modest features; the leader ≥0.69 on public; 0.6–0.7 is a realistic 7-day target with hole-card-aware features.

---

## 9. Recommended solution architecture and 7-day plan

**Thesis: evidence-first.** The 20% component and the hardest part of the 70% component are the same problem — find the manipulated hands. Build the hand scorer first; pair risk is an aggregation of it plus exposure-normalised pair statistics; family heads reuse the same features.

```
seats ⋈ hands (+hand_idx)            actions → per-action context (last aggressor, facing partner?)
        │                                        │
        ▼                                        ▼
pair-hand table (dev: labelled + all eligible; eval: 112,540 pairs × shared hands ≈ 9.6M rows)
        │
        ├── hand scorer (GBDT; +3 family heads)  ──► evidence_hand_1..5 (top-5 per pair)
        │            │ OOF scores
        ▼            ▼
pair features (directed flow, vs-partner rates ÷ vs-others rates, residuals vs player baselines,
               top-k hand scores, burstiness, dev-period baseline of the same pair)
        │
        ├── pair risk model (GBDT, PU training set, GroupKFold by table) ──► risk_score (rank-avg of seeds/models, ≥6 dp)
        └── family heads (one-vs-rest on positives) ──► predicted_behavior = argmax q_F/n_F (never 'none';
             'other_coordination' only for very high risk with flat family posterior — optional, metric-neutral)
```

| Day | Deliverable | Notes |
|---|---|---|
| 1 | Data loaders, pair-hand builder (dev + eval), reference-metric CV harness, **valid baseline submission** (seats-only pair GBDT; evidence = top-5 by partner chip transfer) | Expect ~0.6–0.7 public; confirms the pipeline and format |
| 2 | Action context (last aggressor, facing-partner flags), hole-card strength; hand scorer v1; MAP@5 CV | Target MAP@5 ≥ 0.5 |
| 3 | Family-specific hand heads; postflop evaluator; hand scorer v2; evidence slots from v2 | Target MAP@5 ≥ 0.6 |
| 4 | Pair features incl. OOF hand-score aggregates, vs-partner/vs-others ratios, residuals; pair risk model with PU training set; eval-mirrored PairAP | The number to watch |
| 5 | Family heads + calibrated combination; behaviour routing; open-set flag; per-family CV | |
| 6 | Seeds/model rank-averaging, temporal check, public-split bootstrap; choose 2 finals (best CV, best LB) | |
| 7 | Freeze; reproducibility (`requirements.txt`, seeds, one-command run); draft write-up + 5 case reviews from the submitted evidence | Winner obligations are due 7 days after the private LB |

Model starting points: LightGBM `objective=binary, learning_rate=0.03, num_leaves=15–31, min_data_in_leaf=20–50, feature_fraction=0.7–0.8, bagging 0.8, lambda_l2=5, scale_pos_weight≈n_neg/n_pos, early stopping on eval-mirrored AP`; 5 seeds; rank-average. Optional: CatBoost for diversity.

Ideas from the public notebooks worth keeping: PU-stress evaluation (`detecting-collusive-…`, `suspicious-detection`), family-specialised dual-engine evidence rankers (+0.05 MAP@5), empirical-Bayes shrinkage of pair rates by exposure, "counterfactual action residuals" (what would this player normally do here), nested behaviour activation. Ideas to drop: `none` routing below a risk threshold, per-pair `NO_EVIDENCE`, CV reported on labelled pairs only.

---

## 10. Rules-compliance and leak-avoidance checklist

- [ ] Data never leaves the machine: `.gitignore` all `*.parquet`/`*.csv`; never attach data to a public notebook/repo (the winner code must *read* the Kaggle data, not ship it).
- [ ] No features from `pair_id`/`player_id`/`hand_id`/`table_id` strings, row order, file order, or `started_at` regularities unrelated to gameplay (hand order *within a table* is gameplay and fine; ids are only join keys).
- [ ] No ties in `risk_score`; ≥6 decimals; no clipping to a shared floor.
- [ ] `predicted_behavior` ∈ {directed_transfer, soft_play, coordinated_isolation, other_coordination}; never `none`.
- [ ] Five distinct `hand_id` strings per row, all evaluation-phase and shared by both players; no floats; `NO_EVIDENCE` only as a last resort.
- [ ] Run the reference `score()` against a synthetic solution before every upload; check `pair_id` coverage = 112,540.
- [ ] Keep the pipeline deterministic (seeds), pinned dependencies, single-command reproduction; OSI licence ready.
- [ ] Draft the 5 case reviews *from the submitted evidence* (pair id, hand ids, observable behaviour, benign alternative).

## 11. Risk register

| Risk | Mitigation |
|---|---|
| CV inflated by labelled-only evaluation | Eval-mirrored PairAP on all eligible pairs, per family |
| Two families invisible to seats-only features | Action-level vs-partner features from day 2 |
| Evidence ranker generalises poorly to 4th family | Family-agnostic anomaly head; validate MAP@5 on held-out families (train on 2, test on 1) |
| Public LB noise misleads selection | Bootstrap the public split; select one CV-best and one LB-best |
| Memory blow-ups on 9.6M eval pair-hands | Per-table processing; column selection; write intermediates to parquet |
| Invalid submission (floats, duplicates, NaN) | Validator using the reference metric on every file |
| Rule violation via id artefacts | Explicit feature allow-list; no id-derived columns |
| Positives overfit (372) | Strong regularisation, rank-averaged seeds, few hyper-parameter searches |

## 12. First-hour checklist on any new machine
1. `len(development_labels)`, family counts, evidence per pair (expect 1,860 / 372 / 1,817).
2. `evaluation_pairs.shared_hands.describe()` (min 38, median 76).
3. Confirm no positive player in eval pairs; tables × 5,000 hands; phase split 3,000/2,000.
4. Rebuild the eval-mirrored CV and reproduce PairAP ≈ 0.2 with seats-only features (this report's baseline) before adding features.
5. Reproduce the evidence-anatomy table (§4.4) to make sure action parsing (last aggressor) is right.

## 13. Public resources review (all public material as of 2026-09-14)

### 13.1 Inventory

| Resource | Type | Public LB | What it is |
|---|---|---|---|
| `detecting-collusive-value-transfer-in-6-max-poker.ipynb` | notebook, XGBoost + polars | **0.64469** (its own ladder: 0.46 → 0.51 → 0.61 → 0.62 → 0.645) | Best-documented public solution; PU-stress CV that tracks LB; public ablation |
| `suspicious-detection.ipynb` (V3-H2) | notebook, XGBoost + duckdb + numba | V2.2 anchor **0.66223** | Strictest CV discipline of the set; exact 7-card hand semantics as a second stage; nested behaviour routing |
| `poker-collusion-pu-aware-evidence-ranker.ipynb` (Nomannic) | notebook, XGBoost | **0.55758** (cited by the 0.645 notebook) | Origin of the PU recipe everyone copied; clean evidence EDA |
| `topological-collusion-dynamics-…-eda.ipynb` = `base notebook.ipynb` (v12/v13, 73 lines differ) | notebook, XGB+LGBM+CatBoost | not stated (projected 0.61–0.63) | Triple-GBDT + decoupled family specialists; dual-engine evidence (MAP@5 0.50) |
| `poker-e23-clean-label-pn-submission-candidate.ipynb` | notebook (variant of the above) | not stated (projected 0.63) | Adds a "clean-label P/N" risk branch; graded NDCG evidence ranker (MAP@5 0.36 — worse) |
| `coordinated-collusion-value-transfer-detection.ipynb` | notebook, LightGBM | not stated (OOF composite 0.68 on labelled pairs only) | Weakest: trains on the 1,860 labelled pairs only, heuristic evidence (MAP@5 0.09), `none` below risk 0.45, `NO_EVIDENCE` below risk 0.20 |
| `slash-poker-competition-metric.ipynb` | host notebook | — | The reference metric (reproduced in this report) |
| github.com/phucthaiv02/poker-suspicious-detection | repo | — | README only (no code pushed): evidence-first PU design, OOF stacking, open-set reject |
| Kaggle discussion (4 threads) | forum | — | See 13.4 |

The Code tab lists nothing for logged-out visitors; the 8 notebooks above were downloaded while logged in and appear to be the complete public set.

### 13.2 What the public notebooks converge on (the "public recipe", ≈0.56–0.66 LB)

1. **PU sampling** (Nomannic → everyone): keep the 1,860 labelled pairs; add unlabelled dev pairs with ≥57 shared hands (= 1.5 × min eval `shared_hands`, i.e. the same 1.9% exposure rule found in §4.3), **neither player among the 693 known colluders**, shuffled and capped at **60 per table** (24,000 pairs), trained as negatives with `sample_weight` 0.35 (positives 2.0, confirmed negatives 1.0). One notebook (V3-H2) samples 72/table *without* looking at labels and down-weights positive-player pairs fold-by-fold instead.
2. **Two CV numbers, trust only one**: "labelled AP" (372 vs 1,488) sits at 0.93–0.98 for every notebook; "PU-stress AP" (positives vs the 24k unlabelled, re-weighted to the eval population) sits at 0.52–0.69 and **tracked public LB** (0.617 OOF ↔ 0.645 public).
3. **Features**: BB-normalised per-hand chip transfer between partners (`min(−net_a, net_b)`), one-way flow and its Gini/top-3/top-5 share; "true heads-up" checks/calls between partners (partner still live); outsider-folds-to-pair after partner aggression; partner-folds/calls/raises-to-pair via a `last_aggressor` column (`forward_fill` over hand×street); hole-strength proxies; **partner-vs-field contrasts** (a player's stats on hands where the partner is seated minus hands where the partner is absent — the contrast that "survived ablation"); player-phase baselines; empirical-Bayes shrinkage of contrasts by exposure; 8-bin `phase_progress` burst maxima; pool-relative percent-ranks (duckdb window functions).
4. **Models**: XGBoost/LightGBM/CatBoost classifiers, `StratifiedGroupKFold` on `table_id`; risk = 0.5·overall + 0.5·max(family OvR) (0.645 notebook) or 0.5·triple-GBDT + 0.5·decoupled specialists (topological).
5. **Behaviour**: families written only on a top slice of risk (5% in the 0.645 notebook; 0.8–1.5% "natural density" in the others), `none` elsewhere. **This is metric-wrong** (§5.3): `none` never helps; the 0.645 notebook itself found that tightening the slice from 5% to 2% *lost* 0.009.
6. **Evidence**: XGBRanker (`rank:pairwise`) trained on positive pairs' dev hands, blended with a family heuristic after within-pair z-scoring (0.75/0.25); family-specialised rankers add +0.04–0.05 MAP@5 (0.45 → 0.50); OOF MAP@5 across notebooks 0.35–0.51. Some notebooks only run the model on the top 25k–50k pairs and use heuristics elsewhere; several still emit `NO_EVIDENCE` for low-risk pairs (metric-wrong).

### 13.3 Findings in the notebooks that matter beyond their scores

- **Pseudo-future diagnostic (V3-H2)**: train on the first 70% of the development window, test on the last 30% with 57 features → labelled AP **0.62**, PU-weighted AP **0.09** (vs 0.97 / 0.57 with pool-grouped folds). Signal does not transfer across time nearly as well as across pools — consistent with episodic coordination and with the eval period being a different time window. Any serious CV must include a temporal split.
- **Exact hand semantics (V3-H2 "H2")**: a numba 7-card evaluator + counterfactual action residuals, used as a 25%/35% second-stage blend on the top 35k pairs: **+0.043 MAP@5** (0.349 → 0.391; directed_transfer 0.33 → 0.44) but **−0.003 PairAP**. Hole-card semantics help evidence retrieval immediately; for pair ranking they need to be first-class features, not a tail blend.
- **Cliques**: 693 colluder players across 372 pairs (744 if disjoint) → some players sit in several positive pairs; a graph feature (boost B–C when A–B and A–C look collusive) is justified and un-shipped.
- **Ablation "do not retry" list (0.645 notebook)**: blending `risk_score` with within-table percentiles (Pair AP is global), sampling unlabelled pairs from tables that already have a positive, replacing the pair classifier with a table `XGBRanker`, LightGBM/transformers before street-level templates.
- **Family difficulty is consistent everywhere**: soft_play OvR ≈ 0.86–0.99, directed_transfer ≈ 0.83–0.92, coordinated_isolation ≈ 0.66–0.88 and "fragile" — matching this report's evidence anatomy (isolation is a *sequence*: raise → partner re-raise → outsider folds → fold).
- **Evidence anatomy (Nomannic)** agrees with §4.4: evidence hands are ~4% of positive pairs' shared hands; directed_transfer evidence median pot 34.5 BB vs 10.5; soft_play evidence has 0.80 pair showdowns per hand vs 0.18; isolation evidence action mix is 40% raises (vs 18%).

### 13.4 Discussion threads (4)

| Thread | Content | Implication |
|---|---|---|
| Clarifications (host reply) | External public data and pretrained models are allowed; metric notebook published | Preflop equity tables / hand evaluators are fine |
| CV–LB difference (3 upvotes, unanswered) | "better CV but poorly on LB" | §7.4; use PU-stress + temporal CV |
| Why are most training pairs "unknown"? (unanswered) | Asks whether ~156k unlabelled dev pairs are clean or unlabelled | They are unlabelled (host text); the public recipe uses them as down-weighted negatives |
| Player metadata (host reply, 2026-09-14) | **"Yes, you can use players.parquet, pairwise features included. The rule is about artifacts of how the files were built (ID formats, row ordering, generator internals)."** | Same-region/same-client/account-age-gap features are legal (they are weak, but legal) |

### 13.5 Leaderboard movement (2026-09-13 → 09-14)

#1 unchanged at **0.93798** (last entry 2 days ago); the chasing pack tightened: #2 seantangth 0.91728, #3 Leo 0.91641, #4 Marc Donovici 0.91541, #5 bluey 0.91450, #6 krakaa 0.90913, #7 0.90757, #8 Bình & An 0.90418, #9 0.90291, #10 0.90182, #11 0.90173, #12 0.90030. Twelve teams are now ≥0.90; all public notebooks are ≤0.66. The gap between the public recipe (~0.65) and the pack (~0.90) is 0.25 of final score — that is not tuning; it is (a) Pair AP on the real population going from ~0.6 to ~0.9 and (b) MAP@5 from ~0.45 to ~0.8.

### 13.6 What to take from the public material (and what to leave)

**Take:** the PU sampling recipe (with the label-independent variant), `last_aggressor` action context, partner-vs-field contrasts with shrinkage, top-k/burst aggregations, PU-stress + temporal CV, family-specialised evidence rankers with within-pair normalisation, the strict OOF/nesting discipline of V3-H2, the submission validator that checks every evidence hand is a shared evaluation-phase hand.

**Leave:** `none` for low-risk pairs, `NO_EVIDENCE` for low-risk pairs, evidence models run only on a top-k tail, within-table percentile blends of the risk score, player metadata as a primary signal, training on the 1,860 labelled pairs alone.

**Still unexplored publicly (the likely gap to 0.9+):** hole-card-aware EV-loss features as first-class pair features (not a tail blend); per-hand pair-conditional scoring for *all* 9.6M eval pair-hands with the hand scorer feeding the pair model (evidence-first, §9); explicit episode detection (windowed maxima of hand scores) given the temporal-transfer collapse; clique/graph propagation; per-family calibrated combination instead of 0.5/0.5 blends.

## 14. Related research (papers worth reading, ranked by relevance)

### 14.1 Collusion detection in poker and games — directly on-topic

| # | Paper | What it does | Why it matters here / what to borrow |
|---|---|---|---|
| 1 | **Mazrooei, Archibald & Bowling, "Automating Collusion Detection in Sequential Games", AAAI 2013** — [PDF](https://poker.cs.ualberta.ca/publications/AAAI13.pdf) | Introduces the **collusion table**: for each hand, the impact of player *k*'s actions on player *j*'s expected value, C(j,k) = Σ over k's decisions of [V_j(h·a) − V_j(h)], using a value function V learned by CFR. Two pair scores: **Total Impact** (sum of the pair's impact on itself) and **Marginal Impact** (impact on partner minus average impact on everyone else). Synthetic 3-player limit Hold'em with CFR-trained colluders that value the partner's utility (û_i = u_i + 0.9·u_j), plus "accidental colluders" as confounders; real colluders rank top-4 of 91 pairs; ~100k hands per triple needed. | This is the theoretical version of what this report calls *partner-vs-field* and *EV-loss* features. Their Marginal Impact score is exactly "how a player treats the partner minus how they treat others". Their value function V is the part we can approximate cheaply because **hole cards are visible**: per-action ΔEV of the actor on the partner's equity and on third parties → directed EV-impact features per hand (evidence scorer) and summed per pair (risk). Their confounder design ("accidental colluders") mirrors the host's benign look-alikes. |
| 2 | **Bonjour, Aggarwal & Bhargava, "Information-Theoretic Approach to Detect Collusion in Multi-Agent Games", UAI 2022** — [PDF](https://proceedings.mlr.press/v180/bonjour22a/bonjour22a.pdf) | Domain-independent detector: conditional **mutual information between two players' actions given state**, γ(i;j); **net influence** Γ(i;j) = γ(i;j) − max_k γ(k;j); flag a pair when both Γ(i;j), Γ(j;i) ≥ α (α = 0.05). Tested on 3-player Leduc Hold'em with rule-based and DRL-trained colluders (including *low-payoff* colluders that are invisible from chip flow). 95% detection with ~600–1,800 hands. | A cheap, model-free pair feature: estimate p(a_j | state, a_i) vs p(a_j | state) from action logs (state = street × facing-bet × players_active buckets) and compute the excess over the same player's response to *other* players. It explicitly targets information-sharing collusion and low-payoff collusion — two candidates for the undisclosed 4th family. Sample sizes match this data (38–419 shared eval hands, so noisy but usable with shrinkage). |
| 3 | **Greige et al., "Collusion Detection in Team-Based Multiplayer Games", arXiv 2203.05121 (2022)** — [arXiv](https://arxiv.org/abs/2203.05121) | Builds pairwise cross-team interaction features (who damages/avoids whom, co-occurrence) and scores pairs with **Isolation Forest**; validated on two real datasets (170k+ players). Output is a prioritised list for human review. | Template for the open-set `other_coordination` head: an unsupervised anomaly score over pair-interaction features, fitted on confirmed negatives, flags coordinated pairs that fit none of the three disclosed families. |
| 4 | **Billings & Kan (ICGA J. 2006) DIVAT; Zinkevich et al. (AAAI 2006) advantage-sum estimators; White & Bowling (IJCAI 2009) MIVAT** | Variance-reduced assessment of poker decisions: score each decision by its EV against a baseline strategy rather than by the (lucky) outcome. | Basis for per-hand "how much did this decision cost" features — the loser's EV-loss vs the partner, computed from visible hole cards + board. This is the evidence-hand scorer's core signal for directed_transfer and soft_play. |
| 5 | **Smed, Knuutila & Hakonen, "Can We Prevent Collusion in Multiplayer Online Games?" (SCAI 2006) and "Towards Swift and Accurate Collusion Detection" (GAMEON 2007); Laasonen, Knuutila & Smed, "Eliciting Collusion Features" (SIMUTools 2011)** | Taxonomy of collusion (spectator / assistant / association), the *swiftness vs accuracy* trade-off in detection, and which game features carry collusion signal. | Vocabulary and framing; the "association collusion" type is the soft_play / isolation family; assistant collusion is directed_transfer. Useful for the write-up. |
| 6 | **Johansson, Sönströd & König, "Cheating by sharing information — the doom of online poker?" (2003); Yampolskiy, "Detecting and controlling cheating in online poker" (IEEE CCNC 2008); Yan, "Collusion Detection in Online Bridge" (AAAI 2010)** | Early treatments: hole-card sharing as cheating, taxonomy of poker cheating, and statistical detection in bridge. | Background; card-sharing detectors (folding hands dominated by the partner's holding at abnormal rates) are a concrete 4th-family hypothesis. |
| 7 | **"Collusion in Online Poker Pays Off", ETH Zürich BSc thesis 2012** — [PDF](https://pub.tik.ee.ethz.ch/students/2012-FS/BA-2012-03.pdf) | Built colluding bots and played them on real sites (play and real money) without being detected; documents concrete colluding strategies. | A catalogue of what generator-style collusion strategies look like from the inside — useful when hypothesising the 4th family and when writing benign alternatives for case reviews. |
| 8 | **Industry practice** — e.g. [GuardianStack 2026 overview](https://blog.guardianstack.ai/online-poker-cheating-detection/); US patents [7,604,541](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7604541) ("collusion detection via conditional behavior") and the [Zynga-style "Collusion detection" family](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10699523) | Operators use: win-rate deviations, coordinated folding vs specific opponents, "one player consistently folds to another's raises", session overlap, device fingerprints; patents describe "chip transfer when the player should not have lost given the cards" and "action not in line with the player's determined style". | Confirms the two contrasts this report recommends — *vs-partner vs vs-others* and *vs own baseline* — are what real investigators use. |

### 14.2 Positive-unlabelled learning (the label structure of this competition)

| # | Paper | Use here |
|---|---|---|
| 9 | **Elkan & Noto, "Learning classifiers from only positive and unlabeled data", KDD 2008** | The classic: under SCAR, a classifier trained labelled-vs-unlabelled yields calibrated positives after dividing by c = P(labelled \| positive); also gives a prior estimate. Use to estimate how many hidden positives sit in the ~136k unlabelled dev pairs. |
| 10 | **du Plessis, Niu & Sugiyama (NeurIPS 2014 / ICML 2015)** unbiased PU risk; **Kiryo et al., "Positive-Unlabeled Learning with Non-Negative Risk Estimator", NeurIPS 2017 (nnPU)** | Principled loss when unlabelled data are used as negatives; nnPU prevents the overfitting that the naive "unlabelled = negative" objective causes with flexible models. Worth a try as the pair model's objective once the prior is estimated. |
| 11 | **Mordelet & Vert, "A bagging SVM to learn from positive and unlabeled examples", PRL 2014** | Bagging PU: repeatedly sample unlabelled pairs as negatives and average — the public notebooks' "60 per table" sampling is a single draw of this. |
| 12 | **Bekker & Davis, "Learning from positive and unlabeled data: a survey", Machine Learning 2020** — [arXiv 1811.04820](https://arxiv.org/abs/1811.04820) | Map of the field; crucially discusses the **SCAR vs SAR** assumption. Our positives are "trusted" (clearest cases) → *not* selected at random → SAR; see also **PULSNAR (arXiv 2303.08269)** for class-proportion estimation when SCAR fails. |
| 13 | **Saunders & Freitas, "Evaluating the Predictive Performance of Positive-Unlabelled Classifiers", SIGKDD Explorations 2022** — [arXiv 2206.02423](https://arxiv.org/abs/2206.02423) | How to evaluate PU models honestly: distinguish *prioritisation* (our case — AP) from anomaly detection, report precision and recall separately, never evaluate only on the labelled subset. Formal backing for §7's "eval-mirrored AP". |
| 14 | **Kocsis & György, "Fraud detection by generating positive samples for classification from unlabeled data", ICML-W 2010** | PU fraud detection in games; generating synthetic positives — relevant if you want to augment the 372 positives. |

### 14.3 Multiple-instance learning and evidence retrieval (the 20 % component)

| # | Paper | Use here |
|---|---|---|
| 15 | **Liu, Wu & Zhou, "Key Instance Detection in Multi-Instance Learning", ACML 2012** — [PDF](http://proceedings.mlr.press/v25/liu12b/liu12b.pdf) | The exact formulation of evidence retrieval: bags = pairs, instances = shared hands, find the instances that trigger the bag label. |
| 16 | **Ilse, Tomczak & Welling, "Attention-based Deep Multiple Instance Learning", ICML 2018**; **Li et al., DSMIL, CVPR 2021** | Attention weights over instances give an instance ranking for free while training on bag labels — a neural alternative to the GBDT hand scorer, trainable on *all* unlabelled pairs as negative bags. |
| 17 | **Burges, "From RankNet to LambdaRank to LambdaMART", MSR-TR 2010; Wang et al., "The LambdaLoss Framework for Ranking Metric Optimization", CIKM 2018** | Listwise objectives for within-pair hand ranking (`lambdarank` in LightGBM/XGBoost, grouped by pair) that optimise top-of-list precision — the right objective for MAP@5. |

### 14.4 Anomaly, graph and explainable fraud detection (4th family, cliques, case reviews)

| # | Paper | Use here |
|---|---|---|
| 18 | **Liu, Ting & Zhou, "Isolation Forest", ICDM 2008** | Open-set anomaly score for `other_coordination`. |
| 19 | **Palshikar & Apte, "Collusion set detection using graph clustering", DMKD 2008; Cao et al., "Detecting Wash Trade in Financial Market Using Digraphs and Dynamic Programming", IEEE TNNLS 2016; Wachs & Kertész, "A network approach to cartel detection", Sci. Rep. 2019** | Directed chip-transfer graphs and clique detection: 693 colluder players in 372 pairs means rings exist; graph propagation (A–B and A–C suspicious ⇒ boost B–C) is un-shipped by every public notebook. |
| 20 | **"Collusion Detection with Graph Neural Networks", arXiv 2410.07091 (2024)** — [arXiv](https://arxiv.org/abs/2410.07091) | GNNs on interaction graphs beat MLPs for collusion; a player–hand bipartite GNN is the ambitious version (probably not within 7 days). |
| 21 | **Wolfson & Acar, "Differentiable Inductive Logic Programming for Fraud Detection", arXiv 2410.21928 (2024)** — [arXiv](https://arxiv.org/abs/2410.21928) | Learns explicit rules for fraud relationships/chains; comparable to trees in accuracy but explainable — relevant to the *five case reviews* obligation (rules read like an investigator's note). |

### 14.5 Tooling papers / libraries (allowed: public, free)

- **Hand evaluation**: PHEvaluator (H. Lee, perfect-hash 7-card evaluator, `pip install phevaluator`), `treys` (Cactus Kev / Two-Plus-Two style lookup), `eval7`; preflop 169-class equity tables (public). These give the exact made-hand rank and equity needed for EV-loss features; V3-H2 hand-rolled a numba evaluator instead.
- **Counterfactual Regret Minimization** (Zinkevich et al., NeurIPS 2008) — only if you want a real value function à la Mazrooei; an equity-based proxy is enough for 6-max NLHE here.

### 14.6 How the literature maps onto this competition

| Competition component | Literature to lean on |
|---|---|
| Pair risk (70 %) | Mazrooei 2013 (impact on partner vs others), Bonjour 2022 (net influence), Elkan–Noto / nnPU (training with unlabelled pairs), Bekker–Davis (SAR caveat), Saunders–Freitas (evaluate on the full population) |
| Evidence MAP@5 (20 %) | Key-instance MIL (Liu 2012), attention MIL (Ilse 2018), DIVAT-style per-decision EV loss, LambdaRank/LambdaLoss for the within-pair ranking |
| Behaviour (10 %) | Smed taxonomy (assistant vs association collusion) for feature design; calibrated one-vs-rest (§5.3) |
| `other_coordination` | Isolation Forest (Greige 2022), information-sharing detectors (Johansson 2003, Bonjour 2022 low-payoff collusion) |
| Cliques / rings | Palshikar–Apte, Cao 2016, GNN collusion (2410.07091) |
| Write-up & case reviews | Mazrooei's "screening, not proof" framing; DILP-style explicit rules |

Local copies of the six PDFs that were fetched (Mazrooei 2013, Bonjour 2022, Greige 2022, Bekker–Davis 2020, Saunders–Freitas 2022, ETH 2012) are in `papers/` (git-ignored).

## 15. Sources
- Competition overview/data/rules/leaderboard/discussion pages (rendered 2026-09-13/14).
- Host reference metric notebook (code reproduced in `COMPETITION_FACTS.md`).
- Public notebooks in this folder (reviewed in §13): `detecting-collusive-value-transfer-in-6-max-poker` (public 0.64469), `suspicious-detection` (V3-H2; V2.2 public 0.66223), `poker-collusion-pu-aware-evidence-ranker` (public 0.55758), `topological-collusion-dynamics-value-transfer-eda` / `base notebook`, `poker-e23-clean-label-pn-submission-candidate`, `coordinated-collusion-value-transfer-detection`, `slash-poker-competition-metric`.
- Discussion threads 739839 (clarifications), 741081 (CV–LB), 741280 (unknown pairs), 741306 (player metadata — host: allowed).
- Papers: see §14 (links inline).
- Public baseline README: github.com/phucthaiv02/poker-suspicious-detection.
- This analysis' scripts and tables: scratchpad `eda/` (evidence_anatomy.py, pair_features_all.py, pu_gap.py) and the 7-lens verified dossier (105 findings; 0 refuted).
