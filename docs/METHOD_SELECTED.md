# Method of the selected submissions — full technical description

Selected: **`final_J_bags20_ev18.csv`** (public 0.90025) and **`final_F_allbag_ev18.csv`** (public 0.89964).
Both share the same evidence lists and family labels; they differ only in how the risk vector is blended (§7).
Code: `step9_pipeline.py`, `step18_pipeline.py`, `hyp/*.py`, `reproduce/bag20.py`, `reproduce/make_final.py`.

---

## 1. Data preparation (`step18_pipeline.py` §1–2, inherited from Steps 1–8)

* **Hand chronology.** Hands are sorted by `(table_id, started_at, hand_id)`; `hand_idx` is the position within the table (0–2,999 development, 3,000–4,999 evaluation). No ID or file-order information is used.
* **Action context** (`action_context.parquet`): each action with `street_no`, `is_aggr`, amounts in big blinds and as a fraction of the pot, and `last_aggr` = the player whose bet is currently being faced (forward-filled within hand × street). `to_call > 0 and last_aggr == partner` means "facing the partner's bet".
* **Player-hand table** (`player_hands`): per (hand, player): seat, position relative to the button, hole class (169 classes) and Chen score, VPIP / PFR (action-based), pre-flop entry order, whether a raise was faced, action counts per type, showdown, net chips (bb).
* **Exact hand strength** (`player_strength`): numba 7-card evaluator (verified against the 5-card category counts) → hand value at flop/turn/river/final for all six dealt players (all hole cards are visible in this synthetic data), category (0 = high card … 8 = straight flush) and rank among the six.

## 2. Population action policy and surprise (Step 8; `step18_pipeline.py` §2a′)

A LightGBM 6-class model (fold / check / call / bet / raise / all-in) trained on a 5 M-row sample of the 18.6 M actions with:

* situation: `street_no`, position, `players_active`, `to_call_bb`, `to_call/pot`, `pot_bb`, `stack_bb`, stack-to-pot ratio, raises already made this street, own prior aggressive actions, own prior actions, action index within the street;
* private information: hole ranks (hi, lo), suited, pocket pair, exact made-hand category at the current street, board max rank, board paired, number of hole cards hitting the board, flush-card count, overpair, top-pair flags;
* player style from the **other phase** (VPIP, PFR, aggression per hand, calls per hand, fold-facing rate, showdown rate, check rate, mean pre-flop looseness/tightness surprise). Collusion is phase-specific, so the other phase is a clean style baseline.

Parameters: `learning_rate 0.1, num_leaves 127, min_data_in_leaf 200, feature_fraction 0.8, bagging 0.8, lambda_l2 5, 350 rounds` (hold-out multi-logloss 0.485). Every action gets `p_taken` and `surp = −log p_taken`. Per (hand, player): `surp_sum, surp_max, surp_pf, surp_post, n_surp2 (#actions with surprise > 2), n_surp35, surp_fold/call/aggr/check, p_min, logp_mean`.

A simpler pre-flop policy `P(vpip | hole class, position, faced raise)` (Laplace-smoothed, k = 20) from all 12 M seats gives `loose = −log P(enter)` when the player entered, `tight = −log P(fold)` when a strong hand folded, `junk_vpip = vpip · max(0, 8 − Chen)`, `vpip_resid`.

## 3. Pair-hand features (`step18_pipeline.py` §4; one row per pair × shared hand, ≈ 240 features)

* **Both players' per-hand fields** (`p1_*`, `p2_*`): everything from §1–2.
* **Joint pre-flop deviation:** both entered, both entered with junk (Chen ≤ 5), the second entrant's cards and surprise, `loose_sum/min/max`, `junk_vpip_sum/min`, `vpip_resid_sum`, `both_surprising`, `junk_raise`.
* **Action-surprise pair fields:** `surp_pair_sum/min/max`, pre-/post-flop sums, per-action-type sums, `n_surp2_sum`, `both_surp2`, `both_vpip_surp2`, `p_min_pair`, `logp_mean_sum`; and from the member perspective: surprise of actions **facing the partner** (`surp_vs_partner`, `_max`), facing an outsider, heads-up, folds/calls/raises against the partner, heads-up checks.
* **Exact-strength anatomy:** winner/loser (by net chips) categories and ranks per street, `loser_beats_winner_*`, `fold_better_hand`, `fold_better_to_partner`, `both_made_no_aggr`, `winner_weak_won`, `cat_gap_final`, `dump`, `dump_better_hand`, `checkdown`, `hu_checkdown_strong`, `hu_check_two_pair_plus`.
* **Partner interaction / squeeze:** fold/call/raise to the partner vs to outsiders, by street; outsiders' folds/calls/raises to the pair; `squeeze`, `pf_squeeze`, `squeeze_then_fold(_pf)`, `multi_call_partner`, `pair_overbets`.
* **Chip flow:** `transfer_1_to_2`, `transfer_2_to_1`, `transfer_any`, `transfer_pot_ratio`, `net_gap`, `transfer_x_call`.
* **Within-pair normalisation:** `*_to_max` and percentile ranks `*_prank` over the pair's shared hands (used by the evidence ranker; excluded from the hand models that feed the pair ranker where phase length would bias them).

## 4. Pair features and pair ranker (`step18_pipeline.py` §6; = Step 8 model)

**Aggregation** over a pair's shared hands: means (55 hand fields), maxima, top-3 means, sums; evidence-ranker score aggregates (`hs_mean/max/top3/top5/p90`, counts above the negatives' 95th/99th percentiles, `ev_n10/ev_n50` = number of hands scoring above the 10th/50th percentile of labelled evidence hands); chip-flow direction consistency and asymmetry; per-player **with-partner minus without-partner** contrasts for 17 fields, shrunk by `n/(n+25)`; **other-phase pair baselines** (`ob_*`: the same two players' joint entries, junk entries, surprise sums and transfers in the phase where they are not colluding; contrasts and ratios shrunk with `n_other/(n_other+30)`); within-table percentile ranks of the strongest 60 features. Step 9 adds coincidence z-scores: for joint events (both surprising, both junk entries, both entered), `obs − n·p₁·p₂` and `(obs − exp)/√(exp+0.5)` with each player's own field rate.

**Exposure matching.** The pair model is trained on two 2,000-hand development windows `[0, 2000)` and `[1000, 3000)` (two rows per pair) so counts and sums have the evaluation scale (median 76 shared hands vs 110 for the full 3,000-hand phase). Field baselines are phase-specific.

**Training population:** all eligible development pairs (≥ 57 shared hands ≈ the evaluation rule, no publicly-positive player) + the 1,860 labelled pairs → 122,580 pairs, 244,884 rows. Table-grouped 5-fold CV (`RandomState(42)` permutation).

**Positive–unlabelled, three stages** (LightGBM `learning_rate 0.03, num_leaves 31, min_data_in_leaf 100, feature_fraction 0.5, bagging 0.8, lambda_l2 10, 750 rounds`):
1. weights: confirmed positive 5, confirmed non-target 1, unlabelled 0.1 → OOF₁;
2. unlabelled pairs with OOF₁ ≥ 50th percentile of the labelled positives' OOF become pseudo-positives (weight 1); those ≥ 5th percentile are "ambiguous" and **kept as weak negatives (weight 0.2)** — dropping them, as Steps 3–7 did, removes exactly the loose-bot negatives the evaluation ranking must beat; → OOF₂;
3. repeat with the 25th percentile and five seeds → OOF₃ (the pipeline's risk) and the three **family heads** (one-vs-rest LightGBM, `num_leaves 7`, 400 rounds, trained on confirmed labels only).

Development views (Step 8 features): labelled AP 0.990, AP with all unlabelled as negatives 0.708, AP with a fixed set of 150 colluder-like unlabelled pairs counted positive 0.93.

## 5. Behaviour family (`predict_family`, `BEHAVIOR_RHO`)

`argmax_f P_f / prior_f` over the three heads (priors = labelled family counts 148/132/92). A family is predicted only for the top **5 %** of pairs by risk (`rho` chosen on the development population's host metric with a safety margin for higher evaluation prevalence); all others are `none`. `other_coordination` is never predicted (it scores identically to `none`).

## 6. Evidence hands (`step18_pipeline.py` §5, §5b, §5c)

Trained on the 372 coordinated pairs' 45,129 shared development hands (1,817 labelled evidence vs the rest), table-grouped folds, `HAND_PARAMS = learning_rate 0.05, num_leaves 31, min_data_in_leaf 50, feature_fraction 0.8, bagging 0.8, lambda_l2 10, 400 rounds, 2 seeds`.

**Stage 1 (Step 18 ranker)** on the §3 features **plus** three families computed by `hyp/*.py` (development and evaluation alike):
* `cc_*` (`hyp/category_combo.py`, 44 features): decisive street, who folded, outsiders active at the fold, fold-to-winner, size faced at the fold, loser/winner made-hand categories at the decisive street and per street, loser-better flags (category and kicker), showdown category gap, pocket pairs, outsiders' best category and whether it beat the pair, outsider folds/calls to the pair with better hands, family-specific flags (`cc_dt_flag`, `cc_sp_flag`, `cc_iso_junk_vs_strong`).
* `st_*` (`hyp/street_timing.py`, 53 features): first/last street the partners faced each other, streets both stayed in, heads-up streets, end street, outsiders remaining on the decisive street, first deviating street (surprise > 2), fold-gap between partners, per-street surprise maxima.
* `pp_*` (`hyp/player_policy.py`, 19 features): per-player 6-class policy fitted on **that player's own other-phase** decisions (LightGBM per 100-table chunk with situation features + the player's other-phase conditional rates + player id), scoring in-phase actions: own-policy surprise sums/max/min, `p_min`, counts with p < 0.01 / 0.05, both partners improbable in the same hand, surprise vs population surprise deltas, surprise facing the partner. Within-pair percentile versions are excluded (phase-length dependent).
Development OOF MAP@5: 0.528 → 0.539.

**Stage 2 (pool re-ranker).** For each pair the top-20 hands by the stage-1 score; features = stage-1 features + `hand_score`, score relative to the pool maximum, chronological rank in the pool, phase-relative position, gaps to neighbouring pool hands, number of pool hands within 20 hands, **strong-hand chronology** (strong hands before this one, cumulative score before, cumulative share), the pair's **family** one-hot (true in training, predicted at inference), and fold-safe **target encodings of action-sequence templates** (`tpl_pair` = the partners' ordered actions tagged winner/loser · street · facing partner/outsider/nobody · action; also pre-flop-only and first-three-token variants, each also within family; smoothing m = 10). `POOL_PARAMS = learning_rate 0.03, num_leaves 15, min_data_in_leaf 40, feature_fraction 0.7, bagging 0.8, lambda_l2 10, 500 rounds, 3 seeds`. Pool hands are ranked above all others (`10 + score₂`, else score₁). Development OOF MAP@5: 0.539 → **0.621** (directed_transfer 0.65, soft_play 0.65, coordinated_isolation 0.53).

Five slots are always filled; ties are broken by pot size then hand id, as the host metric does.

## 7. Final assembly (`reproduce/bag20.py`, `reproduce/make_final.py`)

| vector | source |
|---|---|
| `b8` | 20-seed full-data LightGBM bag (seeds 42…269), Step-8 pair features, labels/weights from the stage-3 PU recipe applied to the Step-18 CV OOF |
| `b9` | same with Step-9 pair features and the Step-9 OOF |
| `c8`, `c9` | the fold-CV ensembles (5 folds × 5 seeds) from `step18_pipeline.py` / `step9_pipeline.py` |

* **final_J:** `risk = rank(b8) + rank(b9)` averaged; ties resolved by mean probability then a 10⁻¹³ epsilon (no two pairs equal).
* **final_F:** `risk = rank(c8) + rank(c9) + rank(b8₁₀) + rank(b9₁₀)` averaged (the 10-seed halves of the bags), i.e. 50 % weight on the fold-CV regime as a hedge.
* Both: families from the Step-18 heads gated at the top 5 % of the blended rank; evidence from Step 18.

## 8. Validation and leaderboard decomposition

* Three development views per change (labelled AP, pessimistic AP, hidden-150 AP); a change was kept only if all improved and the public leaderboard confirmed it.
* Probe submission (Step 8 risk with all evidence blanked): 0.77519 → evidence MAP@5 on the leaderboard = (0.89378 − 0.77519)/0.2 = 0.593; with Step 18 evidence ≈ 0.62; implied pair AP ≈ 0.97.
* Public/private split is 30/70 stratified by family; simulation of random splits gives a public–private score SD of ≈ 0.01 for every candidate, an order of magnitude larger than the differences among our top candidates (see `work_step3/final_selection_report.md`).
