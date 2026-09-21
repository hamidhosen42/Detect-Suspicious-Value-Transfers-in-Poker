**Team:** Hack2Publish — Md. Hamid Hosen ([hosen42](https://www.kaggle.com/hosen42)), [esfersami50](https://www.kaggle.com/esfersami50), Foysal Emon Shanto ([foysalemonshanto](https://www.kaggle.com/foysalemonshanto))
**Selected submissions:** `final_J_bags20_ev18.csv` (public 0.90025 / private 0.89598) and `final_F_allbag_ev18.csv` (public 0.89964 / private 0.89566) — private rank 39 / 371
**Code (MIT):** https://github.com/hamidhosen42/Detect-Suspicious-Value-Transfers-in-Poker · reproduction: `reproduce/run_all.sh`

---

## Leaderboard results

Final private score **0.89598** — rank 39 / 371 (top 10.5 %) · winner 0.9407 private.

| selected submission | risk | evidence | public | private |
|---|---|---|---|---|
| **final_J_bags20_ev18** | rank-average of two 20-seed full-data bags (Step-8 and Step-9 feature sets) | Step 18 | 0.90025 | **0.89598** |
| **final_F_allbag_ev18** | same bags (10 seeds each) + the two fold-CV models, equal weights | Step 18 | 0.89964 | 0.89566 |

## 1. The signal

The simulator's bots are strictly card-driven: the population's voluntary-entry rate rises from 2 % for the weakest hole cards (Chen ≤ −1) to 96 % for the strongest (Chen ≥ 12). Reading the 1,817 labelled evidence hands showed that coordination is **two partners acting wrongly for their cards in the same hand**: both entering with junk (8♦3♠ raises, 6♦3♣ calls), calling the partner's bets with nothing, folding a made hand to the partner, checking a set down heads-up, or raising and re-raising each other with 9-2 / 8-2 to squeeze outsiders out. Chip transfer is a consequence, not the signal — chip-flow-only models plateau at 0.69.

Everything below uses gameplay only: hole cards, board, seats/positions, stacks, the ordered action log, and chronology from `started_at` (hands sorted by `table_id, started_at, hand_id` → `hand_idx`). `players.parquet` attributes, ID formats, file order and generator internals are never used. Unlabelled development pairs are treated as unknown.

## 2. Data preparation

* **Action context** (18.6 M rows): each action with `street_no`, `is_aggr` (bet / raise / all-in-for-more), amounts in big blinds and as a fraction of the pot, and `last_aggr` = the player whose bet is currently being faced (forward-filled within hand × street). "Facing the partner" = `to_call > 0 and last_aggr == partner`.
* **Player-hand table** (12 M rows): seat, position relative to the button, 169-class hole class and Chen score, action-based VPIP / PFR, pre-flop entry order, whether a raise was faced, counts of each action type, showdown, net chips (bb).
* **Exact hand strength**: numba 7-card evaluator (verified against the 5-card category counts) → value at flop / turn / river / final for all six dealt players (all hole cards are visible in this synthetic data), category 0–8, and rank among the six.

## 3. Population action policy → per-action surprise

A LightGBM 6-class model (fold / check / call / bet / raise / all-in) trained on a 5 M-row sample of the 18.6 M actions:

* **Situation:** street, position, players active, `to_call_bb`, `to_call/pot`, `pot_bb`, `stack_bb`, stack-to-pot ratio, raises already made this street, own prior aggressive actions, own prior actions, index within the street.
* **Private information:** hole ranks (hi, lo), suited, pocket pair, exact made-hand category at the current street, board max rank, board paired, hole cards hitting the board, flush-card count, overpair / top-pair flags.
* **Player style from the other phase:** VPIP, PFR, aggressive actions per hand, calls per hand, fold-facing rate, showdown rate, check rate, mean pre-flop looseness / tightness surprise. Collusion is phase-specific (labelled development colluders play normally in evaluation and vice-versa), so the other phase is a clean baseline.

`learning_rate 0.1, num_leaves 127, min_data_in_leaf 200, feature_fraction 0.8, bagging 0.8, lambda_l2 5, 350 rounds` (hold-out multi-logloss 0.485). Each action gets `p_taken` and `surp = −log p_taken`; per (hand, player): `surp_sum, surp_max, surp_pf, surp_post, n_surp2 (actions with surprise > 2), n_surp35, surp_fold / call / aggr / check, p_min, logp_mean`.

A simpler pre-flop policy `P(vpip | hole class, position, faced raise)` (Laplace k = 20 over all 12 M seats) yields `loose = −log P(enter)` for entries, `tight = −log P(fold)` for folded strong hands, `junk_vpip = vpip · max(0, 8 − Chen)`, `vpip_resid`.

## 4. Pair-hand features (one row per pair × shared hand, ≈ 240 features)

* **Both players' fields** (`p1_*`, `p2_*`) from §2–3.
* **Joint pre-flop deviation:** both entered, both entered with junk (Chen ≤ 5), the second entrant's cards and surprise, `loose_sum/min/max`, `junk_vpip_sum/min`, `vpip_resid_sum`, `both_surprising`, `junk_raise`.
* **Action-surprise pair fields:** `surp_pair_sum/min/max`, pre-/post-flop sums, per-action-type sums, `n_surp2_sum`, `both_surp2`, `both_vpip_surp2`, `p_min_pair`, `logp_mean_sum`; from the member perspective: surprise of actions **facing the partner** (`surp_vs_partner`, `_max`), facing an outsider, heads-up, folds / calls / raises against the partner, heads-up checks.
* **Exact-strength anatomy:** winner / loser (by net chips) categories and ranks per street, `loser_beats_winner_*`, `fold_better_hand`, `fold_better_to_partner`, `both_made_no_aggr`, `winner_weak_won`, `cat_gap_final`, `dump`, `dump_better_hand`, `checkdown`, `hu_checkdown_strong`, `hu_check_two_pair_plus`.
* **Partner interaction / squeeze:** fold / call / raise to the partner vs. to outsiders, by street; outsiders' folds / calls / raises to the pair; `squeeze`, `pf_squeeze`, `squeeze_then_fold(_pf)`, `multi_call_partner`, `pair_overbets`.
* **Chip flow:** `transfer_1_to_2`, `transfer_2_to_1`, `transfer_any`, `transfer_pot_ratio`, `net_gap`, `transfer_x_call`.
* **Within-pair normalisation:** `*_to_max` and percentile ranks `*_prank` over the pair's shared hands — used by the evidence ranker only; excluded wherever phase length would bias them (see §9).

## 5. Pair features and the pair ranker (70 % of the metric)

**Aggregation** per pair over its shared hands: means of 55 hand fields, maxima, top-3 means, sums; evidence-ranker score aggregates (`hs_mean/max/top3/top5/p90`, counts above the negatives' 95th / 99th percentiles, `ev_n10 / ev_n50` = hands scoring above the 10th / 50th percentile of labelled evidence); chip-flow direction consistency and asymmetry; **with-partner minus without-partner** contrasts for 17 per-player fields, shrunk by `n/(n+25)`; **other-phase pair baselines** (`ob_*`: the same two players' joint entries, junk entries, surprise sums and transfers in the phase where they are not colluding; contrasts and ratios shrunk by `n_other/(n_other+30)`); within-table percentile ranks of the 60 strongest features. The Step-9 feature set adds coincidence z-scores for joint events (both surprising / both junk / both entered): `obs − n·p₁·p₂` and `(obs − exp)/√(exp + 0.5)` with each player's own field rate.

**Exposure matching.** Evaluation tables have 2,000 hands (median 76 shared hands per pair) versus 3,000 in development. The pair model is trained on two 2,000-hand development windows `[0, 2000)` and `[1000, 3000)` (two rows per pair), so every count-like feature has the evaluation scale; field baselines are phase-specific (pooling them across phases cost 0.02).

**Training population:** all eligible development pairs (≥ 57 shared hands ≈ the evaluation rule, no publicly-positive player) plus the 1,860 labelled pairs → 122,580 pairs, 244,884 rows; table-grouped 5-fold CV.

**Positive–unlabelled, three stages.** LightGBM `learning_rate 0.03, num_leaves 31, min_data_in_leaf 100, feature_fraction 0.5, bagging 0.8, lambda_l2 10, 750 rounds`.
1. Weights: confirmed positive 5, confirmed non-target 1, unlabelled 0.1 → OOF₁.
2. Unlabelled pairs with OOF₁ above the 50th percentile of the labelled positives' OOF become pseudo-positives (weight 1); those above the 5th percentile are "ambiguous" and are **kept as weak negatives (0.2)** — a score-mixture estimate puts only ≈ 150 hidden colluders among the 120 k unlabelled pairs (≈ 0.4 % prevalence), so that band is mostly loose-bot negatives, exactly the ones the leaderboard ranking must beat (dropping them, as our Steps 3–7 did, cost ≈ 0.01 AP) → OOF₂.
3. Repeat with the 25th percentile and five seeds → OOF₃ and the family heads.

Development views (Step-8 features): labelled AP 0.990; AP with every unlabelled pair as a negative 0.708; AP with a fixed set of 150 colluder-like unlabelled pairs counted positive 0.93.

## 6. Behaviour family (10 %)

Three one-vs-rest LightGBM heads (`num_leaves 7`, 400 rounds) on the pair features, trained on confirmed labels only. Prediction = `argmax_f P_f / prior_f` (priors 148 / 132 / 92 = labelled family counts) for the top **5 %** of pairs by risk; `none` otherwise (the rate was chosen on the development population's host metric with a safety margin). `other_coordination` is never predicted — it scores identically to `none`.

## 7. Evidence hands (20 %)

Trained on the 372 coordinated pairs' 45,129 shared development hands (1,817 labelled evidence vs. the rest), table-grouped folds, LightGBM `learning_rate 0.05, num_leaves 31, min_data_in_leaf 50, feature_fraction 0.8, bagging 0.8, lambda_l2 10, 400 rounds, 2 seeds`.

**Stage 1** on the §4 features plus three families (each computable for evaluation pairs by the same script):
* `cc_*` (44): decisive street, who folded, outsiders active at the fold, fold-to-winner, size faced at the fold, loser / winner made-hand categories at the decisive street and per street, loser-better flags (category and kicker), showdown category gap, pocket pairs, outsiders' best category and whether it beat the pair, outsider folds / calls with better hands, family-specific flags.
* `st_*` (53): first / last street the partners faced each other, streets both stayed in, heads-up streets, end street, outsiders remaining on the decisive street, first deviating street (surprise > 2), fold-gap between partners, per-street surprise maxima.
* `pp_*` (19): surprise under a **per-player** policy fitted on that player's own other-phase decisions (LightGBM per 100-table chunk with situation features + the player's other-phase conditional rates), scoring in-phase actions: own-policy surprise sums / max / min, `p_min`, counts with p < 0.01 / 0.05, both partners improbable in the same hand, deltas vs. the population surprise, surprise facing the partner.
Development OOF MAP@5: 0.528 → 0.539.

**Stage 2** on each pair's top-20 hands: stage-1 features + `hand_score`, score relative to the pool maximum, chronological rank in the pool, phase-relative position, gaps to neighbouring pool hands, pool hands within 20 hands, **strong-hand chronology** (strong hands before this one, cumulative score before, cumulative share), the pair's **family** one-hot (true in training, predicted at inference), and fold-safe smoothed **target encodings of action-sequence templates** — the partners' ordered actions tagged winner / loser · street · facing partner / outsider / nobody · action (full, pre-flop-only and first-three-token variants, each also within family; m = 10). `learning_rate 0.03, num_leaves 15, min_data_in_leaf 40, feature_fraction 0.7, bagging 0.8, lambda_l2 10, 500 rounds, 3 seeds`. Pool hands rank above all others (`10 + score₂`). Development OOF MAP@5 0.539 → **0.621** (directed_transfer 0.65, soft_play 0.65, coordinated_isolation 0.53). Five slots are always filled; ties broken by pot size then hand id, as the host metric does.

## 8. Final assembly

| vector | source |
|---|---|
| `b8` | 20-seed full-data LightGBM bag (seeds 42 … 269) on the Step-8 pair features; labels / weights from the stage-3 PU recipe applied to the CV OOF |
| `b9` | same on the Step-9 pair features |
| `c8`, `c9` | the fold-CV ensembles (5 folds × 5 seeds) from the two pipelines |

* **final_J** — `risk = mean(rank(b8), rank(b9))`; ties broken by mean probability then a 10⁻¹³ epsilon (no two pairs equal). Most stable candidate.
* **final_F** — `risk = mean(rank(c8), rank(c9), rank(b8₁₀), rank(b9₁₀))` (10-seed halves), i.e. 50 % weight on the fold-CV regime as a hedge against private-set drift.
* Both: families from the Step-18 heads gated at the top 5 % of the blended rank; evidence from Step 18.

## 9. Validation and what did not transfer

* A public probe (Step-8 risk with evidence blanked: 0.775) decomposed our score: pair AP ≈ 0.97, evidence MAP@5 ≈ 0.59 → 0.62 with Step 18, behaviour ≈ 0.96. With our evidence, even a perfect ranking would cap at ≈ 0.92; the leaders' 0.94 requires evidence ≥ 0.70.
* Every change was judged on three development views (labelled AP, pessimistic AP, hidden-150 AP) and confirmed on the public board.
* **Multiple-instance hand model** trained from pair labels: +0.03 on every development view, −0.02 on the public board — its within-pair percentile features scale with phase length (≈ 110 shared hands in development vs. 76 in evaluation). The phase-robust variant (Step 17) scored 0.894 public / **0.900 private**, our best private result; it was not selected because it trailed on the public board although all development views favoured it. With a 30 % public split (≈ 150 positives, score SE ≈ 0.008), the development views were the better guide.
* **Dead ends (verified):** temporal onset / bursts / sessions of the collusion; seat position; stack sizes and stack-fraction transfers; hand timestamps; a common victim; bet-size grids and action-mix fingerprints; equity (EV) transfer; joint action p-values; first-k truncation rules for evidence labels; player-degree features (a development-only artefact: unlabelled dev pairs exclude labelled colluders); family-specific pair rankers; ExtraTrees / logistic / DART ensembles; transductive PU on evaluation pairs. Final-day bagging gains (+0.0003 … +0.0007 public) did not survive the private split.

## 10. Five case reviews (from `final_J`)

**1. `P8FA9939F1886` — rank 1, directed_transfer.** `HDBA527F2C2D221`, `HD9EB30F28B5458`, `HA08CD42774F1F5`, `H950EF8646547F1`, `HE8DFAF2396BF2B`. P1 repeatedly commits the stack to P2 with hopeless holdings: calls from the big blind with 8♥3♥, bets the turn with 8-high, calls a raise to 208 and shoves the river into 7♥7♠ (−299); calls a 3-bet with Q♠7♠ and calls down to an all-in with queen-high against K♦J♠ (−240); calls two re-raises with 4♥3♦ and folds the flop (−109). Over 99 shared hands 526 bb move P1 → P2 and 148 back; with P2 seated P1 enters 20 % of hands with junk, 6 % without. *Benign:* a loose, bluff-heavy player repeatedly running into the partner's strong hands; each hand alone is ordinary bad play.

**2. `PAE6DC4BF00FE` — rank 9, soft_play.** `H715D63EA5BFD96`, `H49C4C83639A21E`, `H0466DD13AAD8F6`, `HF013A54C234011`, `HF3315B7996E692`. Heads-up against P2, P1 never bets made hands: limps 9♠9♥, flops a set on 5♦T♦9♣, checks flop and turn and folds the river to a 2-bb bet; flops a pair of nines and folds to P2's 20-chip river bet holding ace-high; both check every street to showdown. *Benign:* a tight-passive player who slow-plays and folds to any river aggression; check-downs are common between passive players.

**3. `P063AEE30B6A8` — rank 8, coordinated_isolation.** `H9F48475CB65CE2`, `H7D5E4AF2B28B0E`, `HC33F4F889C651D`, `HBD0544815956A5`, `H4F2E56049F11BB`. The pair raises and re-raises each other pre-flop with junk to push outsiders out, then one folds to the other: P1 opens A♦6♠, P2 3-bets 9♦2♦, outsiders fold, P1 folds; P1 opens 6♠2♦, P2 3-bets 8♦2♣, P1 4-bets to 90, P2 folds. *Benign:* two hyper-aggressive players who 3-bet / 4-bet light against every opener; `HBD0544815956A5` actually costs the pair 126 chips to an outsider.

**4. `P8698759D2395` — rank 250, coordinated_isolation, 50 shared hands.** `HBC05929C497F6F`, `H79C01885E6B1A0`, `H9BE917B77C8489`, `HCA7E700AFB342E`, `H5E3E4AF27D7877`. Both partners open very weak hands (9♥5♣, T♠3♦, Q♠7♥) and re-raise each other; in `HCA7E700AFB342E` P2's 4-bet with A♦7♦ inflates a pot that P1 (A♠K♥) then loses to an outsider's rivered flush. *Benign:* two loose-aggressive players; the isolation attempt backfires against a third player, which fits independent aggression as well as coordination. The small sample is why the model ranks it 250th.

**5. `P1944BCEB32C8` — rank 450, directed_transfer, 160 shared hands.** `H7B727FBC6EBAB2`, `HB613D24BED4EF7`, `HF3C572F01DFD65`, `H888B9D10B6DD99`, `HCE7BA51652C427`. P2 enters against P1 with junk (T♠6♥, 3♠T♥, 7♣8♦, 4♦6♦, J♦3♥) and calls large bets with nothing through the river while P1 holds K♥K♣, a pair of fours, A♣A♦; 325 bb move P2 → P1 and 18 back; P2 calls 0.31 times per hand with P1 seated vs. 0.18 without. *Benign:* a calling-station bot that pays off anyone; every hand is an ordinary bad call and P1 simply had the goods.

## Links
* Full technical method: `docs/METHOD_SELECTED.md` · reproduction: `docs/REPRODUCE.md` · experiment log incl. dead ends: `docs/EXPERIMENT_LOG.md` · case reviews with full action logs: `docs/CASE_REVIEWS.md`
* Selected submissions: `deliverables_final/final_J_bags20_ev18.csv`, `deliverables_final/final_F_allbag_ev18.csv`
