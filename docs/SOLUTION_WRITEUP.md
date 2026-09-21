# Solution Write-up — Detect Suspicious Value Transfers in Poker

**Team:** Hack2Publish — Md. Hamid Hosen ([hosen42](https://www.kaggle.com/hosen42)), [esfersami50](https://www.kaggle.com/esfersami50), Foysal Emon Shanto ([foysalemonshanto](https://www.kaggle.com/foysalemonshanto))
**Selected submissions:** `final_J_bags20_ev18.csv` (public 0.90025 / private 0.89598) and `final_F_allbag_ev18.csv` (public 0.89964 / private 0.89566) — private rank 39 / 371
**Code:** https://github.com/hamidhosen42/Detect-Suspicious-Value-Transfers-in-Poker (see `docs/REPRODUCE.md`)

## Leaderboard results

Final private score **0.89598** (rank 39 / 371) · public 0.90025 · winner 0.9407 private.

| submission | risk | evidence | public | private | selected |
|---|---|---|---|---|---|
| **final_J_bags20_ev18** | rank-avg of two 20-seed full-data bags (Step-8 + Step-9 features) | Step 18 | **0.90025** | **0.89598** | ✔ |
| final_H_bagsonly_ev18 | same, 10-seed bags | Step 18 | 0.90022 | 0.89603 | |
| final_G_step8heavy_ev18 | bags + CV models, Step-8 weighted 2:1 | Step 18 | 0.89950 | 0.89589 | |
| **final_F_allbag_ev18** | bags + fold-CV models, equal weights | Step 18 | 0.89964 | 0.89566 | ✔ |
| final_E_blend3_ev18 | Step-8 CV + Step-9 CV + 10-seed Step-8 bag | Step 18 | 0.89907 | 0.89565 | |
| blend8_9_16bag | as E | Step 16 | 0.89716 | 0.89499 | |
| blend89_ev16 | Step-8 CV + Step-9 CV | Step 16 | 0.89690 | 0.89454 | |
| step17_milv3 | Step 16 + phase-robust MIL hand model | Step 16 | 0.89380 | **0.89955** | not selected |
| step16 | Step-8 CV model | Step 16 | 0.89625 | 0.89503 | |
| step8 | Step-8 CV model | Step 8 | 0.89378 | 0.89159 | |
| step9 | Step-9 CV model | Step 9 | 0.89174 | 0.89113 | |
| step6 (exposure matching) | | | 0.84157 | 0.83039 | |
| step3 (first card-aware model) | | | 0.80136 | 0.79778 | |
| best public notebook re-run | | | 0.69243 | 0.69089 | |

Every final-day blend lost ≈ 0.004 from public to private — the bagging and blending gains were inside the noise of the 30 % public split. The one submission that *gained* on private, Step 17 (MIL hand model, +0.006), had trailed on the public board and was not selected, although all three development views had favoured it.

## 1. What the coordination looks like in the data

Every decision in the log is card-conditional: the population's voluntary-entry rate rises from 2 % for the weakest hole cards to 96 % for the strongest. Reading the 1,817 labelled evidence hands showed that the planted behaviour is **two partners acting "wrongly for their cards" in the same hand** — entering pots together with junk, calling the partner's bets with nothing, folding a made hand to the partner, checking a strong hand down heads-up, or raising and re-raising each other pre-flop with 7-2-type holdings to squeeze outsiders out. Chip transfer is a consequence, not the signal: chip-flow-only models plateau near 0.69.

All features are computed from gameplay only: hole cards, board, positions, stacks, the ordered action log, and hand chronology from `started_at`. No player metadata, ID formats, file order or generator internals are used.

## 2. Pipeline (three levels)

### 2.1 Per-action policy surprise
A 6-class LightGBM "behaviour clone" of the population policy (fold / check / call / bet / raise / all-in) is trained on all 18.6 M actions from what the acting bot can see: hole cards (rank, suit, pair), exact 7-card made-hand category at the current street, board texture, position, to-call / pot / stack ratios, raises faced, own prior actions, and the player's **other-phase** style rates (VPIP, PFR, aggression, fold-to-bet, showdown rate). Because collusion is phase-specific, the other phase is a clean baseline for each player. Every action gets `surprise = −log P(taken action)`.

### 2.2 Pair-hand features (one row per pair × shared hand)
~240 features: pre-flop policy surprise and Chen-score junk entries for both partners; exact hand strength per street and rank among the six dealt players; partner-versus-outsider responses (fold / call / raise when the partner set the price vs. when an outsider did); heads-up check-downs; squeeze patterns; chip transfer; sums, minima and maxima of action surprise, and the surprise of actions taken **while facing the partner's bet**; within-pair percentile ranks.

### 2.3 Pair ranking (70 % of the metric)
Pair features aggregate the hand rows: means, sums, top-3 means, counts above thresholds, per-player contrasts with-partner minus without-partner (shrunk by exposure), and **other-phase pair baselines** (how the same two players play together in the phase where they are not colluding). Two design choices mattered most:

* **Exposure matching.** The evaluation phase has 2,000 hands per table (median 76 shared hands per pair) versus 3,000 in development. The pair model is trained on 2,000-hand windows of development so every count-like feature has the same scale it will have at inference.
* **Positive–unlabelled training with hard negatives kept.** Only 1,860 development pairs are labelled. A score-mixture estimate suggests roughly 150 hidden colluders among the 120 k unlabelled pairs (~0.4 % prevalence), so unlabelled pairs are used as weak negatives (weight 0.1) and, after a first pass, the ones that score like confirmed colluders become pseudo-positives. Crucially, the "ambiguous" band of loose-but-honest pairs is kept as weak negatives rather than dropped — those are exactly the negatives the leaderboard ranking has to beat (dropping them cost ≈ 0.01 AP).

The final risk is the rank-average of two 20-seed LightGBM bags trained on all development rows: one on the Step-8 feature set, one on the Step-9 set (adds coordination z-scores: observed joint-deviation counts against the independence expectation n·p₁·p₂ from each player's own field rate).

### 2.4 Behaviour family (10 %)
Three one-vs-rest LightGBM heads on the pair features, trained on labelled positives only; the family with the highest prior-normalised probability is predicted for the top 5 % of pairs by risk, `none` for the rest (the metric treats `none` and `other_coordination` identically, and predicting a family for a false positive costs nothing extra).

### 2.5 Evidence hands (20 %)
Two stages, trained on the coordinated pairs' hands only (labelled evidence vs. their other hands, table-grouped folds):

1. **Stage 1 ranker** on the pair-hand features plus three families added in the last two days: made-hand categories of both partners at the decisive street (who folded, whether the loser held the better hand, outsiders' categories); street timing of the partner interaction (first/last street facing each other, streets both stayed in, outsiders left); and action surprise under a **per-player** policy fitted on that player's other-phase decisions. Development MAP@5 0.528 → 0.539.
2. **Stage 2 re-ranker** on each pair's top-20 pool: pool-relative score, chronology of strong hands, the pair's family, and smoothed target encodings of the pair's ordered action-sequence template ("winner raises, loser calls, winner bets flop, loser folds to partner"). Development MAP@5 0.539 → 0.621.

Five slots are always filled (evidence is never penalised for negatives); ties in risk are removed with a 10⁻¹³ epsilon ordered by the pair's maximum hand score.

## 3. Validation

Public-leaderboard probes decomposed our score (a submission with evidence blanked scored 0.775), giving pair AP ≈ 0.97 and evidence MAP@5 ≈ 0.59–0.62 on the leaderboard, consistent with development. Three development views were tracked for every change: AP on labelled pairs, AP with all unlabelled pairs as negatives, and AP with a fixed set of 150 colluder-like unlabelled pairs counted as positives. Changes were shipped only when all views improved, and each was confirmed on the public leaderboard before being kept.

## 4. What did not work (verified)
* A multiple-instance hand model trained from pair labels: large development gains, −0.02 on the leaderboard; the cause was within-pair percentile features whose scale depends on phase length.
* Player-degree / partner-exclusivity features: a development-only artefact (unlabelled development pairs exclude labelled colluders).
* Temporal clustering or onset of the collusion, bet-size fingerprints, stack-fraction transfers, equity (EV) transfer, a common victim, first-k truncation rules for evidence labels, player metadata, graph features, family-specific pair rankers, ExtraTrees / logistic / DART ensembles.

## 5. Selected submissions
`final_J` (best public, most stable: 20 seeds per bag) and `final_F` (same evidence, risk blended with the fold-CV models — a different training regime as a hedge against private-set drift).
