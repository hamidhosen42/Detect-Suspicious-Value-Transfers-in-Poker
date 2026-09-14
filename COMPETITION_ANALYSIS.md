# Competition analysis: Detect Suspicious Value Transfers in Poker

Analysis date: 14 September 2026. Based on the local competition files, the downloaded reference metric, selected saved notebook outputs, and the official competition page. This is an analysis, not a newly trained solution or leaderboard result.

## Main conclusion

Treat this as **relationship ranking plus evidence retrieval with incomplete labels**, rather than ordinary binary classification. Build action-level evidence features, aggregate them into pair-level risk, and validate on held-out tables. The central risk is obtaining excellent validation scores on curated negatives while ranking ordinary but unusual relationships too highly in the full evaluation population.

The official task also includes an undisclosed coordination family. Consequently, a classifier restricted to the three labeled families is incomplete. Coordination is intermittent, so whole-period averages alone can dilute the signal. [Official overview](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker)

## 1. Verified local data

Counts and schemas below were checked against the actual CSVs and Parquet metadata during this analysis.

| File | Rows | Role |
|---|---:|---|
| `players.parquet` | 12,000 | Account attributes and experience |
| `hands.parquet` | 2,000,000 | Table, phase, time, blinds, board, pot, showdown |
| `seats.parquet` | 12,000,000 | Player cards, stack, contributions, outcome |
| `actions.parquet` | 18,609,028 | Ordered betting decisions and decision context |
| `development_labels.csv` | 1,860 | Confirmed relationship labels |
| `development_evidence.csv` | 1,817 | Labeled supporting pair-hand examples |
| `evaluation_pairs.csv` | 112,540 | Relationships to score |
| `sample_submission.csv` | 112,540 | Eight-column submission template |

There are 400 tables, 1,200,000 development hands, and 800,000 evaluation hands. Labels comprise 1,488 confirmed non-targets and 372 targets: 148 directed transfer, 132 soft play, and 92 coordinated isolation. The observed 20% positive fraction is a property of the labeled sample; it is not an estimate of evaluation prevalence.

The 372 positives involve 693 distinct players. None of those players appears in an evaluation pair, verified directly from the CSVs. A successful model therefore must transfer behavioral patterns to other players.

Evaluation pairs share between 38 and 419 evaluation hands, with median 76 and mean 85.76. The sum is 9,651,820 evaluation pair-hand candidates. This is the relevant retrieval workload: the evidence model needs to select five hands per relationship from that candidate set.

All 1,817 labeled evidence entries join to development hands. Of the positive pairs, 340 have five labeled evidence hands, 21 have four, and 11 have three. **69.29% of evidence hands have no showdown.** Restricting evidence to revealed showdowns would therefore discard most of the public evidence.

| Evidence family | Median final pot, big blinds | Mean final pot, big blinds |
|---|---:|---:|
| Directed transfer | 34.50 | 78.96 |
| Soft play | 17.00 | 54.60 |
| Coordinated isolation | 17.75 | 53.80 |

These summaries describe labeled evidence; they do not establish that large pots imply coordination.

## 2. What the metric actually rewards

The local `slash-poker-competition-metric.ipynb` implements:

`score = 0.70 × pair_AP + 0.20 × evidence_MAP@5 + 0.10 × behavior_MAP`

| Component | Operational consequence |
|---|---|
| Pair AP | Prioritize correctly ordering suspicious relationships, especially near the top. A 0.01 improvement adds 0.007 overall. |
| Evidence MAP@5 | Retrieve the actual labeled evidence IDs in useful order. A 0.01 improvement adds 0.002 overall. |
| Behavior MAP | Route risk scores to one of the three known families accurately. A 0.01 improvement adds 0.001 overall. |

**Important implementation details found by reading the reference code:**

1. Evidence is scored for every true-positive pair, regardless of its submitted risk or predicted behavior. There is no risk threshold gating evidence credit. Return the best available five valid hands even for low-risk pairs.
2. Evidence relevance is binary membership in the truth set. `evidence_rank` in the training CSV does not become a graded relevance label in this metric. Graded rank training is an experiment, not a faithful reproduction of the objective.
3. For each known family, the class score is `risk_score` when that family is predicted and zero otherwise. Behavior MAP is not classification accuracy and is not AP from three freely submitted probabilities.
4. `other_coordination` is excluded from the behavior macro-average, but true unknown-family targets still matter for pair AP and evidence retrieval.
5. AP uses stable, individual-row sorting after sorting by pair ID. This can differ from threshold-grouped AP implementations when predictions tie. Use the supplied scorer for selection; do not exploit IDs to break ties.
6. Evidence AP divides by `min(number_of_relevant_hands, 5)`. With five relevant hands and hits only at ranks 1 and 3, AP is `(1 + 2/3)/5 = 0.3333`.
7. The scorer removes `NO_EVIDENCE` before ranking. Keep valid entries contiguous and place unused sentinels last anyway.

For perspective, perfect pair and behavior scores with zero evidence yield only 0.80. Pair AP 0.90, evidence MAP 0.60, and behavior MAP 0.85 yield 0.835. These are arithmetic scenarios, not forecasts.

Source: local reference notebook, attributed to the [host metric notebook](https://www.kaggle.com/code/florianderoofr/slash-poker-competition-metric). The downloaded code was inspected; its current remote version was not independently retrieved.

## 3. Validation design

Use approximately five outer folds grouped by `table_id`. Keep all rows belonging to a relationship and all players from a table in one fold. Confirm each fold contains all three known target families. Report pooled out-of-fold metrics, per-fold values, and uncertainty using a table-level bootstrap.

Maintain three complementary evaluations:

| Evaluation | Purpose | Limitation |
|---|---|---|
| Confirmed positive vs confirmed negative | Reliable supervised comparison | Curated negatives and label selection may make it optimistic |
| Positive vs held-out unlabeled population, provisionally treated as negative | Stress test ranking against broader background | Hidden positives are counted as errors; this is a contaminated diagnostic, not ground truth |
| Leave one known family out of training | Test transfer to unfamiliar coordination | A held-out known family need not resemble the actual hidden family |

Also test exposure and temporal sensitivity: build features from shorter development windows approaching evaluation exposure, and report low/medium/high shared-hand strata. For window-level experiments, retain only evidence inside the window and distinguish pair identity labels from observed activity inside that window. Do not assume a labeled relationship was active in every subwindow.

Any learned evidence model feeding a pair classifier must produce **out-of-fold training features**. Fit learned normal-play baselines, feature selection, scalers, pseudo-label rules, and ensemble weights inside training folds. A model predicting its own training evidence can make stacked validation look artificially strong.

Do not validate development relationship labels using arbitrary evaluation-period behavior as though activity were guaranteed to persist. Use development features for the primary supervised evaluation; inspect evaluation history as unlabeled data under a clearly documented protocol.

## 4. Feature strategy

### Reconstruct decisions before summarizing them

Join actions to hands and seats, then sort within hand by `action_no`. Use IDs as keys, never predictive values. Normalize monetary features by big blind and, when appropriate, pot or effective stack. Separate all-in calls from aggressive all-ins using `amount` versus `to_call`.

Track action state: current aggressor, who is facing whose bet, active opponents, street, position, pot odds, stack commitment, and prior betting. Validate reconstructed chip accounting against supplied totals, including blinds, returns, and side-pot cases.

Hole cards enable operator-side hand-strength analysis. For decision features, use only the board available on that street; future community cards would introduce hindsight into claims about the quality of a decision. Distinguish actual-card equity from equity against a plausible opponent range. Both are imperfect indicators of intent, especially when bets can induce folds.

### Candidate feature families

| Family | Useful signals | Main confounder to control |
|---|---|---|
| Directed transfer | Directional contribution/outcome asymmetry, expensive weak calls benefiting partner, strong folds to partner, repeated directional loss episodes | Skill gap, tilt, variance, multiway pots |
| Soft play | Reduced aggression against partner relative to other opponents in comparable spots, unusual checks with strong hands, partner-conditioned folds/calls | Passive player style, position, legitimate pot control |
| Coordinated isolation | Ordered sequences where both pressure a third player, third-player folds, low subsequent partner conflict, partner-conditioned aggression | Ordinary squeezes and strong hands |
| General coordination | Persistent partner-specific deviations, unusually concentrated episodes, repeated interaction motifs | Shared schedule, common strategy, exposure |

Do not equate one player's net loss with a direct transfer to the other in a six-player hand. If allocating losses to winners, label it a proxy and handle side pots; it is not an observed transfer ledger.

Compute pair-vs-player-history residuals, excluding the focal relationship where feasible. Compare with matched decision contexts rather than raw population averages. Shrink low-exposure rates toward a background estimate, for example `(events + alpha × background_rate)/(opportunities + alpha)`, with alpha selected using training folds only.

### Aggregate sparse suspicious episodes

For each pair and direction, use top-1/top-3/top-5 evidence scores, upper quantiles, suspicious-event counts per opportunity, local time-window peaks, directional consistency, and excess behavior relative to each player's usual strategy. Combine peak evidence with repeatability: a single extreme hand can be luck, while averages can hide occasional coordination.

Use graph features only after a strong tabular baseline. Prefer graph edges representing behavior residuals; plain shared-table density is largely structural. Table IDs can support grouping and normalization without being model inputs.

## 5. Recommended model architecture

1. **Evidence baseline:** rank pair-hand candidates using interpretable action features. Compare with largest-pot and simple directional-flow baselines.
2. **Supervised evidence heads:** one general head plus family-specific heads. Public evidence hands are positives; shared hands from confirmed non-target pairs provide cleaner negatives. Unlisted hands of positive pairs are not guaranteed benign, so test downweighted or positive-unlabeled treatment separately.
3. **Pair model:** a gradient-boosted tree baseline on behavioral aggregates and cross-fitted evidence summaries. Establish one reproducible implementation before adding model diversity.
4. **Behavior routing:** three known-family heads; compare hard family routing with probability-weighted evidence mixtures and a general evidence fallback.
5. **Unknown-family support:** add partner-specific anomaly signals to the risk ensemble, evaluated with family holdouts. Low confidence among known classes alone is insufficient evidence of coordination.
6. **Ensemble:** retain models only when they improve outer-fold ranking, evidence retrieval, or robustness. Do not optimize weights repeatedly against the public leaderboard.

The binary task has both trusted negatives and unlabeled examples. Start with confirmed positive/negative learning, then compare conservative unlabeled sampling or PU approaches. Treating every unlabeled relationship as a definite negative is unjustified. Conversely, ignoring the broader unlabeled population during diagnostics leaves the main deployment failure mode untested.

## 6. Existing notebook evidence

The following numbers were read from saved outputs, not reproduced by rerunning training. Different validation protocols prevent direct leaderboard comparisons.

| Local notebook | Confirmed-label pair AP | Unlabeled-background diagnostic | Evidence MAP@5 |
|---|---:|---:|---:|
| `suspicious-detection.ipynb` | 0.9737 | Weighted diagnostic; not directly comparable | 0.3488, or 0.3914 with H2 blend |
| `poker-e23-clean-label-pn-submission-candidate.ipynb` | 0.9743 | 0.7060 PU stress AP | 0.3562 |
| `poker-collusion-pu-aware-evidence-ranker.ipynb` | 0.9380 | 0.5173 PU stress AP | 0.4405 |

The E23 notebook's saved projected composite is 0.6318; it is not an observed leaderboard score. The differences show why both population stress testing and retrieval quality deserve attention. They do not establish that one notebook is universally best. Compare them on identical folds and candidate universes before choosing a foundation.

## 7. Efficient implementation

Process selected columns and use integer-coded keys internally. Six players create 15 unordered pairs per hand, so expanding all hands yields 30 million pair-hand rows before action joins. Filter to required pairs early and partition by table or phase. Avoid a global Python-object-heavy join that duplicates action histories across every pair.

Cache player-hand action summaries, filtered pair-hand features, and pair aggregates separately. The evaluation candidate count is about 9.65 million, large enough that memory layout and intermediate persistence matter even though the compressed source files are modest.

Record feature versions, folds, seeds, dependency versions, data fingerprints, all component metrics, and runtime. For every high-ranked alert, preserve the actual action sequence and a plausible benign explanation. This also makes model errors easier to diagnose.

## 8. Experiment order and completion criteria

| Priority | Experiment | Evidence required before advancing |
|---|---|---|
| 1 | Reproduce supplied scorer and make valid baseline output | Correct pair coverage, legal scores/categories, shared evaluation evidence |
| 2 | Fixed table folds and simple pair/evidence baselines | All metric components plus population stress results |
| 3 | Decision reconstruction and opponent-relative features | Improvement beyond exposure, pot size, and raw profit proxies |
| 4 | Cross-fitted family/general evidence models | Held-out MAP@5 gain; inspect individual retrieved hands |
| 5 | Episodic features and conservative unlabeled learning | Broader-background improvement without unstable fold regressions |
| 6 | Unknown-family holdouts and ensemble | Consistent incremental gain and documented uncertainty |
| 7 | Final reproducibility pass | Clean run regenerates the exact selected CSV |

Before export, require exactly 112,540 unique pair IDs matching the template, finite risk in [0,1], allowed behaviors, no empty evidence cells, no within-row duplicate hands, and evidence belonging to the evaluation phase with both players present. Every evaluation pair has enough shared hands for five candidates; supplying all five is useful when they can be ranked, although fewer are permitted.

## 9. Rules, timing, and source limitations

The official overview confirms a $5,000 prize pool and requires prize contenders to publish reproducible code, a writeup of at most 1,500 words, and five evidence case reviews within seven days of private leaderboard release. It prohibits using non-gameplay artifacts such as ID formats or file ordering. Approximately 30% of pairs are public and 70% private. [Official competition information](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker)

The live rules/data pages did not return readable content during this review. Exact closing time, current leaderboard positions, submission limits, external-data permissions, and team restrictions therefore remain unverified here. The older facts file records some of these, but its relative dates and leaderboard snapshot must not be presented as current. Consult the [official rules](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker/rules) and authenticated timeline before relying on administrative details.

## 10. Corrections to the existing facts notes

`COMPETITION_FACTS.md` is preserved as historical context. Its following statements need qualification:

- The workspace is no longer empty; all eight data files are present.
- The statement that no public participant notebooks exist conflicts with its later inventory of downloaded notebooks; current publication status was not checked.
- The proposed evaluation selection rule is described as exact despite 29 unexplained exceptions. It is an observed near-match, not a proven generator rule, and should not become a predictive artifact.
- A low risk score does not itself zero out evidence credit in the inspected metric.
- Public-label AP and a PU-stress composite do not directly estimate private leaderboard performance.
- A historical leader score does not reveal the leader's method, nor prove that a particular architecture is responsible.

The most defensible next implementation is a table-grouped baseline with cross-fitted action-level evidence, opponent-relative pair features, and explicit unlabeled-population diagnostics.
