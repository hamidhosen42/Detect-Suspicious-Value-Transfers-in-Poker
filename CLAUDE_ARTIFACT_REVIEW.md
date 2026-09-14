# Review of the supplied competition-analysis artifact

Reviewed 14 September 2026 using the full text supplied by the user after the artifact URL could not be read. This completes the **content review** of that artifact; its interactive page and any underlying analysis scripts were not available.

## Verdict

লেখাটির evidence-first architecture, action-context features, table-grouped validation এবং প্রতিটি pair-এর জন্য evidence retrieval—এই দিকগুলো কার্যকর starting point। কিন্তু বেশ কয়েকটি সিদ্ধান্ত প্রমাণের চেয়ে বেশি জোর দিয়ে বলা হয়েছে; score decomposition-এ সরাসরি arithmetic error আছে। এটিকে verified final specification হিসেবে ব্যবহার করার আগে নিচের সংশোধনগুলো প্রয়োজন।

The report combines observed data summaries, metric deductions, unprovided experiments, historical web snapshots, and recommendations. Those categories should be explicitly separated. Repeating a result from a local Markdown file or notebook does not independently verify it.

## Independently confirmed in this review

Selected evidence claims were recomputed by joining `development_evidence.csv` to `development_labels.csv` and filtered rows of `seats.parquet`, with many-to-one join validation.

| Claim | Recomputed result | Interpretation |
|---|---:|---|
| Directed evidence has opposite-sign partner net results | 725/725 entries | Correct for public labeled evidence |
| One partner receives a pot share while the other loses | 725/725 entries | Correct; this is not an exact bilateral transfer ledger in a multiway pot |
| Same losing player across a directed pair's evidence | 148/148 pairs | Strong public-evidence feature hypothesis |
| Both soft-play partners reach showdown | 220/632 = 34.81% | Matches the report's approximately 35% |
| Both directed-transfer partners reach showdown | 181/725 = 24.97% | Descriptive, not itself a label rule |
| Both isolation partners reach showdown | 12/460 = 2.61% | Supports looking beyond showdown-only evidence |

The earlier competition review separately checked file sizes/counts, phase counts, the 693 publicly positive players' exclusion from evaluation pairs, candidate exposure, and evidence pot summaries. Those supporting results remain in `COMPETITION_ANALYSIS.md`.

These findings concern **selected public evidence**, not every coordinated hand or evaluation relationship. In particular, 100% loser consistency on selected evidence does not imply the same player loses in every shared hand.

## Claims that need correction

| Artifact claim | Assessment | Correct interpretation |
|---|---|---|
| Evaluation positives are approximately 0.3% | Unestablished | Private evaluation prevalence is unknown. Public labeled-positive counts cannot identify it. |
| “True eligible population” AP 0.19–0.22 | Misleading description | AP with unlabeled pairs provisionally set to zero is a contaminated background diagnostic, not measured evaluation AP. |
| Report only evaluation-mirrored AP | Bad validation policy | Report confirmed-label AP and unlabeled-background diagnostics separately, plus evidence/behavior components. |
| Hidden family “exists only in eval” | Unsupported | Absence from public positive labels does not establish absence from unlabeled development gameplay. |
| Never predict `none` | Not implied by metric | `none` and `other_coordination` have identical numerical effects at fixed risk/evidence. Neither strictly dominates the other. |
| Predicting a known family for almost every pair is near-optimal | Experiment-dependent | No universal guarantee. Wrong high-ranked class assignments can reduce family AP. Validate routing with the exact scorer. |
| Choose family by `argmax q_F / n_F` | Heuristic, not an AP optimizer | Macro normalization alone does not derive this decision rule; AP depends on ranking and false positives across the population. |
| Monotone calibration is “bit-neutral” | Too strong | A strictly increasing transformation preserves ordering in exact arithmetic. Floating-point saturation and non-strict transforms such as isotonic regression can create ties. |
| Never create ties; fixed decimal precision guarantees this | Incorrect as a universal requirement | Ties can arise naturally. Preserve precision and use the host scorer; do not manufacture ID-based tie-breaking features. |
| Exposure will not discriminate “by design” | Overstated | Similar observed class exposure does not prove the mechanism or eliminate nonlinear/interacting exposure effects. Use exposure chiefly for normalization and uncertainty. |
| Pair-selection rule is exact despite 29 unexplained exclusions | Internally inconsistent | Describe it as an observed approximate match, not a proven generator rule. |
| High leaderboard evidence score proves listed hands are nearly all visibly coordinated hands | Invalid inference | Accurate retrieval of selected annotations says nothing definitive about annotation completeness. |
| Seats-only results prove two families are invisible without action features | Too strong | They show the tested seat features/models struggled. Action features are well motivated; impossibility is not established. |
| Public positions 2–10 are within statistical noise | Unestablished | Requires labels, component results, and dependence-aware uncertainty estimates; positive count alone cannot determine AP standard error. |
| Day-one LB 0.6–0.7 and day-three evidence MAP ≥0.6 | Targets, not evidence-backed forecasts | Treat as aspirations; do not schedule success around unvalidated performance assumptions. |

The report's assertions of 1.9 million exhaustive metric cases, simulations, seats-only model scores, action-anatomy tables, and “105 findings; 0 refuted” were not accompanied by executable scripts or outputs. Searches in the current workspace did not find the cited `evidence_anatomy.py`, `pair_features_all.py`, `pu_gap.py`, or a dossier file. Those numerical experiments remain **unreproduced**, not automatically false.

## Arithmetic errors in the leaderboard decomposition

Every illustrative component triple in section 5.4 fails to produce the score beside it:

| Stated score | Given (Pair AP, Evidence MAP, Behavior MAP) | Actual weighted score |
|---:|---|---:|
| 0.938 | (0.95, 0.85, 0.95) | **0.930** |
| 0.916 | (0.93, 0.80, 0.93) | **0.904** |
| 0.899 | (0.91, 0.75, 0.92) | **0.879** |
| 0.879 | (0.89, 0.70, 0.90) | **0.853** |
| 0.837 | (0.85, 0.60, 0.85) | **0.800** |

The separate lower-bound argument is sound **conditional on the historical score being correct**. For S = 0.93798 and all components at most one:

- Pair AP ≥ `(S − 0.30)/0.70` = **0.9114**.
- Evidence MAP ≥ `(S − 0.80)/0.20` = **0.6899**.
- Behavior MAP ≥ `(S − 0.90)/0.10` = **0.3798**.

These bounds constrain possible components but do not identify a leader's architecture, the hidden prevalence, or whether all relevant evidence was annotated. No current leaderboard was verified in this content review.

## Metric advice: what survives scrutiny

**Keep:** fill unused evidence capacity with additional distinct valid candidates, preserving the order of existing candidates. Appending a unique hand at the end of a list shorter than five cannot lower evidence AP: previous contributions remain unchanged, and a new relevant hand adds nonnegative credit. This does not justify inserting a weaker hand earlier or replacing an existing relevant hand. Duplicate IDs invalidate the submission.

**Keep:** evidence scoring is independent of risk and behavior in the inspected host implementation. Low-risk pairs can still earn evidence credit. All evaluation pairs have at least 38 shared hands, so five candidates exist for every pair.

**Keep with qualification:** a constant score gives approximately prevalence under random label ordering, not for every fixed ID ordering. For tied scores, the host's individual-row stable AP differs from sklearn's threshold-grouped AP. This matters especially for zero-masked behavior scores.

**Qualify:** “each positive is worth approximately 0.7/P” is the maximum scale of one positive's own pair-AP contribution. Changing its rank can also change precision at other positives; this is not an exact marginal improvement formula.

**Qualify:** random evidence AP is not a guaranteed floor. For N candidates and R relevant hands, with 1 ≤ R ≤ 5 and a uniformly random top-five ordering, expected AP is:

`H5/N + (R−1) × (5−H5)/(N × (N−1))`, where `H5 = 1 + 1/2 + 1/3 + 1/4 + 1/5`.

For R = 5 this gives approximately 0.0349 at N = 70 and 0.0239 at N = 100. Individual random retrievals can score zero.

**Qualify:** the scorer tolerating empty evidence cells does not establish the precise behavior of Kaggle's upload layer. Follow the stated submission format and use `NO_EVIDENCE` for unused slots.

## Validation recipe: repair before implementation

1. **Retain positive examples explicitly.** Applying “exclude pairs containing a positive player” to the entire training set would remove every positive. Such an exclusion can only apply to the unlabeled-background construction, with labeled positives retained separately. This still creates a surrogate task rather than a perfect copy of private evaluation.
2. **Prevent held-out label use in sample construction.** Use training-fold labels for any label-dependent background exclusion or weighting. Document held-out diagnostic exclusions separately; never feed their results back into training.
3. **Nest the stacked pipeline.** For each outer table fold, fit evidence models using only outer-training tables. Obtain inner-cross-fitted evidence summaries for pair-model training, and score outer validation using evidence models that never saw its labels. A single global set of evidence OOF features followed by another pair-level CV is insufficient to guarantee this isolation.
4. **Treat unlabeled-zero AP as a diagnostic.** Hidden targets, selection effects, and prevalence differences mean it is not a calibrated forecast or guaranteed lower bound on evaluation AP. Do not claim a measured deployment error rate from it.
5. **Match evidence candidate handling.** Randomly retaining two-thirds of hands can remove annotated evidence or change query difficulty. For an exposure stress test, document whether relevant hands are retained, removed, or scored as missing; each choice defines a different diagnostic.
6. **Temporal labels need care.** Pair-level coordination labels do not prove activity in both halves of a development window. Align evidence and candidate windows; do not simply copy the pair label onto every temporal segment.
7. **Bootstrap dependence appropriately.** Resample tables for uncertainty about generalization across pools. A 30%-of-pairs split simulation may explore a hypothetical public split conditional on observed data, but it does not recover hidden leaderboard uncertainty.

Purely unsupervised player-history summaries computed from available validation gameplay are not automatically label leakage. The report should distinguish label-trained baselines, future information, and legitimate retrospective features. The competition scores completed-period investigations, so retrospective data can be available even when it would be inappropriate for a real-time decision model.

## Poker semantics and evidence labels

The recommended partner-conditioned action features are promising. Their interpretation needs more precision:

- A final-board hand evaluator measures showdown strength on that board; it does not supply decision-time equity. Use the street's available board, relevant opponents, pot odds, and remaining-card uncertainty when evaluating an earlier call or fold.
- A Chen score is a coarse preflop summary, not a calibrated equity estimate. Two high-scoring hands do not make passive play suspicious by themselves.
- Last-aggressor tracking must respect street boundaries and action amounts, distinguish all-in calls from raises, and track active players. Forward-filling an aggressor alone does not fully reconstruct betting obligations or side pots.
- Net losses and winnings in a six-player hand do not identify exactly how much one particular opponent transferred to another. Use clearly labeled flow proxies.
- The isolation summary itself reports only 1.43 aggressive partners on average, so “both raise” is an example motif rather than a property of all evidence hands.
- High retrieval scores cannot justify declaring every non-listed hand of a positive pair behaviorally negative. Compare training on confirmed-negative-pair hands with binary annotation-retrieval training and downweighted unlabeled candidates.
- Generic anomaly heads for the fourth family are hypotheses. High uncertainty among known classes alone is not proof of coordination.
- Pure Python at ten million hand evaluations has no demonstrated runtime here. Benchmark the actual evaluator, cache repeated states, and choose compiled/vectorized implementations if needed.

The section 4.5 phrase “137,739 eligible dev pair × hand rows” also conflates pair count with pair-hand count. Report these separately before estimating memory or runtime.

## Resource claims already superseded by the audit

- Public GitHub code exists in [PR #1](https://github.com/phucthaiv02/poker-suspicious-detection/pull/1), even though the main tree contains only a README. Its source-level defects are recorded in `PUBLIC_RESOURCE_REVIEW.md`.
- The coordinated baseline's saved composite uses oracle behavior labels.
- The suspicious-detection notebook's primary saved AP uses pair-random folds; its table-group run was skipped.
- Base/topological are near-duplicate revisions, not independent evidence.
- E23's behavior rate is fixed and its evidence objective is graded NDCG rather than binary AP.

These issues make the artifact's blanket comparisons of public notebook “projected scores” unreliable. Its 0.19 baseline and 0.84 notebook headline also use different populations and cannot support a direct leaderboard comparison.

## Revised practical recommendation

Keep the evidence-first architecture and the useful observed signatures. Start with a reproducible table-grouped evaluation harness, then compare one general evidence model, family-specific retrieval, and a pair model using correctly cross-fitted evidence summaries. Evaluate confirmed negatives and the broader unlabeled population separately. Use the actual scorer for behavior routing; allow all five valid behavior categories.

Use the seven-day schedule as an ordered work list, not a confirmed countdown. Exact deadline, submission limits, eligibility and current leaderboard details remain subject to verification on the official competition pages. There is no newly trained model or measured leaderboard improvement from this review.
