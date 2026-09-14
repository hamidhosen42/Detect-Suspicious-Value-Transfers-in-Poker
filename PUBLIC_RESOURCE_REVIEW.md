# Public resource audit — poker competition

Reviewed 14 September 2026. বাংলা সারাংশ: পাওয়া সব local notebook-এর code ও saved output পর্যালোচনা করা হয়েছে; public web resources খোঁজা হয়েছে; নতুন একটি GitHub PR-এর code-ও পাওয়া গেছে। কয়েকটি বড় validation ও implementation সমস্যা ধরা পড়েছে। Claude artifact-এর content এবং Kaggle-এর পূর্ণ live listing পাওয়া যায়নি, তাই এটিকে internet-এর সব resource-এর সম্পূর্ণ তালিকা বলা যাবে না।

## Scope and evidence standard

This review covers all eight notebook files in the workspace, the official overview and downloaded metric, the public GitHub README **and newly located PR #1**, available discussion references, and attempts to retrieve the supplied Claude artifact. Notebook training was not rerun. Reported model scores are saved outputs, not independently reproduced results. Small metric counterexamples were executed locally.

Notebook cell references below are **zero-based JSON cell indices**, including Markdown cells. The companion `PUBLIC_RESOURCE_INVENTORY.json` records file hashes, code sizes, saved runtime metadata, and locations of key findings. Local notebook provenance is attributed by the existing facts notes; exact original author URLs and current publication versions were not recoverable from their metadata or indexed searches.

## Resource coverage

| Resource | Retrieval status | Use and limitation |
|---|---|---|
| [Official overview](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker/overview) | Indexed text available | Authoritative task description; cached relative dates are not a deadline |
| [Official metric](https://www.kaggle.com/code/florianderoofr/slash-poker-competition-metric) | Local notebook inspected | Exact local implementation; remote current version not retrieved |
| [Kaggle code listing](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker/code) | No readable listing | Cannot certify all currently public notebooks were discovered |
| [Kaggle discussions](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker/discussion) | No readable live listing | Existing notes mention Clarifications and CV–LB difference; not newly verified |
| [Leaderboard](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker/leaderboard) | Empty readable response | No current ranking or winning method inferred |
| [Rules](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker/rules) | Not readable in preceding review | Administrative claims in historical notes remain provisional |
| [Public repository](https://github.com/phucthaiv02/poker-suspicious-detection) | README and main tree readable | Main tree shows README only; this does not mean the repository has no code |
| [Public PR #1](https://github.com/phucthaiv02/poker-suspicious-detection/pull/1) | Description and four-commit patch retrieved | Actual baseline code audited; important defects below |
| [Launch announcement](https://www.reddit.com/r/kaggle/comments/1w8ele6/5k_prize_pool_quick_2week_poker_anomaly_detection/) | Available | Launch context, not modeling evidence |
| [User's Claude artifact](https://claude.ai/code/artifact/6e1ce434-9125-4efc-a26e-dce76915934a?via=auto_preview) | URL content blocked; user subsequently supplied full text | Content reviewed in `CLAUDE_ARTIFACT_REVIEW.md`; underlying scripts unavailable |
| Eight local notebooks | All source extracted; training, scoring, feature and inference paths inspected | Saved-output comparisons require common-fold reproduction |

Searches included the competition title/slug, GitHub repositories, discussion terms, and exact participant-notebook slugs. Many results were unrelated; they are deliberately excluded. Unindexed or inaccessible material remains outside this review.

## Most consequential findings

1. **An apparently strong composite uses true behavior labels.** `coordinated-collusion-value-transfer-detection.ipynb`, cell 33, assigns `row.behavior_family` to every positive in its development submission. Its saved Behavior MAP is 1.00000 and composite 0.68197. That composite is not fully out-of-fold performance.
2. **The most elaborate notebook's primary folds are pair-random.** `suspicious-detection.ipynb`, cell 23, uses `StratifiedKFold`. The table-group robustness run is disabled in cell 1 and explicitly skipped in saved output. Its 0.9737 AP is not table-held-out AP.
3. **E23's “optimal” 2.5% behavior rate is hardcoded.** Cell 19 sets `best_positive_rate = 0.025`; its print label overstates the optimization performed there.
4. **Base and topological notebooks are near-duplicates.** Code-line sequence similarity is 98.57% using `difflib.SequenceMatcher(..., autojunk=False)`. Their results are related experiments, not independent corroboration.
5. **GitHub code exists in a PR, but the default baseline needs repairs.** Its behavior schema detector misses `behavior_family`; its action aliases collide; routing thresholds change scale between validation and inference; stacked validation lacks complete outer-fold isolation.
6. **Local metric helpers differ from the host.** Stable per-row AP versus threshold-grouped AP changes results with tied scores; zero-masked behavior scores create ties routinely. One notebook also handles sentinel gaps differently.

## Notebook-by-notebook comparison

These are stored results from the exact local files. PU columns use different sampling/weighting protocols and should not be directly ranked as equivalent measurements.

| Notebook | Labeled pair AP | Unlabeled-background AP diagnostic | Evidence MAP@5 | Interpretation |
|---|---:|---:|---:|---|
| `coordinated-collusion-value-transfer-detection` | 0.8053 | Not reported in inspected summary | 0.0914 | Simple baseline; composite contaminated by oracle behavior |
| `poker-collusion-pu-aware-evidence-ranker` | 0.9380 | 0.5173 | 0.4405 | Compact learned evidence ranker; weak broad-background pair ranking |
| `detecting-collusive-value-transfer-in-6-max-poker` | 0.9725 overall; 0.9778 selected blend | 0.6888 overall; 0.7407 selected blend | 0.4509 | Useful feature and ablation reference; selected results tuned on OOF |
| `base notebook` | 0.9774 | 0.6518 | 0.4970 | Triple-tree risk, specialist evidence blend |
| `topological-collusion-dynamics-value-transfer-eda` | 0.9653 | 0.6836 | 0.5055 | Related revision, better saved evidence result |
| `poker-e23-clean-label-pn-submission-candidate` | 0.9743 selected; 0.9596 P/N branch alone | 0.7060 selected | 0.3562 | Final model blends clean-label and PU-trained branches |
| `suspicious-detection` | 0.9737 primary | 0.5679 weighted-stable surrogate | 0.3488 routed; 0.3914 H2 blend | Strong engineering ideas; primary split permits table/player overlap |
| `slash-poker-competition-metric` | — | — | — | Scoring authority, not a predictive baseline |

### 1. Coordinated Collusion & Value-Transfer Detection

**Actual implementation:** small LightGBM risk and behavior models, table-grouped folds, pair seat statistics, aggregate action counts, metadata comparisons, and hand retrieval from chip disparity and pot-size heuristics.

**Useful:** readable baseline, inexpensive summaries, clear submission construction, pair-ID-aware AP helper.

**Problems:**

- Cell 33 uses oracle behavior labels in the final development metric, despite separately producing behavior OOF predictions. The 1.0 behavior component must be replaced with actual OOF routing before comparing composite scores.
- The action extractor classifies every `all_in` as aggression, including calls. It does not use `to_call` to distinguish them.
- Counting hands where both players checked is not the same as verifying a heads-up checkdown. Joint aggression counts do not reconstruct a squeeze sequence. Narrative claims are more specific than these implemented features.
- The advertised PU/Elkan–Noto framing is not established by the displayed labeled-data LightGBM training path. Do not infer specialized PU estimation from the title.
- Its evidence helper preserves sentinel slot positions rather than compressing them like the host; duplicates are skipped instead of rejected.

**Decision:** keep as a sanity baseline; do not choose it for evidence quality or trust its composite.

### 2. Poker Collusion — PU-Aware Evidence Ranker

**Actual implementation:** XGBoost pair and behavior models with table-grouped folds, weighted unlabeled background, action-context features, and an `XGBRanker` using pairwise loss and binary evidence targets. The evidence matrix includes within-pair percentiles and family flags.

**Useful:** a compact reference for constructing retrieval queries; training on true family and validating with predicted family makes family-routing error visible.

**Limitations:** outer validation labels are also used for early stopping; behavior rate is selected on the reported OOF sample; all unlisted candidate hands inside positive pairs receive binary zero for retrieval. That last choice can be reasonable for reproducing the public relevance list but does not prove those hands are behaviorally benign. The saved 5% behavior activation rate is not an inferred evaluation prevalence.

**Decision:** useful evidence baseline and feature scaffold; pair ranking needs stronger context features and population diagnostics.

### 3. Detecting Collusive Value Transfer in 6-Max Poker

**Actual implementation:** player-history baselines, one-way flow and concentration features, partner-versus-field contrasts, temporal/burst summaries, overall and family-specific XGBoost detectors, and pairwise evidence ranking. It compares multiple risk combiners explicitly.

**Useful:** strongest directly visible risk-combination experiment among the inspected table-grouped saved outputs: selected blend PU diagnostic 0.7407 versus overall detector 0.6888. Its baseline/residual functions are good implementation references.

**Limitations:** the combiner, family choice, and behavior cutoff are chosen using the same OOF outputs subsequently reported. This creates selection optimism. Behavior activation reaches the upper edge of its 2% search grid, which is a reason to inspect the diagnostic design, not assume 2% is the true optimum. Outer-fold early stopping also makes results less independent than a nested selection protocol.

**Decision:** first reference for pair features and ablations. Freeze folds and rerun selected candidates before importing its numerical conclusions.

### 4–5. Base notebook and Topological Collusion Dynamics

**Actual implementation:** XGBoost/LightGBM/CatBoost risk ensemble; family detectors; partner-history contrasts and shrinkage; global plus family-specific evidence models combining classifiers and rankers.

**Useful:** substantial evidence lift in saved results. Topological global evidence MAP is 0.4629, specialist 0.4973, and final blend 0.5055. Compare these pieces as candidate components.

**Specific differences and cautions:**

- Topological fixes a Boolean-precedence error in `is_pure_dump`. Base implements `A OR (B AND folded)`; topological implements `(A OR B) AND folded`. The base expression treats transfer direction asymmetrically.
- Topological adds three event flags and changes behavior grids and evidence weights. Its final evidence mixture is 75% specialist/25% global, versus base's 60%/40%.
- Narrative descriptions of a 50/50 general-versus-decoupled risk blend do not match the final base/topological code: both use `0.40 XGB + 0.35 LGBM + 0.25 CatBoost` for pair risk.
- The displayed feature formula called a Welch statistic omits the variance denominator; it should be described as a sample-size-scaled contrast, not a Welch test.
- Specialist classifier probabilities and raw ranker scores are blended directly. Their scales are not inherently compatible; compare within-pair percentile blending.
- Specialist ranker rows are re-sorted by pair before positional blending. The original evidence frame is sorted by pair and hand, but the later pair-only sort does not explicitly preserve tie order. This is an **alignment risk to test**, not a proven observed corruption; join predictions using `(pair_id, hand_id)`.
- Learned retrieval is restricted to the top 50,000 risk-ranked pairs, with heuristics elsewhere. An evidence score over all positive development candidates need not represent this production routing. Evaluate the same fallback policy in OOF.

**Decision:** use topological as a specialist-retrieval experiment and base as its comparison revision. Do not ensemble them merely because filenames differ.

### 6. E23 Clean-Label P/N Submission Candidate

**Actual implementation:** retains a PU-trained topological branch, adds a three-seed confirmed-positive/confirmed-negative triple-tree branch, and blends their within-fold ranks. It uses graded `6 - evidence_rank` labels with an NDCG objective for evidence.

**Useful:** explicit comparison of clean-label learning with weighted unlabeled learning. The clean-label branch is a sensible robustness candidate.

**Problems:**

- The final model is not exclusively clean-label-trained; it blends the clean branch with the original branch.
- Saved clean-only AP 0.95964 is below the frozen topological AP 0.97876. The selected blend's AP is 0.97426. Do not describe E23 as a demonstrated improvement in labeled AP.
- Its blend eligibility conditions have a fallback of alpha 0.25 even when no candidate passes. Saved alpha 0.25 has zero improving folds. The fallback does not establish the desired robustness condition.
- Cell 19's 2.5% behavior activation is fixed, although the output calls it optimal.
- Graded NDCG optimizes a different relevance weighting from binary Evidence MAP@5. Saved MAP 0.3562 is materially below the base/topological saved results, although features and validation choices also differ, so the loss alone cannot be blamed.

**Decision:** retain the clean-label ablation, not the full notebook as an unquestioned upgrade. Compare binary versus graded retrieval on identical queries and folds.

### 7. Suspicious Detection / V3-H2

**Actual implementation:** rich action features, pool context, temporal episodes, member-history deltas, nested early-stop/routing splits, family evidence rankers, a hand-strength module, and strict submission checks.

**Useful:** separating early stopping and routing selection inside outer training; weighted population diagnostics; production-matched evidence fallback evaluation; keyed H2 evidence alignment; exact local composite helper and export validation.

**Limitations:** primary outer and inner splits are stratified by pair rather than table. Saved table-group diagnostic was skipped. Nested stopping avoids one type of selection leakage but does not establish unseen-player or unseen-table generalization. The H2 module uses the recorded final board with hand-level action summaries: these are retrospective hand descriptors, not decision-time equity or a causal counterfactual. Reconstruct street-specific information before interpreting them as evidence of an irrational decision.

**Decision:** reuse validation/export engineering and test H2 features separately. Rerun the full chain with table-grouped outer folds before comparing its scores with grouped baselines.

### 8. Host metric

The inspected code is the reference for comparisons. It scores evidence independently of submitted risk and behavior; uses binary relevant-hand membership; and computes known-family AP after zero-masking non-selected families. `none` and `other_coordination` therefore have identical numerical effects when risk and evidence are held fixed. This equivalence does **not** imply that every benign pair should be described as other coordination.

Executed small counterexamples:

| Check | Result |
|---|---|
| Tied scores, labels `[1, 0]` in host order | Host AP 1.0; sklearn threshold-grouped AP 0.5 |
| Relevant hand after one `NO_EVIDENCE` slot | Host evidence AP 1.0; coordinated notebook helper 0.5 |
| Change only `none` to `other_coordination` | Host total unchanged |
| Base dump flag with A=true, B=false, folded=false | Base true; corrected expression false |

Use actual risk scores without artificial ID-based perturbation. Natural ties must be evaluated correctly, not exploited.

## Newly discovered GitHub baseline code

The [open PR](https://github.com/phucthaiv02/poker-suspicious-detection/pull/1) contains four commits ending at `0f48bfc` in the retrieved patch. It supplies schema, feature, model, pipeline, submission, config, and test modules. This corrects the earlier conclusion that only a README was publicly available: that was true of the visible main tree, not the PR branch.

The code attempts a useful evidence-first architecture with clean evidence negatives and grouped cross-fitting. However, static inspection found these issues:

| Severity | Location | Finding and repair |
|---|---|---|
| Blocking default configuration | `configs/baseline.yaml`; `features.py::build_action_player_hand_features` | Both `all_in` and `all-in` normalize to `act_count_all_in`, generating duplicate aggregation aliases. Normalize action values once and deduplicate categories. |
| High | `schema.py::detect_schema` | Behavior candidates omit the actual `behavior_family` column. With default overrides, positive behaviors become `unknown_target`; disclosed heads get no positive examples. Add the schema mapping. |
| High validation risk | `pipeline.py::train` | Evidence OOF features are built once globally, then a second independent GroupKFold fits pair/behavior models. Outer-held-out table labels can influence upstream evidence models used to construct downstream training features. Nest the full evidence training/scoring stage inside each pair outer fold. |
| High inference mismatch | `pipeline.py::train` and `predict_submission` | Behavior thresholds are selected on raw OOF pair probabilities, while evaluation risk is rank-normalized before applying those same thresholds. Apply one consistent scoring transformation during tuning and inference. |
| Metric mismatch | `metrics.py` | sklearn AP differs from host tie handling; helper tests do not cover those cases. Use host scorer. |
| Feature quality | `features.py` | Generic numeric means/sums lose betting order, partner response, street context, and hole-card semantics. These summaries are a baseline, not a complete coordination detector. |
| Runtime | `prepare_features` | Both phases are expanded and featurized before phase filtering; action aggregation is repeated for development and evaluation. Filter early and cache shared player-hand summaries. |
| Validation scope | PR tests | Three small metric/routing/retrieval tests do not exercise schema-to-submission on the real data. Passing them is not evidence the default pipeline runs. |

These are source-level findings; the full PR pipeline was not executed. No leaderboard score or real-data successful run was established. **Recommendation: use its modular layout as a reference after repairs, not as a ready-to-run competitive solution.**

## Discussion, leaderboard, and existing analysis claims

The older notes report a host clarification permitting public external data/pretrained models and a participant CV–LB gap question. Since live discussion content was inaccessible, preserve these as attributed historical notes rather than newly verified rules. The CV–LB question is consistent with the saved population-stress gaps, but does not quantify them independently.

Historical leaderboard numbers cannot establish current rank, hidden prevalence, or competitors' methods. Conditional arithmetic bounds are legitimate: if a score were 0.93798, component bounds imply pair AP at least `(0.93798 - 0.30)/0.70 = 0.9114` and evidence MAP at least `(0.93798 - 0.80)/0.20 = 0.6899`. Those bounds neither verify the score nor reveal how it was obtained. Standard errors cannot be inferred without the relevant labels and dependence structure.

Existing `ANALYSIS.md` and `COMPETITION_FACTS.md` contain useful notes but are not independent public sources. Their claims about exact prevalence, generator selection rules, and verified multi-lens findings need reproducible support before reuse. In particular, “exact” selection rules with unexplained exceptions remain approximate descriptions, and `none`/`other_coordination` metric equivalence is neutrality rather than a strict dominance result.

## What to use in our solution

| Need | Best reference to investigate | Required change |
|---|---|---|
| Pair behavioral features | 6-max notebook | Freeze grouped folds; isolate model/combiner selection |
| Specialist evidence retrieval | Topological notebook | Binary metric; keyed alignment; validate actual routing/fallback |
| Clean-label comparison | E23 branch | Compare P/N and PU under the same selection protocol |
| Validation and submission engineering | Suspicious Detection | Replace primary pair-random folds with grouped folds |
| Modular project structure | GitHub PR | Fix config/schema, nested stacking and score transformation |
| Scoring | Host metric | Reuse exact implementation and tie behavior |

The next experiment should compare these **components on one common validation framework**, not compare their printed headline scores. Keep gameplay-only predictive features, group by table, cross-fit the entire stacked pipeline, retrieve five valid candidate hands for every evaluation pair, and report pair AP, routed evidence MAP@5, and behavior MAP separately. This audit identifies implementation candidates; it does not claim they already improve the leaderboard.

## Claude artifact: content review completed from supplied text

The page's content endpoint returned a 403 challenge. The user subsequently supplied the full artifact text, which has now been reviewed in `CLAUDE_ARTIFACT_REVIEW.md`. Selected evidence statistics were recomputed from the data. The review identifies arithmetic errors, unsupported prevalence and validation claims, and necessary corrections to the proposed pipeline. The interactive page and referenced experiment scripts were not inspected.
