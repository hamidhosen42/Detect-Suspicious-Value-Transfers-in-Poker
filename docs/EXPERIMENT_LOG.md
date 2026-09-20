# Experiment log — every iteration, its leaderboard score, and what it taught

Public leaderboard scores (team submissions). "Dev" numbers are our development views: labelled AP / pessimistic AP (all unlabelled as negatives) / evidence MAP@5.

| step | date | public | change | lesson |
|---|---|---|---|---|
| public notebooks | Sep 7–16 | 0.55–0.69 | chip-flow features, seat/graph features | plateau at 0.69 regardless of model; chip flow is a consequence, not the signal |
| Step 1 | Sep 15 | 0.671 | evidence-first PU baseline | evidence MAP 0.39 |
| Step 2 | — | not submitted | exact 7-card strength, fold-the-better-hand, squeeze flags | superseded by Step 3 |
| Step 3 | Sep 16 | 0.801 | **card-conditional pre-flop policy surprise**, two-step PU on all eligible pairs, `none` rule | the planted signal is both partners entering with junk |
| Step 4 | Sep 16 | 0.820 | evidence stage-2 pool re-ranker, planted-hand counts, third PU round | evidence 0.51 → 0.58 |
| Step 5 | Sep 16 | 0.824 | other-phase pair baselines | small gain |
| Step 6 | Sep 16 | 0.842 | **exposure matching** (2,000-hand training windows), phase-relative pool index | +0.02; scale of count features matters |
| Step 7 | Sep 17 | 0.822 | field baselines pooled over both phases, 3 windows | regression: phase-specific baselines are what isolate phase-specific collusion |
| Step 8 | Sep 17 | **0.894** | **action-level behaviour-clone policy surprise**; ambiguous PU band kept as weak negatives | +0.052; only ~150 hidden colluders in 120k unlabelled pairs |
| probe | Sep 17 | 0.775 | Step 8 with evidence blanked | LB evidence = 0.593, pair AP ≈ 0.97 |
| Step 9 | Sep 18 | 0.892 | coincidence z-scores, family/chronology in stage 2 | neutral on pairs, evidence +0.006 |
| Step 10 | — | not submitted | transductive PU on evaluation pairs | dev views down; not pursued |
| Step 11–13 | Sep 18 | 0.873 (Step 12) | multiple-instance hand model from pair labels + template target encodings | big dev gains, −0.02 LB: MIL relied on within-pair percentile features whose scale depends on phase length |
| Step 14–15 | Sep 18 | 0.872 / 0.854 | leave-table-out eval scoring; within-phase percentiles | did not fix it |
| Step 16 | Sep 19 | 0.896 | Step 8 pair model + template/family evidence re-ranker | evidence transfers (+0.002) |
| Step 17 | Sep 19 | 0.894 | prank-free MIL v3 | dev up, LB flat → MIL abandoned |
| blend89_ev16 | Sep 19 | 0.897 | rank-average Step 8 + Step 9 risk | ensembling works |
| blend3 | Sep 19 | 0.897 | + 10-seed Step 8 bag | |
| blend4 | Sep 19 | 0.894 | + family-specific rankers | hurt on LB (dev hidden-150 view misled) |
| hypothesis sweep | Sep 19 night | — | 8 evidence-label hypotheses in parallel (equity transfer, per-player policy, bet-size grid, provability, stack fractions, made-hand combos, joint p-values, street timing) | three verified on two fold permutations × three seeds |
| Step 18 | Sep 20 | (in E–J) | evidence stage 1 + made-hand combos, street timing, per-player policy | dev evidence 0.613 → 0.621, LB +0.002 |
| final_E | Sep 20 | 0.899 | blend3 risk + Step 18 evidence | |
| final_F | Sep 20 | 0.900 | 8cv + 9cv + 8bag + 9bag + Step 18 evidence | **selected** |
| final_G | Sep 20 | 0.900 | Step-8-heavy weights | neutral |
| final_H | Sep 20 | 0.900 | 10-seed bags only | full-data bags > CV models |
| final_J | Sep 20 | **0.900** | 20-seed bags only | **selected**; bagging saturated |

## Dead ends (verified, do not repeat)

**Evidence label rule.** Chronological first-k truncation; temporal clustering / onset / bursts / seating sessions; seat position and relative position; stack sizes and stack-fraction transfers; hand duration and timestamps; a common victim outsider; bet-size distributions and grids; action-mix fingerprints; success-of-mechanism first-k; equity (EV) transfer; joint action p-values per street; outsider strength (+0.008 only); template target encoding (+0.02, used). The labelled evidence hands are statistically interchangeable with a pair's other collusion-mode hands on every observable we tried; the leader's ≥ 0.70 evidence MAP implies a rule we did not find.

**Pair ranking.** Player-degree / partner-exclusivity features (a development-only artefact: unlabelled dev pairs exclude labelled colluders); family-specific rankers; exposure-weighted blends; ExtraTrees / quantile-logistic / DART and their blends; excluding or pseudo-labelling the top unlabelled pairs; dropping low-exposure training rows; symmetric p1/p2 augmentation; player metadata; the three MIL variants.

## Tooling notes
* Local runs: Apple M-series 10 cores, 26 GB RAM; full pipeline ≈ 90 min (policy model +30 min), all stages cached in `work_step3/`.
* Development-set metrics for stacked hand models are unreliable; the public leaderboard (5 submissions/day, shared across the team) was the only decisive validation, used one change at a time.
