
# >>> DATA-VERIFIED FACTS (added 2026-09-14 02:20 after the competition data was downloaded locally; these SUPERSEDE any estimates below) <<<

Data is now at /Users/md.hamidhosen/Documents/Detect Suspicious Value Transfers in Poker/ (actions.parquet, hands.parquet, seats.parquet, players.parquet, development_labels.csv, development_evidence.csv, evaluation_pairs.csv, sample_submission.csv). Computed with pandas/pyarrow (polars/duckdb NOT installed locally).

## CSV schemas (exact)
- development_labels.csv: 1,860 rows; columns pair_id, player_1, player_2, label (0/1), label_status {confirmed_target: 372, confirmed_non_target: 1488}, behavior_family {none: 1488, directed_transfer: 148, soft_play: 132, coordinated_isolation: 92}. IDs look like P00082F54BA9A / UB2461B374353 (opaque hex).
- development_evidence.csv: 1,817 rows; columns pair_id, evidence_rank (1..5), hand_id (e.g. H2D8EAC9EC7DA02), behavior_family. 372 positive pairs: 340 have 5 evidence hands, 21 have 4, 11 have 3. Evidence hands by family: directed_transfer 725, soft_play 632, coordinated_isolation 460.
- evaluation_pairs.csv: 112,540 rows; columns pair_id, player_1, player_2, shared_hands (int = number of EVALUATION-period hands in which both players were dealt; verified to equal a recomputation from seats.parquet exactly).
- sample_submission.csv: 112,540 rows; risk_score = 0.0, predicted_behavior = none, all evidence slots = NO_EVIDENCE.

## Pool / hand structure (exact)
- 400 table_ids, EXACTLY 5,000 hands each; phase split EXACTLY 3,000 development + 2,000 evaluation per table (1,200,000 / 800,000 overall).
- Every player belongs to exactly one table (12,000 players, 30 per table). Blind level is constant per table: big_blind 2 (218 tables), 4 (143), 10 (39).
- started_at spans 2026-01-01 → 2026-02-02 (simulated). players_dealt always 6.
- Showdown: 84.4% of hands end without showdown; 2-way showdown 13.9%; 3-way 1.5%. 46.7% of hands end preflop (empty board).
- final_pot in BB: median 9.5, mean 32.8, 75th pct 24, max 846.
- Hands per player: development mean 600 (min 126, max 1,243); evaluation mean 400 (min 44, max 906).

## Pair exposure & the evaluation-pair selection rule (exact, reverse-engineered)
- Eval-period pairs with >=1 shared hand: 172,878 (of 174,000 possible). Included in evaluation_pairs.csv: exactly those with shared_hands >= 38 (= 1.9% of the 2,000 eval hands) AND not a publicly labelled pair AND containing no publicly labelled positive player. Excluded: 19,127 pairs containing a positive player; 1,291 labelled non-target pairs; 39,891 pairs with < 38 shared hands; 29 pairs with >=38 shared hands excluded for an unidentified reason (negligible).
- evaluation shared_hands distribution: min 38, 5th pct 41, 25th 56, median 76, mean 85.8, 75th 106, 90th 140, 99th 212, max 419. NO evaluation pair has fewer than 5 shared hands → NO_EVIDENCE is never necessary.
- Development labelled pairs have dev-period shared hands >= 57 (= 1.9% of 3,000) for BOTH positives (min 57, mean 121.3, median 112, max 303) and confirmed negatives (min 57, mean 120.0, median 108.5, max 465) → exposure is by design NOT discriminative between positives and confirmed negatives.
- Labelled pairs exist on 397 of 400 tables (mean 4.7 labelled pairs/table); positives on 245 tables (155 tables with 1 positive, 63 with 2, 18 with 3, 8 with 4, 1 with 5). Eval pairs per table: mean 281 (min 168, max 344).
- Positive players: 693 distinct; NONE appear in any evaluation pair. Confirmed-negative-pair players: 2,599 distinct, 2,433 of them appear in evaluation pairs. 11,302 distinct players appear in evaluation pairs.

## Evidence-hand anatomy (exact)
- 100% of the 1,817 evidence hands are development-phase hands and 100% contain both players of the pair.
- 69.3% of evidence hands end WITHOUT showdown (1,259/1,817); 25.3% 2-way showdown; 5% 3-way; 0.4% 4-way.
- Evidence final_pot in BB — directed_transfer: median 34.5, mean 79, 75th 128.5; soft_play: median 17, mean 54.6; coordinated_isolation: median 17.8, mean 53.8. (Overall hand median is 9.5 BB → evidence hands are larger-than-typical pots, especially directed_transfer.)

## Public notebooks available locally (8 .ipynb in the working directory; all are public Kaggle notebooks for this competition)
Printed out-of-fold numbers (from their own outputs):
- 'suspicious-detection.ipynb' (V3-H2, 162 KB code, xgboost+duckdb+polars): strict OOF labelled Pair AP 0.9737; PU-weighted surrogate AP 0.568; routed Evidence MAP@5 0.349 (0.391 with H2 blend); Behavior MAP 0.912; exact composite on labelled pairs 0.8426.
- 'topological-collusion-dynamics-value-transfer-eda.ipynb' / 'base notebook.ipynb' (triple GBDT XGB+LGBM+CatBoost, empirical-Bayes shrinkage, dual-engine evidence): labelled Pair AP 0.965–0.977; PU-stress Pair AP 0.65–0.68; per-family OOF class AP: directed_transfer 0.60, soft_play 0.72, coordinated_isolation 0.22–0.44; Evidence MAP@5 0.45 (global) → 0.50–0.51 (family-specialised blend); projected composite 0.61–0.63.
- 'poker-e23-clean-label-pn-submission-candidate.ipynb' (E23 clean-label P/N ensemble): labelled AP 0.974, PU-stress 0.706, Evidence MAP@5 0.356 (graded NDCG ranker); projected 0.632.
- 'detecting-collusive-value-transfer-in-6-max-poker.ipynb' (xgboost; title: "why 0.97 OOF is a trap"): labelled AP 0.9725, PU-stress 0.689 (family-max 0.73); per-family detector APs: directed_transfer 0.83–0.92, soft_play 0.86–0.99, coordinated_isolation 0.76–0.88; states "To move 0.645 → 0.80 you need Pair AP ~0.85 and MAP@5 ~0.65".
- 'poker-collusion-pu-aware-evidence-ranker.ipynb' (xgboost, duckdb): labelled AP 0.938, PU-stress 0.517, Evidence MAP@5 0.44.
- 'coordinated-collusion-value-transfer-detection.ipynb' (lightgbm, graph features): OOF Pair AP 0.805, Evidence MAP@5 0.091 (weak), composite 0.68.
- 'slash-poker-competition-metric.ipynb': the host metric (identical to the code below).
Terminology used across notebooks: "labelled AP" = AP restricted to confirmed positives vs confirmed negatives; "PU stress AP" = positives vs ALL unlabelled pairs of held-out pools treated as negative (closer to the real evaluation). Public LB leader is 0.938 — far above every public notebook's projected composite, so the leader has something qualitatively better (most likely far stronger Pair AP under the PU/eval distribution and/or a much stronger evidence ranker).

# Detect Suspicious Value Transfers in Poker — verified public facts (scraped 2026-09-13/14)

Source URLs:
- https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker (overview/data/rules/leaderboard/discussion)
- Metric notebook: https://www.kaggle.com/code/florianderoofr/slash-poker-competition-metric
- Public baseline repo (README only, no code pushed): https://github.com/phucthaiv02/poker-suspicious-detection

## Meta
- Host/Sponsor: Slash (Slash Financial, Inc., San Francisco). Community Prediction Competition. Kaggle competition id 163730.
- Tagline: "Build machine learning models to identify coordinated gameplay patterns"
- Prize pool $5,000: 1st $3,000; 2nd $1,250; 3rd $750. No Kaggle points/medals.
- Timeline: started ~8 days before 2026-09-14 (≈ 2026-09-05/06); closes "7 days to go" (≈ 2026-09-21). Exact deadline on Overview>Timeline (login-gated).
- Participation snapshot: 374 entrants, 216 participants, 210–211 teams, 2,152 submissions.
- Rules: max team size 10; team mergers allowed (merger submission-count constraint); 5 submissions/day; select up to 2 final submissions; winner license Open Source (OSI-approved, no commercial restriction); data "Competition Use Only"; external public data & pretrained models permitted (host confirmed in discussion); AutoML tools allowed.
- Winner verification (within 7 days of private LB release, ANY prize-eligible team): (1) Kaggle Solution Writeup ≤1,500 words; (2) public notebook/repo reproducing the selected submission; (3) five short case reviews from submitted evidence: pair ID, hand IDs, observable behavior, and a plausible benign alternative. Organizer may re-run code in clean environment.
- PROHIBITED: "Solutions must infer coordination from poker activity. Exploiting ID formats, row/file ordering, generator internals, or any artifact unrelated to gameplay is prohibited."
- Public LB ≈30% of evaluation pairs, private 70%, split stratified by behavior family.
- Tags: Retrieval/Ranking, Tabular, Classification, Custom Metric.

## Task description (verbatim-ish from overview)
"Detect coordinated player relationships in 2,000,000 synthetic poker hands. Rank every evaluation pair, identify the likely behavior, and submit the hands that best support each alert."

The challenge: Suspicious-looking play has many innocent explanations: tilt, inexperience, repeated schedules, unusual strategy, or simple luck. Coordinated players may also act only occasionally and mix manipulated hands with normal play. Build a system that can separate these cases. Your submission must detect suspicious pairs and point to concrete hands an investigator could review.

Target behaviors:
- Directed transfer: one player deliberately loses value to another.
- Soft play: partners avoid normal aggression against each other.
- Coordinated isolation: partners pressure other players while limiting conflict between themselves.
- Other coordination: a fourth, undisclosed mechanism that never appears in the public positive labels. Submit it as other_coordination when you detect coordination that fits none of the three disclosed patterns.

Coordination is episodic: a coordinated pair mixes manipulated hands with ordinary play, and a relationship is not necessarily active over the whole timeline.
The dataset also includes difficult non-target patterns such as tilt, weak play, similar strategies, repeated opponent selection, streaks, and strategy changes. Players differ in when and how much they play, so some pairs share far more hands than others.

## Data
- 2,000,000 six-player NLHE hands; 12,000 synthetic players; 12,000,000 player-hand rows (seats); 18,609,028 ordered actions; 112,540 evaluation pairs.
- Players organised in 400 persistent pools of 30. Each pool plays at one table (table_id) for about 5,000 hands. Within each pool, first 60% of hands = development period, final 40% = evaluation period (hands.phase).
- development_labels.csv (132.92 kB): trusted positives and confirmed non-target pairs from the development period. Does NOT list every coordinated pair: every unlisted development pair is UNKNOWN, not negative. (PU setting.)
- development_evidence.csv (87.68 kB): up to five planted evidence hands for each public positive pair. Every listed evidence hand contains a behavior-specific action visible in the public action log; latent scenario activation alone is never evidence.
- evaluation_pairs.csv (5.1 MB): evaluation-period pairs to be scored. Publicly labelled pair IDs and pairs containing a publicly labelled positive player are EXCLUDED. Evaluation labels & evidence private.
- sample_submission.csv (9.34 MB): exact template.
- Total 308.45 MB, 8 files, "22 columns" summary. Column names of the CSVs are login-gated (not scraped); the parquet schemas are below.
- Fully synthetic; hole cards are provided ("operator-side investigation setting").

### players.parquet (12,000 rows, PK player_id)
player_id string; account_age_days int; experience_hands_bucket {new, developing, experienced, veteran}; preferred_stake {micro, low, mid}; region_bucket {americas, apac, europe, other}; client_family {desktop, mobile, web}.

### hands.parquet (2,000,000 rows, PK hand_id)
hand_id string; table_id string (persistent pool id; each player belongs to one pool); started_at UTC timestamp (simulated); phase {development, evaluation}; button_seat int 0–5; small_blind int; big_blind int (levels 1/2, 2/4, 5/10, play-money chips); board_cards string (space-separated; empty if ended preflop; 3–5 cards); final_pot int; players_dealt int (always 6); players_at_showdown int (0 if no showdown). Cards like "As", "Td"; suits c d h s.

### seats.parquet (12,000,000 rows, PK (hand_id, player_id))
hand_id; player_id; seat_no int 0–5; starting_stack int; hole_card_1; hole_card_2; total_contribution int (incl. blinds); net_chips int (awarded − contribution; six values per hand sum to zero); folded bool; went_to_showdown bool; won_share float 0–1.

### actions.parquet (18,609,028 rows, PK (hand_id, action_no), 150.82 MB)
hand_id; action_no int (0-based chronological within hand); street {preflop, flop, turn, river}; player_id; action {fold, check, call, bet, raise, all_in}; amount int (additional chips committed; 0 for check/fold); amount_to int (player's total committed on current street after action); pot_before int; stack_before int; to_call int (0 when checking available); players_active int (non-folded before action). Note: all_in may be a call or aggressive — compare amount with to_call.

## Submission format (one row per pair_id in evaluation_pairs.csv)
pair_id (copied unchanged); risk_score in [0,1]; predicted_behavior ∈ {none, directed_transfer, soft_play, coordinated_isolation, other_coordination}; evidence_hand_1..evidence_hand_5 = up to five evaluation-period hands ranked strongest→weakest, NO_EVIDENCE for unused positions. No empty cells. Do not repeat a hand within a row (invalid submission). Each submitted hand should contain both players. Unknown, non-shared, development-period, or irrelevant hand IDs receive no credit (no penalty stated other than zero credit). Equal risk scores resolved deterministically by pair_id.

## Evaluation
Score = 0.70 × PairAP + 0.20 × EvidenceMAP@5 + 0.10 × BehaviorMAP
- Pair AP: Average Precision over all evaluation pairs using risk_score.
- Evidence MAP@5: mean AP of ranked hand IDs over TRUE target pairs, vs private planted-evidence hands. A missed target contributes zero.
- Behavior MAP: macro one-vs-rest AP, always averaged across the three disclosed families. For each family the class score = risk_score when that family is predicted, else 0. Absent class contributes zero. other_coordination excluded from this component.
- AP = Σ_n (R_n − R_{n−1}) P_n.

### Reference metric code (verbatim from host notebook)
```python
import numpy as np
import pandas as pd

class ParticipantVisibleError(Exception):
    """An error message that Kaggle may safely show to participants."""

ALLOWED_BEHAVIORS = {"none","directed_transfer","soft_play","coordinated_isolation","other_coordination"}
TARGET_BEHAVIORS = ("directed_transfer","soft_play","coordinated_isolation")
EVIDENCE_COLUMNS = tuple(f"evidence_hand_{rank}" for rank in range(1, 6))
NO_EVIDENCE = "NO_EVIDENCE"
REQUIRED_COLUMNS = {"pair_id","risk_score","predicted_behavior",*EVIDENCE_COLUMNS}

def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ranked = y_true[order]
    true_positives = np.cumsum(ranked)
    ranks = np.arange(1, len(ranked) + 1)
    return float(np.sum((true_positives / ranks) * ranked) / positives)

def _clean_evidence(values):
    cleaned = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text != NO_EVIDENCE:
            cleaned.append(text)
    return cleaned

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    if row_id_column_name != "pair_id":
        raise ParticipantVisibleError("The row ID column must be pair_id.")
    if not REQUIRED_COLUMNS.issubset(solution.columns):
        raise ValueError("The private solution has an invalid schema.")
    if not REQUIRED_COLUMNS.issubset(submission.columns):
        missing = sorted(REQUIRED_COLUMNS - set(submission.columns))
        raise ParticipantVisibleError(f"submission.csv is missing columns: {missing}")
    if solution["pair_id"].duplicated().any():
        raise ValueError("The private solution contains duplicate pair IDs.")
    if submission["pair_id"].duplicated().any():
        raise ParticipantVisibleError("pair_id values must be unique.")
    solution_ids = set(solution["pair_id"].astype(str))
    submission_ids = set(submission["pair_id"].astype(str))
    if solution_ids != submission_ids:
        missing = len(solution_ids - submission_ids); extra = len(submission_ids - solution_ids)
        raise ParticipantVisibleError(f"pair_id coverage mismatch: {missing} missing and {extra} extra.")
    truth = solution.set_index("pair_id").sort_index()
    predictions = submission.set_index("pair_id").loc[truth.index]
    risk = pd.to_numeric(predictions["risk_score"], errors="coerce")
    if risk.isna().any() or not risk.between(0, 1).all():
        raise ParticipantVisibleError("risk_score must be numeric and between 0 and 1.")
    predicted_behavior = predictions["predicted_behavior"].astype(str)
    invalid = set(predicted_behavior) - ALLOWED_BEHAVIORS
    if invalid:
        raise ParticipantVisibleError(f"Invalid predicted_behavior values: {sorted(invalid)}")
    for row in predictions.loc[:, EVIDENCE_COLUMNS].itertuples(index=False, name=None):
        evidence = _clean_evidence(list(row))
        if len(evidence) != len(set(evidence)):
            raise ParticipantVisibleError("Evidence hand IDs must not repeat within a pair.")
    y_true = pd.to_numeric(truth["risk_score"], errors="raise").to_numpy(dtype=int)
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Private risk_score values must be binary labels.")
    risk_values = risk.to_numpy(dtype=float)
    pair_ap = _average_precision(y_true, risk_values)
    true_behavior = truth["predicted_behavior"].astype(str).to_numpy()
    predicted_behavior_values = predicted_behavior.to_numpy()
    behavior_scores = []
    for behavior in TARGET_BEHAVIORS:
        behavior_truth = (true_behavior == behavior).astype(int)
        if behavior_truth.sum() == 0:
            behavior_scores.append(0.0); continue
        behavior_risk = np.where(predicted_behavior_values == behavior, risk_values, 0.0)
        behavior_scores.append(_average_precision(behavior_truth, behavior_risk))
    behavior_map = float(np.mean(behavior_scores))
    evidence_scores = []
    for position in np.flatnonzero(y_true == 1):
        relevant = set(_clean_evidence(truth.iloc[position].loc[list(EVIDENCE_COLUMNS)].tolist()))
        submitted = _clean_evidence(predictions.iloc[position].loc[list(EVIDENCE_COLUMNS)].tolist())
        if not relevant:
            evidence_scores.append(0.0); continue
        hits = 0; precision_sum = 0.0
        for rank, hand_id in enumerate(submitted[:5], start=1):
            if hand_id in relevant:
                hits += 1; precision_sum += hits / rank
        evidence_scores.append(precision_sum / min(len(relevant), 5))
    evidence_map = float(np.mean(evidence_scores)) if evidence_scores else 0.0
    final_score = 0.70 * pair_ap + 0.20 * evidence_map + 0.10 * behavior_map
    if not np.isfinite(final_score):
        raise ParticipantVisibleError("The metric produced a non-finite score.")
    return float(final_score)
```
Notebook ran in 27s, 1 input file (the competition data), 0 output files.

## Public leaderboard snapshot (2026-09-13, ~30% of eval pairs)
1 Pardheev Krishna 0.93798 (34 entries); 2 Leo 0.91572; 3 bluey 0.91127; 4 test_spurious 0.90757; 5 seantangth 0.90405; 6 Marc Donovici 0.90352; 7 LuPW 0.90291; 8 yoiisohira 0.90165; 9 Kushal Khemani 0.89897; 10 Nicolas Krusche 0.89885; 11 0.89612; 12 0.89466; 13 0.89300; 14 0.89050; 15 0.88993; 16 0.88958; 17 0.88810; 18 0.88665; 19 0.88527; 20 0.88385; 21 0.88351; 22 0.88225; 23 0.87995; 24 0.87929; 25 0.87867; 26 0.87702; 27 0.87128 (3 entries); 28 0.86948; 29 0.86900; 30 0.86875; 31 0.86835; 32 0.86788; 33 0.86646; 34 0.86396; 35 0.86252; 36 0.85991; 37 0.85745; 38 0.85469; 39 0.85420; 40 0.84636; 41 0.84261; 42 0.84205; 43 0.84197; 44 0.84195; 45 0.83805; 46 Mario Nicolae 0.83792; 47 0.83690; 48 WU YANZH 0.83656 (1 entry); 49 XHMUVW 0.83656 (16 entries); 50–211 not shown. Total 211 teams.

## Discussion (only 2 threads exist)
1. "Clarifications" (Mario Nicolae, 7d ago): Q: external data/pretrained models permitted? metric code timeline? Host Florian DEROO: "Yes, publicly available external data and pretrained models are permitted. The reference implementation is now available here [metric notebook]."
2. "CV-LB difference" (Harish Dhanarajan, 79th, 12h ago, 2 upvotes, 0 comments): "How much of a difference is everyone facing between CV and LB. I'm having better CV score but performing poorly on LB."

## Public Code tab
No public participant notebooks (only host's metric notebook).

## Public baseline repo README (phucthaiv02/poker-suspicious-detection) — approach summary
Treats task as PU pair ranking + multiple-instance evidence retrieval. Pipeline: schema inspection → pair-hand construction (shared hands only for dev-labelled and eval pairs) → action/player-hand aggregation → evidence model (positives = planted evidence hands; negatives = shared hands of confirmed non-target pairs; unlisted pairs never treated as negatives; non-evidence hands of positive pairs not forced negative) → OOF stacking of evidence scores into pair model (GroupKFold by table_id/pair_id) → pair aggregation (top-k evidence stats, exposure features, behavioural aggregates) → pair risk model (LightGBM) → 3 one-vs-rest behavior heads with none_threshold and known_confidence thresholds (open-set reject → other_coordination) → evidence retrieval = top-5 shared eval hands by evidence probability; NO_EVIDENCE only if <5 shared hands. Raw IDs not used as features. Suggested upgrades: pair-vs-player-history residual features; temporal burst/change-point features in eval period; behavior-specific evidence heads; empirical exposure correction using confirmed negatives; action-sequence motifs for coordinated isolation; PU bagging / calibrated pseudo-negatives; LightGBM+CatBoost rank ensemble.

## Local environment facts
- Working dir is EMPTY (no data downloaded). No kaggle CLI, no ~/.kaggle credentials. Python 3.12 with pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9, lightgbm 4.7, xgboost 3.4, catboost 1.2.10. macOS.
