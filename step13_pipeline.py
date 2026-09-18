# %% [markdown]
# # Step 13 — MIL v2 for *Detect Suspicious Value Transfers in Poker*
#
# Step 12 is the base. **New in Step 13 — the multiple-instance hand model, second version.** Negative bags are chosen with a quick pair-level OOF pass
# instead of at random: 12,000 unlabelled pairs that score clean (OOF < 0.001) plus 3,000 *hard* unlabelled pairs (0.001 ≤ OOF < 0.3, the loose-bot
# look-alikes below the pseudo-positive band); every labelled pair contributes **two bags** (both 2,000-hand windows), so the positive side doubles to
# ≈ 60k hands. Ablation on the Step 9 pair features (v1 → v2): pessimistic AP 0.7213 → 0.7315, labelled 0.9988 → 0.9992, hidden-150 0.9245 → 0.9268;
# stacking v1 and v2 together is worse than v2 alone, so v1 is dropped.
#
# ---
#
# # Step 12 — Action-sequence templates in the evidence re-ranker
#
# Step 11 (MIL hand model) is the base. **New in Step 12:** the stage-2 evidence re-ranker also sees the pair's ordered **action-sequence template**
# of the hand — who (winner / loser of the hand) did what, on which street, facing whom (partner / outsider / nobody) — as smoothed, fold-aware target
# encodings (full template, pre-flop part, first three tokens, each also within the family) plus the template length. The labelled evidence hands of
# `directed_transfer` and `soft_play` concentrate on a few templates ("W raises, L calls, W bets the flop, L folds"; the check-down), while the pair's
# other collusion-mode hands spread over hundreds — a sequence identity that aggregate counts cannot express. Dev pool ablation (two seed sets):
# stage-2 MAP@5 0.584 → 0.605 and 0.587 → 0.603.
#
# ---
#
# # Step 11 — Multiple-instance hand model
#
# **Step 8 scored 0.89378** (Step 9 = z-scores + family-aware evidence is the base here; Step 10's transductive round is not included).
#
# **New in Step 11 — a hand model that has seen the negatives.** Every hand score the pair model used so far came from the *evidence ranker*, which is
# trained only inside the coordinated pairs (planted hand vs. the pair's other hands). It has never seen a hand of a loose-bot pair — yet the leaderboard
# loss sits exactly at that boundary. The MIL model learns "hand of a coordinated pair" vs. "hand of an ordinary pair" from the **pair labels**
# (multiple-instance learning): all hands of labelled pairs plus 6,000 random unlabelled pairs; positive bags start as weak positives, then the top-8 hands
# of each positive pair (by the current out-of-fold score) become the positives and the rest weak negatives, three rounds, table-grouped folds.
# Pair features: top-3 / top-8 mean, mean, sum, max and counts above 0.5 / 0.8 of the MIL score over the window's shared hands.
# Ablation on the Step 9 pair features: labelled AP 0.9878 → 0.9990, pessimistic AP 0.6906 → 0.7190.
#
# ---
#
# # Step 9 — Coordination z-scores + evidence chronology
#
# **Step 8 scored 0.89378** (Step 6: 0.84157). Implied leaderboard pair AP ≈ 0.975, evidence MAP@5 ≈ 0.58.
#
# **New in Step 9.**
#
# 1. **Coordination z-scores.** Two loose bots also produce "both surprising in the same hand" events, at rate ≈ n·p₁·p₂ if independent. For every pair the
#    observed count of joint events (both surprising, both junk entries, both entered) is compared with that independence expectation from each player's own
#    phase-level rate: `excess = obs − exp`, `z = (obs − exp)/√(exp+0.5)`, `ratio`. The raw counts were already in the model; the normalised coincidence test is what
#    the low-exposure tail (≈ 50 shared hands, 1–2 visible planted hands) needs.
# 2. **Evidence stage 2 knows the family and the chronology of strong hands.** Colluding pairs carry ≈ 10 planted hands but only ≤ 5 are labelled; the labelled
#    ones skew early (median relative position 0.32 vs 0.56 for the unlabelled planted-looking hands) and are the hands where the mechanism completed (fold to the
#    partner, dump of the better hand). The pool re-ranker gets the pair's family (true family in training, predicted family — 99 % accurate — at inference) and
#    "how many strong hands came before this one".
# 3. OOF pair scores and evaluation risk are saved for blending.
#
# ---
#
# # Step 8 — Action-level policy surprise + hard-negative PU
#
# **Step 6 scored 0.84157** (Step 7, which pooled the field baselines over both phases and used 3 windows, regressed to 0.82150 and is discarded).
#
# **New in Step 8.**
#
# 1. **Population action policy (behaviour clone).** A 6-class LightGBM model of every action in `actions.parquet` (18.6 M decisions) given the private
#    information the bot actually has: hole cards, exact made-hand category at the street, board texture, position, to-call / pot / stack, raises faced,
#    own prior actions, and the player's *other-phase* style (VPIP, PFR, aggression, fold-to-bet, showdown rate — clean of any in-phase collusion).
#    Per action `surprise = -log P(taken action)`; per player-hand and per pair-hand sums / maxima, and — the strongest new feature — the surprise of
#    actions taken **while facing the partner's bet** (`surp_vs_partner`: folding a better hand, flat-calling instead of raising, checking down).
#    Pair level: sums, top-3, partner-vs-field contrasts and other-phase contrasts of the same quantities.
# 2. **Hard negatives stay in training.** A mixture estimate on the stage-1 scores says the unlabelled development pairs hide only ≈ 150 colluders
#    (prevalence ≈ 0.4 %), not thousands. The ≈ 2,600 "ambiguous" unlabelled pairs that Step 3–7 dropped from training are therefore mostly loose-bot
#    pairs — exactly the negatives the evaluation ranking has to beat. They are kept as weak negatives (ablation on the Step 6 features: labelled AP
#    0.9856 → 0.9871, AP with the 150 most colluder-like unlabelled pairs counted as positives 0.931 → 0.942, pessimistic AP unchanged).
#
# ---
#
# # Step 6 — Exposure-matched training
#
# **Step 5 scored 0.82369** (Step 4: 0.81982, Step 3: 0.80136).
#
# **New in Step 6 — two train/test mismatches removed.**
#
# 1. The evaluation phase has 2,000 hands per table, development 3,000: evaluation pairs share a median 76 hands, development pairs 110. Every count, sum and
#    top-k pair feature is therefore ~30 % smaller on evaluation than anything the pair model saw in training. The pair model is now trained on **2,000-hand windows**
#    of the development phase (`[0, 2000)` and `[1000, 3000)`, two rows per pair as augmentation), so training exposure matches evaluation exposure.
# 2. The stage-2 evidence feature `idx_rel` was `hand_idx / 3000`: 0–1 in training but 1.0–1.67 on evaluation hands (indexed 3000–4999). It is now phase-relative.
#
# ---
#
# # Step 5 — Other-phase pair baselines
#
# **Step 4 scored 0.81982 on the public leaderboard** (Step 3: 0.80136, Step 1: 0.671). Implied leaderboard pair AP is still ≈ 0.90.
#
# **New in Step 5 — the pair's own "not colluding" baseline.** The same 30 players play both phases of a table, and collusion is phase-specific: labelled development
# colluders behave normally in evaluation (both-entered rate 0.087 vs 0.193 while colluding), and the predicted evaluation colluders behave normally in development
# (0.102 vs 0.219). So for every pair the *other* phase gives a clean estimate of how these two players naturally play together (two loose bots also enter pots
# together often — that is the main confound). Features: the other-phase rates (`ob_*`), shrunk contrasts `in-phase − other-phase`, and the other-phase sample size.
# On the full development population the single-feature AUC of the policy-surprise rate rises from 0.84 to 0.95 with this contrast.
#
# ---
#
# # Step 4 — Two-stage evidence re-ranking + planted-hand counts
#
# **Step 3 scored 0.80136 on the public leaderboard** (Step 1: 0.671). Implied components: pair AP ≈ 0.90, evidence MAP@5 ≈ 0.51, behaviour MAP ≈ 0.73.
#
# **New in Step 4:**
#
# * **Evidence stage 2.** A colluding pair has ~12 planted-looking hands but only 5 are labelled. The stage-1 ranker already puts 97 % of labelled hands in its top-20;
#   a second ranker on that top-20 pool adds the hand's chronological position among the pool (labelled hands skew early), its score relative to the pool maximum and
#   the spacing to neighbouring pool hands. OOF MAP@5 0.50 → 0.58.
# * **Planted-hand counts as pair features** (`ev_n10`, `ev_n50`, `ev_f10`): number of shared hands whose stage-1 evidence score exceeds the 10th / 50th percentile of
#   labelled evidence scores. Counts are more robust than rates for the shorter evaluation period (median 76 shared hands vs 112 in development).
# * **Third PU round** with a wider pseudo-positive net and five seeds for the final ranker.
#
# ---
# (Step 3 header follows.)
#
# # Step 3 — Card-conditional policy deviation
#
# **Why this step exists.** Reading the labelled evidence hands shows the planted behaviour is not chip flow: the two partners enter pots *together* with junk hole cards
# (8♦3♠ raises, 6♦3♣ calls, 7♣2♥ calls …). The simulator's normal policy is strictly card-driven (VPIP 2 % with Chen −1.5 → 96 % with Chen ≥ 12), so a junk entry is a large
# policy deviation, and two simultaneous deviations by the same two players is the signal. All six players' hole cards are visible, so the deviation is measurable for every hand.
#
# **New in Step 3:**
#
# 0. **Policy-surprise features.** For every player-hand: action-based `vpip_a` / `pfr`, position, whether a raise was faced, hole-card class (169 classes), and the population
#    policy `P(vpip | class, position, faced raise)` estimated on all 12 M seats. `loose = −log P(enter)` when the player entered, `tight = −log P(fold)` when a strong hand folded,
#    `junk_vpip = vpip·max(0, 8 − Chen)`. Pair-hand features: both entered, sum/min/max surprise, junk entries, *second entrant with junk after the partner was already in*.
#    Pair features: rates, top-3, sums, and partner-vs-field contrasts of the same quantities.
# 0b. **Behaviour `none` rule** selected on the host metric on the full development population (predict a family only for the top-ρ risk pairs).
#
# **Inherited from Step 2 (never submitted):**
#
# 1. **Exact 7-card hand strength** (numba evaluator) for every player at flop / turn / river, ranked among the six dealt players → features such as *"the loser folded the better hand to the partner"*, *"the winner did not have the best hand"*, *"both made hands, nobody bet"*.
# 2. **Street-level partner features** for `coordinated_isolation` (pre-flop re-raise of the partner while outsiders are in, outsider folds pre-flop, fold to the partner after the squeeze).
# 3. **Partner-vs-field contrasts** with shrinkage (how a player plays when the partner is seated minus when they are not).
# 4. **Honest CV**: pair features are built for **all** eligible development pairs (≥ 57 shared hands, no publicly-positive player, ≈ 137k pairs), so the pair model is trained and evaluated on the same population as the leaderboard (positives vs everything else, no re-weighting).
#
# Everything else (action context, PU-weighted training, two hand models, family heads, submission validation) is inherited from Step 1.
#
# Metric facts used (verified against the reference metric code): evidence is scored only for true pairs and never penalised for negatives → always fill 5 slots; `none` and `other_coordination` are identical for the behaviour component → always predict a disclosed family; ties in `risk_score` are broken by `pair_id` → avoid ties.

# %%
import os, gc, time, json, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from numba import njit
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
pl.Config.set_tbl_cols(40)

SEED = 42
N_FOLDS = 5
B5 = 15 ** 5
N_TABLE_CHUNKS = 4          # development pair-hands are built in table chunks to bound memory
SHRINK_N = 25.0             # shrinkage constant for partner-vs-field contrasts
PU_WEIGHT = 0.1             # training weight of unlabelled (assumed-negative) pairs (ablation on all eligible pairs: 0.1 > 0.35 > 1.0)
POS_WEIGHT = 5.0            # training weight of confirmed positives
NEG_WEIGHT = 1.0            # training weight of confirmed non-targets

CANDIDATES = [
    Path("/kaggle/input/competitions/detect-suspicious-value-transfers-in-poker"),
    Path("/kaggle/input/detect-suspicious-value-transfers-in-poker"),
    Path(os.environ.get("POKER_DATA_DIR", ".")),
]
DATA_DIR = next(p for p in CANDIDATES if (p / "hands.parquet").exists())
WORK_DIR = Path("/kaggle/working/step13") if Path("/kaggle/working").exists() else Path(os.environ.get("POKER_WORK_DIR", "./work_step6"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
SUB_PATH = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else WORK_DIR / "submission.csv"

T0 = time.time()
def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)

log(f"DATA_DIR={DATA_DIR}  WORK_DIR={WORK_DIR}")

# %% [markdown]
# ## Reference metric (verbatim from the host notebook)

# %%
ALLOWED_BEHAVIORS = {"none", "directed_transfer", "soft_play", "coordinated_isolation", "other_coordination"}
TARGET_BEHAVIORS = ("directed_transfer", "soft_play", "coordinated_isolation")
EVIDENCE_COLUMNS = tuple(f"evidence_hand_{rank}" for rank in range(1, 6))
NO_EVIDENCE = "NO_EVIDENCE"
REQUIRED_COLUMNS = {"pair_id", "risk_score", "predicted_behavior", *EVIDENCE_COLUMNS}


class ParticipantVisibleError(Exception):
    pass


def _average_precision(y_true, scores):
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


def host_score(solution, submission, row_id_column_name="pair_id", return_components=False):
    if row_id_column_name != "pair_id":
        raise ParticipantVisibleError("The row ID column must be pair_id.")
    if not REQUIRED_COLUMNS.issubset(submission.columns):
        raise ParticipantVisibleError(f"submission.csv is missing columns: {sorted(REQUIRED_COLUMNS - set(submission.columns))}")
    if submission["pair_id"].duplicated().any():
        raise ParticipantVisibleError("pair_id values must be unique.")
    if set(solution["pair_id"].astype(str)) != set(submission["pair_id"].astype(str)):
        raise ParticipantVisibleError("pair_id coverage mismatch.")
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
    risk_values = risk.to_numpy(dtype=float)
    pair_ap = _average_precision(y_true, risk_values)
    true_behavior = truth["predicted_behavior"].astype(str).to_numpy()
    pb = predicted_behavior.to_numpy()
    behavior_scores = []
    for behavior in TARGET_BEHAVIORS:
        bt = (true_behavior == behavior).astype(int)
        if bt.sum() == 0:
            behavior_scores.append(0.0)
            continue
        behavior_scores.append(_average_precision(bt, np.where(pb == behavior, risk_values, 0.0)))
    behavior_map = float(np.mean(behavior_scores))
    evidence_scores = []
    for position in np.flatnonzero(y_true == 1):
        relevant = set(_clean_evidence(truth.iloc[position].loc[list(EVIDENCE_COLUMNS)].tolist()))
        submitted = _clean_evidence(predictions.iloc[position].loc[list(EVIDENCE_COLUMNS)].tolist())
        if not relevant:
            evidence_scores.append(0.0)
            continue
        hits, precision_sum = 0, 0.0
        for rank, hand_id in enumerate(submitted[:5], start=1):
            if hand_id in relevant:
                hits += 1
                precision_sum += hits / rank
        evidence_scores.append(precision_sum / min(len(relevant), 5))
    evidence_map = float(np.mean(evidence_scores)) if evidence_scores else 0.0
    final = 0.70 * pair_ap + 0.20 * evidence_map + 0.10 * behavior_map
    if return_components:
        return {"final": final, "pair_ap": pair_ap, "evidence_map5": evidence_map, "behavior_map": behavior_map,
                "behavior_class_ap": dict(zip(TARGET_BEHAVIORS, behavior_scores))}
    return float(final)


def map5_within_pairs(df, score_col="hand_score", target_col="is_evidence", pair_col="pair_id"):
    """Exact per-pair AP@5 (same formula as the host) averaged over pairs that have >=1 relevant hand."""
    vals = []
    d = df.sort_values([pair_col, score_col, "pot_bb", "hand_id"], ascending=[True, False, False, True], kind="mergesort")
    for _, g in d.groupby(pair_col, sort=False):
        rel = g[target_col].to_numpy()
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        top = rel[:5]
        hits = np.cumsum(top)
        vals.append(float(np.sum((hits / np.arange(1, len(top) + 1)) * top) / min(n_rel, 5)))
    return float(np.mean(vals)) if vals else 0.0


log("metric helpers ready")

# %% [markdown]
# ## Load labels, evidence, evaluation pairs, hands

# %%
dev_labels = pl.read_csv(DATA_DIR / "development_labels.csv")
dev_evidence = pl.read_csv(DATA_DIR / "development_evidence.csv")
eval_pairs = pl.read_csv(DATA_DIR / "evaluation_pairs.csv")
sample_sub = pl.read_csv(DATA_DIR / "sample_submission.csv")
assert len(dev_labels) == 1860 and len(dev_evidence) == 1817 and len(eval_pairs) == 112_540

hands = (
    pl.read_parquet(DATA_DIR / "hands.parquet", columns=["hand_id", "table_id", "started_at", "phase", "big_blind", "final_pot", "players_at_showdown", "button_seat"])
    .sort(["table_id", "started_at", "hand_id"])
    .with_columns(pl.int_range(0, pl.len()).over("table_id").cast(pl.Int16).alias("hand_idx"))
)
n_dev_per_table = hands.filter(pl.col("phase") == "development").group_by("table_id").len()["len"].max()
log(f"hands {hands.height:,}; tables {hands['table_id'].n_unique()}; dev hands/table {n_dev_per_table}")

positive_players = set(pl.concat([dev_labels.filter(pl.col("label") == 1)["player_1"], dev_labels.filter(pl.col("label") == 1)["player_2"]]).to_list())
log(f"labels: {dev_labels.filter(pl.col('label')==1).height} positives, {dev_labels.filter(pl.col('label')==0).height} confirmed negatives, {len(positive_players)} positive players")

# %% [markdown]
# ## 1. Ordered action context
# `last_aggr` = the player whose bet/raise is currently being faced (forward-filled within hand × street). An action with `to_call > 0` and `last_aggr == partner` means the actor is facing **the partner's** bet.

# %%
ACTION_CONTEXT = WORK_DIR / "action_context.parquet"
AGGR = ["bet", "raise"]
if not ACTION_CONTEXT.exists():
    (
        pl.scan_parquet(DATA_DIR / "actions.parquet")
        .join(hands.lazy().select(["hand_id", "big_blind", "phase"]), on="hand_id")
        .sort(["hand_id", "action_no"])
        .with_columns(
            (pl.col("action").is_in(AGGR) | ((pl.col("action") == "all_in") & (pl.col("amount") > pl.col("to_call")))).alias("is_aggr"),
            (pl.col("amount") / pl.col("big_blind")).cast(pl.Float32).alias("amount_bb"),
            (pl.col("to_call") / pl.col("big_blind")).cast(pl.Float32).alias("to_call_bb"),
            (pl.col("amount") / pl.max_horizontal("pot_before", "big_blind")).clip(0, 20).cast(pl.Float32).alias("amount_pot_ratio"),
            pl.col("street").replace_strict({"preflop": 0, "flop": 1, "turn": 2, "river": 3}, return_dtype=pl.Int8).alias("street_no"),
        )
        .with_columns(pl.when(pl.col("is_aggr")).then(pl.col("player_id")).otherwise(None).alias("_ag"))
        .with_columns(pl.col("_ag").shift(1).forward_fill().over(["hand_id", "street"]).alias("last_aggr"))
        .select(["hand_id", "phase", "action_no", "street_no", "player_id", "action", "to_call", "players_active",
                 "is_aggr", "last_aggr", "amount_bb", "to_call_bb", "amount_pot_ratio"])
        .sink_parquet(ACTION_CONTEXT)
    )
log(f"action context rows: {pl.scan_parquet(ACTION_CONTEXT).select(pl.len()).collect().item():,}")

# %% [markdown]
# ## 2. Player-hand features and per-player phase baselines

# %%
PLAYER_HANDS = WORK_DIR / "player_hands.parquet"
RANKS = {r: i for i, r in enumerate("23456789TJQKA", start=2)}


def chen(c1, c2):
    r1, s1 = RANKS[c1[0]], c1[1]
    r2, s2 = RANKS[c2[0]], c2[1]
    hi, lo = max(r1, r2), min(r1, r2)
    base = {14: 10, 13: 8, 12: 7, 11: 6}.get(hi, hi / 2)
    if r1 == r2:
        return max(5.0, base * 2)
    sc = base + (2 if s1 == s2 else 0)
    gap = hi - lo - 1
    sc -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 12:
        sc += 1
    return float(sc)


if not PLAYER_HANDS.exists():
    ac = pl.scan_parquet(ACTION_CONTEXT)
    act_by_player = ac.group_by(["hand_id", "player_id"]).agg(
        pl.len().cast(pl.Int16).alias("n_actions"),
        pl.col("is_aggr").sum().cast(pl.Int16).alias("n_aggr"),
        (pl.col("action") == "raise").sum().cast(pl.Int16).alias("n_raise"),
        (pl.col("action") == "call").sum().cast(pl.Int16).alias("n_call"),
        (pl.col("action") == "check").sum().cast(pl.Int16).alias("n_check"),
        (pl.col("action") == "all_in").sum().cast(pl.Int16).alias("n_allin"),
        ((pl.col("action") == "fold") & (pl.col("to_call") > 0)).sum().cast(pl.Int16).alias("n_fold_facing"),
        ((pl.col("street_no") > 0) & (pl.col("action") == "check")).sum().cast(pl.Int16).alias("n_postflop_check"),
        (pl.col("players_active") == 2).sum().cast(pl.Int16).alias("n_hu_actions"),
        pl.col("street_no").max().cast(pl.Int8).alias("last_street"),
        pl.col("amount_bb").max().fill_null(0).cast(pl.Float32).alias("max_amount_bb"),
        pl.col("to_call_bb").max().fill_null(0).cast(pl.Float32).alias("max_to_call_bb"),
        # step 3: pre-flop policy context
        ((pl.col("street_no") == 0) & pl.col("action").is_in(["call", "raise", "bet", "all_in"])).any().alias("vpip_a"),
        ((pl.col("street_no") == 0) & pl.col("is_aggr")).any().alias("pfr"),
        pl.col("action_no").filter((pl.col("street_no") == 0) & pl.col("action").is_in(["call", "raise", "bet", "all_in"])).min().alias("pf_entry_no"),
        (pl.col("to_call_bb").filter(pl.col("street_no") == 0).first() > 1.0).alias("pf_faced_raise"),   # rows are ordered by action_no
        (pl.col("street_no") == 0).any().alias("acted_pf"),
    )
    seats = pl.scan_parquet(DATA_DIR / "seats.parquet")
    cards = seats.select(["hole_card_1", "hole_card_2"]).unique().collect()
    cards = cards.with_columns(pl.Series("chen", [chen(a, b) for a, b in zip(cards["hole_card_1"], cards["hole_card_2"])], dtype=pl.Float32))
    (
        seats.join(hands.lazy().select(["hand_id", "table_id", "phase", "hand_idx", "big_blind", "final_pot", "players_at_showdown", "button_seat"]), on="hand_id")
        .join(cards.lazy(), on=["hole_card_1", "hole_card_2"])
        .with_columns(
            ((pl.col("seat_no") - pl.col("button_seat") + 6) % 6).cast(pl.Int8).alias("pos"),   # 0=BTN 1=SB 2=BB 3=UTG 4=HJ 5=CO
            (pl.max_horizontal(pl.col("hole_card_1").str.slice(0, 1).replace_strict(RANKS, return_dtype=pl.Int16), pl.col("hole_card_2").str.slice(0, 1).replace_strict(RANKS, return_dtype=pl.Int16)) * 15
             + pl.min_horizontal(pl.col("hole_card_1").str.slice(0, 1).replace_strict(RANKS, return_dtype=pl.Int16), pl.col("hole_card_2").str.slice(0, 1).replace_strict(RANKS, return_dtype=pl.Int16))
             + 300 * (pl.col("hole_card_1").str.slice(1, 1) == pl.col("hole_card_2").str.slice(1, 1)).cast(pl.Int16)).cast(pl.Int16).alias("hole_class"),
        )
        .with_columns(
            (pl.col("final_pot") / pl.col("big_blind")).cast(pl.Float32).alias("pot_bb"),
            (pl.col("total_contribution") / pl.col("big_blind")).cast(pl.Float32).alias("contrib_bb"),
            (pl.col("net_chips") / pl.col("big_blind")).cast(pl.Float32).alias("net_bb"),
            (pl.col("total_contribution") > pl.col("big_blind")).alias("vpip"),
        )
        .join(act_by_player, on=["hand_id", "player_id"], how="left")
        .with_columns([pl.col(c).fill_null(0) for c in ["n_actions", "n_aggr", "n_raise", "n_call", "n_check", "n_allin", "n_fold_facing", "n_postflop_check", "n_hu_actions", "last_street", "max_amount_bb", "max_to_call_bb"]])
        .with_columns(pl.col("vpip_a").fill_null(False), pl.col("pfr").fill_null(False), pl.col("acted_pf").fill_null(False),
                      pl.col("pf_entry_no").fill_null(9999).cast(pl.Int32),
                      # a player with no recorded pre-flop action is a blind that saw a limped pot (BB check is recorded, so this is rare) -> treat as not facing a raise
                      pl.col("pf_faced_raise").fill_null(False))
        .select(["hand_id", "player_id", "table_id", "phase", "hand_idx", "big_blind", "pot_bb", "players_at_showdown", "seat_no", "pos", "hole_class",
                 "contrib_bb", "net_bb", "vpip", "vpip_a", "pfr", "pf_entry_no", "pf_faced_raise", "acted_pf", "folded", "went_to_showdown", "won_share", "chen",
                 "n_actions", "n_aggr", "n_raise", "n_call", "n_check", "n_allin", "n_fold_facing", "n_postflop_check", "n_hu_actions", "last_street", "max_amount_bb", "max_to_call_bb"])
        .sink_parquet(PLAYER_HANDS)
    )
    del cards
    gc.collect()

# %% [markdown]
# ## 2a. Population policy and per-player-hand surprise (Step 3)
# `P(vpip | hole class, position, faced raise)` and `P(pfr | …)` from **all** 12 M seats (both phases; colluders are a negligible fraction), smoothed towards the
# class-level mean with 20 pseudo-counts. `loose` is the surprise of entering, `tight` the surprise of folding; `junk_vpip` is a simpler Chen-based version.

# %%
PLAYER_HANDS_POL = WORK_DIR / "player_hands_policy.parquet"
if not PLAYER_HANDS_POL.exists():
    ph_all = pl.scan_parquet(PLAYER_HANDS)
    cls = ph_all.group_by("hole_class").agg(pl.col("vpip_a").cast(pl.Float32).mean().alias("_cls_vpip"), pl.col("pfr").cast(pl.Float32).mean().alias("_cls_pfr"))
    ctx = ph_all.group_by(["hole_class", "pos", "pf_faced_raise"]).agg(pl.len().alias("_n"), pl.col("vpip_a").cast(pl.Float32).sum().alias("_sv"), pl.col("pfr").cast(pl.Float32).sum().alias("_sp"))
    K = 20.0
    policy = (ctx.join(cls, on="hole_class")
              .with_columns(((pl.col("_sv") + K * pl.col("_cls_vpip")) / (pl.col("_n") + K)).clip(1e-4, 1 - 1e-4).cast(pl.Float32).alias("p_vpip"),
                            ((pl.col("_sp") + K * pl.col("_cls_pfr")) / (pl.col("_n") + K)).clip(1e-4, 1 - 1e-4).cast(pl.Float32).alias("p_pfr"))
              .select(["hole_class", "pos", "pf_faced_raise", "p_vpip", "p_pfr"]))
    (
        ph_all.join(policy, on=["hole_class", "pos", "pf_faced_raise"], how="left")
        .with_columns(pl.col("p_vpip").fill_null(0.3), pl.col("p_pfr").fill_null(0.1))
        .with_columns(
            (pl.col("vpip_a").cast(pl.Float32) * (-pl.col("p_vpip").log())).cast(pl.Float32).alias("loose"),
            ((~pl.col("vpip_a")).cast(pl.Float32) * (-(1 - pl.col("p_vpip")).log())).cast(pl.Float32).alias("tight"),
            (pl.col("pfr").cast(pl.Float32) * (-pl.col("p_pfr").log())).cast(pl.Float32).alias("loose_pfr"),
            (pl.col("vpip_a").cast(pl.Float32) * (8.0 - pl.col("chen")).clip(lower_bound=0)).cast(pl.Float32).alias("junk_vpip"),
            (pl.col("pfr").cast(pl.Float32) * (8.0 - pl.col("chen")).clip(lower_bound=0)).cast(pl.Float32).alias("junk_pfr"),
            (pl.col("vpip_a").cast(pl.Float32) - pl.col("p_vpip")).cast(pl.Float32).alias("vpip_resid"),
        )
        .sink_parquet(PLAYER_HANDS_POL)
    )
    del ph_all, cls, ctx, policy
    gc.collect()
PLAYER_HANDS = PLAYER_HANDS_POL
_pol = pl.scan_parquet(PLAYER_HANDS).select(pl.col("vpip_a").cast(pl.Float32).mean(), pl.col("loose").mean(), pl.col("junk_vpip").mean()).collect()
log(f"policy features ready: mean vpip_a={_pol[0,0]:.3f} mean loose={_pol[0,1]:.3f} mean junk_vpip={_pol[0,2]:.3f}")

# %% [markdown]
# ## 2b. Exact hand strength per player and street (numba evaluator)
# Cards are ints `rank*4 + suit`. A hand value is `category·15⁵ + tie-break` (0 = high card … 8 = straight flush); higher is better.
# For every player-hand we evaluate the best 5-of-5/6/7 at flop, turn and river (where the board reached), and rank each player among the six dealt players.
# The evaluator is verified against the known 5-card category counts (1,302,540 high-card … 40 straight flushes).

# %%

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"


def card_to_int(c: str) -> int:
    return RANK_CHARS.index(c[0]) * 4 + SUIT_CHARS.index(c[1])


@njit(cache=True)
def eval5(c0, c1, c2, c3, c4):
    ranks = np.zeros(5, np.int64)
    ranks[0] = c0 >> 2; ranks[1] = c1 >> 2; ranks[2] = c2 >> 2; ranks[3] = c3 >> 2; ranks[4] = c4 >> 2
    flush = ((c0 & 3) == (c1 & 3)) and ((c1 & 3) == (c2 & 3)) and ((c2 & 3) == (c3 & 3)) and ((c3 & 3) == (c4 & 3))
    cnt = np.zeros(13, np.int64)
    for i in range(5):
        cnt[ranks[i]] += 1
    # straight detection (ranks 0..12 = 2..A); wheel = A,2,3,4,5
    mask = 0
    for i in range(5):
        mask |= 1 << ranks[i]
    straight_high = -1
    for hi in range(12, 3, -1):
        if (mask >> (hi - 4)) & 31 == 31:
            straight_high = hi
            break
    if straight_high < 0 and (mask & 0b1000000001111) == 0b1000000001111:
        straight_high = 3  # wheel, 5-high
    # multiplicity pattern
    four = -1; three = -1; pair_hi = -1; pair_lo = -1
    for r in range(12, -1, -1):
        if cnt[r] == 4:
            four = r
        elif cnt[r] == 3:
            three = r
        elif cnt[r] == 2:
            if pair_hi < 0:
                pair_hi = r
            else:
                pair_lo = r
    # kickers descending
    kick = np.zeros(5, np.int64); k = 0
    for r in range(12, -1, -1):
        if cnt[r] == 1:
            kick[k] = r; k += 1
    B = 15
    if straight_high >= 0 and flush:
        return 8 * B**5 + straight_high
    if four >= 0:
        return 7 * B**5 + four * B + kick[0]
    if three >= 0 and pair_hi >= 0:
        return 6 * B**5 + three * B + pair_hi
    if flush:
        v = 5 * B**5
        # all five distinct ranks in a flush (no pairs possible); kickers hold all 5
        for i in range(5):
            v += kick[i] * B ** (4 - i)
        return v
    if straight_high >= 0:
        return 4 * B**5 + straight_high
    if three >= 0:
        return 3 * B**5 + three * B**2 + kick[0] * B + kick[1]
    if pair_hi >= 0 and pair_lo >= 0:
        return 2 * B**5 + pair_hi * B**2 + pair_lo * B + kick[0]
    if pair_hi >= 0:
        return 1 * B**5 + pair_hi * B**3 + kick[0] * B**2 + kick[1] * B + kick[2]
    v = 0
    for i in range(5):
        v += kick[i] * B ** (4 - i)
    return v


@njit(cache=True)
def eval_best(cards, n):
    """Best 5-card value among the first n cards (n = 5, 6 or 7)."""
    best = -1
    if n == 5:
        return eval5(cards[0], cards[1], cards[2], cards[3], cards[4])
    if n == 6:
        for skip in range(6):
            idx = np.empty(5, np.int64); k = 0
            for i in range(6):
                if i != skip:
                    idx[k] = cards[i]; k += 1
            v = eval5(idx[0], idx[1], idx[2], idx[3], idx[4])
            if v > best:
                best = v
        return best
    for s1 in range(7):
        for s2 in range(s1 + 1, 7):
            idx = np.empty(5, np.int64); k = 0
            for i in range(7):
                if i != s1 and i != s2:
                    idx[k] = cards[i]; k += 1
            v = eval5(idx[0], idx[1], idx[2], idx[3], idx[4])
            if v > best:
                best = v
    return best


@njit(cache=True)
def eval_streets(h1, h2, board, nboard):
    """Values at flop (3 board cards), turn (4), river (5); -1 where the board is shorter. Returns (flop, turn, river)."""
    cards = np.empty(7, np.int64)
    cards[0] = h1; cards[1] = h2
    for i in range(nboard):
        cards[2 + i] = board[i]
    vf = eval_best(cards, 5) if nboard >= 3 else -1
    vt = eval_best(cards, 6) if nboard >= 4 else -1
    vr = eval_best(cards, 7) if nboard >= 5 else -1
    return vf, vt, vr


@njit(cache=True)
def eval_table(h1, h2, board, nboard, out):
    """Vectorised: h1,h2 int arrays (N,), board (N,5) ints (-1 padded), nboard (N,), out (N,3) filled with flop/turn/river values."""
    for i in range(h1.shape[0]):
        vf, vt, vr = eval_streets(h1[i], h2[i], board[i], nboard[i])
        out[i, 0] = vf; out[i, 1] = vt; out[i, 2] = vr




PLAYER_STRENGTH = WORK_DIR / "player_strength.parquet"
CARD_MAP = {f"{r}{su}": i * 4 + j for i, r in enumerate(RANK_CHARS) for j, su in enumerate(SUIT_CHARS)}
B5 = 15 ** 5
if not PLAYER_STRENGTH.exists():
    # sanity check of the evaluator on a few known orderings
    _v = lambda cs: eval_best(np.array([card_to_int(c) for c in cs], np.int64), len(cs))
    assert _v(["As", "Ks", "Qs", "Js", "Ts"]) > _v(["Ah", "Ad", "Ac", "As", "Kd"]) > _v(["Kh", "Kd", "Kc", "Qs", "Qd"]) > _v(["Ah", "Th", "7h", "4h", "2h"]) > _v(["Ah", "Kd", "Qc", "Js", "Td"]) > _v(["5h", "4d", "3c", "2s", "Ad"]) > _v(["Ah", "Ad", "Ac", "Ks", "Qd"]) > _v(["Ah", "Ad", "Kc", "Ks", "Qd"]) > _v(["Ah", "Ad", "Kc", "Js", "Qd"]) > _v(["Ah", "Kd", "Qc", "Js", "9d"])
    hb = pl.read_parquet(DATA_DIR / "hands.parquet", columns=["hand_id", "board_cards"]).with_row_index("hrow")
    bl = hb["board_cards"].fill_null("").str.split(" ").list.eval(pl.element().replace_strict(CARD_MAP, default=-1, return_dtype=pl.Int64))
    board = np.column_stack([bl.list.get(i, null_on_oob=True).fill_null(-1).to_numpy() for i in range(5)]).astype(np.int64)
    nboard = (board >= 0).sum(axis=1).astype(np.int64)
    ps = (
        pl.scan_parquet(DATA_DIR / "seats.parquet").select(["hand_id", "player_id", "hole_card_1", "hole_card_2"])
        .with_columns(pl.col("hole_card_1").replace_strict(CARD_MAP, return_dtype=pl.Int64).alias("h1"), pl.col("hole_card_2").replace_strict(CARD_MAP, return_dtype=pl.Int64).alias("h2"))
        .join(hb.lazy().select(["hand_id", "hrow"]), on="hand_id")
        .select(["hand_id", "player_id", "h1", "h2", "hrow"]).collect()
    )
    hrow = ps["hrow"].to_numpy().astype(np.int64)
    out = np.empty((ps.height, 3), np.int64)
    eval_table(ps["h1"].to_numpy().astype(np.int64), ps["h2"].to_numpy().astype(np.int64), board[hrow], nboard[hrow], out)
    nb_p = nboard[hrow]
    v_final = np.where(nb_p >= 5, out[:, 2], np.where(nb_p == 4, out[:, 1], np.where(nb_p == 3, out[:, 0], -1)))
    ps = ps.with_columns(pl.Series("v_flop", out[:, 0]), pl.Series("v_turn", out[:, 1]), pl.Series("v_river", out[:, 2]), pl.Series("v_final", v_final)).drop(["h1", "h2", "hrow"])
    ps = ps.with_columns(
        pl.when(pl.col("v_final") >= 0).then(pl.col("v_final") // B5).otherwise(-1).cast(pl.Int8).alias("cat_final"),
        *[pl.when(pl.col(c) >= 0).then(pl.col(c).rank(descending=True, method="min").over("hand_id")).otherwise(None).cast(pl.Int8).alias(c.replace("v_", "rk_")) for c in ["v_flop", "v_turn", "v_river", "v_final"]],
    )
    ps.write_parquet(PLAYER_STRENGTH)
    del ps, out, board, nboard, hrow, bl, hb
    gc.collect()
log(f"player strength rows: {pl.scan_parquet(PLAYER_STRENGTH).select(pl.len()).collect().item():,}")

# %% [markdown]
# ## 2a′. Action-level population policy and per-action surprise (Step 8)
# A 6-class model of the bot's decision (fold / check / call / bet / raise / all-in) from what the bot sees. `surprise = -log P(taken action)`.
# The player's style covariates come from the **other** phase so that in-phase collusion cannot explain itself away.

# %%
ACTION_SURPRISE = WORK_DIR / "action_surprise.parquet"
PLAYER_HAND_SURPRISE = WORK_DIR / "player_hand_surprise.parquet"
PLAYER_HANDS_V8 = WORK_DIR / "player_hands_v8.parquet"
ACTS = ["fold", "check", "call", "bet", "raise", "all_in"]
SURP_COLS = ["surp_sum", "surp_max", "surp_pf", "surp_post", "n_surp2", "n_surp35", "surp_fold", "surp_call", "surp_aggr", "surp_check", "p_min", "logp_mean"]
if not (ACTION_SURPRISE.exists() and PLAYER_HAND_SURPRISE.exists()):
    _hb = pl.read_parquet(DATA_DIR / "hands.parquet", columns=["hand_id", "phase", "big_blind", "button_seat", "board_cards"])
    _hb = _hb.with_columns(_hb["board_cards"].fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("board"))
    _seats = pl.read_parquet(DATA_DIR / "seats.parquet", columns=["hand_id", "player_id", "seat_no", "hole_card_1", "hole_card_2"])
    _str = pl.read_parquet(PLAYER_STRENGTH, columns=["hand_id", "player_id", "v_flop", "v_turn", "v_river"])
    _ph = pl.scan_parquet(PLAYER_HANDS).select(["player_id", "phase", "vpip_a", "pfr", "n_aggr", "n_call", "n_fold_facing", "went_to_showdown", "n_check", "loose", "tight"])
    _style = (_ph.group_by(["player_id", "phase"]).agg(
                  pl.col("vpip_a").cast(pl.Float32).mean().alias("st_vpip"), pl.col("pfr").cast(pl.Float32).mean().alias("st_pfr"),
                  pl.col("n_aggr").cast(pl.Float32).mean().alias("st_aggr"), pl.col("n_call").cast(pl.Float32).mean().alias("st_call"),
                  pl.col("n_fold_facing").cast(pl.Float32).mean().alias("st_foldf"), pl.col("went_to_showdown").cast(pl.Float32).mean().alias("st_sd"),
                  pl.col("n_check").cast(pl.Float32).mean().alias("st_check"), pl.col("loose").mean().alias("st_loose"), pl.col("tight").mean().alias("st_tight"))
              .with_columns(pl.when(pl.col("phase") == "development").then(pl.lit("evaluation")).otherwise(pl.lit("development")).alias("phase"))   # style from the OTHER phase
              .collect())
    STYLE = [c for c in _style.columns if c.startswith("st_")]
    _a = (pl.scan_parquet(DATA_DIR / "actions.parquet")
          .join(_hb.lazy().select(["hand_id", "phase", "big_blind", "button_seat", "board"]), on="hand_id")
          .join(_seats.lazy(), on=["hand_id", "player_id"]).join(_str.lazy(), on=["hand_id", "player_id"]).join(_style.lazy(), on=["player_id", "phase"], how="left")
          .sort(["hand_id", "action_no"])
          .with_columns(
              pl.col("street").replace_strict({"preflop": 0, "flop": 1, "turn": 2, "river": 3}, return_dtype=pl.Int8).alias("street_no"),
              (pl.col("action").is_in(["bet", "raise"]) | ((pl.col("action") == "all_in") & (pl.col("amount") > pl.col("to_call")))).alias("is_aggr"),
              ((pl.col("seat_no") - pl.col("button_seat") + 6) % 6).cast(pl.Int8).alias("pos"),
              pl.col("hole_card_1").str.slice(0, 1).replace_strict(RANKS, return_dtype=pl.Int8).alias("r1"), pl.col("hole_card_2").str.slice(0, 1).replace_strict(RANKS, return_dtype=pl.Int8).alias("r2"),
              pl.col("hole_card_1").str.slice(1, 1).alias("s1"), pl.col("hole_card_2").str.slice(1, 1).alias("s2"))
          .with_columns(pl.max_horizontal("r1", "r2").alias("hi"), pl.min_horizontal("r1", "r2").alias("lo"), (pl.col("s1") == pl.col("s2")).cast(pl.Int8).alias("suited"),
                        (pl.col("r1") == pl.col("r2")).cast(pl.Int8).alias("pocket"), pl.col("board").list.head(pl.col("street_no") + 2).alias("bd"))
          .with_columns(pl.col("bd").list.eval(pl.element().str.slice(0, 1).replace_strict(RANKS, return_dtype=pl.Int8)).alias("bdr"), pl.col("bd").list.eval(pl.element().str.slice(1, 1)).alias("bds"))
          .with_columns(
              pl.when(pl.col("street_no") == 0).then(-1).otherwise(pl.col("bdr").list.max()).cast(pl.Int8).alias("b_max"),
              pl.when(pl.col("street_no") == 0).then(0).otherwise(pl.col("bdr").list.len() - pl.col("bdr").list.n_unique()).cast(pl.Int8).alias("b_paired"),
              (pl.col("bdr").list.contains(pl.col("r1")).cast(pl.Int8) + pl.col("bdr").list.contains(pl.col("r2")).cast(pl.Int8)).alias("hits"),
              pl.max_horizontal(*[pl.col("bds").list.count_matches(s) + (pl.col("s1") == s).cast(pl.UInt32) + (pl.col("s2") == s).cast(pl.UInt32) for s in "cdhs"]).cast(pl.Int8).alias("flush_cards"),
              pl.when(pl.col("street_no") == 1).then(pl.col("v_flop")).when(pl.col("street_no") == 2).then(pl.col("v_turn")).when(pl.col("street_no") == 3).then(pl.col("v_river")).otherwise(-1).alias("v"))
          .with_columns(
              pl.when(pl.col("v") >= 0).then(pl.col("v") // B5).otherwise(-1).cast(pl.Int8).alias("cat"),
              ((pl.col("pocket") == 1) & (pl.col("hi") > pl.col("b_max")) & (pl.col("street_no") > 0)).cast(pl.Int8).alias("overpair"),
              ((pl.col("hits") >= 1) & (pl.col("hi") == pl.col("b_max")) & (pl.col("street_no") > 0)).cast(pl.Int8).alias("top_pair"),
              (pl.col("to_call") / pl.col("big_blind")).cast(pl.Float32).alias("to_call_bb"),
              (pl.col("to_call") / pl.max_horizontal(pl.col("pot_before"), 1)).clip(0, 5).cast(pl.Float32).alias("to_call_pot"),
              (pl.col("pot_before") / pl.col("big_blind")).cast(pl.Float32).alias("pot_bb"),
              (pl.col("stack_before") / pl.col("big_blind")).cast(pl.Float32).alias("stack_bb"),
              (pl.col("stack_before") / pl.max_horizontal(pl.col("pot_before"), 1)).clip(0, 100).cast(pl.Float32).alias("spr"),
              pl.col("is_aggr").cast(pl.Int8).cum_sum().over(["hand_id", "street_no"]).alias("_ca"),
              pl.col("is_aggr").cast(pl.Int8).cum_sum().over(["hand_id", "player_id"]).alias("_cpa"),
              pl.int_range(0, pl.len()).over(["hand_id", "player_id"]).cast(pl.Int8).alias("own_prior_actions"),
              pl.int_range(0, pl.len()).over(["hand_id", "street_no"]).cast(pl.Int8).alias("street_action_no"))
          .with_columns((pl.col("_ca") - pl.col("is_aggr").cast(pl.Int8)).cast(pl.Int8).alias("aggr_before_street"), (pl.col("_cpa") - pl.col("is_aggr").cast(pl.Int8)).cast(pl.Int8).alias("own_prior_aggr"),
                        pl.col("action").replace_strict({a: i for i, a in enumerate(ACTS)}, return_dtype=pl.Int8).alias("y"))
          .select(["hand_id", "action_no", "player_id", "phase", "street_no", "y", "is_aggr", "pos", "players_active", "to_call_bb", "to_call_pot", "pot_bb", "stack_bb", "spr",
                   "aggr_before_street", "own_prior_aggr", "own_prior_actions", "street_action_no", "hi", "lo", "suited", "pocket", "cat", "b_max", "b_paired", "hits", "flush_cards", "overpair", "top_pair", *STYLE])
          .collect(engine="streaming"))
    POL_FEATS = [c for c in _a.columns if c not in ("hand_id", "action_no", "player_id", "phase", "y", "is_aggr")]
    log(f"action table {_a.shape}; policy features {len(POL_FEATS)}")
    _rng = np.random.RandomState(0); _idx = _rng.choice(_a.height, size=min(5_000_000, _a.height), replace=False)
    _X = _a[_idx].select(POL_FEATS).to_numpy().astype(np.float32); _y = _a[_idx]["y"].to_numpy()
    POL_PARAMS = dict(objective="multiclass", num_class=6, learning_rate=0.1, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=os.cpu_count(), seed=0)
    _m = lgb.train(POL_PARAMS, lgb.Dataset(_X[200_000:], _y[200_000:]), num_boost_round=350, valid_sets=[lgb.Dataset(_X[:200_000], _y[:200_000])], callbacks=[lgb.log_evaluation(100)])
    del _X; gc.collect()
    _P = np.zeros(_a.height, np.float32)
    for s in range(0, _a.height, 2_000_000):
        _c = _a[s:s + 2_000_000]; _pr = _m.predict(_c.select(POL_FEATS).to_numpy().astype(np.float32)); _P[s:s + 2_000_000] = _pr[np.arange(len(_pr)), _c["y"].to_numpy()]
    _a = _a.with_columns(pl.Series("p_taken", _P)).with_columns((-pl.col("p_taken").clip(1e-5, 1).log()).cast(pl.Float32).alias("surp"))
    _a.select(["hand_id", "action_no", "player_id", "street_no", "y", "is_aggr", "p_taken", "surp"]).write_parquet(ACTION_SURPRISE)
    (_a.group_by(["hand_id", "player_id"]).agg(
        pl.col("surp").sum().alias("surp_sum"), pl.col("surp").max().alias("surp_max"),
        pl.col("surp").filter(pl.col("street_no") == 0).sum().alias("surp_pf"), pl.col("surp").filter(pl.col("street_no") > 0).sum().alias("surp_post"),
        (pl.col("surp") > 2.0).sum().cast(pl.Int8).alias("n_surp2"), (pl.col("surp") > 3.5).sum().cast(pl.Int8).alias("n_surp35"),
        pl.col("surp").filter(pl.col("y") == 0).sum().alias("surp_fold"), pl.col("surp").filter(pl.col("y") == 2).sum().alias("surp_call"),
        pl.col("surp").filter(pl.col("is_aggr")).sum().alias("surp_aggr"), pl.col("surp").filter(pl.col("y") == 1).sum().alias("surp_check"),
        pl.col("p_taken").min().alias("p_min"), pl.col("p_taken").log().mean().alias("logp_mean"))
     .with_columns([pl.col(c).fill_null(0.0).cast(pl.Float32) for c in ["surp_pf", "surp_post", "surp_fold", "surp_call", "surp_aggr", "surp_check"]])
     .write_parquet(PLAYER_HAND_SURPRISE))
    log(f"action policy: hold-out multi-logloss {_m.best_score['valid_0']['multi_logloss'] if _m.best_score else float('nan'):.4f}; surprise written")
    del _a, _m, _P, _hb, _seats, _str, _style; gc.collect()
if not PLAYER_HANDS_V8.exists():
    (pl.scan_parquet(PLAYER_HANDS).join(pl.scan_parquet(PLAYER_HAND_SURPRISE), on=["hand_id", "player_id"], how="left")
     .with_columns([pl.col(c).fill_null(0.0) for c in SURP_COLS]).with_columns(pl.col("n_surp2").cast(pl.Int8), pl.col("n_surp35").cast(pl.Int8))
     .sink_parquet(PLAYER_HANDS_V8))
PLAYER_HANDS_V9 = WORK_DIR / "player_hands_v9.parquet"
if not PLAYER_HANDS_V9.exists():
    (pl.scan_parquet(PLAYER_HANDS_V8).with_columns((pl.col("n_surp2") > 0).alias("surp2_any"), (pl.col("vpip_a") & (pl.col("chen") <= 5)).alias("junk_entry"),
                                                   (pl.col("vpip_a") & (pl.col("n_surp2") > 0)).alias("vpip_surp2")).sink_parquet(PLAYER_HANDS_V9))
PLAYER_HANDS = PLAYER_HANDS_V9
_sp = pl.scan_parquet(PLAYER_HANDS).select(pl.col("surp_sum").mean(), pl.col("surp_max").mean(), pl.len()).collect()
log(f"player-hand surprise ready: mean surp_sum={_sp[0,0]:.3f} mean surp_max={_sp[0,1]:.3f} rows {_sp[0,2]:,}")


BASELINE_COLS = ["net_bb", "vpip", "n_aggr", "n_raise", "n_call", "n_fold_facing", "went_to_showdown", "contrib_bb", "vpip_a", "loose", "junk_vpip", "vpip_resid", "pfr", "surp_sum", "surp_max", "surp_post", "surp_pf", "surp_fold", "surp_call", "surp2_any", "junk_entry", "vpip_surp2"]
baselines = (
    pl.scan_parquet(PLAYER_HANDS)
    .group_by(["player_id", "phase"])
    .agg(pl.len().alias("base_hands"), *[pl.col(c).cast(pl.Float32).mean().alias(f"base_{c}") for c in BASELINE_COLS])
    .collect()
)
log(f"player-hand rows: {pl.scan_parquet(PLAYER_HANDS).select(pl.len()).collect().item():,}; baselines: {baselines.height:,}")



# %% [markdown]
# ## 3. Pair selection: **all** eligible development pairs, and all evaluation pairs
# Eligible development pair = ≥ 57 shared development hands (the same 1.9 % exposure rule that defines the evaluation pairs: 38 of 2,000) and no publicly-positive player, plus the 1,860 labelled pairs. Unlabelled pairs are treated as negatives with weight 0.35 in training; the CV positives-vs-everything AP on held-out tables is then computed on the leaderboard's population.

# %%
def pair_hand_map(phase):
    """(p_low, p_high, hand_id, table_id, hand_idx) for every pair dealt into the same hand in `phase`."""
    hp = (
        pl.scan_parquet(PLAYER_HANDS)
        .filter(pl.col("phase") == phase)
        .group_by(["hand_id", "table_id", "hand_idx"])
        .agg(pl.col("player_id").sort_by("seat_no").alias("players"))
    )
    frames = [hp.select(["hand_id", "table_id", "hand_idx", pl.col("players").list.get(i).alias("a"), pl.col("players").list.get(j).alias("b")])
              for i, j in combinations(range(6), 2)]
    return pl.concat(frames).with_columns(pl.min_horizontal("a", "b").alias("p_low"), pl.max_horizontal("a", "b").alias("p_high")).drop(["a", "b"])


DEV_PAIRS = WORK_DIR / "dev_pairs.parquet"
DEV_PAIR_HANDS = WORK_DIR / "dev_pair_hands.parquet"
EVAL_PAIRS_PREP = WORK_DIR / "eval_pairs.parquet"
EVAL_PAIR_HANDS = WORK_DIR / "eval_pair_hands.parquet"

if not all(p.exists() for p in [DEV_PAIRS, DEV_PAIR_HANDS, EVAL_PAIRS_PREP, EVAL_PAIR_HANDS]):
    lab = dev_labels.with_columns(pl.min_horizontal("player_1", "player_2").alias("p_low"), pl.max_horizontal("player_1", "player_2").alias("p_high"))
    dev_all = pair_hand_map("development")
    dev_counts = (dev_all.group_by(["p_low", "p_high"]).agg(pl.col("table_id").first(), pl.len().cast(pl.Int32).alias("shared_hands"))
                  .collect(engine="streaming").sort(["table_id", "p_low", "p_high"]))
    dev_counts = dev_counts.join(lab.select(["pair_id", "p_low", "p_high", "label", "behavior_family"]), on=["p_low", "p_high"], how="left")
    min_shared = int(np.ceil(eval_pairs["shared_hands"].min() * n_dev_per_table / (5000 - n_dev_per_table)))  # 38/2000 -> 57/3000
    known = dev_counts.filter(pl.col("pair_id").is_not_null()).with_columns(pl.lit(True).alias("is_labeled"))
    unknown = (
        dev_counts.filter(pl.col("pair_id").is_null() & (pl.col("shared_hands") >= min_shared)
                          & ~pl.col("p_low").is_in(list(positive_players)) & ~pl.col("p_high").is_in(list(positive_players)))
        .with_columns(pl.concat_str([pl.lit("U"), "p_low", "p_high"], separator="_").alias("pair_id"),
                      pl.lit(0, dtype=pl.Int64).alias("label"), pl.lit("unknown").alias("behavior_family"), pl.lit(False).alias("is_labeled"))
    )
    dev_pairs = pl.concat([known, unknown], how="diagonal_relaxed").sort(["table_id", "pair_id"])
    dev_pairs.write_parquet(DEV_PAIRS)
    (
        dev_all.join(dev_pairs.lazy().select(["pair_id", "p_low", "p_high"]), on=["p_low", "p_high"], how="inner")
        .select(["pair_id", "hand_id", "table_id", "hand_idx", pl.col("p_low").alias("player_1"), pl.col("p_high").alias("player_2")])
        .sink_parquet(DEV_PAIR_HANDS)
    )
    del dev_all, dev_counts, known, unknown
    gc.collect()

    ev = eval_pairs.with_columns(pl.min_horizontal("player_1", "player_2").alias("p_low"), pl.max_horizontal("player_1", "player_2").alias("p_high"))
    eval_all = pair_hand_map("evaluation")
    eph = (
        eval_all.join(ev.lazy().select(["pair_id", "p_low", "p_high"]), on=["p_low", "p_high"], how="inner")
        .select(["pair_id", "hand_id", "table_id", "hand_idx", pl.col("p_low").alias("player_1"), pl.col("p_high").alias("player_2")])
        .collect(engine="streaming")
    )
    eph.write_parquet(EVAL_PAIR_HANDS)
    ev_prep = ev.select(["pair_id", pl.col("p_low").alias("player_1"), pl.col("p_high").alias("player_2"), "shared_hands"]).join(
        eph.group_by("pair_id").agg(pl.col("table_id").first(), pl.len().cast(pl.Int32).alias("shared_calc")), on="pair_id", how="left")
    assert ev_prep.filter(pl.col("shared_hands") != pl.col("shared_calc")).height == 0, "shared_hands reconstruction mismatch"
    ev_prep.write_parquet(EVAL_PAIRS_PREP)
    del eval_all, eph, ev_prep
    gc.collect()

dev_pairs = pl.read_parquet(DEV_PAIRS)
eval_prep = pl.read_parquet(EVAL_PAIRS_PREP)
TABLES = sorted(dev_pairs["table_id"].unique().to_list())
TABLE_CHUNK = {t: i % N_TABLE_CHUNKS for i, t in enumerate(TABLES)}
dev_pairs = dev_pairs.with_columns(pl.col("table_id").replace_strict(TABLE_CHUNK, return_dtype=pl.Int8).alias("chunk"))
log(f"dev pairs: {dev_pairs.height:,} (labelled {dev_pairs['is_labeled'].sum():,}, unlabelled eligible {(~dev_pairs['is_labeled']).sum():,}); dev pair-hands: {pl.scan_parquet(DEV_PAIR_HANDS).select(pl.len()).collect().item():,}")
log(f"eval pairs: {eval_prep.height:,}; eval pair-hands: {pl.scan_parquet(EVAL_PAIR_HANDS).select(pl.len()).collect().item():,}; min shared {eval_prep['shared_hands'].min()}")

# %% [markdown]
# ## 3b. Other-phase pair baselines (Step 5)
# For every development pair: its statistics over the **evaluation** hands the two players share; for every evaluation pair: over their **development** hands.
# Computed from the player-hand table with a self-join on `hand_id` (15 pairs per hand).

# %%
PAIR_PHASE = WORK_DIR / "pair_other_phase_v8.parquet"
OB_STATS = ["both_vpip", "both_junk", "one_junk", "loose_sum", "loose_min", "resid_sum", "junk_vpip_sum", "junk_pfr_sum", "transfer", "n_vpip", "both_fold", "surp_pair", "surp_pair_min", "surp_post_pair"]
if not PAIR_PHASE.exists():
    keys = pl.concat([
        dev_pairs.select(["pair_id", pl.col("p_low"), pl.col("p_high"), pl.lit("evaluation").alias("phase")]),
        eval_prep.select(["pair_id", pl.col("player_1").alias("p_low"), pl.col("player_2").alias("p_high"), pl.lit("development").alias("phase")]),
    ])
    ph_ = pl.scan_parquet(PLAYER_HANDS).select(["hand_id", "player_id", "phase", "vpip_a", "pfr", "chen", "loose", "vpip_resid", "junk_vpip", "junk_pfr", "net_bb", "folded", "surp_sum", "surp_post"])
    (
        ph_.join(ph_, on=["hand_id", "phase"], suffix="_r").filter(pl.col("player_id") < pl.col("player_id_r"))
        .rename({"player_id": "p_low", "player_id_r": "p_high"})
        .join(keys.lazy(), on=["p_low", "p_high", "phase"])
        .with_columns(
            (pl.col("vpip_a") & pl.col("vpip_a_r")).cast(pl.Float32).alias("both_vpip"),
            ((pl.col("vpip_a") & pl.col("vpip_a_r")) & (pl.max_horizontal("chen", "chen_r") <= 5)).cast(pl.Float32).alias("both_junk"),
            ((pl.col("vpip_a") & pl.col("vpip_a_r")) & (pl.min_horizontal("chen", "chen_r") <= 4)).cast(pl.Float32).alias("one_junk"),
            (pl.col("loose") + pl.col("loose_r")).alias("loose_sum"),
            pl.min_horizontal("loose", "loose_r").alias("loose_min"),
            (pl.col("vpip_resid") + pl.col("vpip_resid_r")).alias("resid_sum"),
            (pl.col("junk_vpip") + pl.col("junk_vpip_r")).alias("junk_vpip_sum"),
            (pl.col("junk_pfr") + pl.col("junk_pfr_r")).alias("junk_pfr_sum"),
            pl.max_horizontal(pl.min_horizontal((-pl.col("net_bb")).clip(lower_bound=0), pl.col("net_bb_r").clip(lower_bound=0)),
                              pl.min_horizontal((-pl.col("net_bb_r")).clip(lower_bound=0), pl.col("net_bb").clip(lower_bound=0))).alias("transfer"),
            (pl.col("vpip_a").cast(pl.Float32) + pl.col("vpip_a_r").cast(pl.Float32)).alias("n_vpip"),
            (pl.col("folded") & pl.col("folded_r")).cast(pl.Float32).alias("both_fold"),
            (pl.col("surp_sum") + pl.col("surp_sum_r")).alias("surp_pair"),
            pl.min_horizontal("surp_sum", "surp_sum_r").alias("surp_pair_min"),
            (pl.col("surp_post") + pl.col("surp_post_r")).alias("surp_post_pair"),
        )
        .group_by("pair_id").agg(pl.len().cast(pl.Float32).alias("ob_n"), *[pl.col(c).mean().cast(pl.Float32).alias(f"ob_{c}") for c in OB_STATS])
        .sink_parquet(PAIR_PHASE)
    )
pair_other = pl.read_parquet(PAIR_PHASE)
OB_MEDIAN = {c: float(pair_other[f"ob_{c}"].median()) for c in OB_STATS}
log(f"other-phase baselines: {pair_other.height:,} pairs (dev pairs with eval history: {pair_other.filter(pl.col('pair_id').is_in(dev_pairs['pair_id']))['pair_id'].n_unique():,} / {dev_pairs.height:,}; median other-phase hands {pair_other['ob_n'].median():.0f})")

# %% [markdown]
# ## 4. Pair-conditional hand features (Step 1 set + exact strength + street-level partner features)
# One row per (pair, shared hand). New in Step 2: `loser_beats_winner_*` (the partner who lost money held the better hand at that street), `fold_better_hand` / `fold_better_to_partner` (folded holding the better hand, to the partner's bet), `winner_best_final`, `both_made_no_aggr`, and pre-flop squeeze features (`raise_partner_pf`, `outsider_fold_to_pair_pf`, `pf_squeeze`, `squeeze_then_fold_pf`).

# %%
P_COLS = ["seat_no", "contrib_bb", "net_bb", "vpip", "folded", "went_to_showdown", "won_share", "chen",
          "n_actions", "n_aggr", "n_raise", "n_call", "n_check", "n_allin", "n_fold_facing", "n_postflop_check", "n_hu_actions", "last_street", "max_amount_bb", "max_to_call_bb",
          "pos", "vpip_a", "pfr", "pf_entry_no", "pf_faced_raise", "p_vpip", "loose", "tight", "loose_pfr", "junk_vpip", "junk_pfr", "vpip_resid", *SURP_COLS]
S_COLS = ["v_flop", "v_turn", "v_river", "v_final", "cat_final", "rk_flop", "rk_turn", "rk_river", "rk_final"]


def side(prefix, key):
    ph = pl.scan_parquet(PLAYER_HANDS).select(["hand_id", "player_id", "pot_bb", "players_at_showdown", "big_blind", *P_COLS])
    st = pl.scan_parquet(PLAYER_STRENGTH).select(["hand_id", "player_id", *S_COLS])
    return ph.join(st, on=["hand_id", "player_id"]).select(["hand_id", pl.col("player_id").alias(key), "pot_bb", "players_at_showdown", "big_blind",
                                                            *[pl.col(c).alias(f"{prefix}_{c}") for c in P_COLS + S_COLS]])


def street_value(prefix, street_expr):
    """Hand value of `prefix` at the street given by `street_expr` (0 = pre-flop -> Chen score, 1..3 -> exact value)."""
    return (pl.when(street_expr == 0).then(pl.col(f"{prefix}_chen") * 1000.0)
            .when(street_expr == 1).then(pl.col(f"{prefix}_v_flop").cast(pl.Float64))
            .when(street_expr == 2).then(pl.col(f"{prefix}_v_turn").cast(pl.Float64))
            .otherwise(pl.col(f"{prefix}_v_river").cast(pl.Float64)))


def build_pair_hand_features(ph: pl.LazyFrame, out_path, phase):
    if out_path.exists():
        return
    base = (
        ph.join(side("p1", "player_1"), on=["hand_id", "player_1"])
        .join(side("p2", "player_2").drop(["pot_bb", "players_at_showdown", "big_blind"]), on=["hand_id", "player_2"])
        .with_columns(
            (pl.col("p1_net_bb") - pl.col("p2_net_bb")).alias("signed_net_diff"),
            (pl.col("p1_net_bb") - pl.col("p2_net_bb")).abs().alias("net_gap"),
            pl.min_horizontal((-pl.col("p1_net_bb")).clip(lower_bound=0), pl.col("p2_net_bb").clip(lower_bound=0)).alias("transfer_1_to_2"),
            pl.min_horizontal((-pl.col("p2_net_bb")).clip(lower_bound=0), pl.col("p1_net_bb").clip(lower_bound=0)).alias("transfer_2_to_1"),
            (pl.col("p1_vpip") & pl.col("p2_vpip")).cast(pl.Int8).alias("both_vpip"),
            (pl.col("p1_went_to_showdown") & pl.col("p2_went_to_showdown")).cast(pl.Int8).alias("both_showdown"),
            (pl.col("p1_folded") ^ pl.col("p2_folded")).cast(pl.Int8).alias("one_folded"),
            (((pl.col("p1_won_share") > 0) & (pl.col("p2_net_bb") < 0)) | ((pl.col("p2_won_share") > 0) & (pl.col("p1_net_bb") < 0))).cast(pl.Int8).alias("partner_won_other_lost"),
            (pl.col("p1_net_bb") < pl.col("p2_net_bb")).cast(pl.Int8).alias("loser_is_p1"),
            (pl.col("p1_n_aggr") + pl.col("p2_n_aggr")).alias("pair_aggr"),
            (pl.col("p1_n_raise") + pl.col("p2_n_raise")).alias("pair_raise"),
            (pl.col("p1_n_call") + pl.col("p2_n_call")).alias("pair_call"),
            (pl.col("p1_n_check") + pl.col("p2_n_check")).alias("pair_check"),
            (pl.col("p1_n_postflop_check") + pl.col("p2_n_postflop_check")).alias("pair_postflop_check"),
            ((pl.col("p1_n_aggr") > 0).cast(pl.Int8) + (pl.col("p2_n_aggr") > 0).cast(pl.Int8)).alias("n_partners_aggr"),
            pl.max_horizontal("p1_max_amount_bb", "p2_max_amount_bb").alias("max_amount_bb"),
            pl.max_horizontal("p1_last_street", "p2_last_street").alias("last_street"),
            (pl.col("p1_contrib_bb") + pl.col("p2_contrib_bb")).alias("pair_contrib_bb"),
            # step 3: card-conditional policy deviation of the pair in this hand
            (pl.col("p1_vpip_a") & pl.col("p2_vpip_a")).cast(pl.Int8).alias("both_vpip_a"),
            (pl.col("p1_vpip_a").cast(pl.Int8) + pl.col("p2_vpip_a").cast(pl.Int8)).alias("n_vpip_a"),
            (pl.col("p1_pfr").cast(pl.Int8) + pl.col("p2_pfr").cast(pl.Int8)).alias("n_pfr"),
            (pl.col("p1_loose") + pl.col("p2_loose")).alias("loose_sum"),
            pl.min_horizontal("p1_loose", "p2_loose").alias("loose_min"),
            pl.max_horizontal("p1_loose", "p2_loose").alias("loose_max"),
            pl.max_horizontal("p1_tight", "p2_tight").alias("tight_max"),
            (pl.col("p1_loose_pfr") + pl.col("p2_loose_pfr")).alias("loose_pfr_sum"),
            (pl.col("p1_junk_vpip") + pl.col("p2_junk_vpip")).alias("junk_vpip_sum"),
            pl.min_horizontal("p1_junk_vpip", "p2_junk_vpip").alias("junk_vpip_min"),
            (pl.col("p1_junk_pfr") + pl.col("p2_junk_pfr")).alias("junk_pfr_sum"),
            (pl.col("p1_vpip_resid") + pl.col("p2_vpip_resid")).alias("vpip_resid_sum"),
            pl.max_horizontal("p1_chen", "p2_chen").alias("chen_max_pair"),
            pl.min_horizontal("p1_chen", "p2_chen").alias("chen_min_pair"),
            (pl.col("p1_pf_faced_raise") | pl.col("p2_pf_faced_raise")).cast(pl.Int8).alias("pf_faced_raise_any"),
            (pl.col("p1_pos").is_in([1, 2]).cast(pl.Int8) + pl.col("p2_pos").is_in([1, 2]).cast(pl.Int8)).alias("n_in_blinds"),
            # the member who entered second (after the partner was already in) and that member's cards / surprise
            pl.when(pl.col("p1_vpip_a") & pl.col("p2_vpip_a")).then(
                pl.when(pl.col("p1_pf_entry_no") > pl.col("p2_pf_entry_no")).then(pl.col("p1_chen")).otherwise(pl.col("p2_chen"))).otherwise(99.0).alias("_second_chen"),
            pl.when(pl.col("p1_vpip_a") & pl.col("p2_vpip_a")).then(
                pl.when(pl.col("p1_pf_entry_no") > pl.col("p2_pf_entry_no")).then(pl.col("p1_loose")).otherwise(pl.col("p2_loose"))).otherwise(0.0).alias("second_entrant_loose"),
            # step 8: action-level policy surprise of the pair in this hand
            (pl.col("p1_surp_sum") + pl.col("p2_surp_sum")).alias("surp_pair_sum"),
            pl.min_horizontal("p1_surp_sum", "p2_surp_sum").alias("surp_pair_min"),
            pl.max_horizontal("p1_surp_max", "p2_surp_max").alias("surp_pair_max"),
            (pl.col("p1_surp_pf") + pl.col("p2_surp_pf")).alias("surp_pf_sum"),
            (pl.col("p1_surp_post") + pl.col("p2_surp_post")).alias("surp_post_sum"),
            pl.min_horizontal("p1_surp_post", "p2_surp_post").alias("surp_post_min"),
            (pl.col("p1_surp_fold") + pl.col("p2_surp_fold")).alias("surp_fold_sum"),
            (pl.col("p1_surp_call") + pl.col("p2_surp_call")).alias("surp_call_sum"),
            (pl.col("p1_surp_aggr") + pl.col("p2_surp_aggr")).alias("surp_aggr_sum"),
            (pl.col("p1_surp_check") + pl.col("p2_surp_check")).alias("surp_check_sum"),
            (pl.col("p1_n_surp2") + pl.col("p2_n_surp2")).cast(pl.Int8).alias("n_surp2_sum"),
            ((pl.col("p1_n_surp2") > 0) & (pl.col("p2_n_surp2") > 0)).cast(pl.Int8).alias("both_surp2"),
            (pl.col("p1_vpip_a") & pl.col("p2_vpip_a") & (pl.col("p1_n_surp2") > 0) & (pl.col("p2_n_surp2") > 0)).cast(pl.Int8).alias("both_vpip_surp2"),
            pl.min_horizontal("p1_p_min", "p2_p_min").alias("p_min_pair"),
            (pl.col("p1_logp_mean") + pl.col("p2_logp_mean")).alias("logp_mean_sum"),
        )
        .with_columns(
            ((pl.col("both_vpip_a") == 1) & (pl.col("chen_max_pair") <= 5)).cast(pl.Int8).alias("both_vpip_junk"),
            ((pl.col("both_vpip_a") == 1) & (pl.col("chen_min_pair") <= 4)).cast(pl.Int8).alias("both_vpip_one_junk"),
            (pl.col("_second_chen") <= 4).cast(pl.Int8).alias("second_entrant_junk"),
            ((pl.col("both_vpip_a") == 1) & (pl.col("loose_min") >= 1.5)).cast(pl.Int8).alias("both_surprising"),
            ((pl.col("n_pfr") >= 1) & (pl.col("junk_pfr_sum") >= 4)).cast(pl.Int8).alias("junk_raise"),
            pl.max_horizontal("transfer_1_to_2", "transfer_2_to_1").alias("transfer_any"),
            *[pl.when(pl.col("loser_is_p1") == 1).then(pl.col(f"p1_{c}")).otherwise(pl.col(f"p2_{c}")).alias(f"loser_{c}") for c in ["chen", "contrib_bb", "folded", "last_street", "v_flop", "v_turn", "v_final", "cat_final", "rk_flop", "rk_turn", "rk_final"]],
            *[pl.when(pl.col("loser_is_p1") == 1).then(pl.col(f"p2_{c}")).otherwise(pl.col(f"p1_{c}")).alias(f"winner_{c}") for c in ["chen", "contrib_bb", "folded", "last_street", "v_flop", "v_turn", "v_final", "cat_final", "rk_flop", "rk_turn", "rk_final"]],
            pl.when(pl.col("loser_is_p1") == 1).then(street_value("p1", pl.col("p1_last_street"))).otherwise(street_value("p2", pl.col("p2_last_street"))).alias("_loser_v_fold"),
            pl.when(pl.col("loser_is_p1") == 1).then(street_value("p2", pl.col("p1_last_street"))).otherwise(street_value("p1", pl.col("p2_last_street"))).alias("_winner_v_fold"),
        )
        .with_columns(
            (pl.col("loser_chen") - pl.col("winner_chen")).alias("chen_gap"),
            (pl.col("loser_chen") > pl.col("winner_chen")).cast(pl.Int8).alias("loser_stronger_preflop"),
            (pl.col("transfer_any") / (pl.col("pot_bb") + 1e-3)).clip(0, 1).alias("transfer_pot_ratio"),
            # exact-strength comparisons (only meaningful where the board reached that street)
            ((pl.col("loser_v_flop") >= 0) & (pl.col("loser_v_flop") > pl.col("winner_v_flop"))).cast(pl.Int8).alias("loser_beats_winner_flop"),
            ((pl.col("loser_v_turn") >= 0) & (pl.col("loser_v_turn") > pl.col("winner_v_turn"))).cast(pl.Int8).alias("loser_beats_winner_turn"),
            ((pl.col("loser_v_final") >= 0) & (pl.col("loser_v_final") > pl.col("winner_v_final"))).cast(pl.Int8).alias("loser_beats_winner_final"),
            (pl.col("loser_rk_final") == 1).cast(pl.Int8).alias("loser_best_final"),
            (pl.col("winner_rk_final") == 1).cast(pl.Int8).alias("winner_best_final"),
            (pl.col("winner_cat_final") - pl.col("loser_cat_final")).alias("cat_gap_final"),
            pl.max_horizontal("loser_cat_final", "winner_cat_final").alias("max_cat_final"),
            (pl.col("loser_folded") & (pl.col("_loser_v_fold") > pl.col("_winner_v_fold"))).cast(pl.Int8).alias("fold_better_hand"),
            ((pl.col("loser_cat_final") >= 1) & (pl.col("winner_cat_final") >= 1) & (pl.col("pair_aggr") == 0) & (pl.col("both_showdown") == 1)).cast(pl.Int8).alias("both_made_no_aggr"),
            ((pl.col("winner_rk_final") >= 3) & (pl.col("partner_won_other_lost") == 1)).cast(pl.Int8).alias("winner_weak_won"),
        )
        .drop(["_loser_v_fold", "_winner_v_fold", "_second_chen"])
    )

    # --- member-perspective action features ---------------------------------------------
    members = pl.concat([
        ph.select(["pair_id", "hand_id", pl.col("player_1").alias("member"), pl.col("player_2").alias("partner"), pl.lit(1, dtype=pl.Int8).alias("is_p1")]),
        ph.select(["pair_id", "hand_id", pl.col("player_2").alias("member"), pl.col("player_1").alias("partner"), pl.lit(0, dtype=pl.Int8).alias("is_p1")]),
    ])
    ac = (pl.scan_parquet(ACTION_CONTEXT).filter(pl.col("phase") == phase).select(["hand_id", "action_no", "street_no", "player_id", "action", "to_call", "players_active", "is_aggr", "last_aggr", "amount_pot_ratio"])
          .join(pl.scan_parquet(ACTION_SURPRISE).select(["hand_id", "action_no", "surp"]), on=["hand_id", "action_no"], how="left").with_columns(pl.col("surp").fill_null(0.0)))
    fold_at = ac.filter(pl.col("action") == "fold").group_by(["hand_id", "player_id"]).agg(pl.col("action_no").min().alias("partner_fold_no")).rename({"player_id": "partner"})
    ma = (
        members.join(ac, left_on=["hand_id", "member"], right_on=["hand_id", "player_id"], how="inner")
        .join(fold_at, on=["hand_id", "partner"], how="left")
        .with_columns(
            ((pl.col("to_call") > 0) & (pl.col("last_aggr") == pl.col("partner"))).alias("facing_partner"),
            ((pl.col("to_call") > 0) & pl.col("last_aggr").is_not_null() & (pl.col("last_aggr") != pl.col("partner"))).alias("facing_other"),
            ((pl.col("players_active") == 2) & (pl.col("partner_fold_no").is_null() | (pl.col("partner_fold_no") > pl.col("action_no")))).alias("true_hu"),
            (pl.col("street_no") == 0).alias("pf"),
        )
    )
    inter = ma.group_by(["pair_id", "hand_id"]).agg(
        (pl.col("facing_partner") & (pl.col("action") == "fold")).sum().cast(pl.Int8).alias("fold_to_partner"),
        (pl.col("facing_partner") & (pl.col("action") == "call")).sum().cast(pl.Int8).alias("call_partner"),
        (pl.col("facing_partner") & pl.col("is_aggr")).sum().cast(pl.Int8).alias("raise_partner"),
        (pl.col("facing_other") & (pl.col("action") == "fold")).sum().cast(pl.Int8).alias("fold_to_other"),
        (pl.col("facing_other") & (pl.col("action") == "call")).sum().cast(pl.Int8).alias("call_other"),
        (pl.col("facing_other") & pl.col("is_aggr")).sum().cast(pl.Int8).alias("raise_other"),
        (pl.col("facing_partner") & (pl.col("action") == "fold") & (pl.col("is_p1") == 1)).sum().cast(pl.Int8).alias("p1_fold_to_partner"),
        (pl.col("facing_partner") & (pl.col("action") == "call") & (pl.col("is_p1") == 1)).sum().cast(pl.Int8).alias("p1_call_partner"),
        # street-level
        (pl.col("facing_partner") & (pl.col("action") == "fold") & pl.col("pf")).sum().cast(pl.Int8).alias("fold_to_partner_pf"),
        (pl.col("facing_partner") & (pl.col("action") == "fold") & ~pl.col("pf")).sum().cast(pl.Int8).alias("fold_to_partner_post"),
        (pl.col("facing_partner") & (pl.col("action") == "call") & ~pl.col("pf")).sum().cast(pl.Int8).alias("call_partner_post"),
        (pl.col("facing_partner") & pl.col("is_aggr") & pl.col("pf")).sum().cast(pl.Int8).alias("raise_partner_pf"),
        (pl.col("facing_partner") & pl.col("is_aggr") & ~pl.col("pf")).sum().cast(pl.Int8).alias("raise_partner_post"),
        (pl.col("is_aggr") & pl.col("pf") & (pl.col("is_p1") == 1)).sum().cast(pl.Int8).alias("p1_aggr_pf"),
        (pl.col("is_aggr") & pl.col("pf") & (pl.col("is_p1") == 0)).sum().cast(pl.Int8).alias("p2_aggr_pf"),
        pl.col("true_hu").sum().cast(pl.Int8).alias("hu_actions"),
        (pl.col("true_hu") & (pl.col("action") == "check")).sum().cast(pl.Int8).alias("hu_check"),
        (pl.col("true_hu") & (pl.col("action") == "call")).sum().cast(pl.Int8).alias("hu_call"),
        (pl.col("true_hu") & pl.col("is_aggr")).sum().cast(pl.Int8).alias("hu_aggr"),
        (pl.col("true_hu") & (pl.col("street_no") >= 2) & (pl.col("action") == "check")).sum().cast(pl.Int8).alias("hu_late_check"),
        (pl.col("true_hu") & (pl.col("action") == "fold")).sum().cast(pl.Int8).alias("hu_fold"),
        (pl.col("is_aggr") & (pl.col("amount_pot_ratio") >= 1.0)).sum().cast(pl.Int8).alias("pair_overbets"),
        # step 8: surprise of the member's actions by context
        pl.col("surp").filter(pl.col("facing_partner")).sum().cast(pl.Float32).alias("surp_vs_partner"),
        pl.col("surp").filter(pl.col("facing_partner")).max().cast(pl.Float32).alias("surp_vs_partner_max"),
        pl.col("surp").filter(pl.col("facing_other")).sum().cast(pl.Float32).alias("surp_vs_other"),
        pl.col("surp").filter(pl.col("true_hu")).sum().cast(pl.Float32).alias("surp_hu"),
        pl.col("surp").filter(pl.col("facing_partner") & (pl.col("action") == "fold")).sum().cast(pl.Float32).alias("surp_fold_partner"),
        pl.col("surp").filter(pl.col("facing_partner") & (pl.col("action") == "call")).sum().cast(pl.Float32).alias("surp_call_partner"),
        pl.col("surp").filter(pl.col("facing_partner") & pl.col("is_aggr")).sum().cast(pl.Float32).alias("surp_raise_partner"),
        pl.col("surp").filter(pl.col("true_hu") & (pl.col("action") == "check")).sum().cast(pl.Float32).alias("surp_hu_check"),
        pl.col("surp").filter(pl.col("is_p1") == 1).sum().cast(pl.Float32).alias("p1_surp_act"),
        pl.col("surp").filter(pl.col("is_p1") == 0).sum().cast(pl.Float32).alias("p2_surp_act"),
    )

    # --- outsider responses to the pair's aggression -------------------------------------
    responses = ac.filter((pl.col("to_call") > 0) & pl.col("last_aggr").is_not_null()).select(
        ["hand_id", pl.col("last_aggr").alias("member"), pl.col("player_id").alias("responder"), "action", "is_aggr", (pl.col("street_no") == 0).alias("pf")])
    press = (
        members.join(responses, on=["hand_id", "member"], how="inner")
        .filter(pl.col("responder") != pl.col("partner"))
        .group_by(["pair_id", "hand_id"]).agg(
            (pl.col("action") == "fold").sum().cast(pl.Int8).alias("outsider_fold_to_pair"),
            (pl.col("action") == "call").sum().cast(pl.Int8).alias("outsider_call_to_pair"),
            pl.col("is_aggr").sum().cast(pl.Int8).alias("outsider_raise_to_pair"),
            ((pl.col("action") == "fold") & pl.col("pf")).sum().cast(pl.Int8).alias("outsider_fold_to_pair_pf"),
        )
    )
    fill = ["fold_to_partner", "call_partner", "raise_partner", "fold_to_other", "call_other", "raise_other", "p1_fold_to_partner", "p1_call_partner",
            "fold_to_partner_pf", "fold_to_partner_post", "call_partner_post", "raise_partner_pf", "raise_partner_post", "p1_aggr_pf", "p2_aggr_pf",
            "hu_actions", "hu_check", "hu_call", "hu_aggr", "hu_late_check", "hu_fold", "pair_overbets",
            "outsider_fold_to_pair", "outsider_call_to_pair", "outsider_raise_to_pair", "outsider_fold_to_pair_pf",
            "surp_vs_partner", "surp_vs_partner_max", "surp_vs_other", "surp_hu", "surp_fold_partner", "surp_call_partner", "surp_raise_partner", "surp_hu_check", "p1_surp_act", "p2_surp_act"]
    out = (
        base.join(inter, on=["pair_id", "hand_id"], how="left").join(press, on=["pair_id", "hand_id"], how="left")
        .with_columns([pl.col(c).fill_null(0) for c in fill])
        .with_columns(
            # evidence-anatomy flags
            ((pl.col("fold_to_partner") > 0) & (pl.col("loser_stronger_preflop") == 1)).cast(pl.Int8).alias("fold_stronger_to_partner"),
            ((pl.col("fold_to_partner") > 0) & (pl.col("fold_better_hand") == 1)).cast(pl.Int8).alias("fold_better_to_partner"),
            ((pl.col("loser_chen") >= 6) & (pl.col("winner_chen") >= 6) & (pl.col("raise_partner") == 0) & (pl.col("both_vpip") == 1)).cast(pl.Int8).alias("both_strong_no_raise"),
            (pl.col("call_partner") >= 2).cast(pl.Int8).alias("multi_call_partner"),
            ((pl.col("n_partners_aggr") == 2) & (pl.col("outsider_fold_to_pair") >= 1)).cast(pl.Int8).alias("squeeze"),
            ((pl.col("n_partners_aggr") == 2) & (pl.col("outsider_fold_to_pair") >= 1) & (pl.col("fold_to_partner") >= 1)).cast(pl.Int8).alias("squeeze_then_fold"),
            ((pl.col("raise_partner_pf") >= 1) & (pl.col("outsider_fold_to_pair_pf") >= 1)).cast(pl.Int8).alias("pf_squeeze"),
            ((pl.col("raise_partner_pf") >= 1) & (pl.col("outsider_fold_to_pair_pf") >= 1) & (pl.col("fold_to_partner") >= 1)).cast(pl.Int8).alias("squeeze_then_fold_pf"),
            ((pl.col("partner_won_other_lost") == 1) & (pl.col("call_partner") >= 1) & (pl.col("transfer_any") >= 10)).cast(pl.Int8).alias("dump"),
            ((pl.col("partner_won_other_lost") == 1) & (pl.col("call_partner") >= 1) & (pl.col("loser_beats_winner_final") == 1)).cast(pl.Int8).alias("dump_better_hand"),
            ((pl.col("both_showdown") == 1) & (pl.col("hu_check") >= 2) & (pl.col("pair_aggr") == 0)).cast(pl.Int8).alias("checkdown"),
            ((pl.col("both_showdown") == 1) & (pl.col("hu_late_check") >= 1) & (pl.max_horizontal("loser_chen", "winner_chen") >= 7)).cast(pl.Int8).alias("hu_checkdown_strong"),
            ((pl.col("hu_check") >= 1) & (pl.col("max_cat_final") >= 2) & (pl.col("pair_aggr") == 0)).cast(pl.Int8).alias("hu_check_two_pair_plus"),
            (pl.col("transfer_any") * pl.col("call_partner")).cast(pl.Float32).alias("transfer_x_call"),
        )
        .with_columns(
            (pl.col("transfer_any") / (pl.col("transfer_any").max().over("pair_id") + 1e-3)).cast(pl.Float32).alias("transfer_to_max"),
            (pl.col("net_gap") / (pl.col("net_gap").max().over("pair_id") + 1e-3)).cast(pl.Float32).alias("net_gap_to_max"),
            (pl.col("pot_bb") / (pl.col("pot_bb").max().over("pair_id") + 1e-3)).cast(pl.Float32).alias("pot_to_max"),
            (pl.col("outsider_fold_to_pair") / (pl.col("outsider_fold_to_pair").max().over("pair_id") + 1e-3)).cast(pl.Float32).alias("outsider_fold_to_max"),
            *[(pl.col(c).rank(method="average").over("pair_id") / pl.len().over("pair_id")).cast(pl.Float32).alias(f"{c}_prank") for c in PRANK_BASE],
        )
    )
    out.collect(engine="streaming").write_parquet(out_path)


HAND_RAW = [
    "pot_bb", "players_at_showdown", "both_vpip", "both_showdown", "one_folded", "partner_won_other_lost", "net_gap", "transfer_any", "transfer_pot_ratio",
    "loser_chen", "winner_chen", "chen_gap", "loser_stronger_preflop", "loser_contrib_bb", "pair_contrib_bb",
    "pair_aggr", "pair_raise", "pair_call", "pair_check", "pair_postflop_check", "n_partners_aggr", "max_amount_bb", "last_street", "pair_overbets",
    "fold_to_partner", "call_partner", "raise_partner", "fold_to_other", "call_other", "raise_other",
    "hu_actions", "hu_check", "hu_call", "hu_aggr", "hu_late_check", "hu_fold",
    "outsider_fold_to_pair", "outsider_call_to_pair", "outsider_raise_to_pair",
    # step 2: exact strength
    "loser_cat_final", "winner_cat_final", "loser_rk_final", "winner_rk_final", "loser_rk_flop", "winner_rk_flop", "cat_gap_final", "max_cat_final",
    "loser_beats_winner_flop", "loser_beats_winner_turn", "loser_beats_winner_final", "loser_best_final", "winner_best_final", "fold_better_hand", "both_made_no_aggr", "winner_weak_won",
    # step 2: street-level
    "fold_to_partner_pf", "fold_to_partner_post", "call_partner_post", "raise_partner_pf", "raise_partner_post", "outsider_fold_to_pair_pf",
    # step 3: policy deviation
    "both_vpip_a", "n_vpip_a", "n_pfr", "loose_sum", "loose_min", "loose_max", "tight_max", "loose_pfr_sum", "junk_vpip_sum", "junk_vpip_min", "junk_pfr_sum", "vpip_resid_sum",
    "chen_max_pair", "chen_min_pair", "pf_faced_raise_any", "n_in_blinds", "second_entrant_loose", "both_vpip_junk", "both_vpip_one_junk", "second_entrant_junk", "both_surprising", "junk_raise",
    # step 8: action-level policy surprise
    "surp_pair_sum", "surp_pair_min", "surp_pair_max", "surp_pf_sum", "surp_post_sum", "surp_post_min", "surp_fold_sum", "surp_call_sum", "surp_aggr_sum", "surp_check_sum", "n_surp2_sum", "both_surp2", "p_min_pair", "logp_mean_sum",
    "surp_vs_partner", "surp_vs_partner_max", "surp_vs_other", "surp_hu", "surp_fold_partner", "surp_call_partner", "surp_raise_partner", "surp_hu_check", "both_vpip_surp2",
]
HAND_FLAGS = ["fold_stronger_to_partner", "fold_better_to_partner", "both_strong_no_raise", "multi_call_partner", "squeeze", "squeeze_then_fold", "pf_squeeze", "squeeze_then_fold_pf",
              "dump", "dump_better_hand", "checkdown", "hu_checkdown_strong", "hu_check_two_pair_plus", "transfer_x_call"]
HAND_TOMAX = ["transfer_to_max", "net_gap_to_max", "pot_to_max", "outsider_fold_to_max"]
PRANK_BASE = HAND_RAW + HAND_FLAGS
HAND_FEATS = HAND_RAW + HAND_FLAGS + HAND_TOMAX + [f"{c}_prank" for c in PRANK_BASE]

DEV_HF = [WORK_DIR / f"dev_hand_features_v9_{k}.parquet" for k in range(N_TABLE_CHUNKS)]
EVAL_HF = WORK_DIR / "eval_hand_features_v9.parquet"
for k in range(N_TABLE_CHUNKS):
    chunk_tables = [t for t in TABLES if TABLE_CHUNK[t] == k]
    build_pair_hand_features(pl.scan_parquet(DEV_PAIR_HANDS).filter(pl.col("table_id").is_in(chunk_tables)), DEV_HF[k], "development")
    log(f"dev hand features chunk {k}: {pl.scan_parquet(DEV_HF[k]).select(pl.len()).collect().item():,} rows")
build_pair_hand_features(pl.scan_parquet(EVAL_PAIR_HANDS), EVAL_HF, "evaluation")
log(f"eval hand features: {pl.scan_parquet(EVAL_HF).select(pl.len()).collect().item():,} rows; {len(HAND_FEATS)} hand features")

# %% [markdown]
# ## 5. Hand-level models (GroupKFold by table)
# *Evidence ranker*: trained on the coordinated pairs' hands only (planted evidence vs their other hands) — produces the evidence slots.
# *Suspicious-hand model*: coordinated pairs' hands vs confirmed non-target pairs' hands — its per-pair aggregates are what the pair model uses.

# %%
labelled_ids = set(dev_pairs.filter(pl.col("is_labeled"))["pair_id"].to_list())
dev_hf_lab = pl.concat([pl.scan_parquet(f).filter(pl.col("pair_id").is_in(list(labelled_ids))).collect() for f in DEV_HF])
dev_hf_lab = dev_hf_lab.join(dev_pairs.select(["pair_id", "label", "behavior_family", "is_labeled"]), on="pair_id")
dev_hf_lab = dev_hf_lab.join(dev_evidence.select(["pair_id", "hand_id", pl.lit(True).alias("is_evidence")]), on=["pair_id", "hand_id"], how="left").with_columns(pl.col("is_evidence").fill_null(False))
assert dev_hf_lab["is_evidence"].sum() == 1817, "not every evidence hand was reconstructed"
dh = dev_hf_lab.to_pandas()
dh[HAND_FEATS] = dh[HAND_FEATS].astype("float32")

rng = np.random.RandomState(SEED)
TABLE_FOLD = {t: int(v) for t, v in zip(TABLES, rng.permutation(len(TABLES)) % N_FOLDS)}
dh["fold"] = dh["table_id"].map(TABLE_FOLD)
dev_pairs = dev_pairs.with_columns(pl.col("table_id").replace_strict(TABLE_FOLD, return_dtype=pl.Int8).alias("fold"))

HAND_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=os.cpu_count())
SUS_PARAMS = {**HAND_PARAMS, "scale_pos_weight": 8.0}
HAND_SEEDS = [SEED, SEED + 7]
HAND_ROUNDS = 400
pos_mask = (dh["label"] == 1).to_numpy()
dh["hand_score"] = 0.0
dh["hand_sus"] = 0.0
rank_models, sus_models = {}, {}
for f in range(N_FOLDS):
    tr_pos = pos_mask & (dh["fold"] != f).to_numpy()
    tr_all = (dh["fold"] != f).to_numpy()
    te = (dh["fold"] == f).to_numpy()
    rank_models[f] = [lgb.train({**HAND_PARAMS, "seed": sd}, lgb.Dataset(dh.loc[tr_pos, HAND_FEATS], dh.loc[tr_pos, "is_evidence"].astype(int)), num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
    sus_models[f] = lgb.train(SUS_PARAMS, lgb.Dataset(dh.loc[tr_all, HAND_FEATS], dh.loc[tr_all, "is_evidence"].astype(int)), num_boost_round=HAND_ROUNDS)
    dh.loc[te, "hand_score"] = np.mean([m.predict(dh.loc[te, HAND_FEATS]) for m in rank_models[f]], axis=0)
    dh.loc[te, "hand_sus"] = sus_models[f].predict(dh.loc[te, HAND_FEATS])
    log(f"hand models fold {f}: MAP@5 on positive pairs = {map5_within_pairs(dh[te & pos_mask]):.4f}")

pos_all = dh[pos_mask]
oof_map5 = map5_within_pairs(pos_all)
fam_map5 = {fam: map5_within_pairs(pos_all[pos_all["behavior_family"] == fam]) for fam in TARGET_BEHAVIORS}
log(f"OOF evidence MAP@5 = {oof_map5:.4f}  per family: { {k: round(v, 4) for k, v in fam_map5.items()} }")
rank_models_full = [lgb.train({**HAND_PARAMS, "seed": sd}, lgb.Dataset(dh.loc[pos_mask, HAND_FEATS], dh.loc[pos_mask, "is_evidence"].astype(int)), num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
sus_model_full = lgb.train(SUS_PARAMS, lgb.Dataset(dh[HAND_FEATS], dh["is_evidence"].astype(int)), num_boost_round=HAND_ROUNDS)
neg_scores = dh.loc[(dh["label"] == 0).to_numpy(), "hand_score"].to_numpy()
THR95, THR99 = np.quantile(neg_scores, 0.95), np.quantile(neg_scores, 0.99)
neg_sus = dh.loc[(dh["label"] == 0).to_numpy(), "hand_sus"].to_numpy()
SUS95, SUS99 = np.quantile(neg_sus, 0.95), np.quantile(neg_sus, 0.99)
imp = pd.Series(np.mean([m.feature_importance("gain") for m in rank_models_full], axis=0), index=HAND_FEATS).sort_values(ascending=False)
print("top evidence-ranker features (gain):\n", imp.head(20).round(0).to_string())

# %% [markdown]
# ## 5b. Evidence stage 2 — re-rank the top-K pool of each pair (Step 4)
# Pool = the K hands with the highest stage-1 score per pair. Extra features: chronological rank inside the pool (labelled hands skew early), score relative to the pool
# maximum, position in the table history, spacing to the neighbouring pool hands. Trained on the labelled positive pairs only, same table folds.

# %%
POOL_K = 20
N_DEV_HANDS = int(n_dev_per_table)                     # 3000
N_EVAL_HANDS = int(hands.filter(pl.col("phase") == "evaluation").group_by("table_id").len()["len"].max())   # 2000
TRAIN_WINDOWS = [(0, N_EVAL_HANDS), (N_DEV_HANDS - N_EVAL_HANDS, N_DEV_HANDS)]   # 2,000-hand development windows = evaluation exposure
POOL_EXTRA = ["t_rank_pool", "t_rank_pool_rel", "hand_score", "hs_rel", "idx_rel", "gap_prev_pool", "gap_next_pool", "n_pool_within20", "pool_size",
              "n_strong_before", "score_cum_before", "score_cum_rel", "n_strong_pool", "score_rank_pool", *[f"fam_{f}" for f in TARGET_BEHAVIORS]]
POOL_FEATS = HAND_FEATS + POOL_EXTRA
EV_THR10 = float(np.quantile(dh.loc[pos_mask & dh["is_evidence"].to_numpy(), "hand_score"], 0.10))
EV_THR50 = float(np.quantile(dh.loc[pos_mask & dh["is_evidence"].to_numpy(), "hand_score"], 0.50))


def pool_features(frame: pl.DataFrame) -> pl.DataFrame:
    """frame: pair_id, hand_id, hand_idx, hand_score + HAND_FEATS. Returns the top-K pool per pair with the stage-2 extras."""
    pool = (
        frame.with_columns(pl.col("hand_score").rank(method="ordinal", descending=True).over("pair_id").alias("_hs_rank"))
        .filter(pl.col("_hs_rank") <= POOL_K)
        .sort(["pair_id", "hand_idx"])
        .with_columns(
            pl.len().over("pair_id").cast(pl.Float32).alias("pool_size"),
            pl.col("hand_idx").rank(method="ordinal").over("pair_id").cast(pl.Float32).alias("t_rank_pool"),
            (pl.col("hand_score") / pl.col("hand_score").max().over("pair_id")).cast(pl.Float32).alias("hs_rel"),
            (pl.when(pl.col("hand_idx") < N_DEV_HANDS).then(pl.col("hand_idx").cast(pl.Float32) / N_DEV_HANDS)
             .otherwise((pl.col("hand_idx").cast(pl.Float32) - N_DEV_HANDS) / N_EVAL_HANDS)).alias("idx_rel"),
            pl.col("hand_idx").cast(pl.Int32).diff().over("pair_id").fill_null(9999).cast(pl.Float32).alias("gap_prev_pool"),
            (-pl.col("hand_idx").cast(pl.Int32).diff(-1).over("pair_id")).fill_null(9999).cast(pl.Float32).alias("gap_next_pool"),
            (pl.col("hand_score") > 0.5).cast(pl.Float32).cum_sum().over("pair_id").alias("_nsb"),
            pl.col("hand_score").cum_sum().over("pair_id").alias("_scb"),
            pl.col("hand_score").sum().over("pair_id").alias("_sps"),
            (pl.col("hand_score") > 0.5).sum().over("pair_id").cast(pl.Float32).alias("n_strong_pool"),
            pl.col("hand_score").rank(method="ordinal", descending=True).over("pair_id").cast(pl.Float32).alias("score_rank_pool"),
        )
        .with_columns((pl.col("t_rank_pool") / pl.col("pool_size")).alias("t_rank_pool_rel"),
                      (pl.col("_nsb") - (pl.col("hand_score") > 0.5).cast(pl.Float32)).alias("n_strong_before"),
                      (pl.col("_scb") - pl.col("hand_score")).alias("score_cum_before"),
                      (pl.col("_scb") / pl.col("_sps")).alias("score_cum_rel"))
        .drop(["_nsb", "_scb", "_sps"])
    )
    near = (
        pool.select(["pair_id", "hand_id", "hand_idx"])
        .join(pool.select(["pair_id", pl.col("hand_idx").alias("_o")]), on="pair_id")
        .filter((pl.col("_o") - pl.col("hand_idx")).abs() <= 20)
        .group_by(["pair_id", "hand_id"]).agg(pl.len().cast(pl.Float32).alias("n_pool_within20"))
    )
    return pool.join(near, on=["pair_id", "hand_id"], how="left").with_columns(pl.col("n_pool_within20").fill_null(1.0)).drop("_hs_rank")


POOL_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=os.cpu_count())
POOL_ROUNDS = 500
POOL_SEEDS = [SEED, SEED + 3, SEED + 5]
# --- Step 12: action-sequence templates of the pair (winner / loser roles, street, facing whom, action) ---------------------------------------
TPL_COLS = ["tpl_pair", "tpl_all", "tpl_pf", "tpl_first3"]
TE_K = 10.0


def hand_templates(keys: pl.DataFrame, phase: str) -> pl.DataFrame:
    """keys: pair_id, hand_id, player_1, player_2, loser_is_p1 -> one row per (pair_id, hand_id) with the four template strings."""
    ac_ = pl.scan_parquet(ACTION_CONTEXT).filter(pl.col("phase") == phase).select(["hand_id", "action_no", "street_no", "player_id", "action", "to_call", "last_aggr"])
    a = (keys.lazy().join(ac_, on="hand_id")
         .with_columns(pl.when(pl.col("player_id") == pl.col("player_1")).then(pl.lit("A")).when(pl.col("player_id") == pl.col("player_2")).then(pl.lit("B")).otherwise(pl.lit("O")).alias("who"))
         .with_columns(pl.when(pl.col("who") == "O").then(pl.lit("O")).when(pl.col("loser_is_p1") == 1).then(pl.when(pl.col("who") == "A").then(pl.lit("L")).otherwise(pl.lit("W")))
                       .otherwise(pl.when(pl.col("who") == "A").then(pl.lit("W")).otherwise(pl.lit("L"))).alias("role"),
                       pl.when(pl.col("to_call") == 0).then(pl.lit("-")).when((pl.col("last_aggr") == pl.col("player_1")) | (pl.col("last_aggr") == pl.col("player_2"))).then(pl.lit("p")).otherwise(pl.lit("o")).alias("facing"))
         .sort(["pair_id", "hand_id", "action_no"])
         .with_columns((pl.col("role") + pl.col("street_no").cast(pl.Utf8) + pl.col("facing") + ":" + pl.col("action").str.slice(0, 2)).alias("tok"))
         .group_by(["pair_id", "hand_id"]).agg(pl.col("tok").filter(pl.col("role") != "O").str.join(" ").alias("tpl_pair"), pl.col("tok").str.join(" ").alias("tpl_all"),
                                              pl.col("tok").filter((pl.col("role") != "O") & (pl.col("street_no") == 0)).str.join(" ").alias("tpl_pf"),
                                              pl.col("tok").filter(pl.col("role") != "O").head(3).str.join(" ").alias("tpl_first3"))
         .collect())
    return keys.select(["pair_id", "hand_id"]).join(a, on=["pair_id", "hand_id"], how="left").with_columns([pl.col(c).fill_null("") for c in TPL_COLS])


def te_encode(train_keys: np.ndarray, train_y: np.ndarray, apply_keys: np.ndarray, k: float = TE_K) -> np.ndarray:
    prior = float(train_y.mean()); s = pd.DataFrame({"k": train_keys, "y": train_y}).groupby("k")["y"].agg(["sum", "count"])
    m = (s["sum"] + k * prior) / (s["count"] + k)
    return pd.Series(apply_keys).map(m).fillna(prior).to_numpy()


TE_FEATS = [f"te_{c}" for c in TPL_COLS] + [f"te_{c}_fam" for c in TPL_COLS] + [f"n_{c}" for c in TPL_COLS]
POOL_EXTRA = POOL_EXTRA + TE_FEATS
POOL_FEATS = HAND_FEATS + POOL_EXTRA
_tpl_dev = hand_templates(pl.from_pandas(dh.loc[pos_mask, ["pair_id", "hand_id", "player_1", "player_2", "loser_is_p1"]]), "development")
dh = dh.merge(_tpl_dev.to_pandas(), on=["pair_id", "hand_id"], how="left")
for c in TPL_COLS:
    dh[c] = dh[c].fillna(""); dh[f"n_{c}"] = dh[c].str.count(" ") + (dh[c] != "").astype(int)
    dh[f"te_{c}"] = 0.0; dh[f"te_{c}_fam"] = 0.0
_tk = {c: dh[c].to_numpy() for c in TPL_COLS}; _tkf = {c: (dh["behavior_family"].astype(str) + "|" + dh[c]).to_numpy() for c in TPL_COLS}
_yev = dh["is_evidence"].astype(int).to_numpy(); _fo = dh["fold"].to_numpy()
for f in range(N_FOLDS):
    tr, te = pos_mask & (_fo != f), (_fo == f)
    for c in TPL_COLS:
        dh.loc[te, f"te_{c}"] = te_encode(_tk[c][tr], _yev[tr], _tk[c][te]); dh.loc[te, f"te_{c}_fam"] = te_encode(_tkf[c][tr], _yev[tr], _tkf[c][te])
TE_MAPS = {c: (_tk[c][pos_mask], _yev[pos_mask]) for c in TPL_COLS}; TE_MAPS_FAM = {c: (_tkf[c][pos_mask], _yev[pos_mask]) for c in TPL_COLS}
log(f"templates: {dh.loc[pos_mask, 'tpl_pair'].nunique():,} distinct pair templates over {int(pos_mask.sum()):,} hands of coordinated pairs")
dh_pos = pl.from_pandas(dh.loc[pos_mask, ["pair_id", "hand_id", "hand_idx", "hand_score", "is_evidence", "fold", "pot_bb", "behavior_family", *TE_FEATS] + [c for c in HAND_FEATS if c != "pot_bb"]])
dh_pos = dh_pos.with_columns([(pl.col("behavior_family") == f).cast(pl.Float32).alias(f"fam_{f}") for f in TARGET_BEHAVIORS]).drop("behavior_family")
pool_dev = pool_features(dh_pos).to_pandas()
pool_dev["hand_score2"] = 0.0
pool_models = {}
for f in range(N_FOLDS):
    tr, te = (pool_dev["fold"] != f).to_numpy(), (pool_dev["fold"] == f).to_numpy()
    pool_models[f] = [lgb.train({**POOL_PARAMS, "seed": sd}, lgb.Dataset(pool_dev.loc[tr, POOL_FEATS].astype("float32"), pool_dev.loc[tr, "is_evidence"].astype(int)), num_boost_round=POOL_ROUNDS) for sd in POOL_SEEDS]
    pool_dev.loc[te, "hand_score2"] = np.mean([m.predict(pool_dev.loc[te, POOL_FEATS].astype("float32")) for m in pool_models[f]], axis=0)
pool_models_full = [lgb.train({**POOL_PARAMS, "seed": sd}, lgb.Dataset(pool_dev[POOL_FEATS].astype("float32"), pool_dev["is_evidence"].astype(int)), num_boost_round=POOL_ROUNDS) for sd in POOL_SEEDS]
# final evidence score: pool hands (stage-2) above everything else (stage-1)
dh = dh.merge(pool_dev[["pair_id", "hand_id", "hand_score2"]], on=["pair_id", "hand_id"], how="left")
dh["hand_final"] = np.where(dh["hand_score2"].notna(), 10.0 + dh["hand_score2"], dh["hand_score"])
pos_all = dh[pos_mask]
oof_map5_stage1 = oof_map5
oof_map5 = map5_within_pairs(pos_all, score_col="hand_final")
fam_map5 = {fam: map5_within_pairs(pos_all[pos_all["behavior_family"] == fam], score_col="hand_final") for fam in TARGET_BEHAVIORS}
log(f"evidence stage 2: OOF MAP@5 {oof_map5_stage1:.4f} -> {oof_map5:.4f}  per family: { {k: round(v, 4) for k, v in fam_map5.items()} }")
imp2 = pd.Series(np.mean([m.feature_importance("gain") for m in pool_models_full], axis=0), index=POOL_FEATS).sort_values(ascending=False)
print("top stage-2 features (gain):\n", imp2.head(12).round(0).to_string())

# %% [markdown]
# ## 6. Pair aggregation → pair features for all development pairs → pair risk model + family heads

# %%
AGG_MEAN = ["both_vpip", "both_showdown", "partner_won_other_lost", "transfer_any", "net_gap", "pot_bb", "loser_stronger_preflop", "chen_gap",
            "fold_to_partner", "call_partner", "raise_partner", "fold_to_other", "call_other", "raise_other",
            "hu_actions", "hu_check", "hu_call", "hu_aggr", "hu_late_check", "outsider_fold_to_pair", "outsider_call_to_pair", "outsider_raise_to_pair",
            "n_partners_aggr", "pair_aggr", "pair_raise", "pair_overbets",
            "loser_beats_winner_final", "loser_beats_winner_flop", "loser_best_final", "winner_best_final", "fold_better_hand", "fold_better_to_partner", "both_made_no_aggr", "winner_weak_won", "cat_gap_final",
            "fold_to_partner_pf", "fold_to_partner_post", "raise_partner_pf", "outsider_fold_to_pair_pf", "pf_squeeze", "squeeze_then_fold_pf", "dump_better_hand", "hu_check_two_pair_plus",
            # step 3
            "both_vpip_a", "n_vpip_a", "n_pfr", "loose_sum", "loose_min", "loose_max", "tight_max", "loose_pfr_sum", "junk_vpip_sum", "junk_vpip_min", "junk_pfr_sum", "vpip_resid_sum",
            "second_entrant_loose", "both_vpip_junk", "both_vpip_one_junk", "second_entrant_junk", "both_surprising", "junk_raise",
            # step 8
            "surp_pair_sum", "surp_pair_min", "surp_pair_max", "surp_pf_sum", "surp_post_sum", "surp_post_min", "surp_fold_sum", "surp_call_sum", "surp_aggr_sum", "surp_check_sum", "n_surp2_sum", "both_surp2", "logp_mean_sum",
            "surp_vs_partner", "surp_vs_partner_max", "surp_vs_other", "surp_hu", "surp_fold_partner", "surp_call_partner", "surp_raise_partner", "surp_hu_check"]
AGG_MAX = ["transfer_any", "net_gap", "pot_bb", "outsider_fold_to_pair", "call_partner", "hu_check", "max_amount_bb", "loose_sum", "loose_min", "junk_vpip_sum",
           "surp_pair_sum", "surp_pair_min", "surp_vs_partner", "surp_post_min"]
AGG_TOP3 = ["transfer_any", "net_gap", "outsider_fold_to_pair", "hu_check", "call_partner", "loose_sum", "loose_min", "junk_vpip_sum", "junk_pfr_sum", "second_entrant_loose",
            "surp_pair_sum", "surp_pair_min", "surp_vs_partner", "surp_post_sum", "surp_post_min", "surp_fold_partner", "surp_hu"]
AGG_SUM = ["fold_better_to_partner", "dump_better_hand", "squeeze_then_fold_pf", "loser_beats_winner_final", "both_made_no_aggr",
           "both_vpip_a", "both_vpip_junk", "both_vpip_one_junk", "second_entrant_junk", "both_surprising", "junk_raise",
           "surp_vs_partner", "surp_fold_partner", "surp_call_partner", "both_surp2", "n_surp2_sum", "both_vpip_surp2"]
CONTRAST = ["net_bb", "n_aggr", "n_raise", "n_call", "n_fold_facing", "vpip", "vpip_a", "loose", "junk_vpip", "vpip_resid", "pfr", "surp_sum", "surp_max", "surp_post", "surp_pf", "surp_fold", "surp_call"]
TRANK_COLS = ["vpip_pf_sum", "p1_vpip_pf", "p2_vpip_pf", "vpip_pf_min", "n_call_pf_sum", "p1_n_call_pf", "p2_n_call_pf", "n_call_pf_min", "n_raise_pf_sum", "n_fold_facing_pf_min", "n_aggr_pf_sum", "net_bb_pf_gap",
              "hs_top5", "hs_top3", "hs_mean", "hs_n95", "transfer_rate", "transfer_dominant", "fold_better_to_partner_sum", "call_partner_asym", "hu_aggr_mean", "pair_aggr_mean", "call_other_mean",
              "loser_beats_winner_final_sum", "both_made_no_aggr_sum", "squeeze_then_fold_pf_sum", "pf_squeeze_mean", "outsider_fold_to_pair_pf_mean", "raise_partner_pf_mean", "hu_check_mean", "both_showdown_mean",
              "both_vpip_a_mean", "both_vpip_a_sum", "loose_sum_mean", "loose_sum_top3", "loose_min_top3", "junk_vpip_sum_mean", "junk_vpip_sum_top3", "second_entrant_junk_sum", "second_entrant_loose_top3",
              "both_vpip_junk_sum", "both_surprising_sum", "junk_raise_sum", "vpip_a_pf_sum", "loose_pf_sum", "junk_vpip_pf_sum", "vpip_resid_pf_sum", "vpip_a_pf_min", "loose_pf_min",
              "ev_n10", "ev_n50", "ev_f10", "hs_top10", "hs_sum",
              "both_vpip_vs_other", "both_junk_vs_other", "loose_sum_vs_other", "loose_min_vs_other", "resid_sum_vs_other", "junk_vpip_sum_vs_other", "junk_pfr_sum_vs_other", "transfer_vs_other",
              "loose_sum_ratio_other", "both_vpip_ratio_other",
              "surp_pair_sum_mean", "surp_pair_sum_top3", "surp_pair_min_top3", "surp_vs_partner_sum", "surp_vs_partner_top3", "surp_post_min_top3", "surp_sum_pf_sum", "surp_post_pf_sum", "surp_sum_pf_min",
              "surp_pair_vs_other", "surp_pair_min_vs_other", "surp_post_pair_vs_other", "surp_pair_ratio_other",
              "cosurp_z", "cosurp_excess", "cojunk_z", "cojunk_excess", "covpip_z", "covsurp_z", "covsurp_excess"]


def aggregate_pairs(hf: pl.DataFrame, baselines_phase: pl.DataFrame, pairs: pl.DataFrame) -> pd.DataFrame:
    """hf: polars pair-hand features with hand_score / hand_sus. Returns one row per pair (pandas)."""
    cols = ["pair_id", "hand_id", "table_id", "hand_idx", "player_1", "player_2", "hand_score", "hand_sus", "loser_is_p1", "signed_net_diff",
            "transfer_1_to_2", "transfer_2_to_1", "p1_call_partner", "p1_fold_to_partner"] + [f"p{i}_{c}" for i in (1, 2) for c in CONTRAST] + sorted(set(AGG_MEAN + AGG_MAX + AGG_TOP3 + AGG_SUM))
    lf = hf.select(cols).lazy()
    aggs = [pl.len().alias("n_shared"), pl.col("table_id").first(), pl.col("player_1").first(), pl.col("player_2").first()]
    aggs += [pl.col(c).mean().alias(f"{c}_mean") for c in AGG_MEAN]
    aggs += [pl.col(c).max().alias(f"{c}_max") for c in AGG_MAX]
    aggs += [pl.col(c).top_k(3).mean().alias(f"{c}_top3") for c in AGG_TOP3]
    aggs += [pl.col(c).sum().alias(f"{c}_sum") for c in AGG_SUM]
    aggs += [pl.col(f"p{i}_{c}").cast(pl.Float32).mean().alias(f"_p{i}_{c}") for i in (1, 2) for c in CONTRAST]
    aggs += [
        pl.col("hand_score").mean().alias("hs_mean"), pl.col("hand_score").max().alias("hs_max"),
        pl.col("hand_score").top_k(3).mean().alias("hs_top3"), pl.col("hand_score").top_k(5).mean().alias("hs_top5"), pl.col("hand_score").quantile(0.9).alias("hs_p90"),
        (pl.col("hand_score") > THR95).sum().alias("hs_n95"), (pl.col("hand_score") > THR99).sum().alias("hs_n99"), (pl.col("hand_score") > THR95).mean().alias("hs_f95"),
        (pl.col("hand_score") >= EV_THR10).sum().alias("ev_n10"), (pl.col("hand_score") >= EV_THR50).sum().alias("ev_n50"), (pl.col("hand_score") >= EV_THR10).mean().alias("ev_f10"),
        pl.col("hand_score").top_k(10).mean().alias("hs_top10"), pl.col("hand_score").sum().alias("hs_sum"),
        pl.col("hand_sus").mean().alias("sus_mean"), pl.col("hand_sus").max().alias("sus_max"),
        pl.col("hand_sus").top_k(3).mean().alias("sus_top3"), pl.col("hand_sus").top_k(5).mean().alias("sus_top5"), pl.col("hand_sus").quantile(0.9).alias("sus_p90"),
        (pl.col("hand_sus") > SUS95).sum().alias("sus_n95"), (pl.col("hand_sus") > SUS99).sum().alias("sus_n99"), (pl.col("hand_sus") > SUS95).mean().alias("sus_f95"),
        pl.col("transfer_1_to_2").sum().alias("_t12"), pl.col("transfer_2_to_1").sum().alias("_t21"),
        pl.col("signed_net_diff").sum().alias("_snd"), pl.col("signed_net_diff").abs().sum().alias("_asnd"),
        pl.col("p1_call_partner").sum().alias("_p1_cp"), pl.col("call_partner").sum().alias("_cp"),
        pl.col("p1_fold_to_partner").sum().alias("_p1_fp"), pl.col("fold_to_partner").sum().alias("_fp"),
        pl.col("loser_is_p1").sort_by("hand_score", descending=True).head(5).mean().alias("_top5_loser_p1"),
        pl.col("loser_is_p1").sort_by("hand_sus", descending=True).head(5).mean().alias("_top5s_loser_p1"),
        pl.col("loser_is_p1").sort_by("transfer_any", descending=True).head(5).mean().alias("_top5t_loser_p1"),
    ]
    pf = lf.group_by("pair_id").agg(aggs)
    burst = (
        lf.with_columns(((pl.col("hand_idx").cast(pl.Float32) - pl.col("hand_idx").min().over("pair_id")) / (pl.col("hand_idx").max().over("pair_id") - pl.col("hand_idx").min().over("pair_id") + 1) * 8).floor().clip(0, 7).cast(pl.Int8).alias("_bin"))
        .group_by(["pair_id", "_bin"]).agg(pl.col("hand_score").mean().alias("_b"), pl.col("hand_sus").mean().alias("_bs"), pl.col("transfer_any").mean().alias("_bt"))
        .group_by("pair_id").agg(pl.col("_b").max().alias("hs_burst_max"), (pl.col("_b").max() - pl.col("_b").mean()).alias("hs_burst_excess"),
                                 pl.col("_bs").max().alias("sus_burst_max"), (pl.col("_bs").max() - pl.col("_bs").mean()).alias("sus_burst_excess"),
                                 pl.col("_bt").max().alias("transfer_burst_max"))
    )
    pf = pf.join(burst, on="pair_id").collect()
    bl = baselines_phase
    for pfx, key in (("p1", "player_1"), ("p2", "player_2")):
        pf = pf.join(bl.rename({c: f"{pfx}_{c}" for c in bl.columns if c != "player_id"}).rename({"player_id": key}), on=key, how="left")
    # partner-vs-field contrasts: with-partner mean minus without-partner mean (from the player's phase totals), shrunk by exposure
    exprs = []
    for c in CONTRAST:
        for i in (1, 2):
            field = (pl.col(f"p{i}_base_{c}") * pl.col(f"p{i}_base_hands") - pl.col(f"_p{i}_{c}") * pl.col("n_shared")) / (pl.col(f"p{i}_base_hands") - pl.col("n_shared")).clip(lower_bound=1)
            exprs.append(((pl.col(f"_p{i}_{c}") - field) * pl.col("n_shared") / (pl.col("n_shared") + SHRINK_N)).cast(pl.Float32).alias(f"p{i}_{c}_pf"))
    pf = pf.with_columns(exprs).with_columns(
        *[(pl.col(f"p1_{c}_pf") + pl.col(f"p2_{c}_pf")).alias(f"{c}_pf_sum") for c in CONTRAST],
        *[(pl.col(f"p1_{c}_pf") - pl.col(f"p2_{c}_pf")).abs().alias(f"{c}_pf_gap") for c in CONTRAST],
        *[pl.min_horizontal(f"p1_{c}_pf", f"p2_{c}_pf").alias(f"{c}_pf_min") for c in CONTRAST],
        ((pl.col("_t12") - pl.col("_t21")).abs() / (pl.col("_t12") + pl.col("_t21") + 1.0)).alias("transfer_imbalance"),
        ((pl.col("_t12") + pl.col("_t21")) / pl.col("n_shared")).alias("transfer_rate"),
        pl.max_horizontal("_t12", "_t21").alias("transfer_dominant"),
        (pl.col("_snd").abs() / (pl.col("_asnd") + 1.0)).alias("direction_consistency"),
        pl.max_horizontal(pl.col("_top5_loser_p1"), 1 - pl.col("_top5_loser_p1")).alias("top5_same_loser"),
        pl.max_horizontal(pl.col("_top5s_loser_p1"), 1 - pl.col("_top5s_loser_p1")).alias("top5s_same_loser"),
        pl.max_horizontal(pl.col("_top5t_loser_p1"), 1 - pl.col("_top5t_loser_p1")).alias("top5t_same_loser"),
        (pl.max_horizontal(pl.col("_p1_cp"), pl.col("_cp") - pl.col("_p1_cp")) / (pl.col("_cp") + 1.0)).alias("call_partner_asym"),
        (pl.max_horizontal(pl.col("_p1_fp"), pl.col("_fp") - pl.col("_p1_fp")) / (pl.col("_fp") + 1.0)).alias("fold_partner_asym"),
        (pl.col("_p1_net_bb") - pl.col("p1_base_net_bb")).alias("p1_net_resid"), (pl.col("_p2_net_bb") - pl.col("p2_base_net_bb")).alias("p2_net_resid"),
    ).with_columns(
        (pl.col("p1_net_resid") - pl.col("p2_net_resid")).abs().alias("net_resid_gap"),
        pl.min_horizontal("p1_net_resid", "p2_net_resid").alias("net_resid_min"),
        pl.max_horizontal("p1_net_resid", "p2_net_resid").alias("net_resid_max"),
    )
    # in-phase pair rates that have an other-phase counterpart (Step 5)
    pf = pf.with_columns(
        pl.col("both_vpip_a_mean").alias("_ip_both_vpip"), pl.col("both_vpip_junk_mean").alias("_ip_both_junk"), pl.col("both_vpip_one_junk_mean").alias("_ip_one_junk"),
        pl.col("loose_sum_mean").alias("_ip_loose_sum"), pl.col("loose_min_mean").alias("_ip_loose_min"), pl.col("vpip_resid_sum_mean").alias("_ip_resid_sum"),
        pl.col("junk_vpip_sum_mean").alias("_ip_junk_vpip_sum"), pl.col("junk_pfr_sum_mean").alias("_ip_junk_pfr_sum"), pl.col("transfer_any_mean").alias("_ip_transfer"),
        pl.col("n_vpip_a_mean").alias("_ip_n_vpip"),
        pl.col("surp_pair_sum_mean").alias("_ip_surp_pair"), pl.col("surp_pair_min_mean").alias("_ip_surp_pair_min"), pl.col("surp_post_sum_mean").alias("_ip_surp_post_pair"),
    )
    pf = pf.join(pair_other, on="pair_id", how="left").with_columns(pl.col("ob_n").fill_null(0.0), *[pl.col(f"ob_{c}").fill_null(OB_MEDIAN[c]) for c in OB_STATS])
    OB_K = 30.0
    pf = pf.with_columns((pl.col("ob_n") / (pl.col("ob_n") + OB_K)).alias("_ob_w")).with_columns(
        *[(pl.col(f"_ip_{c}") - pl.col("_ob_w") * pl.col(f"ob_{c}") - (1 - pl.col("_ob_w")) * OB_MEDIAN[c]).cast(pl.Float32).alias(f"{c}_vs_other") for c in OB_STATS if c != "both_fold"],
        *[((pl.col(f"_ip_{c}") + 0.02) / (pl.col("_ob_w") * pl.col(f"ob_{c}") + (1 - pl.col("_ob_w")) * OB_MEDIAN[c] + 0.02)).cast(pl.Float32).alias(f"{c}_ratio_other") for c in ["both_vpip", "both_junk", "loose_sum", "junk_vpip_sum", "transfer", "surp_pair"]],
    )
    # step 9: coincidence tests against the independence expectation n·p1·p2 (player rates from the phase field)
    zex = []
    for obs, r1, r2, nm in [("both_surp2_sum", "surp2_any", "surp2_any", "cosurp"), ("both_vpip_junk_sum", "junk_entry", "junk_entry", "cojunk"),
                            ("both_vpip_a_sum", "vpip_a", "vpip_a", "covpip"), ("both_vpip_surp2_sum", "vpip_surp2", "vpip_surp2", "covsurp")]:
        exp_ = pl.col("n_shared") * pl.col(f"p1_base_{r1}") * pl.col(f"p2_base_{r2}")
        zex += [(pl.col(obs) - exp_).cast(pl.Float32).alias(f"{nm}_excess"), ((pl.col(obs) - exp_) / (exp_ + 0.5).sqrt()).cast(pl.Float32).alias(f"{nm}_z"),
                ((pl.col(obs) + 0.5) / (exp_ + 0.5)).cast(pl.Float32).alias(f"{nm}_ratio"), exp_.cast(pl.Float32).alias(f"{nm}_expected")]
    pf = pf.with_columns(zex)
    drop = [c for c in pf.columns if c.startswith("_") or c.startswith("p1_base_") or c.startswith("p2_base_")]
    pf = pf.drop(drop)
    # within-table percentile ranks of the strongest features ("how unusual is this pair for its pool")
    pf = pf.with_columns([(pl.col(c).rank(method="average").over("table_id") / pl.len().over("table_id")).cast(pl.Float32).alias(f"{c}_trank") for c in TRANK_COLS if c in pf.columns])
    pf = pf.join(pairs.select([c for c in pairs.columns if c in ("pair_id", "label", "behavior_family", "is_labeled", "fold", "shared_hands", "chunk")]), on="pair_id", how="left")
    return pf.to_pandas()


base_dev = baselines.filter(pl.col("phase") == "development").drop("phase")
DEV_PF = WORK_DIR / "dev_pair_features_v9.parquet"   # Step 9 aggregates are reused; MIL columns are merged below
if not DEV_PF.exists():
    parts = []
    for k in range(N_TABLE_CHUNKS):
        hf = pl.read_parquet(DEV_HF[k])
        fold_of_row = hf["table_id"].replace_strict(TABLE_FOLD, return_dtype=pl.Int8).to_numpy()
        X = hf.select(HAND_FEATS).to_pandas().astype("float32")
        hs = np.zeros(hf.height, np.float32); su = np.zeros(hf.height, np.float32)
        for f in range(N_FOLDS):
            m = fold_of_row == f
            if m.any():
                hs[m] = np.mean([mm.predict(X[m]) for mm in rank_models[f]], axis=0)
                su[m] = sus_models[f].predict(X[m])
        hf = hf.with_columns(pl.Series("hand_score", hs), pl.Series("hand_sus", su))
        for w, (lo, hi) in enumerate(TRAIN_WINDOWS):
            agg = aggregate_pairs(hf.filter((pl.col("hand_idx") >= lo) & (pl.col("hand_idx") < hi)), base_dev, dev_pairs)
            agg["window"] = w
            agg["pair_id_orig"] = agg["pair_id"]
            agg["pair_id"] = agg["pair_id"] + f"_w{w}"
            parts.append(agg)
            log(f"dev chunk {k} window {w} [{lo},{hi}): {len(agg):,} pairs (median shared hands {agg['n_shared'].median():.0f})")
        del hf, X
        gc.collect()
    dev_pf = pd.concat(parts, ignore_index=True)
    dev_pf.to_parquet(DEV_PF, index=False)
dev_pf = pd.read_parquet(DEV_PF)

# %% [markdown]
# ## 6b. Multiple-instance hand model, v2 (Step 13): hand of a coordinated pair vs. hand of an ordinary pair
# Trained from the **pair labels**: bags = (pair, window); positive bags start as weak positives, then the top-8 hands of each positive bag (by the
# out-of-fold score) become the positives and the rest weak negatives (three rounds). Negative bags = 12,000 clean + 3,000 hard unlabelled pairs, chosen
# with a quick pair-level OOF pass on the Step 9 pair features. Table-grouped folds throughout; development aggregates are out-of-fold by table.

# %%
MIL_DEV = [WORK_DIR / f"mil_dev_w{w}_v13.parquet" for w in range(len(TRAIN_WINDOWS))]
MIL_EVAL = WORK_DIR / "mil_eval_v13.parquet"
MIL_COLS = ["mil_top3", "mil_top8", "mil_mean", "mil_n50", "mil_n80", "mil_f50", "mil_sum", "mil_max"]
MIL_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=os.cpu_count())
MIL_ROUNDS, MIL_K, MIL_ITERS, MIL_CLEAN_PAIRS, MIL_HARD_PAIRS = 300, 8, 3, 12000, 3000
_META0 = {"pair_id", "table_id", "player_1", "player_2", "label", "behavior_family", "is_labeled", "fold", "shared_hands", "chunk", "pred_family", "window", "pair_id_orig"}
if not (all(p.exists() for p in MIL_DEV) and MIL_EVAL.exists()):
    # quick pair-level OOF (Step 9 features, one seed) to pick the negative bags
    _pre = [c for c in dev_pf.columns if c not in _META0 and not c.startswith("sus_") and c != "top5s_same_loser" and c not in MIL_COLS]
    _y0 = dev_pf["label"].fillna(0).astype(int).to_numpy(); _lab0 = dev_pf["is_labeled"].to_numpy(); _f0 = dev_pf["fold"].to_numpy()
    _w0 = np.where(_lab0, np.where(_y0 == 1, 5.0, 1.0), 0.1); _X0 = dev_pf[_pre].astype("float32"); _o0 = np.zeros(len(_y0))
    for f in range(N_FOLDS):
        tr, te = _f0 != f, _f0 == f
        _o0[te] = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=os.cpu_count()),
                            lgb.Dataset(_X0[tr], _y0[tr], weight=_w0[tr]), num_boost_round=400).predict(_X0[te])
    _q = pd.DataFrame({"pair_id": dev_pf["pair_id_orig"], "o": _o0, "lab": _lab0, "w": dev_pf["window"]}); _q = _q[_q["w"] == 0]
    log(f"MIL v2 negative selection: quick pair OOF labelled AP={average_precision_score(_y0[_lab0 & (dev_pf['window'] == 0).to_numpy()], _o0[_lab0 & (dev_pf['window'] == 0).to_numpy()]):.4f}")
    _rs = np.random.RandomState(SEED + 7)
    _clean = _q[(~_q["lab"]) & (_q["o"] < 0.001)]["pair_id"].to_numpy(); _hard = _q[(~_q["lab"]) & (_q["o"] >= 0.001) & (_q["o"] < 0.3)]["pair_id"].to_numpy()
    _keep = set(dev_pairs.filter(pl.col("is_labeled"))["pair_id"].to_list()) | set(_rs.choice(_clean, min(MIL_CLEAN_PAIRS, len(_clean)), replace=False)) | set(_rs.choice(_hard, min(MIL_HARD_PAIRS, len(_hard)), replace=False))
    _lab = dict(zip(dev_pairs["pair_id"], dev_pairs["label"].fill_null(0)))
    _hs = pl.concat([pl.scan_parquet(f).filter(pl.col("pair_id").is_in(list(_keep))).select(["pair_id", "hand_id", "table_id", "hand_idx", *HAND_FEATS]).collect() for f in DEV_HF])
    _hs = pl.concat([_hs.filter((pl.col("hand_idx") >= lo) & (pl.col("hand_idx") < hi)).with_columns(pl.lit(w, dtype=pl.Int8).alias("win")) for w, (lo, hi) in enumerate(TRAIN_WINDOWS)])
    _hs = _hs.with_columns(pl.col("pair_id").replace_strict(_lab, return_dtype=pl.Int8).alias("plabel"), pl.col("table_id").replace_strict(TABLE_FOLD, return_dtype=pl.Int8).alias("fold"),
                           (pl.col("pair_id") + "_" + pl.col("win").cast(pl.Utf8)).alias("bag"))
    log(f"MIL v2 hand sample {_hs.shape}; hands of coordinated pairs {(_hs['plabel'] == 1).sum():,}; bags {_hs['bag'].n_unique():,} ({len(_clean):,} clean / {len(_hard):,} hard unlabelled pairs available)")
    _X = _hs.select(HAND_FEATS).to_numpy().astype(np.float32); _pl = _hs["plabel"].to_numpy(); _fold = _hs["fold"].to_numpy(); _bag = _hs["bag"].to_numpy()
    _y = _pl.astype(int); _w = np.where(_pl == 1, 0.3, 1.0)
    for it in range(MIL_ITERS):
        _s = np.zeros(len(_y))
        for f in range(N_FOLDS):
            tr, te = _fold != f, _fold == f
            _s[te] = lgb.train(MIL_PARAMS, lgb.Dataset(_X[tr], _y[tr], weight=_w[tr]), num_boost_round=MIL_ROUNDS).predict(_X[te])
        _df = pd.DataFrame({"bag": _bag, "s": _s, "pl": _pl}); _df["rk"] = _df.groupby("bag")["s"].rank(ascending=False, method="first")
        _y = np.where((_df["pl"] == 1) & (_df["rk"] <= MIL_K), 1, 0); _w = np.where(_df["pl"] == 1, np.where(_df["rk"] <= MIL_K, 1.0, 0.2), 1.0)
        _agg = _df.groupby("bag").agg(top3=("s", lambda g: g.nlargest(3).mean()), pl=("pl", "first"))
        log(f"MIL v2 round {it + 1}: bag AP from the top-3 mean hand score (all sampled bags) = {average_precision_score(_agg['pl'], _agg['top3']):.4f}")
    mil_models = {f: lgb.train(MIL_PARAMS, lgb.Dataset(_X[_fold != f], _y[_fold != f], weight=_w[_fold != f]), num_boost_round=MIL_ROUNDS) for f in range(N_FOLDS)}
    mil_full = lgb.train(MIL_PARAMS, lgb.Dataset(_X, _y, weight=_w), num_boost_round=MIL_ROUNDS)
    del _X, _hs, _X0; gc.collect()

    def mil_aggregate(lf: pl.LazyFrame, oof_by_table: bool) -> pl.DataFrame:
        parts = []; n = lf.select(pl.len()).collect().item()
        for st in range(0, n, 1_500_000):
            c = lf.slice(st, 1_500_000).collect(); Xc = c.select(HAND_FEATS).to_numpy().astype(np.float32); sc = np.zeros(c.height)
            if oof_by_table:
                fo = c["table_id"].replace_strict(TABLE_FOLD, return_dtype=pl.Int8).to_numpy()
                for f in range(N_FOLDS):
                    m = fo == f
                    if m.any(): sc[m] = mil_models[f].predict(Xc[m])
            else:
                sc = mil_full.predict(Xc)
            parts.append(c.select(["pair_id"]).with_columns(pl.Series("mil", sc)))
        return pl.concat(parts).group_by("pair_id").agg(
            pl.col("mil").top_k(3).mean().alias("mil_top3"), pl.col("mil").top_k(8).mean().alias("mil_top8"), pl.col("mil").mean().alias("mil_mean"),
            (pl.col("mil") > 0.5).sum().alias("mil_n50"), (pl.col("mil") > 0.8).sum().alias("mil_n80"), (pl.col("mil") > 0.5).mean().alias("mil_f50"),
            pl.col("mil").sum().alias("mil_sum"), pl.col("mil").max().alias("mil_max"))

    for w, (lo, hi) in enumerate(TRAIN_WINDOWS):
        mil_aggregate(pl.concat([pl.scan_parquet(f) for f in DEV_HF]).filter((pl.col("hand_idx") >= lo) & (pl.col("hand_idx") < hi)), True).write_parquet(MIL_DEV[w])
        log(f"MIL v2 aggregates for development window {w} written")
    mil_aggregate(pl.scan_parquet(EVAL_HF), False).write_parquet(MIL_EVAL)
    log("MIL v2 aggregates for evaluation written")
    del mil_models, mil_full; gc.collect()
_mil = [pd.read_parquet(p).set_index("pair_id") for p in MIL_DEV]
_vals = np.stack([_mil[w].reindex(dev_pf["pair_id_orig"]).to_numpy() for w in range(len(TRAIN_WINDOWS))])   # (windows, rows, cols)
_wi = dev_pf["window"].to_numpy()
for i, c in enumerate(MIL_COLS):
    dev_pf[c] = _vals[_wi, np.arange(len(dev_pf)), i]
log(f"MIL columns merged into the development pair features (missing: {int(dev_pf[MIL_COLS[0]].isna().sum())})")
META = {"pair_id", "table_id", "player_1", "player_2", "label", "behavior_family", "is_labeled", "fold", "shared_hands", "chunk", "pred_family", "window", "pair_id_orig"}
PAIR_FEATS = [c for c in dev_pf.columns if c not in META and not c.startswith("sus_") and c != "top5s_same_loser"]   # sus_* aggregates hurt on the honest population (ablation)
log(f"dev pair features: {dev_pf.shape}; {len(PAIR_FEATS)} features")

# --- pair risk model on ALL eligible development pairs -----------------------------------
y = dev_pf["label"].fillna(0).astype(int).to_numpy()
labm = dev_pf["is_labeled"].to_numpy()
w_train = np.where(labm, np.where(y == 1, POS_WEIGHT, NEG_WEIGHT), PU_WEIGHT)
folds = dev_pf["fold"].to_numpy()
PAIR_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=os.cpu_count())
PAIR_SEEDS = [SEED, SEED + 11, SEED + 23]
PAIR_ROUNDS = 750
X = dev_pf[PAIR_FEATS].astype("float32")


def fit_pair_cv(y_fit, w_fit, tag, seeds=PAIR_SEEDS, rounds=PAIR_ROUNDS, with_family=True):
    """Grouped 5-fold CV of the pair ranker (+ family heads). Returns OOF risk, models, OOF family probabilities, family models."""
    oof_ = np.zeros(len(dev_pf)); models = []
    fam_m = {fam: [] for fam in TARGET_BEHAVIORS}; oof_f = {fam: np.zeros(len(dev_pf)) for fam in TARGET_BEHAVIORS}
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        tr_w = tr & (w_fit > 0)
        for sd in seeds:
            m = lgb.train({**PAIR_PARAMS, "seed": sd}, lgb.Dataset(X[tr_w], y_fit[tr_w], weight=w_fit[tr_w]), num_boost_round=rounds)
            oof_[te] += m.predict(X[te]) / len(seeds)
            models.append(m)
        if with_family:
            tr_f = tr_w & ~((y_fit == 1) & ~labm)   # pseudo-positives have no known family -> excluded from the family heads
            for fam in TARGET_BEHAVIORS:
                yf = (dev_pf["behavior_family"] == fam).to_numpy().astype(int)
                mf = lgb.train({**PAIR_PARAMS, "num_leaves": 7}, lgb.Dataset(X[tr_f], yf[tr_f], weight=w_fit[tr_f]), num_boost_round=400)
                oof_f[fam][te] = mf.predict(X[te])
                fam_m[fam].append(mf)
        lab = labm & te
        log(f"{tag} fold {f}: labelled AP={average_precision_score(y[lab], oof_[lab]):.4f} | eval-mirrored AP (all pairs)={average_precision_score(y[te], oof_[te]):.4f}")
    return oof_, models, oof_f, fam_m


# ---- stage 1: plain PU (unlabelled = weak negatives) ------------------------------------------------
oof1, _, _, _ = fit_pair_cv(y, w_train, "stage-1 pair model", seeds=PAIR_SEEDS[:1], rounds=PAIR_ROUNDS, with_family=False)
# ---- stage 2: two-step PU -----------------------------------------------------------------------------
# Unlabelled pairs that score like confirmed colluders are (by the host's own statement) unknown, not clean. On the development population they have the same
# feature profile as the labelled positives (same both-entered rate, junk-entry count and transfer rate), so they are removed from the negative set:
#   * score >= 50th percentile of the labelled positives' OOF  -> pseudo-positive (weight PSEUDO_POS_WEIGHT)
#   * score >= 5th percentile of the labelled positives' OOF   -> ambiguous, excluded from training (weight 0)
#   * otherwise                                                -> reliable negative (weight PU_WEIGHT)
PSEUDO_POS_WEIGHT = 1.0
AMBIG_WEIGHT = 0.2      # step 8: "ambiguous" unlabelled pairs are mostly loose-bot negatives (≈150 hidden colluders in 120k pairs) -> keep them as weak negatives
pos_oof = oof1[labm & (y == 1)]
thr_ambig, thr_pseudo = np.quantile(pos_oof, 0.05), np.quantile(pos_oof, 0.50)
unl = ~labm
ambiguous = unl & (oof1 >= thr_ambig) & (oof1 < thr_pseudo)
pseudo = unl & (oof1 >= thr_pseudo)
print(f"two-step PU: thresholds ambiguous>={thr_ambig:.4f} pseudo-positive>={thr_pseudo:.4f} | unlabelled: {int(pseudo.sum())} pseudo-positives, {int(ambiguous.sum())} excluded, {int((unl & ~ambiguous & ~pseudo).sum())} reliable negatives")
y2 = np.where(pseudo, 1, y)
w_train2 = np.where(labm, np.where(y == 1, POS_WEIGHT, NEG_WEIGHT), np.where(pseudo, PSEUDO_POS_WEIGHT, np.where(ambiguous, AMBIG_WEIGHT, PU_WEIGHT)))
oof2_, _, _, _ = fit_pair_cv(y2, w_train2, "stage-2 pair model", seeds=PAIR_SEEDS[:1], with_family=False)
# ---- stage 3: repeat with the stage-2 scores and a wider pseudo-positive net (25th percentile of the labelled positives) --------
pos_oof2 = oof2_[labm & (y == 1)]
thr_ambig3, thr_pseudo3 = np.quantile(pos_oof2, 0.05), np.quantile(pos_oof2, 0.25)
ambiguous = unl & (oof2_ >= thr_ambig3) & (oof2_ < thr_pseudo3)
pseudo = unl & (oof2_ >= thr_pseudo3)
print(f"stage 3 PU: thresholds ambiguous>={thr_ambig3:.4f} pseudo-positive>={thr_pseudo3:.4f} | unlabelled: {int(pseudo.sum())} pseudo-positives, {int(ambiguous.sum())} excluded")
y2 = np.where(pseudo, 1, y)
w_train2 = np.where(labm, np.where(y == 1, POS_WEIGHT, NEG_WEIGHT), np.where(pseudo, PSEUDO_POS_WEIGHT, np.where(ambiguous, AMBIG_WEIGHT, PU_WEIGHT)))
oof, pair_models, oof_fam, fam_models = fit_pair_cv(y2, w_train2, "stage-3 pair model", seeds=PAIR_SEEDS + [SEED + 37, SEED + 51])
clean = ~ambiguous & ~pseudo
print(f"stage-1 eval-mirrored AP {average_precision_score(y, oof1):.4f} -> stage-2 {average_precision_score(y, oof):.4f} (all unlabelled as negatives, pessimistic)")
print(f"stage-2 AP on the population without the {int((ambiguous | pseudo).sum())} colluder-like unlabelled pairs: {average_precision_score(y[clean], oof[clean]):.4f}")
print(f"stage-2 AP if pseudo-positives are counted as positives: {average_precision_score(y2[~ambiguous], oof[~ambiguous]):.4f}")

pd.DataFrame({"pair_id": dev_pf["pair_id"], "pair_id_orig": dev_pf["pair_id_orig"], "window": dev_pf["window"], "label": y, "is_labeled": labm, "oof": oof,
              **{f"oof_{fam}": oof_fam[fam] for fam in TARGET_BEHAVIORS}}).to_parquet(WORK_DIR / "dev_oof_v13.parquet", index=False)
labelled_ap = average_precision_score(y[labm], oof[labm])
pu_ap = average_precision_score(y, oof)
print(f"\nOOF labelled Pair AP (positives vs confirmed negatives, OPTIMISTIC): {labelled_ap:.4f}")
print(f"OOF eval-mirrored Pair AP (positives vs ALL eligible pairs, tracks LB): {pu_ap:.4f}")
for fam in TARGET_BEHAVIORS:
    sel = (dev_pf["behavior_family"] == fam).to_numpy() | (y == 0)
    print(f"   {fam:22s} eval-mirrored AP = {average_precision_score(y[sel], oof[sel]):.4f}")

fam_prior = {fam: (dev_pf["behavior_family"] == fam).sum() for fam in TARGET_BEHAVIORS}
def predict_family(fam_probs: dict) -> np.ndarray:
    mat = np.column_stack([fam_probs[fam] / fam_prior[fam] for fam in TARGET_BEHAVIORS])
    return np.array(TARGET_BEHAVIORS)[mat.argmax(axis=1)]
dev_pf["pred_family"] = predict_family(oof_fam)
pos = dev_pf["label"] == 1
print(f"family accuracy on positives: {(dev_pf.loc[pos, 'pred_family'] == dev_pf.loc[pos, 'behavior_family']).mean():.3f}")

# host metric on the full development population (unlabelled pairs count as negatives; evidence from the OOF ranker on labelled pairs)
top5 = (dh.sort_values(["pair_id", "hand_final", "pot_bb", "hand_id"], ascending=[True, False, False, True]).groupby("pair_id")["hand_id"].apply(lambda s: list(s.head(5))))
W0 = (dev_pf["window"] == 0).to_numpy()
sol = pd.DataFrame({"pair_id": dev_pf.loc[W0, "pair_id_orig"].to_numpy(), "risk_score": y[W0], "predicted_behavior": np.where(y[W0] == 1, dev_pf.loc[W0, "behavior_family"], "none")})
ev_map = dev_evidence.to_pandas().groupby("pair_id")["hand_id"].apply(list)
for i, c in enumerate(EVIDENCE_COLUMNS):
    sol[c] = [(ev_map[p][i] if (p in ev_map.index and i < len(ev_map[p])) else NO_EVIDENCE) for p in sol["pair_id"]]
sub_dev = pd.DataFrame({"pair_id": dev_pf.loc[W0, "pair_id_orig"].to_numpy(), "risk_score": oof[W0], "predicted_behavior": dev_pf.loc[W0, "pred_family"].to_numpy()})
for i, c in enumerate(EVIDENCE_COLUMNS):
    sub_dev[c] = [(top5[p][i] if (p in top5.index and i < len(top5[p])) else NO_EVIDENCE) for p in sub_dev["pair_id"]]
comp = host_score(sol, sub_dev, return_components=True)
print("host metric on the full development population (family for every pair):", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in comp.items()}, default=str))
# behaviour `none` rule: a family is predicted only for the top-rho fraction of risk; everything else is `none` (class score 0 for all three families)
rank_pct = pd.Series(oof[W0]).rank(ascending=False, method="first").to_numpy() / int(W0.sum())
NONE_GRID = [1.0, 0.10, 0.05, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.004, 0.003]
none_scores = {}
for rho in NONE_GRID:
    sd = sub_dev.copy()
    sd["predicted_behavior"] = np.where(rank_pct <= rho, dev_pf.loc[W0, "pred_family"].to_numpy(), "none")
    none_scores[rho] = host_score(sol, sd, return_components=True)
    print(f"  rho={rho:<7} final={none_scores[rho]['final']:.4f} behavior_map={none_scores[rho]['behavior_map']:.4f}")
_best = max(v["final"] for v in none_scores.values())
BEHAVIOR_RHO = max(r for r in NONE_GRID if r <= 0.05 and none_scores[r]["final"] >= _best - 0.0005)   # largest rate that is not measurably worse: safety margin for a higher evaluation prevalence
comp = none_scores[BEHAVIOR_RHO]
print(f"selected behaviour rate rho={BEHAVIOR_RHO} -> host metric on the development population: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in comp.items()}, default=str)}")
imp = pd.Series(np.mean([m.feature_importance("gain") for m in pair_models], axis=0), index=PAIR_FEATS).sort_values(ascending=False)
print("top pair features (gain):\n", imp.head(25).round(0).to_string())

# %% [markdown]
# ## 7. Evaluation inference: hand scores → pair features → risk, family, evidence

# %%
EVAL_SCORED = WORK_DIR / "eval_hand_scored_v9.parquet"
if not EVAL_SCORED.exists():
    parts = []
    lf = pl.scan_parquet(EVAL_HF)
    n = lf.select(pl.len()).collect().item()
    CH = 2_000_000
    for start in range(0, n, CH):
        chunk = lf.slice(start, CH).collect()
        Xc = chunk.select(HAND_FEATS).to_pandas().astype("float32")
        s1 = np.mean([m.predict(Xc) for m in rank_models_full], axis=0)
        s2 = sus_model_full.predict(Xc)
        parts.append(chunk.with_columns(pl.Series("hand_score", s1, dtype=pl.Float32), pl.Series("hand_sus", s2, dtype=pl.Float32)))
        log(f"scored eval hands {min(start + CH, n):,}/{n:,}")
    pl.concat(parts).write_parquet(EVAL_SCORED)
    del parts, chunk, Xc
    gc.collect()
eh = pl.read_parquet(EVAL_SCORED)
base_eval = baselines.filter(pl.col("phase") == "evaluation").drop("phase")
eval_pf = aggregate_pairs(eh, base_eval, eval_prep)
eval_pf = eval_pf.merge(pd.read_parquet(MIL_EVAL), on="pair_id", how="left")
assert len(eval_pf) == 112_540
eval_pf.to_parquet(WORK_DIR / "eval_pair_features_v13.parquet", index=False)
Xe = eval_pf[PAIR_FEATS].astype("float32")
risk = np.mean([m.predict(Xe) for m in pair_models], axis=0)
fam_probs = {fam: np.mean([m.predict(Xe) for m in fam_models[fam]], axis=0) for fam in TARGET_BEHAVIORS}
eval_pf["risk_score"] = risk
pd.DataFrame({"pair_id": eval_pf["pair_id"], "risk": risk, **{f"p_{fam}": fam_probs[fam] for fam in TARGET_BEHAVIORS}}).to_parquet(WORK_DIR / "eval_risk_v13.parquet", index=False)
eval_rank_pct = pd.Series(risk).rank(ascending=False, method="first").to_numpy() / len(risk)
eval_pf["predicted_behavior"] = np.where(eval_rank_pct <= BEHAVIOR_RHO, predict_family(fam_probs), "none")

eval_fam = pl.DataFrame({"pair_id": eval_pf["pair_id"].to_numpy(), "_fam": predict_family(fam_probs)})
pool_eval = pool_features(eh.select(["pair_id", "hand_id", "hand_idx", "hand_score", "pot_bb"] + [c for c in HAND_FEATS if c != "pot_bb"]).join(eval_fam, on="pair_id", how="left")
                          .with_columns([(pl.col("_fam") == f).cast(pl.Float32).alias(f"fam_{f}") for f in TARGET_BEHAVIORS]).drop("_fam"))
_keys_e = pool_eval.select(["pair_id", "hand_id"]).join(eh.select(["pair_id", "hand_id", "player_1", "player_2", "loser_is_p1"]), on=["pair_id", "hand_id"]).join(eval_fam, on="pair_id", how="left")
_tpl_e = hand_templates(_keys_e.select(["pair_id", "hand_id", "player_1", "player_2", "loser_is_p1"]), "evaluation").join(_keys_e.select(["pair_id", "hand_id", "_fam"]), on=["pair_id", "hand_id"])
_te_e = {}
for c in TPL_COLS:
    _te_e[f"te_{c}"] = te_encode(*TE_MAPS[c], _tpl_e[c].to_numpy()); _te_e[f"te_{c}_fam"] = te_encode(*TE_MAPS_FAM[c], (_tpl_e["_fam"].cast(pl.Utf8) + "|" + _tpl_e[c]).to_numpy())
    _te_e[f"n_{c}"] = (_tpl_e[c].str.count_matches(" ") + (_tpl_e[c] != "").cast(pl.UInt32)).to_numpy().astype(np.float32)
_tpl_e = _tpl_e.select(["pair_id", "hand_id"]).with_columns([pl.Series(k, v) for k, v in _te_e.items()])
pool_eval = pool_eval.join(_tpl_e, on=["pair_id", "hand_id"], how="left").with_columns([pl.col(c).fill_null(0.0) for c in TE_FEATS])
log(f"evaluation pool templates encoded for {pool_eval.height:,} hands")
s2 = np.mean([m.predict(pool_eval.select(POOL_FEATS).to_pandas().astype("float32")) for m in pool_models_full], axis=0)
pool_eval = pool_eval.with_columns(pl.Series("hand_score2", s2, dtype=pl.Float32))
eh = eh.join(pool_eval.select(["pair_id", "hand_id", "hand_score2"]), on=["pair_id", "hand_id"], how="left").with_columns(
    pl.when(pl.col("hand_score2").is_not_null()).then(10.0 + pl.col("hand_score2")).otherwise(pl.col("hand_score")).cast(pl.Float32).alias("hand_final"))
log(f"evidence stage 2 applied to {pool_eval.height:,} pool hands of {eh['pair_id'].n_unique():,} pairs")
top = (
    eh.select(["pair_id", "hand_id", pl.col("hand_final").alias("hand_score"), "pot_bb"])
    .sort(["pair_id", "hand_score", "pot_bb", "hand_id"], descending=[False, True, True, False])
    .group_by("pair_id", maintain_order=True).head(5)
    .with_columns(pl.int_range(0, pl.len()).over("pair_id").alias("rank"))
    .pivot(on="rank", index="pair_id", values="hand_id")
)
top = top.rename({str(i): f"evidence_hand_{i+1}" for i in range(5)}) if "0" in top.columns else top.rename({i: f"evidence_hand_{i+1}" for i in range(5)})
log(f"evidence lists built for {top.height:,} pairs")

# %% [markdown]
# ## 8. Build and validate `submission.csv`

# %%
sub = eval_pf[["pair_id", "risk_score", "predicted_behavior"]].merge(top.to_pandas(), on="pair_id", how="left")
# no ties: round to 9 dp, then separate exact duplicates by a sub-grid epsilon ordered by a gameplay key (max hand score)
sub["risk_score"] = sub["risk_score"].clip(0, 1).round(9)
sec = eval_pf.set_index("pair_id").loc[sub["pair_id"], "hs_max"].to_numpy()
order = pd.DataFrame({"r": sub["risk_score"], "sec": sec}).sort_values(["r", "sec"], ascending=[True, False], kind="mergesort")
dup_rank = order.groupby("r").cumcount().reindex(sub.index)
sub["risk_score"] = np.clip(sub["risk_score"] + 1e-13 * dup_rank.to_numpy(), 0, 1)
assert sub["risk_score"].is_unique, "risk_score still has ties"
for c in EVIDENCE_COLUMNS:
    sub[c] = sub[c].fillna(NO_EVIDENCE).astype(str)
sub = sample_sub.select("pair_id").to_pandas().merge(sub, on="pair_id", how="left")

# --- validation --------------------------------------------------------------------------
assert len(sub) == 112_540 and sub["pair_id"].is_unique
assert sub.isna().sum().sum() == 0
assert sub["risk_score"].between(0, 1).all()
assert set(sub["predicted_behavior"]) <= set(TARGET_BEHAVIORS) | {"none"}
assert (sub[list(EVIDENCE_COLUMNS)] == NO_EVIDENCE).sum().sum() == 0, "every pair has >= 38 shared hands; no NO_EVIDENCE expected"
ev_long = sub.melt(id_vars="pair_id", value_vars=list(EVIDENCE_COLUMNS), value_name="hand_id")
assert not ev_long.duplicated(["pair_id", "hand_id"]).any(), "duplicate evidence hand within a pair"
shared_keys = eh.select(["pair_id", "hand_id"]).to_pandas()
assert ev_long.merge(shared_keys, on=["pair_id", "hand_id"], how="left", indicator=True)["_merge"].eq("both").all(), "evidence hand not a shared evaluation hand"
print("risk ties:", sub["risk_score"].duplicated().sum(), "| behaviours:", sub["predicted_behavior"].value_counts().to_dict())
# the host metric must accept the file (fake solution)
fake = sub[["pair_id"]].copy()
rs = np.random.RandomState(0)
fake["risk_score"] = (rs.rand(len(fake)) < 0.003).astype(int)
fake["predicted_behavior"] = np.where(fake["risk_score"] == 1, rs.choice(TARGET_BEHAVIORS, len(fake)), "none")
for c in EVIDENCE_COLUMNS:
    fake[c] = np.where(fake["risk_score"] == 1, sub[c], NO_EVIDENCE)
print("host metric accepts the file; score vs a random fake solution:", round(host_score(fake, sub), 4))

sub.to_csv(SUB_PATH, index=False)
chk = pd.read_csv(SUB_PATH, dtype={c: str for c in EVIDENCE_COLUMNS})
assert chk.shape == (112_540, 8)
log(f"wrote {SUB_PATH} ({SUB_PATH.stat().st_size/1e6:.1f} MB)")
print(chk.head(3).to_string())
print(json.dumps({"behaviour_rho": BEHAVIOR_RHO, "oof_labelled_pair_ap": round(labelled_ap, 4), "oof_eval_mirrored_pair_ap": round(pu_ap, 4), "oof_evidence_map5_stage1": round(oof_map5_stage1, 4), "oof_evidence_map5": round(oof_map5, 4),
                  "evidence_map5_by_family": {k: round(v, 4) for k, v in fam_map5.items()}, "dev_population_host_metric": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in comp.items()}}, indent=2, default=str))
