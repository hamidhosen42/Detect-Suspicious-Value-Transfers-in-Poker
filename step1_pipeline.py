# %% [markdown]
# # Step 1 — Evidence-first baseline for *Detect Suspicious Value Transfers in Poker*
#
# **What this notebook does (one `Run All`, CPU only, ~30–45 min on Kaggle):**
#
# 1. Builds an ordered **action context** (`last_aggressor`, facing-partner flags, true heads-up flags).
# 2. Builds **player-hand** features (chips in BB, hole-card strength, action counts) and per-player phase baselines.
# 3. Selects **development pairs** = 1,860 labelled pairs + a PU sample of unlabelled pairs (60 per table, ≥ 57 shared hands, no publicly-positive player), and **all 112,540 evaluation pairs**.
# 4. Builds **pair-conditional hand features** for every (pair, shared hand): chip transfer between the partners, who lost, fold/call/raise *to the partner* vs *to outsiders*, outsider folds after partner aggression, heads-up check-downs, hole-card strength of loser vs winner.
# 5. Trains a **hand-level evidence model** on the planted evidence hands (GroupKFold by table) → out-of-fold hand scores → **MAP@5** on development.
# 6. Aggregates hand features + hand scores into **pair features**, trains the **pair risk model** (PU-weighted) and three **family heads**.
# 7. Reports the honest CV numbers: *labelled* AP (positives vs confirmed negatives — optimistic) and *PU-weighted* AP (positives vs the unlabelled population — tracks the leaderboard).
# 8. Writes `/kaggle/working/submission.csv` with **5 evidence hands for every pair, never `none`, no tied risk scores**, and validates it with the host's reference metric.
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
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
pl.Config.set_tbl_cols(40)

SEED = 42
N_FOLDS = 5
UNKNOWN_PER_TABLE = 60      # PU sample size per table
PU_WEIGHT = 0.35            # training weight of unlabelled (assumed-negative) pairs
POS_WEIGHT = 2.0            # training weight of confirmed positives
NEG_WEIGHT = 1.0            # training weight of confirmed non-targets

CANDIDATES = [
    Path("/kaggle/input/competitions/detect-suspicious-value-transfers-in-poker"),
    Path("/kaggle/input/detect-suspicious-value-transfers-in-poker"),
    Path(os.environ.get("POKER_DATA_DIR", ".")),
]
DATA_DIR = next(p for p in CANDIDATES if (p / "hands.parquet").exists())
WORK_DIR = Path("/kaggle/working/step1") if Path("/kaggle/working").exists() else Path(os.environ.get("POKER_WORK_DIR", "./work_step1"))
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
    pl.read_parquet(DATA_DIR / "hands.parquet", columns=["hand_id", "table_id", "started_at", "phase", "big_blind", "final_pot", "players_at_showdown"])
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
    )
    seats = pl.scan_parquet(DATA_DIR / "seats.parquet")
    cards = seats.select(["hole_card_1", "hole_card_2"]).unique().collect()
    cards = cards.with_columns(pl.Series("chen", [chen(a, b) for a, b in zip(cards["hole_card_1"], cards["hole_card_2"])], dtype=pl.Float32))
    (
        seats.join(hands.lazy().select(["hand_id", "table_id", "phase", "hand_idx", "big_blind", "final_pot", "players_at_showdown"]), on="hand_id")
        .join(cards.lazy(), on=["hole_card_1", "hole_card_2"])
        .with_columns(
            (pl.col("final_pot") / pl.col("big_blind")).cast(pl.Float32).alias("pot_bb"),
            (pl.col("total_contribution") / pl.col("big_blind")).cast(pl.Float32).alias("contrib_bb"),
            (pl.col("net_chips") / pl.col("big_blind")).cast(pl.Float32).alias("net_bb"),
            (pl.col("total_contribution") > pl.col("big_blind")).alias("vpip"),
        )
        .join(act_by_player, on=["hand_id", "player_id"], how="left")
        .with_columns([pl.col(c).fill_null(0) for c in ["n_actions", "n_aggr", "n_raise", "n_call", "n_check", "n_allin", "n_fold_facing", "n_postflop_check", "n_hu_actions", "last_street", "max_amount_bb", "max_to_call_bb"]])
        .select(["hand_id", "player_id", "table_id", "phase", "hand_idx", "big_blind", "pot_bb", "players_at_showdown", "seat_no",
                 "contrib_bb", "net_bb", "vpip", "folded", "went_to_showdown", "won_share", "chen",
                 "n_actions", "n_aggr", "n_raise", "n_call", "n_check", "n_allin", "n_fold_facing", "n_postflop_check", "n_hu_actions", "last_street", "max_amount_bb", "max_to_call_bb"])
        .sink_parquet(PLAYER_HANDS)
    )
    del cards
    gc.collect()

BASELINE_COLS = ["net_bb", "vpip", "n_aggr", "n_raise", "n_call", "n_fold_facing", "went_to_showdown", "contrib_bb"]
baselines = (
    pl.scan_parquet(PLAYER_HANDS)
    .group_by(["player_id", "phase"])
    .agg(pl.len().alias("base_hands"), *[pl.col(c).cast(pl.Float32).mean().alias(f"base_{c}") for c in BASELINE_COLS])
    .collect()
)
log(f"player-hand rows: {pl.scan_parquet(PLAYER_HANDS).select(pl.len()).collect().item():,}; baselines: {baselines.height:,}")

# %% [markdown]
# ## 3. Pair selection: labelled + PU sample (development) and all evaluation pairs

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
    dev_counts = dev_all.group_by(["p_low", "p_high"]).agg(pl.col("table_id").first(), pl.len().cast(pl.Int32).alias("shared_hands")).collect(engine="streaming").sort(["table_id", "p_low", "p_high"])
    dev_counts = dev_counts.join(lab.select(["pair_id", "p_low", "p_high", "label", "behavior_family"]), on=["p_low", "p_high"], how="left")

    min_shared = int(np.ceil(eval_pairs["shared_hands"].min() * n_dev_per_table / (5000 - n_dev_per_table)))  # 38/2000 -> 57/3000
    known = dev_counts.filter(pl.col("pair_id").is_not_null()).with_columns(
        pl.lit(True).alias("is_labeled"), pl.lit(1.0, dtype=pl.Float32).alias("population_weight"))
    unlabeled = dev_counts.filter(
        pl.col("pair_id").is_null() & (pl.col("shared_hands") >= min_shared)
        & ~pl.col("p_low").is_in(list(positive_players)) & ~pl.col("p_high").is_in(list(positive_players)))
    pop_n = unlabeled.group_by("table_id").len().rename({"len": "_pop_n"})
    unknown = (
        unlabeled.sort(["table_id", "p_low", "p_high"]).sample(fraction=1.0, shuffle=True, seed=SEED)   # sort first: deterministic PU sample
        .group_by("table_id", maintain_order=True).head(UNKNOWN_PER_TABLE)
    )
    samp_n = unknown.group_by("table_id").len().rename({"len": "_samp_n"})
    unknown = (
        unknown.join(pop_n, on="table_id").join(samp_n, on="table_id")
        .with_columns(
            pl.concat_str([pl.lit("U"), "p_low", "p_high"], separator="_").alias("pair_id"),
            pl.lit(0, dtype=pl.Int64).alias("label"), pl.lit("unknown").alias("behavior_family"),
            pl.lit(False).alias("is_labeled"),
            (pl.col("_pop_n") / pl.col("_samp_n")).cast(pl.Float32).alias("population_weight"),
        ).drop(["_pop_n", "_samp_n"])
    )
    dev_pairs = pl.concat([known, unknown], how="diagonal_relaxed").sort(["table_id", "pair_id"])
    dev_pairs.write_parquet(DEV_PAIRS)
    (
        dev_all.join(dev_pairs.lazy().select(["pair_id", "p_low", "p_high"]), on=["p_low", "p_high"], how="inner")
        .select(["pair_id", "hand_id", "table_id", "hand_idx", pl.col("p_low").alias("player_1"), pl.col("p_high").alias("player_2")])
        .sink_parquet(DEV_PAIR_HANDS)
    )
    del dev_all, dev_counts, known, unlabeled, unknown
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
log(f"dev pairs: {dev_pairs.height:,} (labelled {dev_pairs['is_labeled'].sum():,}, PU {(~dev_pairs['is_labeled']).sum():,}); dev pair-hands: {pl.scan_parquet(DEV_PAIR_HANDS).select(pl.len()).collect().item():,}")
log(f"eval pairs: {eval_prep.height:,}; eval pair-hands: {pl.scan_parquet(EVAL_PAIR_HANDS).select(pl.len()).collect().item():,}; min shared {eval_prep['shared_hands'].min()}")

# %% [markdown]
# ## 4. Pair-conditional hand features
# One row per (pair, shared hand). Everything is expressed *relative to the pair*: which partner lost, whether chips moved from one partner to the other, whether a partner folded/called/raised **against the partner** versus against outsiders, whether outsiders folded to the pair's aggression, and what each partner was holding.

# %%
P_COLS = ["seat_no", "contrib_bb", "net_bb", "vpip", "folded", "went_to_showdown", "won_share", "chen",
          "n_actions", "n_aggr", "n_raise", "n_call", "n_check", "n_allin", "n_fold_facing", "n_postflop_check", "n_hu_actions", "last_street", "max_amount_bb", "max_to_call_bb"]


def side(prefix, key):
    return pl.scan_parquet(PLAYER_HANDS).select(["hand_id", pl.col("player_id").alias(key), "pot_bb", "players_at_showdown", "big_blind",
                                                 *[pl.col(c).alias(f"{prefix}_{c}") for c in P_COLS]])


def build_pair_hand_features(pair_hands_path, out_path, phase):
    if out_path.exists():
        return
    ph = pl.scan_parquet(pair_hands_path)
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
        )
        .with_columns(
            pl.max_horizontal("transfer_1_to_2", "transfer_2_to_1").alias("transfer_any"),
            pl.when(pl.col("loser_is_p1") == 1).then(pl.col("p1_chen")).otherwise(pl.col("p2_chen")).alias("loser_chen"),
            pl.when(pl.col("loser_is_p1") == 1).then(pl.col("p2_chen")).otherwise(pl.col("p1_chen")).alias("winner_chen"),
            pl.when(pl.col("loser_is_p1") == 1).then(pl.col("p1_contrib_bb")).otherwise(pl.col("p2_contrib_bb")).alias("loser_contrib_bb"),
        )
        .with_columns(
            (pl.col("loser_chen") - pl.col("winner_chen")).alias("chen_gap"),
            (pl.col("loser_chen") > pl.col("winner_chen")).cast(pl.Int8).alias("loser_stronger_preflop"),
            (pl.col("transfer_any") / (pl.col("pot_bb") + 1e-3)).clip(0, 1).alias("transfer_pot_ratio"),
        )
    )

    # --- member-perspective action features ---------------------------------------------
    members = pl.concat([
        ph.select(["pair_id", "hand_id", pl.col("player_1").alias("member"), pl.col("player_2").alias("partner"), pl.lit(1, dtype=pl.Int8).alias("is_p1")]),
        ph.select(["pair_id", "hand_id", pl.col("player_2").alias("member"), pl.col("player_1").alias("partner"), pl.lit(0, dtype=pl.Int8).alias("is_p1")]),
    ])
    ac = pl.scan_parquet(ACTION_CONTEXT).filter(pl.col("phase") == phase).select(["hand_id", "action_no", "street_no", "player_id", "action", "to_call", "players_active", "is_aggr", "last_aggr", "amount_pot_ratio"])
    fold_at = ac.filter(pl.col("action") == "fold").group_by(["hand_id", "player_id"]).agg(pl.col("action_no").min().alias("partner_fold_no")).rename({"player_id": "partner"})
    ma = (
        members.join(ac, left_on=["hand_id", "member"], right_on=["hand_id", "player_id"], how="inner")
        .join(fold_at, on=["hand_id", "partner"], how="left")
        .with_columns(
            ((pl.col("to_call") > 0) & (pl.col("last_aggr") == pl.col("partner"))).alias("facing_partner"),
            ((pl.col("to_call") > 0) & pl.col("last_aggr").is_not_null() & (pl.col("last_aggr") != pl.col("partner"))).alias("facing_other"),
            ((pl.col("players_active") == 2) & (pl.col("partner_fold_no").is_null() | (pl.col("partner_fold_no") > pl.col("action_no")))).alias("true_hu"),
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
        pl.col("true_hu").sum().cast(pl.Int8).alias("hu_actions"),
        (pl.col("true_hu") & (pl.col("action") == "check")).sum().cast(pl.Int8).alias("hu_check"),
        (pl.col("true_hu") & (pl.col("action") == "call")).sum().cast(pl.Int8).alias("hu_call"),
        (pl.col("true_hu") & pl.col("is_aggr")).sum().cast(pl.Int8).alias("hu_aggr"),
        (pl.col("true_hu") & (pl.col("street_no") >= 2) & (pl.col("action") == "check")).sum().cast(pl.Int8).alias("hu_late_check"),
        (pl.col("true_hu") & (pl.col("action") == "fold")).sum().cast(pl.Int8).alias("hu_fold"),
        (pl.col("is_aggr") & (pl.col("amount_pot_ratio") >= 1.0)).sum().cast(pl.Int8).alias("pair_overbets"),
    )

    # --- outsider responses to the pair's aggression -------------------------------------
    responses = ac.filter((pl.col("to_call") > 0) & pl.col("last_aggr").is_not_null()).select(
        ["hand_id", pl.col("last_aggr").alias("member"), pl.col("player_id").alias("responder"), "action", "is_aggr"])
    press = (
        members.join(responses, on=["hand_id", "member"], how="inner")
        .filter(pl.col("responder") != pl.col("partner"))
        .group_by(["pair_id", "hand_id"]).agg(
            (pl.col("action") == "fold").sum().cast(pl.Int8).alias("outsider_fold_to_pair"),
            (pl.col("action") == "call").sum().cast(pl.Int8).alias("outsider_call_to_pair"),
            pl.col("is_aggr").sum().cast(pl.Int8).alias("outsider_raise_to_pair"),
        )
    )
    fill = ["fold_to_partner", "call_partner", "raise_partner", "fold_to_other", "call_other", "raise_other", "p1_fold_to_partner", "p1_call_partner",
            "hu_actions", "hu_check", "hu_call", "hu_aggr", "hu_late_check", "hu_fold", "pair_overbets",
            "outsider_fold_to_pair", "outsider_call_to_pair", "outsider_raise_to_pair"]
    out = (
        base.join(inter, on=["pair_id", "hand_id"], how="left").join(press, on=["pair_id", "hand_id"], how="left")
        .with_columns([pl.col(c).fill_null(0) for c in fill])
        .with_columns(
            # evidence-anatomy flags (from the EDA of the planted evidence hands)
            ((pl.col("fold_to_partner") > 0) & (pl.col("loser_stronger_preflop") == 1)).cast(pl.Int8).alias("fold_stronger_to_partner"),
            ((pl.col("loser_chen") >= 6) & (pl.col("winner_chen") >= 6) & (pl.col("raise_partner") == 0) & (pl.col("both_vpip") == 1)).cast(pl.Int8).alias("both_strong_no_raise"),
            (pl.col("call_partner") >= 2).cast(pl.Int8).alias("multi_call_partner"),
            ((pl.col("n_partners_aggr") == 2) & (pl.col("outsider_fold_to_pair") >= 1)).cast(pl.Int8).alias("squeeze"),
            ((pl.col("n_partners_aggr") == 2) & (pl.col("outsider_fold_to_pair") >= 1) & (pl.col("fold_to_partner") >= 1)).cast(pl.Int8).alias("squeeze_then_fold"),
            ((pl.col("partner_won_other_lost") == 1) & (pl.col("call_partner") >= 1) & (pl.col("transfer_any") >= 10)).cast(pl.Int8).alias("dump"),
            ((pl.col("both_showdown") == 1) & (pl.col("hu_check") >= 2) & (pl.col("pair_aggr") == 0)).cast(pl.Int8).alias("checkdown"),
            ((pl.col("both_showdown") == 1) & (pl.col("hu_late_check") >= 1) & (pl.max_horizontal("loser_chen", "winner_chen") >= 7)).cast(pl.Int8).alias("hu_checkdown_strong"),
            (pl.col("transfer_any") * pl.col("call_partner")).cast(pl.Float32).alias("transfer_x_call"),
        )
        .with_columns(
            # within-pair normalisations: "how extreme is this hand *for this pair*" (max-ratio and percentile rank)
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
]
HAND_FLAGS = ["fold_stronger_to_partner", "both_strong_no_raise", "multi_call_partner", "squeeze", "squeeze_then_fold", "dump", "checkdown", "hu_checkdown_strong", "transfer_x_call"]
HAND_TOMAX = ["transfer_to_max", "net_gap_to_max", "pot_to_max", "outsider_fold_to_max"]
PRANK_BASE = HAND_RAW + HAND_FLAGS
HAND_FEATS = HAND_RAW + HAND_FLAGS + HAND_TOMAX + [f"{c}_prank" for c in PRANK_BASE]

DEV_HF = WORK_DIR / "dev_hand_features.parquet"
EVAL_HF = WORK_DIR / "eval_hand_features.parquet"
build_pair_hand_features(DEV_PAIR_HANDS, DEV_HF, "development")
log(f"dev hand features: {pl.scan_parquet(DEV_HF).select(pl.len()).collect().item():,} rows")
build_pair_hand_features(EVAL_PAIR_HANDS, EVAL_HF, "evaluation")
log(f"eval hand features: {pl.scan_parquet(EVAL_HF).select(pl.len()).collect().item():,} rows")

# %% [markdown]
# ## 5. Hand-level evidence model (GroupKFold by table) → OOF hand scores → MAP@5

# %%
dev_hf = pl.read_parquet(DEV_HF).join(dev_pairs.select(["pair_id", "label", "behavior_family", "is_labeled", "population_weight"]), on="pair_id")
dev_hf = dev_hf.join(dev_evidence.select(["pair_id", "hand_id", pl.lit(True).alias("is_evidence")]), on=["pair_id", "hand_id"], how="left").with_columns(pl.col("is_evidence").fill_null(False))
assert dev_hf["is_evidence"].sum() == 1817, "not every evidence hand was reconstructed"
dh = dev_hf.to_pandas()
dh[HAND_FEATS] = dh[HAND_FEATS].astype("float32")

tables = np.sort(dh["table_id"].unique())
rng = np.random.RandomState(SEED)
table_fold = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % N_FOLDS)}
dh["fold"] = dh["table_id"].map(table_fold)
dev_pairs = dev_pairs.with_columns(pl.col("table_id").replace_strict(table_fold, return_dtype=pl.Int8).alias("fold"))

HAND_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=os.cpu_count())
HAND_SEEDS = [SEED, SEED + 7]
HAND_ROUNDS = 400
lab_mask = dh["is_labeled"].to_numpy()
pos_mask = lab_mask & (dh["label"] == 1).to_numpy()   # the evidence model learns "which of a coordinated pair's hands are the manipulated ones"
dh["hand_score"] = 0.0
for f in range(N_FOLDS):
    tr = pos_mask & (dh["fold"] != f).to_numpy()
    te = (dh["fold"] == f).to_numpy()
    for sd in HAND_SEEDS:
        m = lgb.train({**HAND_PARAMS, "seed": sd}, lgb.Dataset(dh.loc[tr, HAND_FEATS], dh.loc[tr, "is_evidence"].astype(int)), num_boost_round=HAND_ROUNDS)
        dh.loc[te, "hand_score"] += m.predict(dh.loc[te, HAND_FEATS]) / len(HAND_SEEDS)
    log(f"hand model fold {f}: MAP@5 on positive pairs = {map5_within_pairs(dh[te & pos_mask]):.4f}")

pos_all = dh[pos_mask]
oof_map5 = map5_within_pairs(pos_all)
fam_map5 = {fam: map5_within_pairs(pos_all[pos_all["behavior_family"] == fam]) for fam in TARGET_BEHAVIORS}
log(f"OOF evidence MAP@5 = {oof_map5:.4f}  per family: { {k: round(v, 4) for k, v in fam_map5.items()} }")
hand_models_full = [lgb.train({**HAND_PARAMS, "seed": sd}, lgb.Dataset(dh.loc[pos_mask, HAND_FEATS], dh.loc[pos_mask, "is_evidence"].astype(int)), num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]

# second hand model: "does this hand look like a coordinated pair's hand?" (positives' hands vs confirmed negatives' hands) — used only for pair aggregation
SUS_PARAMS = {**HAND_PARAMS, "scale_pos_weight": 8.0}
dh["hand_sus"] = 0.0
for f in range(N_FOLDS):
    tr = lab_mask & (dh["fold"] != f).to_numpy()
    te = (dh["fold"] == f).to_numpy()
    m = lgb.train(SUS_PARAMS, lgb.Dataset(dh.loc[tr, HAND_FEATS], dh.loc[tr, "is_evidence"].astype(int)), num_boost_round=HAND_ROUNDS)
    dh.loc[te, "hand_sus"] = m.predict(dh.loc[te, HAND_FEATS])
sus_model_full = lgb.train(SUS_PARAMS, lgb.Dataset(dh.loc[lab_mask, HAND_FEATS], dh.loc[lab_mask, "is_evidence"].astype(int)), num_boost_round=HAND_ROUNDS)
log("suspicious-hand model (pair aggregation) trained")
neg_scores = dh.loc[lab_mask & (dh["label"] == 0).to_numpy(), "hand_score"].to_numpy()
THR95, THR99 = np.quantile(neg_scores, 0.95), np.quantile(neg_scores, 0.99)
neg_sus = dh.loc[lab_mask & (dh["label"] == 0).to_numpy(), "hand_sus"].to_numpy()
SUS95, SUS99 = np.quantile(neg_sus, 0.95), np.quantile(neg_sus, 0.99)
imp = pd.Series(np.mean([m.feature_importance("gain") for m in hand_models_full], axis=0), index=HAND_FEATS).sort_values(ascending=False)
print("top hand features (gain):\n", imp.head(15).round(0).to_string())

# %% [markdown]
# ## 6. Pair aggregation → pair risk model + family heads

# %%
AGG_MEAN = ["both_vpip", "both_showdown", "partner_won_other_lost", "transfer_any", "net_gap", "pot_bb", "loser_stronger_preflop", "chen_gap",
            "fold_to_partner", "call_partner", "raise_partner", "fold_to_other", "call_other", "raise_other",
            "hu_actions", "hu_check", "hu_call", "hu_aggr", "hu_late_check", "outsider_fold_to_pair", "outsider_call_to_pair", "outsider_raise_to_pair",
            "n_partners_aggr", "pair_aggr", "pair_raise", "pair_overbets"]
AGG_MAX = ["transfer_any", "net_gap", "pot_bb", "outsider_fold_to_pair", "call_partner", "hu_check", "max_amount_bb"]
AGG_TOP3 = ["transfer_any", "net_gap", "outsider_fold_to_pair", "hu_check", "call_partner"]


def aggregate_pairs(hf: pl.DataFrame, baselines_phase: pl.DataFrame, pairs: pl.DataFrame) -> pd.DataFrame:
    """hf: polars pair-hand features with hand_score. Returns one row per pair."""
    lf = hf.select(["pair_id", "hand_id", "table_id", "hand_idx", "player_1", "player_2", "hand_score", "hand_sus", "loser_is_p1", "signed_net_diff",
                            "transfer_1_to_2", "transfer_2_to_1", "p1_net_bb", "p2_net_bb", "p1_n_aggr", "p2_n_aggr", "p1_n_fold_facing", "p2_n_fold_facing",
                            "p1_vpip", "p2_vpip", "p1_call_partner", "p1_fold_to_partner"] + sorted(set(AGG_MEAN + AGG_MAX + AGG_TOP3))).lazy()
    aggs = [pl.len().alias("n_shared"), pl.col("table_id").first(), pl.col("player_1").first(), pl.col("player_2").first()]
    aggs += [pl.col(c).mean().alias(f"{c}_mean") for c in AGG_MEAN]
    aggs += [pl.col(c).max().alias(f"{c}_max") for c in AGG_MAX]
    aggs += [pl.col(c).top_k(3).mean().alias(f"{c}_top3") for c in AGG_TOP3]
    aggs += [
        pl.col("hand_score").mean().alias("hs_mean"), pl.col("hand_score").max().alias("hs_max"),
        pl.col("hand_score").top_k(3).mean().alias("hs_top3"), pl.col("hand_score").top_k(5).mean().alias("hs_top5"),
        pl.col("hand_score").quantile(0.9).alias("hs_p90"),
        (pl.col("hand_score") > THR95).sum().alias("hs_n95"), (pl.col("hand_score") > THR99).sum().alias("hs_n99"),
        (pl.col("hand_score") > THR95).mean().alias("hs_f95"),
        pl.col("hand_sus").mean().alias("sus_mean"), pl.col("hand_sus").max().alias("sus_max"),
        pl.col("hand_sus").top_k(3).mean().alias("sus_top3"), pl.col("hand_sus").top_k(5).mean().alias("sus_top5"), pl.col("hand_sus").quantile(0.9).alias("sus_p90"),
        (pl.col("hand_sus") > SUS95).sum().alias("sus_n95"), (pl.col("hand_sus") > SUS99).sum().alias("sus_n99"), (pl.col("hand_sus") > SUS95).mean().alias("sus_f95"),
        pl.col("loser_is_p1").sort_by("hand_sus", descending=True).head(5).mean().alias("_top5s_loser_p1"),
        pl.col("transfer_1_to_2").sum().alias("_t12"), pl.col("transfer_2_to_1").sum().alias("_t21"),
        pl.col("signed_net_diff").sum().alias("_snd"), pl.col("signed_net_diff").abs().sum().alias("_asnd"),
        pl.col("p1_net_bb").mean().alias("_p1_net"), pl.col("p2_net_bb").mean().alias("_p2_net"),
        pl.col("p1_n_aggr").mean().alias("_p1_aggr"), pl.col("p2_n_aggr").mean().alias("_p2_aggr"),
        pl.col("p1_n_fold_facing").mean().alias("_p1_ff"), pl.col("p2_n_fold_facing").mean().alias("_p2_ff"),
        pl.col("p1_vpip").cast(pl.Float32).mean().alias("_p1_vpip"), pl.col("p2_vpip").cast(pl.Float32).mean().alias("_p2_vpip"),
        pl.col("p1_call_partner").sum().alias("_p1_cp"), pl.col("call_partner").sum().alias("_cp"),
        pl.col("p1_fold_to_partner").sum().alias("_p1_fp"), pl.col("fold_to_partner").sum().alias("_fp"),
        # directedness of the top-5 scored hands: same loser?
        pl.col("loser_is_p1").sort_by("hand_score", descending=True).head(5).mean().alias("_top5_loser_p1"),
        pl.col("loser_is_p1").sort_by("transfer_any", descending=True).head(5).mean().alias("_top5t_loser_p1"),
    ]
    pf = lf.group_by("pair_id").agg(aggs)
    # burst: max over 8 chronological bins of the mean hand score
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
    pf = pf.with_columns(
        ((pl.col("_t12") - pl.col("_t21")).abs() / (pl.col("_t12") + pl.col("_t21") + 1.0)).alias("transfer_imbalance"),
        ((pl.col("_t12") + pl.col("_t21")) / pl.col("n_shared")).alias("transfer_rate"),
        pl.max_horizontal("_t12", "_t21").alias("transfer_dominant"),
        (pl.col("_snd").abs() / (pl.col("_asnd") + 1.0)).alias("direction_consistency"),
        pl.max_horizontal(pl.col("_top5_loser_p1"), 1 - pl.col("_top5_loser_p1")).alias("top5_same_loser"),
        pl.max_horizontal(pl.col("_top5t_loser_p1"), 1 - pl.col("_top5t_loser_p1")).alias("top5t_same_loser"),
        pl.max_horizontal(pl.col("_top5s_loser_p1"), 1 - pl.col("_top5s_loser_p1")).alias("top5s_same_loser"),
        (pl.max_horizontal(pl.col("_p1_cp"), pl.col("_cp") - pl.col("_p1_cp")) / (pl.col("_cp") + 1.0)).alias("call_partner_asym"),
        (pl.max_horizontal(pl.col("_p1_fp"), pl.col("_fp") - pl.col("_p1_fp")) / (pl.col("_fp") + 1.0)).alias("fold_partner_asym"),
        # residuals vs each player's own phase baseline
        (pl.col("_p1_net") - pl.col("p1_base_net_bb")).alias("p1_net_resid"), (pl.col("_p2_net") - pl.col("p2_base_net_bb")).alias("p2_net_resid"),
        (pl.col("_p1_aggr") - pl.col("p1_base_n_aggr")).alias("p1_aggr_resid"), (pl.col("_p2_aggr") - pl.col("p2_base_n_aggr")).alias("p2_aggr_resid"),
        (pl.col("_p1_ff") - pl.col("p1_base_n_fold_facing")).alias("p1_ff_resid"), (pl.col("_p2_ff") - pl.col("p2_base_n_fold_facing")).alias("p2_ff_resid"),
        (pl.col("_p1_vpip") - pl.col("p1_base_vpip")).alias("p1_vpip_resid"), (pl.col("_p2_vpip") - pl.col("p2_base_vpip")).alias("p2_vpip_resid"),
    ).with_columns(
        (pl.col("p1_net_resid") - pl.col("p2_net_resid")).abs().alias("net_resid_gap"),
        pl.min_horizontal("p1_net_resid", "p2_net_resid").alias("net_resid_min"),
        pl.max_horizontal("p1_net_resid", "p2_net_resid").alias("net_resid_max"),
        (pl.col("p1_aggr_resid") + pl.col("p2_aggr_resid")).alias("aggr_resid_sum"),
        pl.min_horizontal("p1_aggr_resid", "p2_aggr_resid").alias("aggr_resid_min"),
        (pl.col("p1_ff_resid") + pl.col("p2_ff_resid")).alias("ff_resid_sum"),
        (pl.col("p1_vpip_resid") + pl.col("p2_vpip_resid")).alias("vpip_resid_sum"),
    )
    drop = [c for c in pf.columns if c.startswith("_") or c.startswith("p1_base_") or c.startswith("p2_base_")]
    pf = pf.drop(drop).join(pairs.select([c for c in pairs.columns if c in ("pair_id", "label", "behavior_family", "is_labeled", "population_weight", "fold", "shared_hands")]), on="pair_id", how="left")
    return pf.to_pandas()


base_dev = baselines.filter(pl.col("phase") == "development").drop("phase")
dev_hf = dev_hf.with_columns(pl.Series("hand_score", dh["hand_score"].to_numpy(), dtype=pl.Float32), pl.Series("hand_sus", dh["hand_sus"].to_numpy(), dtype=pl.Float32))
dev_pf = aggregate_pairs(dev_hf, base_dev, dev_pairs)
PAIR_FEATS = [c for c in dev_pf.columns if c not in ("pair_id", "table_id", "player_1", "player_2", "label", "behavior_family", "is_labeled", "population_weight", "fold", "shared_hands")]
dev_pf.to_parquet(WORK_DIR / "dev_pair_features.parquet", index=False)
log(f"dev pair features: {dev_pf.shape}; {len(PAIR_FEATS)} features")

# --- pair risk model ---------------------------------------------------------------------
y = dev_pf["label"].fillna(0).astype(int).to_numpy()
w_train = np.where(dev_pf["is_labeled"], np.where(y == 1, POS_WEIGHT, NEG_WEIGHT), PU_WEIGHT)
w_pop = np.where(dev_pf["is_labeled"], 1.0, dev_pf["population_weight"].to_numpy())
folds = dev_pf["fold"].to_numpy()
PAIR_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=os.cpu_count())
oof = np.zeros(len(dev_pf))
pair_models = []
fam_models = {fam: [] for fam in TARGET_BEHAVIORS}
oof_fam = {fam: np.zeros(len(dev_pf)) for fam in TARGET_BEHAVIORS}
X = dev_pf[PAIR_FEATS].astype("float32")
for f in range(N_FOLDS):
    tr, te = folds != f, folds == f
    m = lgb.train(PAIR_PARAMS, lgb.Dataset(X[tr], y[tr], weight=w_train[tr]), num_boost_round=600)
    oof[te] = m.predict(X[te])
    pair_models.append(m)
    for fam in TARGET_BEHAVIORS:
        yf = ((dev_pf["behavior_family"] == fam).to_numpy()).astype(int)
        mf = lgb.train({**PAIR_PARAMS, "num_leaves": 7}, lgb.Dataset(X[tr], yf[tr], weight=w_train[tr]), num_boost_round=400)
        oof_fam[fam][te] = mf.predict(X[te])
        fam_models[fam].append(mf)
    lab = dev_pf["is_labeled"].to_numpy() & te
    log(f"pair model fold {f}: labelled AP={average_precision_score(y[lab], oof[lab]):.4f} | PU-weighted AP={average_precision_score(y[te], oof[te], sample_weight=w_pop[te]):.4f}")

labm = dev_pf["is_labeled"].to_numpy()
labelled_ap = average_precision_score(y[labm], oof[labm])
pu_ap = average_precision_score(y, oof, sample_weight=w_pop)
print(f"\nOOF labelled Pair AP (positives vs confirmed negatives, OPTIMISTIC): {labelled_ap:.4f}")
print(f"OOF PU-weighted Pair AP (positives vs unlabelled population, tracks LB): {pu_ap:.4f}")
for fam in TARGET_BEHAVIORS:
    sel = (dev_pf["behavior_family"] == fam).to_numpy() | (y == 0)
    print(f"   {fam:22s} PU-weighted AP = {average_precision_score(y[sel], oof[sel], sample_weight=w_pop[sel]):.4f}")

# behaviour: argmax of the three heads, scaled by inverse family frequency (macro AP rewards rare families)
fam_prior = {fam: (dev_pf["behavior_family"] == fam).sum() for fam in TARGET_BEHAVIORS}
def predict_family(fam_probs: dict) -> np.ndarray:
    mat = np.column_stack([fam_probs[fam] / fam_prior[fam] for fam in TARGET_BEHAVIORS])
    return np.array(TARGET_BEHAVIORS)[mat.argmax(axis=1)]
dev_pf["pred_family"] = predict_family(oof_fam)
pos = dev_pf["label"] == 1
print(f"family accuracy on positives: {(dev_pf.loc[pos, 'pred_family'] == dev_pf.loc[pos, 'behavior_family']).mean():.3f}")

# composite with the host metric on the dev sample (labelled + PU treated as negatives; unweighted -> optimistic)
top5 = (dh.sort_values(["pair_id", "hand_score", "pot_bb", "hand_id"], ascending=[True, False, False, True]).groupby("pair_id")["hand_id"].apply(lambda s: list(s.head(5))))
sol = pd.DataFrame({"pair_id": dev_pf["pair_id"], "risk_score": y, "predicted_behavior": np.where(y == 1, dev_pf["behavior_family"], "none")})
ev_map = dev_evidence.to_pandas().groupby("pair_id")["hand_id"].apply(list)
for i, c in enumerate(EVIDENCE_COLUMNS):
    sol[c] = [ (ev_map[p][i] if (p in ev_map.index and i < len(ev_map[p])) else NO_EVIDENCE) for p in sol["pair_id"]]
sub_dev = pd.DataFrame({"pair_id": dev_pf["pair_id"], "risk_score": oof, "predicted_behavior": dev_pf["pred_family"]})
for i, c in enumerate(EVIDENCE_COLUMNS):
    sub_dev[c] = [ (top5[p][i] if i < len(top5[p]) else NO_EVIDENCE) for p in sub_dev["pair_id"]]
comp = host_score(sol, sub_dev, return_components=True)
print("host metric on dev sample (labelled + PU sample as negatives):", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in comp.items()}, default=str))
imp = pd.Series(np.mean([m.feature_importance("gain") for m in pair_models], axis=0), index=PAIR_FEATS).sort_values(ascending=False)
print("top pair features (gain):\n", imp.head(20).round(0).to_string())

# %% [markdown]
# ## 7. Evaluation inference: hand scores → pair features → risk, family, evidence

# %%
EVAL_SCORED = WORK_DIR / "eval_hand_scored.parquet"
if not EVAL_SCORED.exists():
    parts = []
    lf = pl.scan_parquet(EVAL_HF)
    n = lf.select(pl.len()).collect().item()
    CH = 2_000_000
    for start in range(0, n, CH):
        chunk = lf.slice(start, CH).collect()
        Xc = chunk.select(HAND_FEATS).to_pandas().astype("float32")
        s = np.mean([m.predict(Xc) for m in hand_models_full], axis=0)
        s2 = sus_model_full.predict(Xc)
        parts.append(chunk.with_columns(pl.Series("hand_score", s, dtype=pl.Float32), pl.Series("hand_sus", s2, dtype=pl.Float32)))
        log(f"scored eval hands {min(start + CH, n):,}/{n:,}")
    pl.concat(parts).write_parquet(EVAL_SCORED)
    del parts, chunk, Xc
    gc.collect()
eh = pl.read_parquet(EVAL_SCORED)
base_eval = baselines.filter(pl.col("phase") == "evaluation").drop("phase")
eval_pf = aggregate_pairs(eh, base_eval, eval_prep)
assert len(eval_pf) == 112_540
eval_pf.to_parquet(WORK_DIR / "eval_pair_features.parquet", index=False)
Xe = eval_pf[PAIR_FEATS].astype("float32")
risk = np.mean([m.predict(Xe) for m in pair_models], axis=0)
fam_probs = {fam: np.mean([m.predict(Xe) for m in fam_models[fam]], axis=0) for fam in TARGET_BEHAVIORS}
eval_pf["risk_score"] = risk
eval_pf["predicted_behavior"] = predict_family(fam_probs)

# evidence: top-5 distinct shared evaluation hands by hand score (ties: bigger pot, then hand_id)
top = (
    eh.select(["pair_id", "hand_id", "hand_score", "pot_bb"])
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
assert set(sub["predicted_behavior"]) <= set(TARGET_BEHAVIORS)
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
print(json.dumps({"oof_labelled_pair_ap": round(labelled_ap, 4), "oof_pu_weighted_pair_ap": round(pu_ap, 4), "oof_evidence_map5": round(oof_map5, 4),
                  "evidence_map5_by_family": {k: round(v, 4) for k, v in fam_map5.items()}, "dev_sample_host_metric": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in comp.items()}}, indent=2, default=str))
