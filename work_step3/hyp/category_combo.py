"""
Hypothesis `category_combo`: the evidence-label rule is conditioned on the made-hand CATEGORIES of the two
partners (and of the outsiders) at the DECISIVE street -- the street on which the losing partner folded, or the
final street when the pair went to showdown.

Per-(pair_id, hand_id) features (prefix cc_), all built ONLY from cards / actions (never from labels):
  * decisive-street anatomy   : cc_dec_street, cc_who_folded, cc_n_out_active_at_fold, cc_fold_to_winner,
                                cc_fold_to_call_bb, cc_fold_surp
  * partner categories        : cc_loser_cat_dec, cc_winner_cat_dec (category = value // 15**5, -2 = pre-flop),
                                cc_cat_gap_dec, cc_loser_better_dec (exact value), cc_loser_cat_better,
                                cc_loser_kicker_better, cc_fold_cat_better, cc_fold_cat_better_hu,
                                cc_sd_cat_gap, cc_sd_winner_much_better, per-street categories
                                cc_{loser,winner}_cat_{flop,turn,river}, cc_{loser,winner}_cat_own_last,
                                cc_{loser,winner}_pocket
  * outsider categories       : cc_o_cat_dec_max, cc_o_beats_pair_dec, cc_o_active_beats_pair_dec,
                                cc_out_fold_to_pair_n, cc_out_fold_all4, cc_out_fold_frac,
                                cc_out_fold_better_to_pair, cc_out_fold_strong_pf, cc_out_call_better_to_pair,
                                cc_aggr_chen_min_pf, cc_iso_junk_vs_strong
  * family-style flags        : cc_dt_flag, cc_sp_sd_flag, cc_sp_flag, cc_ci_flag
  * categorical combo codes   : cc_combo_a (who_folded x dec_street x loser_cat x winner_cat),
                                cc_combo_b (who_folded x dec_street x gap-sign/kicker),
                                cc_combo_o (who_folded x outsider folds x outsider better-folds)
  * target encodings          : cc_te_a/b/o and family-crossed cc_te_fa/fb/fo.  In DEV these are computed
                                fold-safely (outer table-fold -> encode from training folds only; training rows
                                get inner 5-fold OOF encodings).  The parquet stores the outer-fold OOF values.

HOW TO COMPUTE FOR EVALUATION PAIRS
    python3 work_step3/hyp/category_combo.py --phase eval --pairs <parquet with pair_id column> [--out <path>]
  It reads work_step3/eval_hand_features_v9.parquet (same columns as the dev tables), player_strength.parquet,
  player_hands_v9.parquet, action_context.parquet (+ action_surprise.parquet), data/seats.parquet, builds the
  identical raw features with `build_features`, and fits the target encodings on ALL dev positive-pair hands
  (labels from development_evidence.csv) with the same smoothing (m=30) before applying them to eval rows.
  The family-crossed encodings need a behavior_family column for eval pairs (use the predicted family); if it
  is absent they are filled with the family-agnostic encodings.
    python3 work_step3/hyp/category_combo.py --phase dev      (default) builds the dev table, evaluates.
"""
import argparse, os, sys, time, warnings
import numpy as np, polars as pl, pandas as pd
warnings.filterwarnings("ignore")

ROOT = "/Users/md.hamidhosen/Documents/Kaggle/Detect Suspicious Value Transfers in Poker/"
D = ROOT + "data/"; W = ROOT + "work_step3/"; H = W + "hyp/"
B5 = 15 ** 5
NTHREADS = 3
TE_M = 30.0
RANKS = "23456789TJQKA"

BASE_COLS = ["pair_id", "hand_id", "table_id", "hand_idx", "player_1", "player_2", "pot_bb", "players_at_showdown",
             "loser_is_p1", "last_street", "loser_folded", "winner_folded", "loser_last_street", "winner_last_street",
             "both_showdown", "pair_aggr", "n_partners_aggr", "surp_pair_max"] + \
            [f"{p}_{c}" for p in ("p1", "p2") for c in ("chen", "v_flop", "v_turn", "v_river", "v_final", "cat_final",
                                                        "folded", "last_street", "pfr", "vpip_a")]
RAW_FEATS = ["cc_dec_street", "cc_who_folded", "cc_n_out_active_at_fold", "cc_fold_to_winner", "cc_fold_to_call_bb", "cc_fold_surp",
             "cc_loser_cat_dec", "cc_winner_cat_dec", "cc_cat_gap_dec", "cc_loser_better_dec", "cc_loser_cat_better", "cc_loser_kicker_better",
             "cc_fold_cat_better", "cc_fold_cat_better_hu", "cc_sd_cat_gap", "cc_sd_winner_much_better",
             "cc_loser_cat_flop", "cc_loser_cat_turn", "cc_loser_cat_river", "cc_winner_cat_flop", "cc_winner_cat_turn", "cc_winner_cat_river",
             "cc_loser_cat_own_last", "cc_winner_cat_own_last", "cc_loser_pocket", "cc_winner_pocket",
             "cc_o_cat_dec_max", "cc_o_beats_pair_dec", "cc_o_active_beats_pair_dec", "cc_out_fold_to_pair_n", "cc_out_fold_all4", "cc_out_fold_frac",
             "cc_out_fold_better_to_pair", "cc_out_fold_strong_pf", "cc_out_call_better_to_pair", "cc_aggr_chen_min_pf", "cc_iso_junk_vs_strong",
             "cc_dt_flag", "cc_sp_sd_flag", "cc_sp_flag", "cc_ci_flag"]
COMBO_COLS = ["cc_combo_a", "cc_combo_b", "cc_combo_o"]
TE_FEATS = ["cc_te_a", "cc_te_b", "cc_te_o", "cc_te_fa", "cc_te_fb", "cc_te_fo"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# feature construction (phase-agnostic)
# ----------------------------------------------------------------------------------------------------------------
def _v_at(prefix, s):
    """exact value of `prefix` at street s (0 = pre-flop -> Chen*1000, 1..3 exact 15**5-coded value; -1 if undealt)"""
    return (pl.when(s == 0).then(pl.col(f"{prefix}chen").cast(pl.Float64) * 1000.0)
            .when(s == 1).then(pl.col(f"{prefix}v_flop").cast(pl.Float64))
            .when(s == 2).then(pl.col(f"{prefix}v_turn").cast(pl.Float64))
            .otherwise(pl.col(f"{prefix}v_river").cast(pl.Float64)))


def _cat_of(expr):
    return pl.when(expr >= 0).then(expr // B5).otherwise(-1).cast(pl.Int8)


def _cat_at(prefix, s):
    """made-hand category at street s: -2 pre-flop, -1 street not dealt, 0..8 otherwise"""
    return (pl.when(s == 0).then(pl.lit(-2, dtype=pl.Int8))
            .when(s == 1).then(_cat_of(pl.col(f"{prefix}v_flop")))
            .when(s == 2).then(_cat_of(pl.col(f"{prefix}v_turn")))
            .otherwise(_cat_of(pl.col(f"{prefix}v_river"))))


def load_side_tables(hand_ids: pl.Series, phase: str):
    """per-player strength+cards, actions (+surprise) and hole cards for the given hands"""
    hs = hand_ids.implode()
    st = pl.scan_parquet(W + "player_strength.parquet").filter(pl.col("hand_id").is_in(hs))
    ph = (pl.scan_parquet(W + "player_hands_v9.parquet").filter(pl.col("hand_id").is_in(hs))
          .select(["hand_id", "player_id", "chen", "vpip_a", "pfr", "folded", "last_street"]))
    players = st.join(ph, on=["hand_id", "player_id"]).collect()
    ac = (pl.scan_parquet(W + "action_context.parquet").filter((pl.col("phase") == phase) & pl.col("hand_id").is_in(hs))
          .select(["hand_id", "action_no", "street_no", "player_id", "action", "to_call", "players_active", "is_aggr", "last_aggr", "to_call_bb"]))
    su = pl.scan_parquet(W + "action_surprise.parquet").filter(pl.col("hand_id").is_in(hs)).select(["hand_id", "action_no", "surp"])
    actions = ac.join(su, on=["hand_id", "action_no"], how="left").with_columns(pl.col("surp").fill_null(0.0)).collect()
    seats = (pl.scan_parquet(D + "seats.parquet").filter(pl.col("hand_id").is_in(hs)).select(["hand_id", "player_id", "hole_card_1", "hole_card_2"])
             .with_columns((pl.col("hole_card_1").str.slice(0, 1) == pl.col("hole_card_2").str.slice(0, 1)).cast(pl.Int8).alias("pocket"))
             .select(["hand_id", "player_id", "pocket"]).collect())
    return players, actions, seats


def build_features(base: pl.DataFrame, players: pl.DataFrame, actions: pl.DataFrame, seats: pl.DataFrame) -> pl.DataFrame:
    """base: one row per (pair_id, hand_id) with BASE_COLS (from dev_hand_features_v9_* / eval_hand_features_v9)."""
    b = base.select(BASE_COLS).with_columns(
        pl.when(pl.col("loser_is_p1") == 1).then(pl.col("player_1")).otherwise(pl.col("player_2")).alias("loser"),
        pl.when(pl.col("loser_is_p1") == 1).then(pl.col("player_2")).otherwise(pl.col("player_1")).alias("winner"),
        pl.col("loser_folded").cast(pl.Boolean), pl.col("winner_folded").cast(pl.Boolean),
    )
    for side, who in (("l_", "loser"), ("w_", "winner")):
        src = "p1" if who == "loser" else "p2"
        oth = "p2" if who == "loser" else "p1"
        b = b.with_columns([pl.when(pl.col("loser_is_p1") == 1).then(pl.col(f"{src}_{c}")).otherwise(pl.col(f"{oth}_{c}")).alias(f"{side}{c}")
                            for c in ("chen", "v_flop", "v_turn", "v_river", "v_final", "cat_final", "last_street", "pfr", "vpip_a")])
    b = b.with_columns(
        pl.when(pl.col("loser_folded")).then(pl.col("loser_last_street")).otherwise(pl.col("last_street")).cast(pl.Int8).alias("cc_dec_street"),
        (pl.col("loser_folded").cast(pl.Int8) + 2 * pl.col("winner_folded").cast(pl.Int8)).alias("cc_who_folded"),
    ).with_columns(
        _cat_at("l_", pl.col("cc_dec_street")).alias("cc_loser_cat_dec"), _cat_at("w_", pl.col("cc_dec_street")).alias("cc_winner_cat_dec"),
        _v_at("l_", pl.col("cc_dec_street")).alias("l_v_dec"), _v_at("w_", pl.col("cc_dec_street")).alias("w_v_dec"),
        _cat_of(pl.col("l_v_flop")).alias("cc_loser_cat_flop"), _cat_of(pl.col("l_v_turn")).alias("cc_loser_cat_turn"), _cat_of(pl.col("l_v_river")).alias("cc_loser_cat_river"),
        _cat_of(pl.col("w_v_flop")).alias("cc_winner_cat_flop"), _cat_of(pl.col("w_v_turn")).alias("cc_winner_cat_turn"), _cat_of(pl.col("w_v_river")).alias("cc_winner_cat_river"),
        _cat_at("l_", pl.col("l_last_street")).alias("cc_loser_cat_own_last"), _cat_at("w_", pl.col("w_last_street")).alias("cc_winner_cat_own_last"),
    ).with_columns(
        pl.when(pl.col("cc_dec_street") == 0).then(0).otherwise(pl.col("cc_loser_cat_dec") - pl.col("cc_winner_cat_dec")).cast(pl.Int8).alias("cc_cat_gap_dec"),
        (pl.col("l_v_dec") > pl.col("w_v_dec")).cast(pl.Int8).alias("cc_loser_better_dec"),
    ).with_columns(
        (pl.col("cc_cat_gap_dec") > 0).cast(pl.Int8).alias("cc_loser_cat_better"),
        ((pl.col("cc_cat_gap_dec") == 0) & (pl.col("cc_loser_better_dec") == 1) & (pl.col("cc_dec_street") >= 1)).cast(pl.Int8).alias("cc_loser_kicker_better"),
        ((pl.col("cc_who_folded") == 1) & (pl.col("cc_dec_street") >= 1) & (pl.col("cc_cat_gap_dec") > 0)).cast(pl.Int8).alias("cc_fold_cat_better"),
        pl.when(pl.col("cc_who_folded") == 0).then(pl.col("cc_cat_gap_dec")).otherwise(0).cast(pl.Int8).alias("cc_sd_cat_gap"),
        ((pl.col("both_showdown") == 1) & (pl.col("cc_cat_gap_dec") <= -1)).cast(pl.Int8).alias("cc_sd_winner_much_better"),
    )
    # ---- pocket pairs --------------------------------------------------------------------------------------------
    b = (b.join(seats.rename({"player_id": "loser", "pocket": "cc_loser_pocket"}), on=["hand_id", "loser"], how="left")
         .join(seats.rename({"player_id": "winner", "pocket": "cc_winner_pocket"}), on=["hand_id", "winner"], how="left")
         .with_columns(pl.col("cc_loser_pocket").fill_null(0), pl.col("cc_winner_pocket").fill_null(0)))
    # ---- loser's fold action: outsiders still active, who set the price, price, surprise ------------------------
    folds = actions.filter(pl.col("action") == "fold").select(["hand_id", "player_id", "action_no", "players_active", "last_aggr", "to_call_bb", "surp"])
    lf = (b.select(["pair_id", "hand_id", "loser", "winner"])
          .join(folds.rename({"player_id": "loser"}), on=["hand_id", "loser"], how="inner")
          .join(folds.select(["hand_id", pl.col("player_id").alias("winner"), pl.col("action_no").alias("w_fold_no")]), on=["hand_id", "winner"], how="left")
          .with_columns(
              (pl.col("w_fold_no").is_null() | (pl.col("w_fold_no") > pl.col("action_no"))).cast(pl.Int8).alias("winner_active"),
              (pl.col("last_aggr") == pl.col("winner")).cast(pl.Int8).alias("cc_fold_to_winner"),
              pl.col("to_call_bb").cast(pl.Float32).alias("cc_fold_to_call_bb"), pl.col("surp").cast(pl.Float32).alias("cc_fold_surp"))
          .with_columns((pl.col("players_active") - 1 - pl.col("winner_active")).clip(0, 4).cast(pl.Int8).alias("cc_n_out_active_at_fold"))
          .select(["pair_id", "hand_id", "cc_n_out_active_at_fold", "cc_fold_to_winner", "cc_fold_to_call_bb", "cc_fold_surp"]))
    b = b.join(lf, on=["pair_id", "hand_id"], how="left").with_columns(
        pl.col("cc_n_out_active_at_fold").fill_null(-1), pl.col("cc_fold_to_winner").fill_null(-1),
        pl.col("cc_fold_to_call_bb").fill_null(-1.0), pl.col("cc_fold_surp").fill_null(-1.0),
    ).with_columns(((pl.col("cc_fold_cat_better") == 1) & (pl.col("cc_n_out_active_at_fold") == 0)).cast(pl.Int8).alias("cc_fold_cat_better_hu"))
    # ---- outsiders' cards at the decisive street -----------------------------------------------------------------
    keys = b.select(["pair_id", "hand_id", "player_1", "player_2", "cc_dec_street", "l_v_dec", "w_v_dec"])
    out = (keys.join(players, on="hand_id").filter((pl.col("player_id") != pl.col("player_1")) & (pl.col("player_id") != pl.col("player_2")))
           .with_columns(_v_at("", pl.col("cc_dec_street")).alias("o_v_dec"), _cat_at("", pl.col("cc_dec_street")).alias("o_cat_dec"),
                         (~pl.col("folded") | (pl.col("last_street") >= pl.col("cc_dec_street"))).alias("o_active"))
           .with_columns((pl.col("o_v_dec") > pl.max_horizontal("l_v_dec", "w_v_dec")).alias("o_beats"))
           .group_by(["pair_id", "hand_id"]).agg(
               pl.col("o_cat_dec").max().alias("cc_o_cat_dec_max"),
               pl.col("o_beats").any().cast(pl.Int8).alias("cc_o_beats_pair_dec"),
               (pl.col("o_beats") & pl.col("o_active")).any().cast(pl.Int8).alias("cc_o_active_beats_pair_dec")))
    b = b.join(out, on=["pair_id", "hand_id"], how="left").with_columns(
        pl.col("cc_o_cat_dec_max").fill_null(-1), pl.col("cc_o_beats_pair_dec").fill_null(0), pl.col("cc_o_active_beats_pair_dec").fill_null(0))
    # ---- outsider responses to the pair's aggression, with cards of responder and aggressor --------------------
    pc = players.select(["hand_id", "player_id", "chen", "v_flop", "v_turn", "v_river"])
    resp = (b.select(["pair_id", "hand_id", "player_1", "player_2"]).join(actions, on="hand_id")
            .filter((pl.col("player_id") != pl.col("player_1")) & (pl.col("player_id") != pl.col("player_2")) & (pl.col("to_call") > 0)
                    & ((pl.col("last_aggr") == pl.col("player_1")) | (pl.col("last_aggr") == pl.col("player_2"))))
            .join(pc, on=["hand_id", "player_id"], how="left")
            .join(pc.rename({"player_id": "last_aggr", "chen": "a_chen", "v_flop": "a_v_flop", "v_turn": "a_v_turn", "v_river": "a_v_river"}), on=["hand_id", "last_aggr"], how="left")
            .with_columns(_v_at("", pl.col("street_no")).alias("o_v"), _v_at("a_", pl.col("street_no")).alias("a_v"))
            .with_columns((pl.col("o_v") > pl.col("a_v")).alias("o_better"), (pl.col("action") == "fold").alias("is_fold"), (pl.col("action") == "call").alias("is_call"))
            .group_by(["pair_id", "hand_id"]).agg(
                pl.col("is_fold").sum().cast(pl.Int8).alias("cc_out_fold_to_pair_n"),
                pl.len().cast(pl.Int8).alias("n_resp"),
                (pl.col("is_fold") & pl.col("o_better")).sum().cast(pl.Int8).alias("cc_out_fold_better_to_pair"),
                (pl.col("is_fold") & (pl.col("street_no") == 0) & (pl.col("chen") >= 8)).sum().cast(pl.Int8).alias("cc_out_fold_strong_pf"),
                (pl.col("is_call") & pl.col("o_better")).sum().cast(pl.Int8).alias("cc_out_call_better_to_pair"),
                pl.col("a_chen").filter(pl.col("street_no") == 0).min().cast(pl.Float32).alias("cc_aggr_chen_min_pf"),
                (pl.col("is_fold") & (pl.col("street_no") == 0) & (pl.col("chen") >= 8) & (pl.col("a_chen") <= 4)).any().cast(pl.Int8).alias("cc_iso_junk_vs_strong")))
    b = b.join(resp, on=["pair_id", "hand_id"], how="left").with_columns(
        pl.col("cc_out_fold_to_pair_n").fill_null(0), pl.col("n_resp").fill_null(0), pl.col("cc_out_fold_better_to_pair").fill_null(0),
        pl.col("cc_out_fold_strong_pf").fill_null(0), pl.col("cc_out_call_better_to_pair").fill_null(0),
        pl.col("cc_aggr_chen_min_pf").fill_null(99.0), pl.col("cc_iso_junk_vs_strong").fill_null(0),
    ).with_columns(
        (pl.col("cc_out_fold_to_pair_n") >= 4).cast(pl.Int8).alias("cc_out_fold_all4"),
        pl.when(pl.col("n_resp") > 0).then(pl.col("cc_out_fold_to_pair_n") / pl.col("n_resp")).otherwise(-1.0).cast(pl.Float32).alias("cc_out_fold_frac"),
    )
    # ---- family-style flags + combo codes ------------------------------------------------------------------------
    gap_sign = (pl.when(pl.col("cc_cat_gap_dec") > 0).then(3).when((pl.col("cc_cat_gap_dec") == 0) & (pl.col("cc_loser_better_dec") == 1)).then(2)
                .when(pl.col("cc_cat_gap_dec") == 0).then(1).otherwise(0))
    b = b.with_columns(
        pl.col("cc_fold_cat_better").alias("cc_dt_flag"),
        pl.col("cc_sd_winner_much_better").alias("cc_sp_sd_flag"),
        ((pl.col("cc_fold_cat_better") == 1) | (pl.col("cc_sd_winner_much_better") == 1)).cast(pl.Int8).alias("cc_sp_flag"),
        pl.col("cc_out_fold_all4").alias("cc_ci_flag"),
        (pl.col("cc_who_folded").cast(pl.Int32) * 1000 + pl.col("cc_dec_street").cast(pl.Int32) * 100
         + (pl.col("cc_loser_cat_dec").cast(pl.Int32) + 2).clip(0, 6) * 10 + (pl.col("cc_winner_cat_dec").cast(pl.Int32) + 2).clip(0, 6)).alias("cc_combo_a"),
        (pl.col("cc_who_folded").cast(pl.Int32) * 100 + pl.col("cc_dec_street").cast(pl.Int32) * 10 + gap_sign).alias("cc_combo_b"),
        (pl.col("cc_who_folded").cast(pl.Int32) * 100 + pl.col("cc_out_fold_to_pair_n").clip(0, 4).cast(pl.Int32) * 10
         + pl.col("cc_out_fold_better_to_pair").clip(0, 4).cast(pl.Int32)).alias("cc_combo_o"),
    )
    return b.select(["pair_id", "hand_id"] + RAW_FEATS + COMBO_COLS)


# ----------------------------------------------------------------------------------------------------------------
# fold-safe target encoding
# ----------------------------------------------------------------------------------------------------------------
def te_fit(codes, y, m=TE_M):
    prior = y.mean()
    s = pd.DataFrame({"c": codes, "y": y}).groupby("c")["y"].agg(["sum", "count"])
    enc = (s["sum"] + m * prior) / (s["count"] + m)
    return enc.to_dict(), prior


def te_apply(codes, enc, prior):
    return pd.Series(codes).map(enc).fillna(prior).to_numpy(dtype=np.float32)


def te_oof_train(codes, y, groups, n_inner=5, seed=0):
    """inner-fold OOF encoding for training rows (grouped by table)"""
    out = np.zeros(len(y), np.float32)
    ug = np.array(sorted(set(groups))); rng = np.random.RandomState(seed); gf = {g: int(v) for g, v in zip(ug, rng.permutation(len(ug)) % n_inner)}
    f = np.array([gf[g] for g in groups])
    for k in range(n_inner):
        tr, te = f != k, f == k
        enc, prior = te_fit(codes[tr], y[tr]); out[te] = te_apply(codes[te], enc, prior)
    return out


FAM_CODE = {"coordinated_isolation": 0, "directed_transfer": 1, "soft_play": 2}   # fixed so dev and eval encodings agree


def combo_codes(df: pd.DataFrame):
    fam = df["behavior_family"].map(FAM_CODE).fillna(3).to_numpy().astype(np.int64) if "behavior_family" in df else np.zeros(len(df), np.int64)
    a, b_, o = df["cc_combo_a"].to_numpy().astype(np.int64), df["cc_combo_b"].to_numpy().astype(np.int64), df["cc_combo_o"].to_numpy().astype(np.int64)
    return {"cc_te_a": a, "cc_te_b": b_, "cc_te_o": o, "cc_te_fa": fam * 100000 + a, "cc_te_fb": fam * 100000 + b_, "cc_te_fo": fam * 100000 + o}


def add_te_outer(df: pd.DataFrame, y, fold, groups):
    """dev: outer-fold-safe TE columns (training rows inner-OOF, test rows from training folds)"""
    codes = combo_codes(df)
    for name, c in codes.items():
        col = np.zeros(len(df), np.float32)
        for k in np.unique(fold):
            tr, te = fold != k, fold == k
            enc, prior = te_fit(c[tr], y[tr]); col[te] = te_apply(c[te], enc, prior)
        df[name] = col
    return df


def te_fold_features(df_tr: pd.DataFrame, y_tr, groups_tr, df_te: pd.DataFrame):
    """inside the harness: TE columns for train (inner OOF) and test (from full train)"""
    ctr, cte = combo_codes(df_tr), combo_codes(df_te)
    Xtr = np.column_stack([te_oof_train(ctr[n], y_tr, groups_tr) for n in TE_FEATS])
    Xte = np.column_stack([te_apply(cte[n], *te_fit(ctr[n], y_tr)) for n in TE_FEATS])
    return Xtr.astype(np.float32), Xte.astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------
# harness (identical to work_step3/scripts/hand_oof.py: table folds rng 42 %5, LightGBM 400 rounds, seeds 42/49)
# ----------------------------------------------------------------------------------------------------------------
def map5(df, score_col, target_col="is_ev", pair_col="pair_id"):
    vals = []
    d = df.sort_values([pair_col, score_col, "pot_bb", "hand_id"], ascending=[True, False, False, True], kind="mergesort")
    for _, g in d.groupby(pair_col, sort=False):
        rel = g[target_col].to_numpy().astype(int); n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        top = rel[:5]; hits = np.cumsum(top)
        vals.append(float(np.sum((hits / np.arange(1, len(top) + 1)) * top) / min(n_rel, 5)))
    return float(np.mean(vals))


def run_harness(df: pd.DataFrame, feats, y, fold, groups, with_te=False, seeds=(42, 49), rounds=400):
    import lightgbm as lgb
    P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=10.0, verbose=-1, num_threads=NTHREADS)
    X = df[feats].to_numpy().astype(np.float32); oof = np.zeros(len(y)); imp = {}
    for k in range(5):
        tr, te = fold != k, fold == k
        Xtr, Xte = X[tr], X[te]
        if with_te:
            Ttr, Tte = te_fold_features(df[tr], y[tr], groups[tr], df[te]); Xtr, Xte = np.hstack([Xtr, Ttr]), np.hstack([Xte, Tte])
        for sd in seeds:
            m = lgb.train({**P, "seed": sd}, lgb.Dataset(Xtr, y[tr]), num_boost_round=rounds); oof[te] += m.predict(Xte) / len(seeds)
            for n, g in zip(feats + (TE_FEATS if with_te else []), m.feature_importance("gain")):
                imp[n] = imp.get(n, 0.0) + g
    return oof, imp


def main_dev(args):
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    lab = pl.read_csv(D + "development_labels.csv"); ev = pl.read_csv(D + "development_evidence.csv")
    pos = lab.filter(pl.col("label") == 1); ids = pos["pair_id"].implode()
    base = pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(ids)).collect() for i in range(4)])
    base = (base.join(ev.select(["pair_id", "hand_id", pl.lit(True).alias("is_ev")]), on=["pair_id", "hand_id"], how="left")
            .with_columns(pl.col("is_ev").fill_null(False)).join(pos.select(["pair_id", "behavior_family"]), on="pair_id"))
    log("base", base.shape)
    players, actions, seats = load_side_tables(base["hand_id"].unique(), "development")
    feat = build_features(base, players, actions, seats)
    log("features", feat.shape, f"{time.time() - t0:.0f}s")
    src = open(ROOT + "step9_pipeline.py").read(); ns = {}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")], ns); HF = ns["HAND_FEATS"]
    df = base.join(feat, on=["pair_id", "hand_id"], how="left")
    tables = sorted(df["table_id"].unique().to_list()); rng = np.random.RandomState(42); TF = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % 5)}
    df = df.with_columns(pl.col("table_id").replace_strict(TF, return_dtype=pl.Int8).alias("fold")).sort(["pair_id", "hand_idx"])
    for f in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        df = df.with_columns((pl.col("behavior_family") == f).cast(pl.Float32).alias("fam_" + f))
    FE = HF + ["fam_directed_transfer", "fam_soft_play", "fam_coordinated_isolation"]
    pdf = df.to_pandas(); y = pdf["is_ev"].to_numpy().astype(int); fold = pdf["fold"].to_numpy(); groups = pdf["table_id"].to_numpy()
    # outer-fold OOF target encodings -> stored in the artifact
    pdf = add_te_outer(pdf, y, fold, groups)
    art = pdf[["pair_id", "hand_id"] + RAW_FEATS + COMBO_COLS + TE_FEATS]
    art.to_parquet(H + "category_combo.parquet", index=False); log("artifact saved", art.shape)
    # (a) univariate AUCs
    conf = (~pdf["is_ev"]) & (pdf["surp_pair_max"] >= 3)
    auc_all, auc_conf = {}, {}
    for c in RAW_FEATS + TE_FEATS:
        x = pdf[c].to_numpy().astype(np.float64)
        auc_all[c] = float(roc_auc_score(y, x))
        msk = pdf["is_ev"].to_numpy() | conf.to_numpy()
        auc_conf[c] = float(roc_auc_score(y[msk], x[msk]))
    print("\nfeature                          AUC(ev vs all)  AUC(ev vs confusers)")
    for c in RAW_FEATS + TE_FEATS:
        print(f"{c:32s} {auc_all[c]:.3f}           {auc_conf[c]:.3f}")
    # (b) harness without / with
    oof0, imp0 = run_harness(pdf, FE, y, fold, groups, with_te=False); pdf["s0"] = oof0; m0 = map5(pdf, "s0"); log(f"baseline MAP@5 = {m0:.4f}")
    oof1, imp1 = run_harness(pdf, FE + RAW_FEATS, y, fold, groups, with_te=True); pdf["s1"] = oof1; m1 = map5(pdf, "s1"); log(f"with cc (+TE) MAP@5 = {m1:.4f}  delta {m1 - m0:+.4f}")
    oof2, _ = run_harness(pdf, FE + RAW_FEATS, y, fold, groups, with_te=False); pdf["s2"] = oof2; m2 = map5(pdf, "s2"); log(f"with cc raw only MAP@5 = {m2:.4f}  delta {m2 - m0:+.4f}")
    print("\nper-family MAP@5 (baseline / with cc+TE / with cc raw):")
    for f in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        q = pdf[pdf["behavior_family"] == f]
        print(f"  {f:24s} {map5(q, 's0'):.4f} / {map5(q, 's1'):.4f} / {map5(q, 's2'):.4f}   (n pairs {q['pair_id'].nunique()})")
    top = sorted(imp1.items(), key=lambda kv: -kv[1])[:25]
    print("\ntop-25 gain importances (with):"); [print(f"  {n:32s} {g:12.0f}") for n, g in top]
    pdf[["pair_id", "hand_id", "hand_idx", "behavior_family", "is_ev", "fold", "s0", "s1", "s2"]].to_parquet(H + "category_combo_oof.parquet", index=False)
    print(f"\nRESULT map5_baseline={m0:.4f} map5_with={m1:.4f} delta={m1 - m0:+.4f} map5_raw_only={m2:.4f}  total {time.time() - t0:.0f}s")


def main_eval(args):
    t0 = time.time()
    pairs = pl.read_parquet(args.pairs)
    cols = BASE_COLS + (["behavior_family"] if "behavior_family" in pl.scan_parquet(W + "eval_hand_features_v9.parquet").collect_schema().names() else [])
    base = pl.scan_parquet(W + "eval_hand_features_v9.parquet").select(cols).filter(pl.col("pair_id").is_in(pairs["pair_id"].implode())).collect()
    if "behavior_family" not in base.columns and "behavior_family" in pairs.columns:
        base = base.join(pairs.select(["pair_id", "behavior_family"]), on="pair_id", how="left")
    log("eval base", base.shape)
    players, actions, seats = load_side_tables(base["hand_id"].unique(), "evaluation")
    feat = build_features(base, players, actions, seats).to_pandas()
    if "behavior_family" in base.columns:
        feat = feat.merge(base.select(["pair_id", "hand_id", "behavior_family"]).to_pandas(), on=["pair_id", "hand_id"], how="left")
    # target encodings fitted on all dev positive-pair hands
    dev = pd.read_parquet(H + "category_combo.parquet")
    lab = pd.read_csv(D + "development_labels.csv"); ev = pd.read_csv(D + "development_evidence.csv")
    dev = dev.merge(lab[["pair_id", "behavior_family"]], on="pair_id").merge(ev[["pair_id", "hand_id"]].assign(is_ev=1), on=["pair_id", "hand_id"], how="left")
    dev["is_ev"] = dev["is_ev"].fillna(0).astype(int)
    cd = combo_codes(dev); ce = combo_codes(feat) if "behavior_family" in feat else None
    for n in TE_FEATS:
        if ce is None and n in ("cc_te_fa", "cc_te_fb", "cc_te_fo"):
            feat[n] = feat[n.replace("_f", "_")]; continue
        codes_e = ce[n] if ce is not None else combo_codes(feat)[n]
        feat[n] = te_apply(codes_e, *te_fit(cd[n], dev["is_ev"].to_numpy()))
    out = args.out or (H + "category_combo_eval.parquet")
    feat[["pair_id", "hand_id"] + RAW_FEATS + COMBO_COLS + TE_FEATS].to_parquet(out, index=False); log("saved", out, feat.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="dev", choices=["dev", "eval"]); ap.add_argument("--pairs", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    main_dev(a) if a.phase == "dev" else main_eval(a)
