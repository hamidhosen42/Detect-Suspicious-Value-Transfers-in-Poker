"""
HYPOTHESIS street_timing: labelled evidence hands are the ones where the collusion mechanism
executed on its "canonical" street (dump post-flop, check-down reaching the river, isolation squeeze
pre-flop), while confusers execute it on another street.

Street-resolved partner-interaction features per (pair_id, hand_id):
  co-presence   : n_streets_both_in, pair_exit_street, pair_fold_gap, both_in_end, end_street(_act)
  heads-up      : hu_s1..hu_s3 (pair heads-up at start of flop/turn/river), n_hu_streets, hu_street_first,
                  hu_at_end, isolate_street (street on which the last outsider folded while the pair was still in)
  facing        : first_face_street, last_face_street, n_face_streets, outsiders_at_first_face, first_face_hu,
                  fold_to_partner_street, call_partner_street, raise_partner_street, fold_to_partner_hu,
                  first_aggr_street, last_aggr_street, n_aggr_streets
  outsiders     : outsiders_s1..outsiders_s3 (outsiders still in at start of street), outsiders_at_end,
                  outsiders_showdown, loser_fold_street
  deviation     : dev_first_min/max/gap (first street where each partner had surp>2), both_dev_same_street,
                  n_dev_streets, last_dev_street, max_surp_street, surp_s0..surp_s3, surpmax_s0..surpmax_s3,
                  dev_after_face, dev_at_end, loser_dev_street, loser_surp_end
  derived       : end_minus_face, end_minus_hu, face_eq_end

None of the features uses is_ev / evidence_rank / the pair label.

RESULT (dev, 372 positive pairs, standard stage-1 harness HAND_FEATS+family, GroupKFold by table, seeds 42/49):
  baseline MAP@5 0.5183 -> all 53 features 0.5246 (+0.0063, paired bootstrap 95% CI [-0.003,+0.016]);
  deviation-timing subset (surp_s*/surpmax_s*/dev_*/max_surp_street/loser_*) alone: -0.0055 (harmful);
  structural subset (co-presence / heads-up / facing / outsider streets) alone: +0.0067, gain almost all in
  coordinated_isolation (0.378 -> 0.402). RECOMMENDED: use STRUCT_FEATURES only (see bottom of file).

USAGE
  python3 work_step3/hyp/street_timing.py                # dev experiment (positive pairs), writes
                                                          #   work_step3/hyp/street_timing.parquet + .log
  python3 work_step3/hyp/street_timing.py --phase evaluation --pairs work_step3/eval_pair_hands.parquet \
        --out work_step3/hyp/street_timing_eval.parquet  # same features for evaluation pairs
  From another script: from street_timing import build_street_features
        feats = build_street_features(pair_hands_df, phase)  # pair_hands_df: pair_id, hand_id, player_1, player_2
  Inputs: work_step3/action_context.parquet, work_step3/action_surprise.parquet, data/hands.parquet, data/seats.parquet.
  The features are absolute per-hand quantities (no within-pair ranks), so they are phase-length independent.
"""
import argparse, os, sys, time, warnings
import numpy as np, pandas as pd, polars as pl

warnings.filterwarnings("ignore")
ROOT = os.environ.get("POKER_ROOT", ".").rstrip("/")
D = os.environ.get("POKER_DATA_DIR", f"{ROOT}/data").rstrip("/") + "/"; W = os.environ.get("POKER_WORK_DIR", f"{ROOT}/work_step3").rstrip("/") + "/"; H = W + "hyp/"
NOFOLD = 99  # sentinel for "never folded"
T0 = time.time()


def log(msg, fh=None):
    line = f"[{time.time() - T0:6.0f}s] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n"); fh.flush()


# --------------------------------------------------------------------------------------------------
# feature builder
# --------------------------------------------------------------------------------------------------
def build_street_features(ph: pl.DataFrame, phase: str) -> pl.DataFrame:
    """ph: DataFrame with pair_id, hand_id, player_1, player_2 (one row per pair-hand). Returns pair_id, hand_id + features."""
    ph = ph.select(["pair_id", "hand_id", "player_1", "player_2"]).unique()
    hids = ph["hand_id"].unique()

    # hand-level: end street from the board (0 pre-flop .. 3 river), showdown count
    hands = (pl.scan_parquet(D + "hands.parquet").filter(pl.col("hand_id").is_in(hids))
             .select(["hand_id", "board_cards", "players_at_showdown"])
             .with_columns(pl.col("board_cards").fill_null("").str.strip_chars().alias("bc"))
             .with_columns(pl.when(pl.col("bc") == "").then(0).otherwise(pl.col("bc").str.count_matches(" ") + 1).alias("nb"))
             .with_columns(pl.when(pl.col("nb") >= 5).then(3).when(pl.col("nb") == 4).then(2).when(pl.col("nb") == 3).then(1).otherwise(0).cast(pl.Int8).alias("end_street"))
             .select(["hand_id", "end_street", "players_at_showdown"]).collect())

    seats = (pl.scan_parquet(D + "seats.parquet").filter(pl.col("hand_id").is_in(hids))
             .select(["hand_id", "player_id", "net_chips", "went_to_showdown"]).collect())

    ac = (pl.scan_parquet(W + "action_context.parquet").filter((pl.col("phase") == phase) & pl.col("hand_id").is_in(hids))
          .select(["hand_id", "action_no", "street_no", "player_id", "action", "to_call", "players_active", "is_aggr", "last_aggr"])
          .join(pl.scan_parquet(W + "action_surprise.parquet").select(["hand_id", "action_no", "surp"]), on=["hand_id", "action_no"], how="left")
          .with_columns(pl.col("surp").fill_null(0.0).cast(pl.Float32), pl.col("street_no").cast(pl.Int16)).collect())

    # per (hand, player): street and action_no of the fold (if any)
    folds = (ac.filter(pl.col("action") == "fold").group_by(["hand_id", "player_id"])
             .agg(pl.col("street_no").min().alias("fold_street"), pl.col("action_no").min().alias("fold_no")))
    end_act = ac.group_by("hand_id").agg(pl.col("street_no").max().cast(pl.Int8).alias("end_street_act"))

    # pair-level fold streets of both members
    pf = (ph.join(folds.rename({"player_id": "player_1", "fold_street": "f1", "fold_no": "fn1"}), on=["hand_id", "player_1"], how="left")
            .join(folds.rename({"player_id": "player_2", "fold_street": "f2", "fold_no": "fn2"}), on=["hand_id", "player_2"], how="left")
            .with_columns(pl.col("f1").fill_null(NOFOLD), pl.col("f2").fill_null(NOFOLD), pl.col("fn1").fill_null(10**9), pl.col("fn2").fill_null(10**9))
            .join(hands, on="hand_id", how="left").join(end_act, on="hand_id", how="left")
            .with_columns(pl.col("end_street_act").fill_null(0), pl.col("end_street").fill_null(0)))

    # number of folds per street per hand (all players) -> active count at start of each street = 6 - folds before it
    fps = (folds.group_by("hand_id").agg(*[(pl.col("fold_street") == s).sum().alias(f"nf{s}") for s in range(4)]))
    pf = pf.join(fps, on="hand_id", how="left").with_columns([pl.col(f"nf{s}").fill_null(0) for s in range(4)])
    pf = pf.with_columns(
        pl.lit(6).alias("act0"),
        (6 - pl.col("nf0")).alias("act1"),
        (6 - pl.col("nf0") - pl.col("nf1")).alias("act2"),
        (6 - pl.col("nf0") - pl.col("nf1") - pl.col("nf2")).alias("act3"),
    )
    # outsider fold streets: the isolation street = street of the last outsider fold while both partners were still in,
    # provided the pair ends up heads-up (all four outsiders folded)
    of = (folds.join(ph.select(["pair_id", "hand_id", "player_1", "player_2"]), on="hand_id", how="inner")
          .filter((pl.col("player_id") != pl.col("player_1")) & (pl.col("player_id") != pl.col("player_2")))
          .group_by(["pair_id", "hand_id"]).agg(pl.col("fold_street").max().alias("out_last_fold_street"), pl.col("fold_no").max().alias("out_last_fold_no"), pl.len().alias("n_out_folds")))
    pf = pf.join(of, on=["pair_id", "hand_id"], how="left").with_columns(pl.col("n_out_folds").fill_null(0), pl.col("out_last_fold_street").fill_null(-1), pl.col("out_last_fold_no").fill_null(-1))

    exprs = []
    for s in range(4):
        both = (pl.col("f1") >= s) & (pl.col("f2") >= s) & (pl.col("end_street") >= s)
        exprs.append(both.cast(pl.Int8).alias(f"both_in_s{s}"))
        members_in = (pl.col("f1") >= s).cast(pl.Int8) + (pl.col("f2") >= s).cast(pl.Int8)
        outs = (pl.col(f"act{s}") - members_in)
        exprs.append(pl.when(pl.col("end_street") >= s).then(outs).otherwise(-1).cast(pl.Int8).alias(f"outsiders_s{s}"))
        exprs.append((both & (outs == 0)).cast(pl.Int8).alias(f"hu_s{s}"))
    pf = pf.with_columns(exprs)
    pf = pf.with_columns(
        (pl.col("both_in_s0") + pl.col("both_in_s1") + pl.col("both_in_s2") + pl.col("both_in_s3")).cast(pl.Int8).alias("n_streets_both_in"),
        (pl.col("hu_s0") + pl.col("hu_s1") + pl.col("hu_s2") + pl.col("hu_s3")).cast(pl.Int8).alias("n_hu_streets"),
        pl.when(pl.col("hu_s0") == 1).then(0).when(pl.col("hu_s1") == 1).then(1).when(pl.col("hu_s2") == 1).then(2).when(pl.col("hu_s3") == 1).then(3).otherwise(-1).cast(pl.Int8).alias("hu_street_first"),
        pl.min_horizontal(pl.col("f1"), pl.col("f2")).clip(upper_bound=4).cast(pl.Int8).alias("pair_exit_street"),
        (pl.col("f1").clip(upper_bound=4) - pl.col("f2").clip(upper_bound=4)).abs().cast(pl.Int8).alias("pair_fold_gap"),
        # decisive street = end street: both still in / outsiders still in / heads-up
        pl.when(pl.col("end_street") == 0).then(pl.col("both_in_s0")).when(pl.col("end_street") == 1).then(pl.col("both_in_s1")).when(pl.col("end_street") == 2).then(pl.col("both_in_s2")).otherwise(pl.col("both_in_s3")).cast(pl.Int8).alias("both_in_end"),
        pl.when(pl.col("end_street") == 0).then(pl.col("outsiders_s0")).when(pl.col("end_street") == 1).then(pl.col("outsiders_s1")).when(pl.col("end_street") == 2).then(pl.col("outsiders_s2")).otherwise(pl.col("outsiders_s3")).cast(pl.Int8).alias("outsiders_at_end"),
        pl.when(pl.col("end_street") == 0).then(pl.col("hu_s0")).when(pl.col("end_street") == 1).then(pl.col("hu_s1")).when(pl.col("end_street") == 2).then(pl.col("hu_s2")).otherwise(pl.col("hu_s3")).cast(pl.Int8).alias("hu_at_end"),
        # isolation street: all 4 outsiders folded and both partners were still in when the last one folded
        pl.when((pl.col("n_out_folds") == 4) & (pl.col("fn1") > pl.col("out_last_fold_no")) & (pl.col("fn2") > pl.col("out_last_fold_no")))
          .then(pl.col("out_last_fold_street")).otherwise(-1).cast(pl.Int8).alias("isolate_street"),
    )

    # member-perspective action features
    members = pl.concat([
        ph.select(["pair_id", "hand_id", pl.col("player_1").alias("member"), pl.col("player_2").alias("partner"), pl.lit(1, dtype=pl.Int8).alias("is_p1")]),
        ph.select(["pair_id", "hand_id", pl.col("player_2").alias("member"), pl.col("player_1").alias("partner"), pl.lit(0, dtype=pl.Int8).alias("is_p1")]),
    ])
    ma = (members.join(ac, left_on=["hand_id", "member"], right_on=["hand_id", "player_id"], how="inner")
          .join(folds.select(["hand_id", pl.col("player_id").alias("partner"), pl.col("fold_no").alias("partner_fold_no")]), on=["hand_id", "partner"], how="left")
          .with_columns(
              ((pl.col("to_call") > 0) & (pl.col("last_aggr") == pl.col("partner"))).alias("facing_partner"),
              (pl.col("partner_fold_no").is_null() | (pl.col("partner_fold_no") > pl.col("action_no"))).alias("partner_in"),
              (pl.col("surp") > 2.0).alias("dev"),
          ).with_columns(((pl.col("players_active") == 2) & pl.col("partner_in")).alias("true_hu")))

    def first_street(cond, name):
        return pl.col("street_no").filter(cond).min().fill_null(-1).cast(pl.Int8).alias(name)

    def last_street(cond, name):
        return pl.col("street_no").filter(cond).max().fill_null(-1).cast(pl.Int8).alias(name)

    ma = ma.sort(["pair_id", "hand_id", "action_no"])
    inter = ma.group_by(["pair_id", "hand_id"]).agg(
        first_street(pl.col("facing_partner"), "first_face_street"),
        last_street(pl.col("facing_partner"), "last_face_street"),
        pl.col("street_no").filter(pl.col("facing_partner")).n_unique().fill_null(0).cast(pl.Int8).alias("n_face_streets"),
        (pl.col("players_active").filter(pl.col("facing_partner")).first() - 2).fill_null(-1).cast(pl.Int8).alias("outsiders_at_first_face"),
        first_street(pl.col("facing_partner") & (pl.col("action") == "fold"), "fold_to_partner_street"),
        first_street(pl.col("facing_partner") & (pl.col("action") == "call"), "call_partner_street"),
        first_street(pl.col("facing_partner") & pl.col("is_aggr"), "raise_partner_street"),
        (pl.col("facing_partner") & (pl.col("action") == "fold") & (pl.col("players_active") == 2)).sum().cast(pl.Int8).alias("fold_to_partner_hu"),
        first_street(pl.col("is_aggr"), "first_aggr_street"),
        last_street(pl.col("is_aggr"), "last_aggr_street"),
        pl.col("street_no").filter(pl.col("is_aggr")).n_unique().fill_null(0).cast(pl.Int8).alias("n_aggr_streets"),
        first_street(pl.col("true_hu"), "hu_act_street_first"),
        # deviation timing
        pl.col("street_no").filter(pl.col("dev") & (pl.col("is_p1") == 1)).min().fill_null(NOFOLD).alias("_d1"),
        pl.col("street_no").filter(pl.col("dev") & (pl.col("is_p1") == 0)).min().fill_null(NOFOLD).alias("_d2"),
        pl.col("street_no").filter(pl.col("dev")).n_unique().fill_null(0).cast(pl.Int8).alias("n_dev_streets"),
        last_street(pl.col("dev"), "last_dev_street"),
        # street of the max-surprise member action (ties -> earliest)
        pl.when(pl.col("surp").max() > 0).then(pl.col("street_no").sort_by(["surp", "action_no"], descending=[True, False]).first()).otherwise(-1).cast(pl.Int8).alias("max_surp_street"),
        *[pl.col("surp").filter(pl.col("street_no") == s).sum().fill_null(0.0).cast(pl.Float32).alias(f"surp_s{s}") for s in range(4)],
        *[pl.col("surp").filter(pl.col("street_no") == s).max().fill_null(0.0).cast(pl.Float32).alias(f"surpmax_s{s}") for s in range(4)],
        pl.any_horizontal([(pl.col("dev") & (pl.col("is_p1") == 1) & (pl.col("street_no") == s)).any() & (pl.col("dev") & (pl.col("is_p1") == 0) & (pl.col("street_no") == s)).any() for s in range(4)]).alias("_bsame"),
        # loser-specific (loser = lower net chips) is done below; keep per-member first dev street and surp on the last street
        pl.col("surp").filter((pl.col("is_p1") == 1) & (pl.col("street_no") == pl.col("street_no").max())).sum().fill_null(0.0).cast(pl.Float32).alias("_p1_surp_end"),
        pl.col("surp").filter((pl.col("is_p1") == 0) & (pl.col("street_no") == pl.col("street_no").max())).sum().fill_null(0.0).cast(pl.Float32).alias("_p2_surp_end"),
    )
    inter = inter.with_columns(
        pl.min_horizontal("_d1", "_d2").clip(upper_bound=4).cast(pl.Int8).alias("dev_first_min"),
        pl.max_horizontal("_d1", "_d2").clip(upper_bound=4).cast(pl.Int8).alias("dev_first_max"),
        (pl.col("_d1").clip(upper_bound=4) - pl.col("_d2").clip(upper_bound=4)).abs().cast(pl.Int8).alias("dev_first_gap"),
        pl.col("_bsame").fill_null(False).cast(pl.Int8).alias("both_dev_same_street"),
    )

    # loser side (lower net chips) from seats
    ls = (ph.join(seats.rename({"player_id": "player_1", "net_chips": "n1", "went_to_showdown": "sd1"}), on=["hand_id", "player_1"], how="left")
            .join(seats.rename({"player_id": "player_2", "net_chips": "n2", "went_to_showdown": "sd2"}), on=["hand_id", "player_2"], how="left")
            .with_columns((pl.col("n1") < pl.col("n2")).alias("loser_is_p1"), (pl.col("sd1").cast(pl.Int8) + pl.col("sd2").cast(pl.Int8)).alias("_n_sd")))

    out = (pf.join(inter, on=["pair_id", "hand_id"], how="left").join(ls.select(["pair_id", "hand_id", "loser_is_p1", "_n_sd"]), on=["pair_id", "hand_id"], how="left")
           .with_columns(
               pl.when(pl.col("loser_is_p1")).then(pl.col("f1")).otherwise(pl.col("f2")).clip(upper_bound=4).cast(pl.Int8).alias("loser_fold_street"),
               pl.when(pl.col("loser_is_p1")).then(pl.col("_d1")).otherwise(pl.col("_d2")).clip(upper_bound=4).cast(pl.Int8).alias("loser_dev_street"),
               pl.when(pl.col("loser_is_p1")).then(pl.col("_p1_surp_end")).otherwise(pl.col("_p2_surp_end")).cast(pl.Float32).alias("loser_surp_end"),
               (pl.col("players_at_showdown") - pl.col("_n_sd")).clip(lower_bound=0).cast(pl.Int8).alias("outsiders_showdown"),
           )
           .with_columns(
               pl.when(pl.col("first_face_street") >= 0).then(pl.col("end_street") - pl.col("first_face_street")).otherwise(-1).cast(pl.Int8).alias("end_minus_face"),
               pl.when(pl.col("hu_street_first") >= 0).then(pl.col("end_street") - pl.col("hu_street_first")).otherwise(-1).cast(pl.Int8).alias("end_minus_hu"),
               ((pl.col("first_face_street") >= 0) & (pl.col("first_face_street") == pl.col("end_street"))).cast(pl.Int8).alias("face_eq_end"),
               ((pl.col("dev_first_min") < 4) & (pl.col("first_face_street") >= 0) & (pl.col("dev_first_min") >= pl.col("first_face_street"))).cast(pl.Int8).alias("dev_after_face"),
               ((pl.col("last_dev_street") >= 0) & (pl.col("last_dev_street") == pl.col("end_street"))).cast(pl.Int8).alias("dev_at_end"),
           ))
    feats = FEATURES
    out = out.select(["pair_id", "hand_id", *feats])
    fill = {c: (0.0 if out.schema[c] == pl.Float32 else -1) for c in feats}
    out = out.with_columns([pl.col(c).fill_null(v) for c, v in fill.items()])
    return out


FEATURES = [
    "end_street", "end_street_act", "n_streets_both_in", "both_in_end", "pair_exit_street", "pair_fold_gap",
    "hu_s1", "hu_s2", "hu_s3", "n_hu_streets", "hu_street_first", "hu_at_end", "isolate_street", "hu_act_street_first",
    "outsiders_s1", "outsiders_s2", "outsiders_s3", "outsiders_at_end", "outsiders_showdown",
    "first_face_street", "last_face_street", "n_face_streets", "outsiders_at_first_face",
    "fold_to_partner_street", "call_partner_street", "raise_partner_street", "fold_to_partner_hu",
    "first_aggr_street", "last_aggr_street", "n_aggr_streets", "loser_fold_street",
    "dev_first_min", "dev_first_max", "dev_first_gap", "both_dev_same_street", "n_dev_streets", "last_dev_street", "max_surp_street",
    "surp_s0", "surp_s1", "surp_s2", "surp_s3", "surpmax_s0", "surpmax_s1", "surpmax_s2", "surpmax_s3",
    "loser_dev_street", "loser_surp_end", "end_minus_face", "end_minus_hu", "face_eq_end", "dev_after_face", "dev_at_end",
]


# --------------------------------------------------------------------------------------------------
# evaluation harness (dev positive pairs)
# --------------------------------------------------------------------------------------------------
def map5(df, score_col="hand_score", target_col="is_ev", pair_col="pair_id"):
    vals = []
    d = df.sort_values([pair_col, score_col, "pot_bb", "hand_id"], ascending=[True, False, False, True], kind="mergesort")
    for _, g in d.groupby(pair_col, sort=False):
        rel = g[target_col].to_numpy(); n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        top = rel[:5]; hits = np.cumsum(top)
        vals.append(float(np.sum((hits / np.arange(1, len(top) + 1)) * top) / min(n_rel, 5)))
    return float(np.mean(vals)) if vals else 0.0


def run_harness(df: pl.DataFrame, feats, fh, tag):
    import lightgbm as lgb
    X = df.select(feats).to_numpy().astype(np.float32); y = df["is_ev"].to_numpy().astype(int); fold = df["fold"].to_numpy()
    P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=3)
    oof = np.zeros(len(y)); imp = np.zeros(len(feats))
    for f in range(5):
        tr, te = fold != f, fold == f
        for sd in (42, 49):
            m = lgb.train({**P, "seed": sd}, lgb.Dataset(X[tr], y[tr]), num_boost_round=400)
            oof[te] += m.predict(X[te]) / 2; imp += m.feature_importance("gain") / 10
    pdf = df.select(["pair_id", "hand_id", "pot_bb", "is_ev", "behavior_family"]).to_pandas(); pdf["hand_score"] = oof
    m_all = map5(pdf); fam = {k: map5(pdf[pdf.behavior_family == k]) for k in ["directed_transfer", "soft_play", "coordinated_isolation"]}
    log(f"{tag}: MAP@5 = {m_all:.4f}  per family: { {k: round(v, 4) for k, v in fam.items()} }", fh)
    return m_all, fam, oof, dict(zip(feats, imp))


def main_dev():
    from sklearn.metrics import roc_auc_score
    fh = open(H + "street_timing.log", "w")
    lab = pl.read_csv(D + "development_labels.csv"); ev = pl.read_csv(D + "development_evidence.csv")
    pos = lab.filter(pl.col("label") == 1); ids = pos["pair_id"].to_list()
    ph = pl.scan_parquet(W + "dev_pair_hands.parquet").filter(pl.col("pair_id").is_in(ids)).collect()
    log(f"positive pair-hands: {ph.shape}", fh)
    feats = build_street_features(ph, "development")
    log(f"features built: {feats.shape}; nulls: {int(feats.null_count().sum_horizontal().sum())}", fh)
    feats.write_parquet(H + "street_timing.parquet")

    # base table: v9 features of positive pairs + label + family + folds
    src = open(f"{ROOT}/step9_pipeline.py").read(); ns = {}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")], ns); HF = ns["HAND_FEATS"]
    df = pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(ids)).collect() for i in range(4)])
    df = (df.join(ev.select(["pair_id", "hand_id", pl.lit(True).alias("is_ev")]), on=["pair_id", "hand_id"], how="left").with_columns(pl.col("is_ev").fill_null(False))
            .join(pos.select(["pair_id", "behavior_family"]), on="pair_id").join(feats, on=["pair_id", "hand_id"], how="left"))
    assert df["is_ev"].sum() == 1817 and df.select(FEATURES).null_count().sum_horizontal().sum() == 0
    tables = sorted(df["table_id"].unique().to_list()); rng = np.random.RandomState(42); TF = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % 5)}
    df = df.with_columns(pl.col("table_id").replace_strict(TF, return_dtype=pl.Int8).alias("fold")).sort(["pair_id", "hand_idx"])
    for f in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        df = df.with_columns((pl.col("behavior_family") == f).cast(pl.Float32).alias("fam_" + f))
    FAM = ["fam_directed_transfer", "fam_soft_play", "fam_coordinated_isolation"]

    # (a) univariate AUCs
    y = df["is_ev"].to_numpy(); conf = (~df["is_ev"]) & (df["surp_pair_max"] >= 3.0); conf = conf.to_numpy(); fam_arr = df["behavior_family"].to_numpy()
    log(f"is_ev={y.sum()}  confusers(non-ev & surp_pair_max>=3)={conf.sum()}  all non-ev={(~y).sum()}", fh)
    rows = []
    for c in FEATURES:
        x = df[c].to_numpy().astype(float)
        r = dict(feature=c, auc_all=roc_auc_score(y, x), auc_conf=roc_auc_score(y[y | conf], x[y | conf]), ev_mean=x[y].mean(), conf_mean=x[conf].mean(), non_mean=x[~y].mean())
        for fm in ["directed_transfer", "soft_play", "coordinated_isolation"]:
            mk = fam_arr == fm; r[f"auc_all_{fm[:4]}"] = roc_auc_score(y[mk], x[mk]); mk2 = mk & (y | conf); r[f"auc_conf_{fm[:4]}"] = roc_auc_score(y[mk2], x[mk2])
        rows.append(r)
    auc = pd.DataFrame(rows).set_index("feature")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    log("univariate AUC (is_ev vs all non-ev | is_ev vs confusers), overall and per family (dire/soft/coor):\n" + auc.round(3).to_string(), fh)
    auc.to_csv(H + "street_timing_auc.csv")

    # (b) harness
    base_feats = HF + FAM
    m0, f0, oof0, _ = run_harness(df, base_feats, fh, "BASELINE  HAND_FEATS+family")
    m1, f1, oof1, imp = run_harness(df, base_feats + FEATURES, fh, "WITH street_timing")
    log(f"delta MAP@5 = {m1 - m0:+.4f}   per family: { {k: round(f1[k] - f0[k], 4) for k in f0} }", fh)
    top = sorted(((imp[c], c) for c in FEATURES), reverse=True)[:20]
    log("gain importance of new features (top 20): " + ", ".join(f"{c}={g:.0f}" for g, c in top), fh)
    # paired bootstrap over pairs for the delta
    pdf = df.select(["pair_id", "hand_id", "pot_bb", "is_ev", "behavior_family"]).to_pandas()
    pairs = pdf.pair_id.unique(); rs = np.random.RandomState(0)
    per0 = {}; per1 = {}
    for name, oof, store in (("b", oof0, per0), ("w", oof1, per1)):
        pdf["hand_score"] = oof
        d = pdf.sort_values(["pair_id", "hand_score", "pot_bb", "hand_id"], ascending=[True, False, False, True], kind="mergesort")
        for pid, g in d.groupby("pair_id", sort=False):
            rel = g.is_ev.to_numpy(); top5 = rel[:5]; hits = np.cumsum(top5); store[pid] = float(np.sum((hits / np.arange(1, 6)) * top5) / min(int(rel.sum()), 5))
    dv = np.array([per1[p] - per0[p] for p in pairs]); boots = [dv[rs.randint(0, len(dv), len(dv))].mean() for _ in range(2000)]
    log(f"paired bootstrap of delta over {len(pairs)} pairs: mean {dv.mean():+.4f}, 95% CI [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}], P(delta<=0)={np.mean(np.array(boots) <= 0):.3f}", fh)
    # (c) surprise-only variants: does replacing the pf/post split by the street-resolved surprise help alone?
    sub = [c for c in FEATURES if c.startswith("surp") or c.startswith("dev") or c in ("max_surp_street", "last_dev_street", "n_dev_streets", "both_dev_same_street", "loser_dev_street", "loser_surp_end")]
    m2, f2, _, _ = run_harness(df, base_feats + sub, fh, "WITH deviation-timing subset only")
    sub2 = [c for c in FEATURES if c not in sub]
    m3, f3, _, _ = run_harness(df, base_feats + sub2, fh, "WITH structural street subset only")
    log(f"subset deltas: deviation-timing {m2 - m0:+.4f}, structural {m3 - m0:+.4f}", fh)
    fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="development"); ap.add_argument("--pairs", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.pairs is None:
        main_dev()
    else:
        ph = pl.read_parquet(a.pairs, columns=["pair_id", "hand_id", "player_1", "player_2"])
        out = build_street_features(ph, a.phase)
        out.write_parquet(a.out or H + f"street_timing_{a.phase}.parquet"); log(f"wrote {out.shape} -> {a.out}")

DEV_FEATURES = [c for c in FEATURES if c.startswith("surp") or c.startswith("dev") or c in ("max_surp_street", "last_dev_street", "n_dev_streets", "both_dev_same_street", "loser_dev_street", "loser_surp_end")]
STRUCT_FEATURES = [c for c in FEATURES if c not in DEV_FEATURES]  # the subset that carries the (small) gain
