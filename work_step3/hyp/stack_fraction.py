"""
HYPOTHESIS stack_fraction: evidence hands are the ones where the value moved is large RELATIVE TO
STACKS (fraction of loser's starting stack lost / winner's stack gained / pot vs effective stack /
partner put all-in or covered), not in absolute chips.

Features are built ONLY from data/seats.parquet, data/actions.parquet, data/hands.parquet and the
(pair_id, hand_id, player_1, player_2) pair-hand table.  No label, evidence or is_ev information is used
anywhere in `compute_stack_fraction`.

HOW TO COMPUTE FOR EVALUATION PAIRS
-----------------------------------
    from work_step3.hyp.stack_fraction import compute_stack_fraction
    ph = pl.read_parquet("work_step3/eval_pair_hands.parquet")   # pair_id, hand_id, player_1, player_2
    feats = compute_stack_fraction(ph)                            # pair_id, hand_id + RAW + *_prank
The function is phase-agnostic: it filters seats/actions/hands to the hand_ids present in `ph`,
builds per-(hand, player) stack-relative quantities, joins them for player_1/player_2, derives the
loser/winner-oriented pair features and finally adds within-pair percentile ranks (rank(average)/len
over pair_id, exactly as the pipeline's *_prank features).  For eval process in chunks of pair_ids if
memory is a concern (each chunk must contain whole pairs so the pranks are correct).

Run as a script (from the project root) it computes the features for all hands of the 372 labelled
positive dev pairs, saves work_step3/hyp/stack_fraction.parquet and reports
  (a) univariate AUC of each feature: is_ev vs all other hands / is_ev vs confusers (surp_pair_max>=3)
  (b) MAP@5 of the standard stage-1 harness (HAND_FEATS + family one-hot) without vs with the features.
"""
import os, sys, time, warnings, json
os.environ.setdefault("POLARS_MAX_THREADS", "4")
import numpy as np, polars as pl, pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data"); W = os.path.join(ROOT, "work_step3")

AGGR = ["bet", "raise", "all_in"]

RAW = [
    # value moved relative to the loser's / winner's starting stack
    "loss_frac", "gain_frac", "transfer_frac_loser", "transfer_frac_winner",
    "contrib_frac_loser", "contrib_frac_winner", "contrib_frac_max", "contrib_frac_min",
    "loser_end_frac", "loser_busted",
    # pot relative to effective stack
    "pot_frac_eff", "pot_frac_loser", "pot_frac_winner", "spr_flop_eff",
    # covered / stack asymmetry within pair, pair stack vs table
    "loser_covered", "log_stack_ratio_lw", "eff_stack_rel_table", "stack_gap_frac",
    # all-in / facing all-in decisions, pressure relative to stack
    "allin_loser", "allin_winner", "allin_any", "allin_both", "faced_allin_loser", "faced_allin_winner",
    "max_tocall_frac_loser", "max_tocall_frac_winner", "max_tocall_frac_pair",
    "max_bet_frac_loser", "max_bet_frac_winner", "max_bet_frac_pair",
    "tocall_frac_vs_partner_max", "fold_frac_vs_partner", "call_frac_vs_partner", "bet_frac_vs_partner",
    "commit_frac_loser_vs_partner",
    # outsiders relative to their stacks
    "out_loss_frac_max", "out_contrib_frac_max", "out_allin", "out_faced_allin", "out_end_frac_min",
]
PRANK = [c for c in RAW if c not in ("loser_busted", "loser_covered", "allin_loser", "allin_winner", "allin_any",
                                       "allin_both", "faced_allin_loser", "faced_allin_winner", "out_allin", "out_faced_allin")]
FEATS = RAW + [f"{c}_prank" for c in PRANK]


def _player_action_feats(hand_ids):
    """Per (hand_id, player_id) stack-relative action quantities + per (hand_id, player_id, last_aggr) ones."""
    a = (pl.scan_parquet(os.path.join(D, "actions.parquet"))
         .filter(pl.col("hand_id").is_in(hand_ids))
         .select(["hand_id", "action_no", "street", "player_id", "action", "amount", "to_call", "pot_before", "stack_before"])
         .collect())
    # who set the current price (last aggressor before this action, same street) -> last_aggr
    a = a.sort(["hand_id", "action_no"]).with_columns(
        pl.when(pl.col("action").is_in(AGGR)).then(pl.col("player_id")).otherwise(None).alias("_ag"))
    a = a.with_columns(pl.col("_ag").shift(1).forward_fill().over(["hand_id", "street"]).alias("last_aggr")).drop("_ag")
    sb = pl.col("stack_before").clip(lower_bound=1).cast(pl.Float64)
    a = a.with_columns(
        (pl.col("to_call") / sb).alias("tocall_frac"),
        (pl.col("amount") / sb).alias("amt_frac"),
        (pl.col("to_call") >= pl.col("stack_before")).alias("faced_allin"),
        (pl.col("action") == "all_in").alias("is_allin"),
    )
    pf = a.group_by(["hand_id", "player_id"]).agg(
        pl.col("is_allin").any().alias("allin"),
        pl.col("faced_allin").any().alias("faced_allin"),
        pl.col("tocall_frac").max().alias("max_tocall_frac"),
        pl.when(pl.col("action").is_in(AGGR)).then(pl.col("amt_frac")).otherwise(0.0).max().alias("max_bet_frac"),
        pl.when(pl.col("street") == "flop").then(pl.col("stack_before")).otherwise(None).max().alias("stack_at_flop"),
    )
    # facing the partner: aggregate per (hand, actor, price-setter)
    vp = a.filter(pl.col("last_aggr").is_not_null() & (pl.col("last_aggr") != pl.col("player_id"))).group_by(["hand_id", "player_id", "last_aggr"]).agg(
        pl.col("tocall_frac").max().alias("tocall_frac_vs"),
        pl.when(pl.col("action") == "fold").then(pl.col("tocall_frac")).otherwise(0.0).max().alias("fold_frac_vs"),
        pl.when(pl.col("action") == "call").then(pl.col("tocall_frac")).otherwise(0.0).max().alias("call_frac_vs"),
        pl.when(pl.col("action").is_in(AGGR)).then(pl.col("amt_frac")).otherwise(0.0).max().alias("bet_frac_vs"),
        pl.when(pl.col("action").is_in(["call"] + AGGR)).then(pl.col("amount") / pl.col("stack_before").clip(lower_bound=1)).otherwise(0.0).sum().alias("commit_frac_vs"),
    )
    # pot at start of flop
    flop_pot = a.filter(pl.col("street") == "flop").group_by("hand_id").agg(pl.col("pot_before").min().alias("pot_flop"))
    return pf, vp, flop_pot


def compute_stack_fraction(ph: pl.DataFrame) -> pl.DataFrame:
    """ph: pair_id, hand_id, player_1, player_2 (+ anything else, ignored). Returns pair_id, hand_id + FEATS."""
    ph = ph.select(["pair_id", "hand_id", "player_1", "player_2"]).unique()
    hand_ids = ph["hand_id"].unique()
    seats = (pl.scan_parquet(os.path.join(D, "seats.parquet")).filter(pl.col("hand_id").is_in(hand_ids))
             .select(["hand_id", "player_id", "starting_stack", "total_contribution", "net_chips", "folded"]).collect())
    hands = (pl.scan_parquet(os.path.join(D, "hands.parquet")).filter(pl.col("hand_id").is_in(hand_ids))
             .select(["hand_id", "big_blind", "final_pot"]).collect())
    pf, vp, flop_pot = _player_action_feats(hand_ids)
    seats = seats.join(pf, on=["hand_id", "player_id"], how="left").with_columns(
        pl.col("allin").fill_null(False), pl.col("faced_allin").fill_null(False),
        pl.col("max_tocall_frac").fill_null(0.0), pl.col("max_bet_frac").fill_null(0.0))
    st = pl.col("starting_stack").clip(lower_bound=1).cast(pl.Float64)
    seats = seats.with_columns(
        st.alias("stack"),
        (pl.col("total_contribution") / st).alias("contrib_frac"),
        ((-pl.col("net_chips")).clip(lower_bound=0) / st).alias("loss_frac_p"),
        ((pl.col("starting_stack") + pl.col("net_chips")) / st).alias("end_frac"),
    )
    tbl = seats.group_by("hand_id").agg(pl.col("stack").mean().alias("table_stack_mean"))
    pcols = ["stack", "net_chips", "total_contribution", "contrib_frac", "loss_frac_p", "end_frac", "allin", "faced_allin",
             "max_tocall_frac", "max_bet_frac", "stack_at_flop"]
    s1 = seats.select(["hand_id", "player_id"] + pcols).rename({"player_id": "player_1", **{c: f"p1_{c}" for c in pcols}})
    s2 = seats.select(["hand_id", "player_id"] + pcols).rename({"player_id": "player_2", **{c: f"p2_{c}" for c in pcols}})
    df = ph.join(s1, on=["hand_id", "player_1"], how="left").join(s2, on=["hand_id", "player_2"], how="left")
    df = df.join(hands, on="hand_id", how="left").join(tbl, on="hand_id", how="left").join(flop_pot, on="hand_id", how="left")
    # facing partner (both directions)
    v12 = vp.rename({"player_id": "player_1", "last_aggr": "player_2", **{c: f"p1_{c}" for c in ["tocall_frac_vs", "fold_frac_vs", "call_frac_vs", "bet_frac_vs", "commit_frac_vs"]}})
    v21 = vp.rename({"player_id": "player_2", "last_aggr": "player_1", **{c: f"p2_{c}" for c in ["tocall_frac_vs", "fold_frac_vs", "call_frac_vs", "bet_frac_vs", "commit_frac_vs"]}})
    df = df.join(v12, on=["hand_id", "player_1", "player_2"], how="left").join(v21, on=["hand_id", "player_1", "player_2"], how="left")
    for p in ("p1", "p2"):
        for c in ["tocall_frac_vs", "fold_frac_vs", "call_frac_vs", "bet_frac_vs", "commit_frac_vs"]:
            df = df.with_columns(pl.col(f"{p}_{c}").fill_null(0.0))
    # outsiders (players in the hand who are not p1/p2)
    outs = (seats.join(ph.select(["pair_id", "hand_id", "player_1", "player_2"]), on="hand_id")
            .filter((pl.col("player_id") != pl.col("player_1")) & (pl.col("player_id") != pl.col("player_2")))
            .group_by(["pair_id", "hand_id"]).agg(
                pl.col("loss_frac_p").max().alias("out_loss_frac_max"),
                pl.col("contrib_frac").max().alias("out_contrib_frac_max"),
                pl.col("allin").any().cast(pl.Float32).alias("out_allin"),
                pl.col("faced_allin").any().cast(pl.Float32).alias("out_faced_allin"),
                pl.col("end_frac").min().alias("out_end_frac_min")))
    df = df.join(outs, on=["pair_id", "hand_id"], how="left")
    # loser / winner orientation (net chips; tie -> larger contribution; tie -> p1)
    loser_is_p1 = (pl.col("p1_net_chips") < pl.col("p2_net_chips")) | (
        (pl.col("p1_net_chips") == pl.col("p2_net_chips")) & (pl.col("p1_total_contribution") >= pl.col("p2_total_contribution")))
    df = df.with_columns(loser_is_p1.alias("loser_is_p1"))
    def L(c): return pl.when(pl.col("loser_is_p1")).then(pl.col(f"p1_{c}")).otherwise(pl.col(f"p2_{c}"))
    def Wn(c): return pl.when(pl.col("loser_is_p1")).then(pl.col(f"p2_{c}")).otherwise(pl.col(f"p1_{c}"))
    transfer = pl.min_horizontal((-L("net_chips")).clip(lower_bound=0), Wn("net_chips").clip(lower_bound=0)).cast(pl.Float64)
    eff = pl.min_horizontal("p1_stack", "p2_stack")
    df = df.with_columns(
        L("loss_frac_p").alias("loss_frac"),
        (Wn("net_chips").clip(lower_bound=0) / Wn("stack")).alias("gain_frac"),
        (transfer / L("stack")).alias("transfer_frac_loser"),
        (transfer / Wn("stack")).alias("transfer_frac_winner"),
        L("contrib_frac").alias("contrib_frac_loser"), Wn("contrib_frac").alias("contrib_frac_winner"),
        pl.max_horizontal("p1_contrib_frac", "p2_contrib_frac").alias("contrib_frac_max"),
        pl.min_horizontal("p1_contrib_frac", "p2_contrib_frac").alias("contrib_frac_min"),
        L("end_frac").alias("loser_end_frac"), (L("end_frac") <= 0.0).cast(pl.Float32).alias("loser_busted"),
        (pl.col("final_pot") / eff).alias("pot_frac_eff"),
        (pl.col("final_pot") / L("stack")).alias("pot_frac_loser"), (pl.col("final_pot") / Wn("stack")).alias("pot_frac_winner"),
        (pl.min_horizontal("p1_stack_at_flop", "p2_stack_at_flop") / pl.col("pot_flop").clip(lower_bound=1)).fill_null(-1.0).alias("spr_flop_eff"),
        (L("stack") <= Wn("stack")).cast(pl.Float32).alias("loser_covered"),
        (L("stack") / Wn("stack")).log().alias("log_stack_ratio_lw"),
        (eff / pl.col("table_stack_mean")).alias("eff_stack_rel_table"),
        ((pl.col("p1_stack") - pl.col("p2_stack")).abs() / pl.max_horizontal("p1_stack", "p2_stack")).alias("stack_gap_frac"),
        L("allin").cast(pl.Float32).alias("allin_loser"), Wn("allin").cast(pl.Float32).alias("allin_winner"),
        (pl.col("p1_allin") | pl.col("p2_allin")).cast(pl.Float32).alias("allin_any"),
        (pl.col("p1_allin") & pl.col("p2_allin")).cast(pl.Float32).alias("allin_both"),
        L("faced_allin").cast(pl.Float32).alias("faced_allin_loser"), Wn("faced_allin").cast(pl.Float32).alias("faced_allin_winner"),
        L("max_tocall_frac").alias("max_tocall_frac_loser"), Wn("max_tocall_frac").alias("max_tocall_frac_winner"),
        pl.max_horizontal("p1_max_tocall_frac", "p2_max_tocall_frac").alias("max_tocall_frac_pair"),
        L("max_bet_frac").alias("max_bet_frac_loser"), Wn("max_bet_frac").alias("max_bet_frac_winner"),
        pl.max_horizontal("p1_max_bet_frac", "p2_max_bet_frac").alias("max_bet_frac_pair"),
        pl.max_horizontal("p1_tocall_frac_vs", "p2_tocall_frac_vs").alias("tocall_frac_vs_partner_max"),
        pl.max_horizontal("p1_fold_frac_vs", "p2_fold_frac_vs").alias("fold_frac_vs_partner"),
        pl.max_horizontal("p1_call_frac_vs", "p2_call_frac_vs").alias("call_frac_vs_partner"),
        pl.max_horizontal("p1_bet_frac_vs", "p2_bet_frac_vs").alias("bet_frac_vs_partner"),
        L("commit_frac_vs").alias("commit_frac_loser_vs_partner"),
        pl.col("out_loss_frac_max").fill_null(0.0), pl.col("out_contrib_frac_max").fill_null(0.0),
        pl.col("out_allin").fill_null(0.0), pl.col("out_faced_allin").fill_null(0.0), pl.col("out_end_frac_min").fill_null(1.0),
    )
    df = df.with_columns([(pl.col(c).rank(method="average").over("pair_id") / pl.len().over("pair_id")).cast(pl.Float32).alias(f"{c}_prank") for c in PRANK])
    return df.select(["pair_id", "hand_id"] + FEATS).with_columns([pl.col(c).cast(pl.Float32) for c in FEATS])


# ----------------------------------------------------------------------------------------------------
def map5_within_pairs(df, score_col="hand_score", target_col="is_ev", pair_col="pair_id"):
    vals = []
    d = df.sort_values([pair_col, score_col, "pot_bb", "hand_id"], ascending=[True, False, False, True], kind="mergesort")
    for _, g in d.groupby(pair_col, sort=False):
        rel = g[target_col].to_numpy(); n_rel = int(rel.sum())
        if n_rel == 0: continue
        top = rel[:5]; hits = np.cumsum(top)
        vals.append(float(np.sum((hits / np.arange(1, len(top) + 1)) * top) / min(n_rel, 5)))
    return float(np.mean(vals)) if vals else 0.0


def run_harness(df, FE, seeds=(42, 49), rounds=400, threads=3):
    import lightgbm as lgb
    X = df.select(FE).to_numpy().astype(np.float32); y = df["is_ev"].to_numpy().astype(int); fold = df["fold"].to_numpy()
    P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=threads)
    oof = np.zeros(len(y)); imp = np.zeros(len(FE))
    for f in range(5):
        tr, te = fold != f, fold == f
        for sd in seeds:
            m = lgb.train({**P, "seed": sd}, lgb.Dataset(X[tr], y[tr]), num_boost_round=rounds)
            oof[te] += m.predict(X[te]) / len(seeds); imp += m.feature_importance("gain")
    return oof, imp


def main():
    from sklearn.metrics import roc_auc_score
    os.chdir(ROOT); t0 = time.time()
    lab = pl.read_csv(os.path.join(D, "development_labels.csv")); ev = pl.read_csv(os.path.join(D, "development_evidence.csv"))
    pos = lab.filter(pl.col("label") == 1); ids = set(pos["pair_id"])
    ph = pl.read_parquet(os.path.join(W, "dev_pair_hands.parquet")).filter(pl.col("pair_id").is_in(list(ids)))
    print("positive-pair hands:", ph.height)
    feats = compute_stack_fraction(ph)
    out = os.path.join(W, "hyp", "stack_fraction.parquet"); feats.write_parquet(out)
    print(f"features computed & saved -> {out}  ({feats.height} rows, {len(FEATS)} feats) in {time.time()-t0:.0f}s")

    # ---- load standard harness frame (labels are used ONLY for evaluation from here on) ----
    df = pl.concat([pl.scan_parquet(os.path.join(W, f"dev_hand_features_v9_{i}.parquet")).filter(pl.col("pair_id").is_in(list(ids))).collect() for i in range(4)])
    df = (df.join(ev.select(["pair_id", "hand_id", pl.lit(True).alias("is_ev")]), on=["pair_id", "hand_id"], how="left")
            .with_columns(pl.col("is_ev").fill_null(False)).join(pos.select(["pair_id", "behavior_family"]), on="pair_id"))
    src = open(os.path.join(ROOT, "step9_pipeline.py")).read(); ns = {}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")], ns); HF = ns["HAND_FEATS"]
    tables = sorted(df["table_id"].unique().to_list()); rng = np.random.RandomState(42); TF = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % 5)}
    df = df.with_columns(pl.col("table_id").replace_strict(TF, return_dtype=pl.Int8).alias("fold")).sort(["pair_id", "hand_idx"])
    for f in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        df = df.with_columns((pl.col("behavior_family") == f).cast(pl.Float32).alias("fam_" + f))
    df = df.join(feats, on=["pair_id", "hand_id"], how="left")
    assert df.select(pl.col(FEATS[0]).is_null().sum()).item() == 0, "feature join failed"
    print("harness rows:", df.height, " evidence:", int(df["is_ev"].sum()))

    # ---- (a) univariate AUCs ----
    y_all = df["is_ev"].to_numpy(); conf_mask = (~df["is_ev"]).to_numpy() & (df["surp_pair_max"].fill_null(0).to_numpy() >= 3.0)
    sub = y_all | conf_mask
    print(f"confusers: {int(conf_mask.sum())}")
    rows = []
    for c in FEATS:
        x = df[c].fill_null(-9).to_numpy().astype(float)
        a1 = roc_auc_score(y_all, x); a2 = roc_auc_score(y_all[sub], x[sub])
        rows.append((c, a1, a2, float(np.median(x[y_all])), float(np.median(x[conf_mask]))))
    auc = pd.DataFrame(rows, columns=["feature", "auc_ev_vs_all", "auc_ev_vs_conf", "med_ev", "med_conf"])
    auc["max_dev_all"] = (auc["auc_ev_vs_all"] - 0.5).abs(); auc = auc.sort_values("max_dev_all", ascending=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print(auc.drop(columns="max_dev_all").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    auc.to_csv(os.path.join(W, "hyp", "stack_fraction_auc.csv"), index=False)

    # ---- (b) harness MAP@5 without / with ----
    FAM = ["fam_directed_transfer", "fam_soft_play", "fam_coordinated_isolation"]
    base_FE = HF + FAM; with_FE = HF + FAM + FEATS
    pdf = df.select(["pair_id", "hand_id", "pot_bb", "is_ev", "behavior_family", "fold"]).to_pandas()
    t1 = time.time(); oof0, _ = run_harness(df, base_FE); pdf["hand_score"] = oof0
    m0 = map5_within_pairs(pdf); fam0 = {f: map5_within_pairs(pdf[pdf.behavior_family == f]) for f in ["directed_transfer", "soft_play", "coordinated_isolation"]}
    print(f"BASELINE  MAP@5 = {m0:.4f}  per family {json.dumps({k: round(v,4) for k,v in fam0.items()})}  ({time.time()-t1:.0f}s)")
    t1 = time.time(); oof1, imp1 = run_harness(df, with_FE); pdf["hand_score"] = oof1
    m1 = map5_within_pairs(pdf); fam1 = {f: map5_within_pairs(pdf[pdf.behavior_family == f]) for f in ["directed_transfer", "soft_play", "coordinated_isolation"]}
    print(f"WITH      MAP@5 = {m1:.4f}  per family {json.dumps({k: round(v,4) for k,v in fam1.items()})}  ({time.time()-t1:.0f}s)")
    print(f"DELTA = {m1-m0:+.4f}")
    # per-fold deltas for a sense of noise
    pdf["s0"] = oof0; pdf["s1"] = oof1
    for f in range(5):
        g = pdf[pdf.fold == f]
        print(f"  fold {f}: base {map5_within_pairs(g.assign(hand_score=g.s0)):.4f}  with {map5_within_pairs(g.assign(hand_score=g.s1)):.4f}")
    imp = pd.Series(imp1, index=with_FE).sort_values(ascending=False); imp = imp / imp.sum()
    print("top-25 gain importance (WITH model):"); print(imp.head(25).to_string(float_format=lambda v: f"{v:.4f}"))
    print("new-feature share of gain:", float(imp[FEATS].sum()), " top new:", imp[FEATS].sort_values(ascending=False).head(10).round(4).to_dict())
    pdf[["pair_id", "hand_id", "is_ev", "s0", "s1"]].to_parquet(os.path.join(W, "hyp", "stack_fraction_oof.parquet"))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
