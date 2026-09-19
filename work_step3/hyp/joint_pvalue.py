"""
HYPOTHESIS joint_pvalue: the simulator picks the labelled evidence hands as the most extreme JOINT deviation of the pair
under the population action policy -- e.g. the product of the two partners' action probabilities on the decisive street,
the minimum over streets of the joint log-probability (optionally normalised by the number of decisions), or the joint
surprise of actions the partners take against EACH OTHER rather than against outsiders.

All features are built ONLY from work_step3/action_surprise.parquet (p_taken / surp per action, from the 6-class population
policy model) joined with work_step3/action_context.parquet (to_call, last_aggr, players_active, action) restricted to the two
partners' actions.  No label, evidence file or evidence_rank is used anywhere.

USAGE
  cd <project>; python3 work_step3/hyp/joint_pvalue.py            # dev positive pairs: build + evaluate (this report)
  from work_step3.hyp.joint_pvalue import build_features           # or exec the file and call build_features(...)

EVAL RECIPE (evaluation pairs)
  ph = pl.read_parquet("work_step3/eval_hand_features_v9.parquet", columns=["pair_id","hand_id","player_1","player_2"])
  feats = build_features(ph)               # -> one row per (pair_id, hand_id) with the JP_FEATS columns below
  join on (pair_id, hand_id) to the eval hand table; add the within-pair percentile ranks with add_pranks(feats) if the
  ranker was trained with them.  action_surprise/action_context already contain both phases, so nothing else changes.
"""
import polars as pl, numpy as np, os, sys, time, warnings
from scipy.stats import chi2
warnings.filterwarnings("ignore")
W = "work_step3/"; D = "data/"
OUT = W + "hyp/joint_pvalue.parquet"
NT = 3

def _nlog10_sf(stat, k):
    """Fisher combination: -log10 P(chi2_{2k} >= stat).  stat = 2 * sum(-log p).  k = number of actions."""
    stat = np.asarray(stat, dtype=np.float64); k = np.asarray(k, dtype=np.float64)
    out = np.zeros_like(stat)
    ok = k > 0
    out[ok] = -chi2.logsf(stat[ok], 2 * k[ok]) / np.log(10.0)
    return np.clip(out, 0, 50)

def build_features(ph: pl.DataFrame) -> pl.DataFrame:
    """ph: DataFrame with pair_id, hand_id, player_1, player_2 (one row per pair-hand).  Returns per (pair_id, hand_id) features."""
    t0 = time.time()
    hids = ph["hand_id"].unique()
    a = (pl.scan_parquet(W + "action_surprise.parquet").filter(pl.col("hand_id").is_in(hids))
         .join(pl.scan_parquet(W + "action_context.parquet").select(["hand_id", "action_no", "action", "to_call", "players_active", "last_aggr"]),
               on=["hand_id", "action_no"], how="left")
         .sort(["hand_id", "action_no"]).collect())
    # population (label-free) surprise mean / std for the z-normalisation
    st = pl.scan_parquet(W + "action_surprise.parquet").select(pl.col("surp").mean().alias("mu"), pl.col("surp").std().alias("sd")).collect()
    MU, SD = float(st["mu"][0]), float(st["sd"][0])
    # link every response to the specific bet/raise it faces (same street, same forward-fill rule as the pipeline's last_aggr)
    a = (a.with_columns(pl.when(pl.col("is_aggr")).then(pl.col("action_no")).otherwise(None).alias("_agno"))
          .with_columns(pl.col("_agno").shift(1).forward_fill().over(["hand_id", "street_no"]).alias("last_aggr_no"))
          .join(a.select(["hand_id", pl.col("action_no").alias("last_aggr_no"), pl.col("surp").alias("surp_faced")]), on=["hand_id", "last_aggr_no"], how="left")
          .with_columns(pl.col("surp_faced").fill_null(0.0)))
    hand_last = a.group_by("hand_id").agg(pl.col("street_no").max().alias("hand_last_street"))
    fold_at = a.filter(pl.col("action") == "fold").group_by(["hand_id", "player_id"]).agg(pl.col("action_no").min().alias("partner_fold_no")).rename({"player_id": "partner"})
    members = pl.concat([
        ph.select(["pair_id", "hand_id", pl.col("player_1").alias("member"), pl.col("player_2").alias("partner"), pl.lit(1, dtype=pl.Int8).alias("is_p1")]),
        ph.select(["pair_id", "hand_id", pl.col("player_2").alias("member"), pl.col("player_1").alias("partner"), pl.lit(0, dtype=pl.Int8).alias("is_p1")]),
    ])
    ma = (members.join(a, left_on=["hand_id", "member"], right_on=["hand_id", "player_id"], how="inner")
          .join(fold_at, on=["hand_id", "partner"], how="left")
          .with_columns(
              ((pl.col("to_call") > 0) & (pl.col("last_aggr") == pl.col("partner"))).alias("facing_partner"),
              ((pl.col("to_call") > 0) & pl.col("last_aggr").is_not_null() & (pl.col("last_aggr") != pl.col("partner"))).alias("facing_other"),
              (pl.col("to_call") == 0).alias("unfaced"),
              (pl.col("partner_fold_no").is_null() | (pl.col("partner_fold_no") > pl.col("action_no"))).alias("partner_active"),
          ))
    # ---- per street ----------------------------------------------------------------------------------
    S = (ma.group_by(["pair_id", "hand_id", "street_no"]).agg(
            pl.col("surp").sum().alias("s_sum"), pl.len().alias("s_n"),
            pl.col("surp").filter(pl.col("is_p1") == 1).sum().alias("s1_sum"), pl.col("surp").filter(pl.col("is_p1") == 0).sum().alias("s2_sum"),
            pl.col("surp").filter(pl.col("is_p1") == 1).max().alias("s1_max"), pl.col("surp").filter(pl.col("is_p1") == 0).max().alias("s2_max"),
            (pl.col("is_p1") == 1).sum().alias("s1_n"), (pl.col("is_p1") == 0).sum().alias("s2_n"),
            pl.col("surp").filter(pl.col("partner_active")).sum().alias("s_act_sum"), pl.col("partner_active").sum().alias("s_act_n"))
         .with_columns(
            ((pl.col("s1_n") > 0) & (pl.col("s2_n") > 0)).alias("both_act"),
            pl.min_horizontal("s1_sum", "s2_sum").alias("s_min12"),
            (pl.col("s1_max").fill_null(0) + pl.col("s2_max").fill_null(0)).alias("s_worst12")))
    S = S.with_columns(pl.Series("s_fisher", _nlog10_sf(2 * S["s_sum"].to_numpy(), S["s_n"].to_numpy())))
    S = S.join(hand_last, on="hand_id", how="left")
    both_end = S.filter(pl.col("both_act")).group_by(["pair_id", "hand_id"]).agg(pl.col("street_no").max().alias("pair_end_street"))
    S = S.join(both_end, on=["pair_id", "hand_id"], how="left")
    per_street = S.group_by(["pair_id", "hand_id"]).agg(
        *[pl.col("s_sum").filter(pl.col("street_no") == s).sum().alias(f"jp_js_s{s}") for s in range(4)],
        pl.col("s_sum").max().alias("jp_js_max_street"),
        (pl.col("s_sum") / pl.col("s_n")).max().alias("jp_js_max_street_norm"),
        pl.col("street_no").sort_by("s_sum", descending=True).first().cast(pl.Int8).alias("jp_js_argmax_street"),
        pl.col("s_fisher").max().alias("jp_fisher_street_max"),
        pl.col("s_min12").filter(pl.col("both_act")).max().alias("jp_js_both_street_max"),
        pl.col("s_min12").filter(pl.col("both_act")).sum().alias("jp_js_both_street_sum"),
        pl.col("s_worst12").filter(pl.col("both_act")).max().alias("jp_js_worst_same_street"),
        pl.col("both_act").sum().cast(pl.Int8).alias("jp_n_both_streets"),
        pl.col("s_sum").filter(pl.col("street_no") == pl.col("hand_last_street")).sum().alias("jp_js_last_street"),
        pl.col("s_sum").filter(pl.col("street_no") == pl.col("pair_end_street")).sum().alias("jp_js_pair_end_street"),
        pl.col("s_fisher").filter(pl.col("street_no") == pl.col("pair_end_street")).sum().alias("jp_fisher_pair_end_street"),
        pl.col("s_act_sum").sum().alias("jp_js_partner_active"), pl.col("s_act_n").sum().alias("jp_n_partner_active"),
        pl.col("s_sum").filter(pl.col("street_no") > 0).max().alias("jp_js_max_post_street"),
    )
    # ---- per hand -------------------------------------------------------------------------------------
    H = ma.group_by(["pair_id", "hand_id"]).agg(
        pl.col("surp").sum().alias("js_sum"), pl.len().alias("jp_n_pair_actions"),
        pl.col("surp").filter(pl.col("is_p1") == 1).max().alias("s1_max"), pl.col("surp").filter(pl.col("is_p1") == 0).max().alias("s2_max"),
        pl.col("surp").sort(descending=True).head(2).sum().alias("jp_js_top2"),
        pl.col("surp").filter(pl.col("facing_partner")).sum().alias("jp_js_face_partner"), pl.col("facing_partner").sum().alias("n_fp"),
        pl.col("surp").filter(pl.col("facing_other")).sum().alias("jp_js_face_other"), pl.col("facing_other").sum().alias("n_fo"),
        pl.col("surp").filter(pl.col("unfaced")).sum().alias("jp_js_unfaced"),
        pl.col("surp").filter(pl.col("unfaced") & pl.col("partner_active")).sum().alias("jp_js_unfaced_pactive"),
        pl.col("surp").filter(~pl.col("partner_active")).sum().alias("jp_js_after_partner_fold"),
        # exchanges: a response facing the partner's bet + the surprise of that bet itself
        (pl.col("surp") + pl.col("surp_faced")).filter(pl.col("facing_partner")).max().alias("jp_exchange_max"),
        (pl.col("surp") + pl.col("surp_faced")).filter(pl.col("facing_partner")).sum().alias("jp_exchange_sum"),
        pl.min_horizontal("surp", "surp_faced").filter(pl.col("facing_partner")).max().alias("jp_exchange_both_max"),
        pl.col("p_taken").filter(pl.col("facing_partner")).min().alias("jp_p_min_face_partner"),
        pl.col("p_taken").filter(pl.col("facing_other")).min().alias("jp_p_min_face_other"),
    )
    H = H.with_columns(
        (pl.col("s1_max").fill_null(0) + pl.col("s2_max").fill_null(0)).alias("jp_worst_prod"),        # -log(p1_min * p2_min)
        pl.Series("jp_fisher_all", _nlog10_sf(2 * H["js_sum"].to_numpy(), H["jp_n_pair_actions"].to_numpy())),
        pl.Series("jp_fisher_face_partner", _nlog10_sf(2 * H["jp_js_face_partner"].fill_null(0).to_numpy(), H["n_fp"].to_numpy())),
        ((pl.col("js_sum") - pl.col("jp_n_pair_actions") * MU) / (SD * pl.col("jp_n_pair_actions").cast(pl.Float64).sqrt())).alias("jp_js_z"),
        (pl.col("js_sum") / pl.col("jp_n_pair_actions")).alias("jp_js_mean"),
        (pl.col("jp_js_face_partner").fill_null(0) / pl.max_horizontal(pl.col("n_fp"), 1) - pl.col("jp_js_face_other").fill_null(0) / pl.max_horizontal(pl.col("n_fo"), 1)).alias("jp_face_partner_minus_other"),
        pl.col("n_fp").cast(pl.Int8).alias("jp_n_face_partner"),
    )
    out = ph.select(["pair_id", "hand_id"]).join(H, on=["pair_id", "hand_id"], how="left").join(per_street, on=["pair_id", "hand_id"], how="left")
    out = out.with_columns(
        pl.Series("jp_fisher_partner_active", _nlog10_sf(2 * out["jp_js_partner_active"].fill_null(0).to_numpy(), out["jp_n_partner_active"].fill_null(0).to_numpy())),
        (pl.col("jp_js_partner_active").fill_null(0) / pl.max_horizontal(pl.col("jp_n_partner_active").fill_null(0), 1)).alias("jp_js_partner_active_mean"),
    )
    feats = [c for c in out.columns if c.startswith("jp_")]
    out = out.select(["pair_id", "hand_id"] + feats).with_columns([pl.col(c).fill_null(0.0).cast(pl.Float32) for c in feats])
    print(f"build_features: {out.shape} in {time.time()-t0:.1f}s; {len(feats)} features", flush=True)
    return out

def add_pranks(df: pl.DataFrame, cols):
    return df.with_columns([(pl.col(c).rank(method="average").over("pair_id") / pl.len().over("pair_id")).cast(pl.Float32).alias(f"{c}_prank") for c in cols])

# =====================================================================================================
if __name__ == "__main__":
    import lightgbm as lgb, pandas as pd
    from sklearn.metrics import roc_auc_score
    lab = pl.read_csv(D + "development_labels.csv"); ev = pl.read_csv(D + "development_evidence.csv")
    pos = lab.filter(pl.col("label") == 1); ids = set(pos["pair_id"])
    df = pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).collect() for i in range(4)])
    df = (df.join(ev.select(["pair_id", "hand_id", pl.lit(True).alias("is_ev")]), on=["pair_id", "hand_id"], how="left").with_columns(pl.col("is_ev").fill_null(False))
            .join(pos.select(["pair_id", "behavior_family"]), on="pair_id"))
    F = build_features(df.select(["pair_id", "hand_id", "player_1", "player_2"]))
    F.write_parquet(OUT); print("saved", OUT, flush=True)
    JP = [c for c in F.columns if c.startswith("jp_")]
    df = df.join(F, on=["pair_id", "hand_id"], how="left")
    df = add_pranks(df, JP); JPP = [f"{c}_prank" for c in JP]
    src = open("step9_pipeline.py").read(); ns = {}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")], ns); HF = ns["HAND_FEATS"]
    tables = sorted(df["table_id"].unique().to_list()); rng = np.random.RandomState(42); TF = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % 5)}
    df = df.with_columns(pl.col("table_id").replace_strict(TF, return_dtype=pl.Int8).alias("fold")).sort(["pair_id", "hand_idx"])
    FAM = ["directed_transfer", "soft_play", "coordinated_isolation"]
    for f in FAM: df = df.with_columns((pl.col("behavior_family") == f).cast(pl.Float32).alias("fam_" + f))
    FAMC = ["fam_" + f for f in FAM]
    pdf = df.to_pandas(); y = pdf.is_ev.values.astype(int); fold = pdf.fold.values; pid = pdf.pair_id.values
    conf = (~pdf.is_ev.values) & (pdf.surp_pair_max.values >= 3)
    fam = pdf.behavior_family.values

    def map5(s, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        dd = pd.DataFrame({"p": pid[m], "s": s[m], "y": y[m], "pot": pdf.pot_bb.values[m], "h": pdf.hand_id.values[m]}).sort_values(["p", "s", "pot", "h"], ascending=[True, False, False, True]); vals = []
        for _, g in dd.groupby("p", sort=False):
            r = g.y.values; n = r.sum()
            if n == 0: continue
            t = r[:5]; vals.append((np.cumsum(t) / np.arange(1, len(t) + 1) * t).sum() / min(n, 5))
        return float(np.mean(vals))

    # (a) univariate AUCs + sort-key MAP@5 --------------------------------------------------------------
    print("\n== univariate: AUC(ev vs all non-ev) | AUC(ev vs confusers surp_pair_max>=3) | MAP@5 as sole within-pair sort key (all | DT | SP | CI)")
    ref = ["surp_pair_max", "surp_pair_sum", "surp_pair_min", "surp_vs_partner", "p_min_pair", "logp_mean_sum", "surp_post_sum"]
    auc_all, auc_conf, rows = {}, {}, []
    for c in ref + JP:
        v = pdf[c].values.astype(np.float64)
        if c == "p_min_pair": v = -v
        a1 = roc_auc_score(y, v); a2 = roc_auc_score(y[y.astype(bool) | conf], v[y.astype(bool) | conf])
        mk = map5(v); mf = [map5(v, fam == f) for f in FAM]
        rows.append((c, a1, a2, mk, *mf))
        if c in JP: auc_all[c] = round(a1, 4); auc_conf[c] = round(a2, 4)
    for r in sorted(rows, key=lambda r: -r[3]): print(f"{r[0]:32s} auc_all={r[1]:.4f} auc_conf={r[2]:.4f} map5={r[3]:.4f}  DT={r[4]:.4f} SP={r[5]:.4f} CI={r[6]:.4f}")

    # (b) harness ---------------------------------------------------------------------------------------
    P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=NT)
    def run(feats, rounds=400, seeds=(42, 49)):
        X = pdf[feats].values.astype(np.float32); oof = np.zeros(len(y)); imp = np.zeros(len(feats))
        for f in range(5):
            tr, te = fold != f, fold == f
            for sd in seeds:
                m = lgb.train({**P, "seed": sd}, lgb.Dataset(X[tr], y[tr]), num_boost_round=rounds); oof[te] += m.predict(X[te]) / len(seeds); imp += m.feature_importance("gain")
        return oof, imp
    def report(name, oof):
        print(f"{name:44s} MAP@5={map5(oof):.4f}  DT={map5(oof, fam==FAM[0]):.4f} SP={map5(oof, fam==FAM[1]):.4f} CI={map5(oof, fam==FAM[2]):.4f}  AUC_conf={roc_auc_score(y[y.astype(bool)|conf], oof[y.astype(bool)|conf]):.4f}", flush=True)
    t = time.time()
    o0, _ = run(HF + FAMC); report("stage-1 baseline (HAND_FEATS+family)", o0)
    o1, imp1 = run(HF + FAMC + JP); report("+ joint_pvalue raw", o1)
    o2, imp2 = run(HF + FAMC + JP + JPP); report("+ joint_pvalue raw + prank", o2)
    o3, _ = run(HF + FAMC + JPP); report("+ joint_pvalue prank only", o3)
    print(f"harness time {time.time()-t:.0f}s")
    top = sorted(zip(imp2, HF + FAMC + JP + JPP), reverse=True)[:25]; print("top gain (raw+prank run):", [(f, int(v)) for v, f in top])
    print("jp gain share (raw run): %.3f" % (imp1[len(HF + FAMC):].sum() / imp1.sum()))
    np.save(W + "hyp/joint_pvalue_oofs.npy", np.vstack([o0, o1, o2, o3]))
    import json; json.dump({"auc_all": auc_all, "auc_conf": auc_conf, "map5_baseline": map5(o0), "map5_raw": map5(o1), "map5_raw_prank": map5(o2), "map5_prank": map5(o3)}, open(W + "hyp/joint_pvalue_results.json", "w"), indent=1)
