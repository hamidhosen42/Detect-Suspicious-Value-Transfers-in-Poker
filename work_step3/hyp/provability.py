"""
Hypothesis "provability": the host labels as evidence the hands where the collusion is PROVABLE from what an
auditor would see -- showdown exposure, a folded best hand (to the partner), the partner winning with the worse
hand, an uncontested pot to the partner right after the other partner folded the best hand, etc.

Per-hand features for (pair_id, hand_id).  No labels / evidence files are used to build any feature.

HOW TO COMPUTE THE SAME FEATURES FOR EVALUATION PAIRS
-----------------------------------------------------
    import polars as pl; from provability import compute_provability
    pairs = pl.scan_parquet("work_step3/eval_hand_features.parquet").select(["pair_id","hand_id","player_1","player_2"]).collect()
    feats = compute_provability(pairs)          # -> DataFrame[pair_id, hand_id, <PROV_FEATS>]
compute_provability() reads only raw data (data/seats.parquet, data/hands.parquet) and the label-free per-player
caches (work_step3/action_context.parquet, work_step3/player_strength.parquet, work_step3/player_hands_v9.parquet
for chen / vpip_a), filtered to the hands of the pairs passed in; it works identically for dev and eval hands.
The loser/winner convention is the pipeline's: loser = player_1 if p1_net < p2_net else player_2.

Run as a script (dev): computes the features for all 45,129 positive-pair hands, saves
work_step3/hyp/provability.parquet, prints univariate AUCs and the stage-1 MAP@5 without / with the features.
"""
import polars as pl, numpy as np, pandas as pd, lightgbm as lgb, os, sys, time, warnings
warnings.filterwarnings("ignore")
D = "data/"; W = "work_step3/"; OUT = W + "hyp/provability.parquet"
NT = 3   # threads (shared machine)

STREET_V = {1: "v_flop", 2: "v_turn", 3: "v_river"}

def compute_provability(pairs: pl.DataFrame) -> pl.DataFrame:
    """pairs: DataFrame with pair_id, hand_id, player_1, player_2 (one row per pair-hand)."""
    hids = pairs["hand_id"].unique().to_list()
    hd = pl.scan_parquet(D + "hands.parquet").filter(pl.col("hand_id").is_in(hids)).select(["hand_id", "big_blind", "players_at_showdown"]).collect()
    se = pl.scan_parquet(D + "seats.parquet").filter(pl.col("hand_id").is_in(hids)).select(
        ["hand_id", "player_id", "folded", "went_to_showdown", "won_share", "net_chips", "total_contribution"]).collect()
    st = pl.scan_parquet(W + "player_strength.parquet").filter(pl.col("hand_id").is_in(hids)).select(
        ["hand_id", "player_id", "v_flop", "v_turn", "v_river", "v_final", "cat_final", "rk_final"]).collect()
    ph = pl.scan_parquet(W + "player_hands_v9.parquet").filter(pl.col("hand_id").is_in(hids)).select(["hand_id", "player_id", "chen", "vpip_a"]).collect()
    ac = pl.scan_parquet(W + "action_context.parquet").filter(pl.col("hand_id").is_in(hids)).select(
        ["hand_id", "action_no", "street_no", "player_id", "action", "to_call", "players_active", "last_aggr", "to_call_bb"]).collect()

    # ---- per-player table ----------------------------------------------------------------------------------
    folds = (ac.filter(pl.col("action") == "fold").sort(["hand_id", "action_no"]).group_by(["hand_id", "player_id"]).first()
             .select(["hand_id", "player_id", pl.col("action_no").alias("fold_action_no"), pl.col("street_no").cast(pl.Int8).alias("fold_street"),
                      pl.col("last_aggr").alias("fold_facing"), pl.col("players_active").cast(pl.Int8).alias("fold_n_active"), pl.col("to_call_bb").alias("fold_to_call_bb")]))
    last = ac.sort(["hand_id", "action_no"]).group_by("hand_id").last().select(["hand_id", pl.col("player_id").alias("last_player"), (pl.col("action") == "fold").alias("last_is_fold")])
    # calls made while facing a specific aggressor: (hand, caller, aggressor) -> n
    calls = (ac.filter((pl.col("action") == "call") & pl.col("last_aggr").is_not_null()).group_by(["hand_id", "player_id", "last_aggr"]).len()
             .rename({"player_id": "caller", "last_aggr": "aggressor", "len": "n_calls_facing"}))
    P = (se.join(st, on=["hand_id", "player_id"], how="left").join(ph, on=["hand_id", "player_id"], how="left")
         .join(folds, on=["hand_id", "player_id"], how="left").join(hd, on="hand_id", how="left")
         .with_columns(pl.col("chen").fill_null(0.0), pl.col("vpip_a").fill_null(0).cast(pl.Int8),
                       pl.when(pl.col("fold_street") == 1).then(pl.col("v_flop")).when(pl.col("fold_street") == 2).then(pl.col("v_turn"))
                         .when(pl.col("fold_street") == 3).then(pl.col("v_river")).otherwise(None).alias("v_at_fold")))
    # remaining-at-end aggregates (players who never folded) and showdown aggregates
    agg = P.group_by("hand_id").agg(
        pl.col("v_final").filter(~pl.col("folded")).max().alias("h_vmax_rem"),
        pl.col("cat_final").filter(~pl.col("folded")).max().alias("h_catmax_rem"),
        pl.col("v_final").filter(pl.col("went_to_showdown")).max().alias("h_vmax_sd"),
        pl.col("v_final").max().alias("h_vmax_all"),
        (~pl.col("folded")).sum().cast(pl.Int8).alias("h_n_notfold"),
        pl.col("net_chips").filter(pl.col("net_chips") < 0).sum().alias("h_total_lost"))
    # for every fold: best hand among players still active AFTER that fold (folded later or never)
    fj = (folds.join(P.select(["hand_id", pl.col("player_id").alias("q"), pl.col("fold_action_no").alias("q_fold_no"), pl.col("v_final").alias("q_vf"),
                              pl.col("v_flop").alias("q_v1"), pl.col("v_turn").alias("q_v2"), pl.col("v_river").alias("q_v3"), pl.col("chen").alias("q_chen"), pl.col("cat_final").alias("q_cat")]), on="hand_id")
          .filter((pl.col("q") != pl.col("player_id")) & (pl.col("q_fold_no").is_null() | (pl.col("q_fold_no") > pl.col("fold_action_no"))))
          .with_columns(pl.when(pl.col("fold_street") == 1).then(pl.col("q_v1")).when(pl.col("fold_street") == 2).then(pl.col("q_v2")).when(pl.col("fold_street") == 3).then(pl.col("q_v3")).otherwise(None).alias("q_vs"))
          .group_by(["hand_id", "player_id"]).agg(pl.col("q_vf").max().alias("rem_vmax_final"), pl.col("q_vs").max().alias("rem_vmax_street"), pl.col("q_chen").max().alias("rem_chen_max"), pl.col("q_cat").max().alias("rem_catmax")))
    P = P.join(fj, on=["hand_id", "player_id"], how="left").join(agg, on="hand_id", how="left").join(last, on="hand_id", how="left")
    P = P.with_columns(
        (pl.col("won_share") > 0).alias("won"),
        (pl.col("fold_action_no").is_not_null() & (pl.col("last_player") == pl.col("player_id")) & pl.col("last_is_fold")).alias("fold_ends_hand"),
        (pl.col("folded") & (pl.col("rk_final") == 1)).alias("fold_best6"),
        (pl.col("folded") & (pl.col("v_final") > pl.col("h_vmax_rem").fill_null(-1))).alias("fold_best_rem"),
        (pl.col("folded") & (pl.col("v_final") > pl.col("rem_vmax_final").fill_null(-1))).alias("fold_best_at_fold"),
        (pl.col("folded") & (pl.col("fold_street") >= 1) & (pl.col("v_at_fold") > pl.col("rem_vmax_street").fill_null(-1))).alias("fold_best_street"),
        (pl.col("folded") & (pl.col("fold_street") == 0) & (pl.col("chen") > pl.col("rem_chen_max").fill_null(-99))).alias("fold_best_pf_chen"),
        (pl.col("folded") & (pl.col("went_to_showdown").not_()) & (pl.col("v_final") > pl.col("h_vmax_sd").fill_null(-1)) & (pl.col("players_at_showdown") >= 1)).alias("fold_beats_showdown"),
        pl.when(pl.col("folded")).then(pl.col("cat_final") - pl.col("rem_catmax").fill_null(0)).otherwise(0).cast(pl.Float32).alias("fold_cat_margin"),
        (pl.col("total_contribution").cast(pl.Float64) / pl.col("big_blind")).cast(pl.Float32).alias("contrib_bb"),
    )
    cols = ["folded", "went_to_showdown", "won", "won_share", "net_chips", "contrib_bb", "chen", "vpip_a", "v_final", "cat_final", "rk_final",
            "fold_street", "fold_facing", "fold_n_active", "fold_to_call_bb", "fold_ends_hand", "fold_best6", "fold_best_rem", "fold_best_at_fold",
            "fold_best_street", "fold_best_pf_chen", "fold_beats_showdown", "fold_cat_margin"]
    hcols = ["players_at_showdown", "h_vmax_rem", "h_vmax_all", "h_n_notfold", "h_total_lost", "big_blind"]

    # ---- pair-hand table -----------------------------------------------------------------------------------
    X = pairs.select(["pair_id", "hand_id", "player_1", "player_2"])
    X = X.join(P.select(["hand_id", pl.col("player_id").alias("player_1")] + [pl.col(c).alias("a_" + c) for c in cols] + hcols), on=["hand_id", "player_1"], how="left")
    X = X.join(P.select(["hand_id", pl.col("player_id").alias("player_2")] + [pl.col(c).alias("b_" + c) for c in cols]), on=["hand_id", "player_2"], how="left")
    X = X.with_columns((pl.col("a_net_chips") < pl.col("b_net_chips")).alias("loser_is_p1"))
    L = lambda c: pl.when(pl.col("loser_is_p1")).then(pl.col("a_" + c)).otherwise(pl.col("b_" + c))
    Wn = lambda c: pl.when(pl.col("loser_is_p1")).then(pl.col("b_" + c)).otherwise(pl.col("a_" + c))
    X = X.with_columns(pl.when(pl.col("loser_is_p1")).then(pl.col("player_1")).otherwise(pl.col("player_2")).alias("loser_id"),
                       pl.when(pl.col("loser_is_p1")).then(pl.col("player_2")).otherwise(pl.col("player_1")).alias("winner_id"))
    X = X.with_columns([L(c).alias("L_" + c) for c in cols] + [Wn(c).alias("W_" + c) for c in cols])
    # calls by loser while facing the winner's bet, and by winner facing loser
    X = (X.join(calls.rename({"caller": "loser_id", "aggressor": "winner_id", "n_calls_facing": "L_calls_facing_W"}), on=["hand_id", "loser_id", "winner_id"], how="left")
          .join(calls.rename({"caller": "winner_id", "aggressor": "loser_id", "n_calls_facing": "W_calls_facing_L"}), on=["hand_id", "winner_id", "loser_id"], how="left")
          .with_columns(pl.col("L_calls_facing_W").fill_null(0), pl.col("W_calls_facing_L").fill_null(0)))
    # outsider (non-partner) facts: folded facing a partner's bet / with a strong hand / after investing chips
    out = (P.join(X.select(["pair_id", "hand_id", "player_1", "player_2", "L_v_final", "W_v_final"]), on="hand_id")
             .filter((pl.col("player_id") != pl.col("player_1")) & (pl.col("player_id") != pl.col("player_2")))
             .with_columns(((pl.col("fold_facing") == pl.col("player_1")) | (pl.col("fold_facing") == pl.col("player_2"))).fill_null(False).alias("o_fold_to_pair"))
             .group_by(["pair_id", "hand_id"]).agg(
                 pl.col("o_fold_to_pair").sum().cast(pl.Int8).alias("n_out_fold_to_pair"),
                 (pl.col("o_fold_to_pair") & (pl.col("contrib_bb") > 1.0)).sum().cast(pl.Int8).alias("n_out_fold_to_pair_invested"),
                 (pl.col("o_fold_to_pair") & (pl.col("chen") >= 8)).sum().cast(pl.Int8).alias("n_out_fold_to_pair_strong_pf"),
                 (pl.col("o_fold_to_pair") & (pl.col("v_final") > pl.max_horizontal("L_v_final", "W_v_final"))).sum().cast(pl.Int8).alias("n_out_fold_to_pair_best"),
                 pl.col("contrib_bb").filter(pl.col("folded")).sum().cast(pl.Float32).alias("out_lost_bb"),
                 pl.col("went_to_showdown").sum().cast(pl.Int8).alias("n_out_sd"),
                 pl.col("won").any().alias("out_won")))
    X = X.join(out, on=["pair_id", "hand_id"], how="left")
    i8 = lambda e: e.fill_null(False).cast(pl.Int8)
    nsd = pl.col("players_at_showdown")
    F = X.with_columns(
        # --- A. showdown exposure (public information) ---
        (pl.col("L_went_to_showdown").cast(pl.Int8) + pl.col("W_went_to_showdown").cast(pl.Int8)).alias("pv_n_partner_sd"),
        pl.when(pl.col("L_went_to_showdown")).then(nsd).otherwise(0).cast(pl.Int8).alias("pv_n_saw_loser_cards"),
        pl.when(pl.col("W_went_to_showdown")).then(nsd).otherwise(0).cast(pl.Int8).alias("pv_n_saw_winner_cards"),
        i8(pl.col("L_went_to_showdown") & pl.col("W_won")).alias("pv_loser_sd_lost_to_partner"),
        i8(pl.col("L_went_to_showdown") & pl.col("W_won") & (pl.col("L_calls_facing_W") >= 1)).alias("pv_loser_sd_called_partner_lost"),
        i8(pl.col("W_went_to_showdown") & pl.col("W_won")).alias("pv_winner_sd_won"),
        pl.when(pl.col("W_went_to_showdown")).then(pl.col("W_cat_final")).otherwise(-1).cast(pl.Int8).alias("pv_winner_sd_cat"),
        i8(pl.col("W_went_to_showdown") & pl.col("W_won") & (pl.col("W_cat_final") <= 1)).alias("pv_winner_sd_won_weak"),
        ((pl.col("L_went_to_showdown") & (pl.col("L_chen") <= 5) & (pl.col("L_vpip_a") == 1)).cast(pl.Int8) + (pl.col("W_went_to_showdown") & (pl.col("W_chen") <= 5) & (pl.col("W_vpip_a") == 1)).cast(pl.Int8)).alias("pv_n_partner_sd_junk"),
        i8(pl.col("W_went_to_showdown") & pl.col("W_won") & (pl.col("W_chen") <= 5) & (pl.col("W_vpip_a") == 1)).alias("pv_winner_sd_junk_won"),
        i8(pl.col("L_went_to_showdown") & pl.col("W_went_to_showdown") & (pl.col("n_out_sd") == 0)).alias("pv_pair_only_sd"),
        # --- B. folded best hand (uses folded hole cards = hidden information) ---
        i8(pl.col("L_fold_best6")).alias("pv_L_fold_best6"),
        i8(pl.col("L_fold_best_rem")).alias("pv_L_fold_best_rem"),
        i8(pl.col("L_fold_best_at_fold")).alias("pv_L_fold_best_at_fold"),
        i8(pl.col("L_fold_best_street")).alias("pv_L_fold_best_street"),
        i8(pl.col("L_fold_best_pf_chen")).alias("pv_L_fold_best_pf_chen"),
        i8(pl.col("L_fold_beats_showdown")).alias("pv_L_fold_beats_showdown"),
        pl.col("L_fold_cat_margin").fill_null(0).alias("pv_L_fold_cat_margin"),
        i8(pl.col("L_fold_best_rem") & (pl.col("L_fold_facing") == pl.col("winner_id"))).alias("pv_L_fold_best_to_partner"),
        i8(pl.col("L_fold_best_at_fold") & (pl.col("L_fold_facing") == pl.col("winner_id"))).alias("pv_L_fold_best_at_fold_to_partner"),
        i8(pl.col("L_fold_best_rem") & (pl.col("L_fold_facing") == pl.col("winner_id")) & (pl.col("L_fold_n_active") == 2)).alias("pv_L_fold_best_hu_partner"),
        i8(pl.col("L_folded") & (pl.col("L_fold_facing") == pl.col("winner_id")) & (pl.col("L_fold_n_active") == 2)).alias("pv_L_fold_to_partner_hu"),
        pl.when(pl.col("L_folded") & (pl.col("L_fold_facing") == pl.col("winner_id"))).then(pl.col("L_fold_street")).otherwise(-1).cast(pl.Int8).alias("pv_L_fold_to_partner_street"),
        pl.when(pl.col("L_folded") & (pl.col("L_fold_facing") == pl.col("winner_id"))).then(pl.col("L_fold_to_call_bb")).otherwise(0).cast(pl.Float32).alias("pv_L_fold_to_partner_tocall_bb"),
        i8(pl.col("L_fold_ends_hand")).alias("pv_L_fold_ends_hand"),
        i8(pl.col("L_fold_ends_hand") & pl.col("W_won") & (nsd == 0)).alias("pv_L_fold_ends_hand_to_partner"),
        i8(pl.col("W_fold_best_rem")).alias("pv_W_fold_best_rem"),   # winner (net-better partner) also folded a best hand (both folded / tie cases)
        # --- C. partner won with the worse hand ---
        i8(pl.col("W_won") & (pl.col("L_v_final") > pl.col("W_v_final"))).alias("pv_W_won_worse_than_L"),
        i8(pl.col("W_went_to_showdown") & pl.col("W_won") & pl.col("L_folded") & (pl.col("L_v_final") > pl.col("W_v_final"))).alias("pv_W_sd_won_vs_folded_better_L"),
        i8(pl.col("W_won") & (pl.col("W_rk_final") > 1)).alias("pv_W_won_not_best6"),
        i8(pl.col("W_won") & (nsd == 0) & (pl.col("h_vmax_all") > pl.col("W_v_final"))).alias("pv_W_nosd_won_vs_better_folded_any"),
        pl.when(pl.col("W_won")).then(pl.col("W_rk_final")).otherwise(0).cast(pl.Int8).alias("pv_W_won_rank6"),
        # --- D. uncontested pot to partner ---
        i8((nsd == 0) & pl.col("W_won")).alias("pv_nosd_partner_won"),
        i8((nsd == 0) & pl.col("W_won") & pl.col("L_fold_best_rem")).alias("pv_nosd_partner_won_after_best_fold"),
        i8((nsd == 0) & pl.col("W_won") & pl.col("L_folded") & (pl.col("L_fold_facing") == pl.col("winner_id"))).alias("pv_nosd_partner_won_after_fold_to_partner"),
        i8((nsd == 0) & pl.col("W_won") & pl.col("L_fold_best_rem") & (pl.col("L_fold_facing") == pl.col("winner_id"))).alias("pv_nosd_partner_won_after_best_fold_to_partner"),
        # --- E. visible outcome ---
        pl.col("W_won_share").fill_null(0).cast(pl.Float32).alias("pv_W_won_share"),
        i8(pl.col("W_won") & (pl.col("W_won_share") >= 1.0)).alias("pv_W_won_full"),
        i8(pl.col("W_won") & (pl.col("L_net_chips") < 0)).alias("pv_W_won_L_lost"),
        i8(pl.col("W_won") & (pl.col("L_net_chips") < 0) & (pl.col("h_n_notfold") + pl.col("L_folded").cast(pl.Int8) == 2)).alias("pv_W_won_L_lost_last_two"),
        pl.when(pl.col("h_total_lost") < 0).then(pl.col("L_net_chips").cast(pl.Float64) / pl.col("h_total_lost")).otherwise(0).cast(pl.Float32).alias("pv_L_share_of_losses"),
        i8(pl.col("out_won")).alias("pv_out_won"),
        # --- F. outsider victims (visible harm) ---
        pl.col("n_out_fold_to_pair").fill_null(0).alias("pv_n_out_fold_to_pair"),
        pl.col("n_out_fold_to_pair_invested").fill_null(0).alias("pv_n_out_fold_to_pair_invested"),
        pl.col("n_out_fold_to_pair_strong_pf").fill_null(0).alias("pv_n_out_fold_to_pair_strong_pf"),
        pl.col("n_out_fold_to_pair_best").fill_null(0).alias("pv_n_out_fold_to_pair_best"),
        pl.col("out_lost_bb").fill_null(0).alias("pv_out_lost_bb"),
    )
    # --- G. composites: "provable" per mechanism ---
    F = F.with_columns(
        i8((pl.col("pv_W_won_L_lost") == 1) & ((pl.col("pv_L_fold_best_to_partner") == 1) | (pl.col("pv_loser_sd_called_partner_lost") == 1) | (pl.col("pv_W_won_worse_than_L") == 1))).alias("pv_prov_transfer"),
        i8((pl.col("pv_n_partner_sd") == 2) | ((pl.col("pv_winner_sd_won") == 1) & (pl.col("pv_L_fold_best_rem") == 1))).alias("pv_prov_softplay"),
        i8((pl.col("pv_n_out_fold_to_pair") >= 2) & (pl.col("pv_nosd_partner_won") == 1)).alias("pv_prov_isolation"),
        i8((pl.col("pv_n_partner_sd") >= 1) | (pl.col("pv_L_fold_ends_hand_to_partner") == 1)).alias("pv_prov_public_any"),
    )
    feats = [c for c in F.columns if c.startswith("pv_")]
    return F.select(["pair_id", "hand_id"] + [pl.col(c).fill_null(0).fill_nan(0) if F.schema[c].is_float() else pl.col(c).fill_null(0) for c in feats])

PUBLIC = ["pv_n_partner_sd", "pv_n_saw_loser_cards", "pv_n_saw_winner_cards", "pv_loser_sd_lost_to_partner", "pv_loser_sd_called_partner_lost", "pv_winner_sd_won",
          "pv_winner_sd_cat", "pv_winner_sd_won_weak", "pv_n_partner_sd_junk", "pv_winner_sd_junk_won", "pv_pair_only_sd", "pv_L_fold_to_partner_hu", "pv_L_fold_to_partner_street",
          "pv_L_fold_to_partner_tocall_bb", "pv_L_fold_ends_hand", "pv_L_fold_ends_hand_to_partner", "pv_nosd_partner_won", "pv_nosd_partner_won_after_fold_to_partner",
          "pv_W_won_share", "pv_W_won_full", "pv_W_won_L_lost", "pv_W_won_L_lost_last_two", "pv_L_share_of_losses", "pv_out_won", "pv_n_out_fold_to_pair", "pv_n_out_fold_to_pair_invested",
          "pv_out_lost_bb", "pv_prov_public_any"]

# ============================================================ dev evaluation ============================================================
def map5(pid, s, y, pot, hid):
    dd = pd.DataFrame({"p": pid, "s": s, "y": y, "pot": pot, "h": hid}).sort_values(["p", "s", "pot", "h"], ascending=[True, False, False, True]); vals = {}
    for p, g in dd.groupby("p", sort=False):
        r = g.y.values; n = r.sum()
        if n == 0: continue
        t = r[:5]; vals[p] = (np.cumsum(t) / np.arange(1, 6) * t).sum() / min(n, 5)
    return vals

if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    lab = pl.read_csv(D + "development_labels.csv"); ev = pl.read_csv(D + "development_evidence.csv"); pos = lab.filter(pl.col("label") == 1); ids = list(set(pos["pair_id"]))
    src = open("step9_pipeline.py").read(); ns = {}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")], ns); HF = ns["HAND_FEATS"]
    df = pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(ids)).collect() for i in range(4)])
    df = (df.join(ev.select(["pair_id", "hand_id", pl.lit(True).alias("is_ev")]), on=["pair_id", "hand_id"], how="left").with_columns(pl.col("is_ev").fill_null(False))
            .join(pos.select(["pair_id", "behavior_family"]), on="pair_id"))
    print("loaded", df.shape, f"{time.time()-t0:.0f}s", flush=True)
    feats = compute_provability(df.select(["pair_id", "hand_id", "player_1", "player_2"]))
    PV = [c for c in feats.columns if c.startswith("pv_")]
    feats.write_parquet(OUT); print("features", feats.shape, "->", OUT, f"{time.time()-t0:.0f}s", flush=True)
    df = df.join(feats, on=["pair_id", "hand_id"], how="left")
    assert df.select(pl.col(PV[0]).is_null().sum()).item() == 0
    # consistency check vs pipeline's own convention
    print("loser_is_p1 agreement with pipeline:", (df["loser_is_p1"] == (df["p1_net_bb"] < df["p2_net_bb"]).cast(pl.Int8)).mean())
    tables = sorted(df["table_id"].unique().to_list()); rng = np.random.RandomState(42); TF = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % 5)}
    df = df.with_columns(pl.col("table_id").replace_strict(TF, return_dtype=pl.Int8).alias("fold")).sort(["pair_id", "hand_idx"])
    FAM = ["directed_transfer", "soft_play", "coordinated_isolation"]
    for f in FAM: df = df.with_columns((pl.col("behavior_family") == f).cast(pl.Float32).alias("fam_" + f))
    FAMC = ["fam_" + f for f in FAM]
    pdf = df.to_pandas(); y = pdf.is_ev.values.astype(int); fold = pdf.fold.values; pid = pdf.pair_id.values
    conf = (~pdf.is_ev) & (pdf.surp_pair_max >= 3)
    # ---- (a) univariate AUCs ----
    def auc(mask_pos, mask_neg, c):
        x = np.nan_to_num(pdf[c].values.astype(float)); yy = np.r_[np.ones(mask_pos.sum()), np.zeros(mask_neg.sum())]; xx = np.r_[x[mask_pos], x[mask_neg]]
        return roc_auc_score(yy, xx) if len(np.unique(xx)) > 1 else 0.5
    rows = []
    print(f"\n{'feature':48s} {'mean_ev':>8s} {'mean_cf':>8s} {'mean_ot':>8s} | {'AUC_nonev':>9s} {'AUC_conf':>8s} | per-family AUC ev/conf: DT    SP    CI")
    for c in PV:
        a1 = auc(pdf.is_ev.values, ~pdf.is_ev.values, c); a2 = auc(pdf.is_ev.values, conf.values, c)
        fa = [auc((pdf.is_ev & (pdf.behavior_family == f)).values, (conf & (pdf.behavior_family == f)).values, c) for f in FAM]
        rows.append((c, a1, a2, *fa))
        print(f"{c:48s} {pdf.loc[pdf.is_ev, c].mean():8.3f} {pdf.loc[conf, c].mean():8.3f} {pdf.loc[~pdf.is_ev & ~conf, c].mean():8.3f} | {a1:9.3f} {a2:8.3f} | {fa[0]:.3f} {fa[1]:.3f} {fa[2]:.3f}", flush=True)
    pd.DataFrame(rows, columns=["feature", "auc_nonev", "auc_conf", "auc_conf_DT", "auc_conf_SP", "auc_conf_CI"]).to_csv(W + "hyp/provability_auc.csv", index=False)
    # ---- (b) stage-1 harness ----
    P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=NT)
    def run(F, rounds=400, seeds=(42, 49)):
        X = pdf[F].values.astype(np.float32); oof = np.zeros(len(y)); imp = np.zeros(len(F))
        for f in range(5):
            tr, te = fold != f, fold == f
            for sd in seeds:
                m = lgb.train({**P, "seed": sd}, lgb.Dataset(X[tr], y[tr]), num_boost_round=rounds); oof[te] += m.predict(X[te]) / len(seeds); imp += m.feature_importance("gain")
        return oof, imp
    def report(name, oof):
        v = map5(pid, oof, y, pdf.pot_bb.values, pdf.hand_id.values); fam = pdf.groupby("pair_id").behavior_family.first()
        per = {f: np.mean([s for p, s in v.items() if fam[p] == f]) for f in FAM}
        print(f"{name:34s} MAP@5={np.mean(list(v.values())):.4f}  DT={per['directed_transfer']:.4f} SP={per['soft_play']:.4f} CI={per['coordinated_isolation']:.4f}", flush=True); return v
    o0, _ = run(HF + FAMC); v0 = report("baseline HAND_FEATS+family", o0)
    o1, imp1 = run(HF + FAMC + PV); v1 = report("+ provability (all)", o1)
    o2, _ = run(HF + FAMC + PUBLIC); v2 = report("+ provability (public-only)", o2)
    top = sorted(zip(imp1, HF + FAMC + PV), reverse=True)[:25]; print("top gain (with):", [(f, int(g)) for g, f in top if f.startswith("pv_")][:15])
    print("rank of pv_ feats in gain:", [i for i, (g, f) in enumerate(sorted(zip(imp1, HF + FAMC + PV), reverse=True)) if f.startswith("pv_")][:20])
    d = np.array([v1[p] - v0[p] for p in v0]); print(f"delta all = {np.mean(d):+.4f}  (paired sd/sqrt(n) = {d.std(ddof=1)/np.sqrt(len(d)):.4f}); pairs up={int((d>0).sum())} down={int((d<0).sum())}")
    np.save(W + "hyp/provability_oofs.npy", np.vstack([o0, o1, o2]))
    print(f"total {time.time()-t0:.0f}s")
