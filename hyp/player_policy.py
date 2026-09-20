"""
HYPOTHESIS `player_policy` — per-PLAYER action policy fitted on the player's OWN other-phase decisions.

Idea: the simulator's bots have fixed per-player style parameters, so a policy fitted on a player's other-phase
hands (dev colluders play normally in eval, and vice-versa) should flag in-phase actions that are near-impossible
*for that specific player* more sharply than the population policy (`action_surprise.parquet`).

Model: one 6-class LightGBM (fold/check/call/bet/raise/all_in) trained ONLY on OTHER-phase actions of the players of
the target pairs, with (a) the same situation features as the population policy (cards, made-hand category, board
texture, position, to-call / pot / stack, raises faced, own prior actions), (b) the player's OTHER-phase conditional
style rates (`r_*`, smoothed; e.g. P(enter | junk, unraised), P(fold | facing bet, strong made hand)) and
(c) `pid` = player identity as a categorical feature, so the trees can learn player-specific thresholds.
Chosen variant is set by MODEL_VARIANT below (selected by other-phase hold-out log-loss, see log).

Scoring: every IN-phase action of the two partners gets p_own = P_player(taken action), surp_own = -log p_own,
delta = surp_own - surp_pop (population surprise from action_surprise.parquet). Aggregated per (pair_id, hand_id):
  pp_surp_sum / pp_surp_max / pp_surp_min (min over partners of their max)   own-policy surprise
  pp_pmin (min p_own in the hand), pp_pmin_max (max over partners of their min p_own: small => BOTH improbable)
  pp_n_p01 / pp_n_p05 (# actions with p_own < 0.01 / 0.05), pp_both_p01 / pp_both_p05 (both partners have one)
  pp_delta_sum / pp_delta_max / pp_delta_min                                own-minus-population surprise
  pp_vs_partner_sum / pp_vs_partner_max  own-policy surprise of actions taken while the PARTNER set the price
  pp_logp_mean, pp_p1_surp, pp_p2_surp, pp_pf_sum, pp_post_sum
  + within-pair percentile ranks (`_prank`) of the main ones (label-free, computed over all shared hands of the pair).

Leakage: no evidence file, evidence_rank, is_ev or pair label is used anywhere. development_labels.csv is read only
to enumerate which pairs (and hence which players / shared hands) to compute rows for.

EVALUATION RECIPE (same features for evaluation pairs):
    PHASE=evaluation python3 work_step3/hyp/player_policy.py
  -> pairs from data/evaluation_pairs.csv, players = their union; the policy is fitted on the players' DEVELOPMENT-
     phase actions (their other phase) and evaluation-phase actions of every shared hand are scored. Players are
     processed in chunks of tables (TABLES_PER_CHUNK) so one model never sees more than ~3M rows / 3k players.
     Output: work_step3/hyp/player_policy_eval.parquet with identical columns.
  Development run (default): PHASE=development, pairs = the 372 labelled-positive pairs, policy fitted on their
     evaluation-phase actions, rows for all 45,129 shared development hands -> work_step3/hyp/player_policy.parquet.
"""
import polars as pl, numpy as np, lightgbm as lgb, os, sys, time, gc, warnings
warnings.filterwarnings("ignore")
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

PHASE = os.environ.get("PHASE", "development")
OTHER = "evaluation" if PHASE == "development" else "development"
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "sit+rates+pid")   # sit | sit+rates | sit+pid | sit+rates+pid
TABLES_PER_CHUNK = int(os.environ.get("TABLES_PER_CHUNK", "100"))
NT = int(os.environ.get("NUM_THREADS", "3"))
d = os.environ.get("POKER_DATA_DIR", "data").rstrip("/") + "/"; W = os.environ.get("POKER_WORK_DIR", "work_step3").rstrip("/") + "/"; OUT = W + "hyp/" + ("player_policy.parquet" if PHASE == "development" else "player_policy_eval.parquet")
RANKS = {r: i for i, r in enumerate("23456789TJQKA", start=2)}; B5 = 15 ** 5
ACTS = ["fold", "check", "call", "bet", "raise", "all_in"]
SIT = ["street_no", "pos", "players_active", "to_call_bb", "to_call_pot", "pot_bb", "stack_bb", "spr", "aggr_before_street", "own_prior_aggr",
       "own_prior_actions", "street_action_no", "hi", "lo", "suited", "pocket", "cat", "b_max", "b_paired", "hits", "flush_cards", "overpair", "top_pair"]

# ---------------------------------------------------------------- 1. target pairs and players
if PHASE == "development":
    lab = pl.read_csv(d + "development_labels.csv")
    pairs = lab.filter(pl.col("label") == 1).select(["pair_id", "player_1", "player_2"])      # membership only; label never used as a feature
else:
    pairs = pl.read_csv(d + "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])
players = sorted(set(pairs["player_1"]) | set(pairs["player_2"]))
log(f"phase={PHASE} other={OTHER} pairs={pairs.height} players={len(players)} variant={MODEL_VARIANT}")

hb = pl.read_parquet(d + "hands.parquet", columns=["hand_id", "table_id", "phase", "big_blind", "button_seat", "board_cards"])
hb = hb.with_columns(hb["board_cards"].fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("board")).drop("board_cards")
seats = pl.read_parquet(d + "seats.parquet", columns=["hand_id", "player_id", "seat_no", "hole_card_1", "hole_card_2"]).filter(pl.col("player_id").is_in(players))
strg = pl.scan_parquet(W + "player_strength.parquet").filter(pl.col("player_id").is_in(players)).select(["hand_id", "player_id", "v_flop", "v_turn", "v_river"]).collect()
ptab = seats.join(hb.select(["hand_id", "table_id"]), on="hand_id").group_by("player_id").agg(pl.col("table_id").first())   # each player sits at one table

# ---------------------------------------------------------------- 2. per-action situation features (identical to the step-8/9 population policy)
def build_actions(pl_set):
    return (pl.scan_parquet(d + "actions.parquet").filter(pl.col("player_id").is_in(pl_set))
        .join(hb.lazy(), on="hand_id").join(seats.lazy(), on=["hand_id", "player_id"]).join(strg.lazy(), on=["hand_id", "player_id"])
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
        .select(["hand_id", "action_no", "player_id", "phase", "table_id", "street_no", "y", "is_aggr", *SIT[1:]])
        .collect(engine="streaming"))

# ---------------------------------------------------------------- 3. other-phase conditional style rates per player (smoothed)
def rates(df):
    pf1 = (pl.col("street_no") == 0) & (pl.col("own_prior_actions") == 0); nor = pl.col("aggr_before_street") == 0; post = pl.col("street_no") > 0
    junk = (pl.col("hi") <= 9) & (pl.col("pocket") == 0); strong = pl.col("cat") >= 2; weak = pl.col("cat") <= 0
    tc = pl.col("to_call_bb") > 0; y = pl.col("y"); ag = pl.col("is_aggr"); pf = pl.col("street_no") == 0
    def r(name, ev, cond, prior, k=5): return ((((ev) & (cond)).sum() + prior * k) / ((cond).sum() + k)).cast(pl.Float32).alias(name)
    return df.group_by("player_id").agg(
        r("r_vpip_junk", y != 0, pf1 & nor & junk, 0.15), r("r_vpip_nor", y != 0, pf1 & nor, 0.3), r("r_vpip_fr", y != 0, pf1 & ~nor, 0.2), r("r_pfr", y >= 4, pf1, 0.15),
        r("r_bet_post", ag, post & ~tc, 0.3), r("r_fold_post", y == 0, post & tc, 0.4), r("r_raise_post", ag, post & tc, 0.15), r("r_check_strong", y == 1, post & ~tc & strong, 0.3),
        r("r_bet_weak", ag, post & ~tc & weak, 0.2), r("r_fold_strong", y == 0, post & tc & strong, 0.1), r("r_call_weak", y == 2, post & tc & weak, 0.2), r("r_allin", y == 5, pl.lit(True), 0.02),
        r("r_3bet", ag, pf & ~nor, 0.05), pl.len().cast(pl.Float32).alias("r_n"))
RATES = ["r_vpip_junk", "r_vpip_nor", "r_vpip_fr", "r_pfr", "r_bet_post", "r_fold_post", "r_raise_post", "r_check_strong", "r_bet_weak", "r_fold_strong", "r_call_weak", "r_allin", "r_3bet", "r_n"]
FEATS = SIT + (RATES if "rates" in MODEL_VARIANT else []) + (["pid"] if "pid" in MODEL_VARIANT else [])
CAT = ["pid"] if "pid" in MODEL_VARIANT else []
P = dict(objective="multiclass", num_class=6, learning_rate=0.08, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=5.0, verbose=-1, num_threads=NT, seed=0, cat_smooth=20, min_data_per_group=50, max_cat_to_onehot=4)

# ---------------------------------------------------------------- 4. fit on OTHER phase, score IN phase, chunked by table
tables = sorted(ptab["table_id"].unique().to_list()); chunks = [tables[i:i + TABLES_PER_CHUNK] for i in range(0, len(tables), TABLES_PER_CHUNK)]
scored = []
for ci, tb in enumerate(chunks):
    pls = ptab.filter(pl.col("table_id").is_in(tb))["player_id"].to_list()
    a = build_actions(pls)
    oth = a.filter(pl.col("phase") == OTHER); inp = a.filter(pl.col("phase") == PHASE)
    R = rates(oth); pid = {p: i for i, p in enumerate(sorted(pls))}
    oth = oth.join(R, on="player_id").with_columns(pl.col("player_id").replace_strict(pid, return_dtype=pl.Int32).alias("pid"))
    inp = inp.join(R, on="player_id", how="left").with_columns(pl.col("player_id").replace_strict(pid, return_dtype=pl.Int32).alias("pid"))
    # internal hand-level hold-out (10%) of the OTHER phase only for early stopping, then refit on all other-phase rows
    hs = oth["hand_id"].unique().sort().to_numpy(); rng = np.random.RandomState(7); ho = set(hs[rng.rand(len(hs)) < 0.1].tolist())
    te = oth["hand_id"].is_in(list(ho)).to_numpy(); tr = ~te
    X = oth.select(FEATS).to_numpy().astype(np.float32); y = oth["y"].to_numpy()
    ds = lgb.Dataset(X[tr], y[tr], feature_name=FEATS, categorical_feature=CAT, free_raw_data=False)
    m = lgb.train(P, ds, num_boost_round=800, valid_sets=[lgb.Dataset(X[te], y[te], reference=ds)], callbacks=[lgb.early_stopping(30, verbose=False)])
    pte = m.predict(X[te], num_iteration=m.best_iteration); ll = float(-np.log(np.clip(pte[np.arange(te.sum()), y[te]], 1e-5, 1)).mean())
    nb = int(m.best_iteration * 1.1) + 1
    m = lgb.train(P, lgb.Dataset(X, y, feature_name=FEATS, categorical_feature=CAT), num_boost_round=nb)
    Xi = inp.select(FEATS).to_numpy().astype(np.float32); pr = m.predict(Xi); p_own = pr[np.arange(len(pr)), inp["y"].to_numpy()]
    scored.append(inp.select(["hand_id", "action_no", "player_id", "street_no", "y", "is_aggr"]).with_columns(pl.Series("p_own", p_own.astype(np.float32))))
    log(f"chunk {ci+1}/{len(chunks)}: tables={len(tb)} players={len(pls)} other rows={oth.height:,} in rows={inp.height:,} holdout logloss={ll:.4f} rounds={nb}")
    del a, oth, inp, X, Xi, m, ds; gc.collect()
S = pl.concat(scored); del scored
S = S.with_columns((-pl.col("p_own").clip(1e-5, 1).log()).cast(pl.Float32).alias("surp_own"))
S = S.join(pl.scan_parquet(W + "action_surprise.parquet").select(["hand_id", "action_no", "player_id", "p_taken"]).filter(pl.col("hand_id").is_in(S["hand_id"].unique().implode())).collect(),
           on=["hand_id", "action_no", "player_id"], how="left")
S = S.with_columns((-pl.col("p_taken").clip(1e-5, 1).log()).cast(pl.Float32).alias("surp_pop")).with_columns((pl.col("surp_own") - pl.col("surp_pop")).alias("delta"))
S = S.join(pl.scan_parquet(W + "action_context.parquet").select(["hand_id", "action_no", "player_id", "last_aggr"]).filter(pl.col("hand_id").is_in(S["hand_id"].unique().implode())).collect(),
           on=["hand_id", "action_no", "player_id"], how="left")
log(f"scored in-phase actions: {S.height:,}; mean surp_own={S['surp_own'].mean():.3f} mean surp_pop={S['surp_pop'].mean():.3f} frac p_own<0.01={(S['p_own']<0.01).mean():.4f}")

# ---------------------------------------------------------------- 5. pair-hand rows = all IN-phase hands where both partners were dealt
ph = seats.join(hb.select(["hand_id", "phase", "table_id"]), on="hand_id").filter(pl.col("phase") == PHASE).select(["hand_id", "player_id"])
rows = (pairs.join(ph.rename({"player_id": "player_1"}), on="player_1").join(ph.rename({"player_id": "player_2"}), on=["player_2", "hand_id"]))
log(f"pair-hand rows: {rows.height:,}")

def agg_player(S, who):
    part = pl.col("partner"); return (S.group_by(["pair_id", "hand_id"]).agg(
        pl.col("surp_own").sum().alias(f"{who}_surp"), pl.col("surp_own").max().alias(f"{who}_surp_max"), pl.col("p_own").min().alias(f"{who}_pmin"),
        (pl.col("p_own") < 0.01).sum().cast(pl.Int8).alias(f"{who}_n01"), (pl.col("p_own") < 0.05).sum().cast(pl.Int8).alias(f"{who}_n05"),
        pl.col("delta").sum().alias(f"{who}_delta"), pl.col("delta").max().alias(f"{who}_delta_max"),
        pl.col("surp_own").filter(pl.col("last_aggr") == part).sum().fill_null(0.0).alias(f"{who}_vsp"), pl.col("surp_own").filter(pl.col("last_aggr") == part).max().fill_null(0.0).alias(f"{who}_vsp_max"),
        pl.col("surp_own").filter(pl.col("street_no") == 0).sum().fill_null(0.0).alias(f"{who}_pf"), pl.col("surp_own").filter(pl.col("street_no") > 0).sum().fill_null(0.0).alias(f"{who}_post"),
        pl.col("p_own").clip(1e-5, 1).log().sum().alias(f"{who}_logp"), pl.len().cast(pl.Int8).alias(f"{who}_n")))
A1 = agg_player(rows.join(S, left_on=["player_1", "hand_id"], right_on=["player_id", "hand_id"]).with_columns(pl.col("player_2").alias("partner")), "p1")
A2 = agg_player(rows.join(S, left_on=["player_2", "hand_id"], right_on=["player_id", "hand_id"]).with_columns(pl.col("player_1").alias("partner")), "p2")
F = rows.select(["pair_id", "hand_id"]).join(A1, on=["pair_id", "hand_id"], how="left").join(A2, on=["pair_id", "hand_id"], how="left")
F = F.with_columns([pl.col(c).fill_null(0) for c in F.columns if c[:3] in ("p1_", "p2_")]).with_columns([pl.col(c).fill_null(1.0) for c in ("p1_pmin", "p2_pmin")])
F = F.with_columns(
    (pl.col("p1_surp") + pl.col("p2_surp")).alias("pp_surp_sum"), pl.max_horizontal("p1_surp_max", "p2_surp_max").alias("pp_surp_max"), pl.min_horizontal("p1_surp_max", "p2_surp_max").alias("pp_surp_min"),
    pl.min_horizontal("p1_pmin", "p2_pmin").alias("pp_pmin"), pl.max_horizontal("p1_pmin", "p2_pmin").alias("pp_pmin_max"),
    (pl.col("p1_n01") + pl.col("p2_n01")).cast(pl.Int8).alias("pp_n_p01"), (pl.col("p1_n05") + pl.col("p2_n05")).cast(pl.Int8).alias("pp_n_p05"),
    ((pl.col("p1_n01") > 0) & (pl.col("p2_n01") > 0)).cast(pl.Int8).alias("pp_both_p01"), ((pl.col("p1_n05") > 0) & (pl.col("p2_n05") > 0)).cast(pl.Int8).alias("pp_both_p05"),
    (pl.col("p1_delta") + pl.col("p2_delta")).alias("pp_delta_sum"), pl.max_horizontal("p1_delta_max", "p2_delta_max").alias("pp_delta_max"), pl.min_horizontal("p1_delta_max", "p2_delta_max").alias("pp_delta_min"),
    (pl.col("p1_vsp") + pl.col("p2_vsp")).alias("pp_vs_partner_sum"), pl.max_horizontal("p1_vsp_max", "p2_vsp_max").alias("pp_vs_partner_max"),
    ((pl.col("p1_logp") + pl.col("p2_logp")) / pl.max_horizontal(pl.col("p1_n") + pl.col("p2_n"), 1)).alias("pp_logp_mean"),
    (pl.col("p1_pf") + pl.col("p2_pf")).alias("pp_pf_sum"), (pl.col("p1_post") + pl.col("p2_post")).alias("pp_post_sum"),
    pl.col("p1_surp").alias("pp_p1_surp"), pl.col("p2_surp").alias("pp_p2_surp"))
PP = [c for c in F.columns if c.startswith("pp_")]
F = F.with_columns([(pl.col(c).rank("average").over("pair_id") / pl.len().over("pair_id")).cast(pl.Float32).alias(c + "_prank") for c in ["pp_surp_sum", "pp_surp_max", "pp_surp_min", "pp_delta_sum", "pp_pmin_max", "pp_vs_partner_sum"]])
F = F.select(["pair_id", "hand_id"] + [c for c in F.columns if c.startswith("pp_")]).with_columns([pl.col(c).cast(pl.Float32) for c in F.columns if c.startswith("pp_")])
os.makedirs(W + "hyp", exist_ok=True); F.write_parquet(OUT)
log(f"saved {OUT} {F.shape}; features: {[c for c in F.columns if c.startswith('pp_')]}")

# ---------------------------------------------------------------- 6. (development only) evaluation: EVAL=1 python3 work_step3/hyp/player_policy.py
# Other-phase hold-out log-loss of policy variants (evaluation-phase actions of the 693 positive-pair players, 20 % of hands held out):
#   population policy (step 8, 5M rows, other-phase style covariates) 0.4695 | sit only 0.5237 | sit+st 0.4807 | sit+rates 0.4764 |
#   sit+pid 0.4911 | sit+rates+pid 0.4905 (chunked by 100 tables, as run here: 0.49-0.53).  Fraction of clean hold-out actions with
#   p<0.01 under the player's OWN policy: 0.0008 for every variant -> the bots are stochastic, not deterministic per player.
if PHASE == "development" and os.environ.get("EVAL", "0") == "1":
    from sklearn.metrics import roc_auc_score
    ev = pl.read_csv(d + "development_evidence.csv"); lab = pl.read_csv(d + "development_labels.csv"); pos = lab.filter(pl.col("label") == 1); ids = set(pos["pair_id"])
    src = open("step9_pipeline.py").read(); ns = {}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")], ns); HF = ns["HAND_FEATS"]
    cols = ["pair_id", "hand_id", "table_id", "hand_idx", "surp_pair_max", "pot_bb"] + [c for c in HF if c not in ("pot_bb", "surp_pair_max")]
    df = pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(cols).collect(engine="streaming") for i in range(4)])
    df = (df.join(ev.select(["pair_id", "hand_id", pl.lit(True).alias("is_ev")]), on=["pair_id", "hand_id"], how="left").with_columns(pl.col("is_ev").fill_null(False))
            .join(pos.select(["pair_id", "behavior_family"]), on="pair_id").join(F, on=["pair_id", "hand_id"], how="left"))
    y = df["is_ev"].to_numpy().astype(int); evm = df["is_ev"].to_numpy(); conf = (~evm) & (df["surp_pair_max"] >= 3).to_numpy()
    for c in PP + [x + "_prank" for x in ["pp_surp_sum", "pp_surp_max", "pp_surp_min", "pp_delta_sum", "pp_pmin_max", "pp_vs_partner_sum"]]:
        x = df[c].fill_null(0).to_numpy().astype(float); m = evm | conf; log(f"AUC {c:28s} ev-vs-all {roc_auc_score(y, x):.4f}  ev-vs-confusers {roc_auc_score(y[m], x[m]):.4f}")
    tables = sorted(df["table_id"].unique().to_list()); rng = np.random.RandomState(42); TF = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % 5)}
    df = df.with_columns(pl.col("table_id").replace_strict(TF, return_dtype=pl.Int8).alias("fold")).sort(["pair_id", "hand_idx"])
    for f in ["directed_transfer", "soft_play", "coordinated_isolation"]: df = df.with_columns((pl.col("behavior_family") == f).cast(pl.Float32).alias("fam_" + f))
    FE = HF + ["fam_directed_transfer", "fam_soft_play", "fam_coordinated_isolation"]; fold = df["fold"].to_numpy(); y = df["is_ev"].to_numpy().astype(int)
    PB = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=NT)
    def map5(dd, score):
        dd = dd.with_columns(pl.Series("s", score)).sort(["pair_id", "s", "pot_bb", "hand_id"], descending=[False, True, True, False]); res = []
        for _, g in dd.group_by("pair_id", maintain_order=True):
            rel = g["is_ev"].to_numpy(); n = rel.sum()
            if n == 0: continue
            hits = 0; ap = 0.0
            for i, r in enumerate(rel[:5]):
                if r: hits += 1; ap += hits / (i + 1)
            res.append(ap / min(n, 5))
        return float(np.mean(res))
    def run(feats, name):
        X = df.select(feats).fill_null(0).to_numpy().astype(np.float32); oof = np.zeros(len(y))
        for f in range(5):
            tr, te = fold != f, fold == f
            for sd in (42, 49): m = lgb.train({**PB, "seed": sd}, lgb.Dataset(X[tr], y[tr]), num_boost_round=400); oof[te] += m.predict(X[te]) / 2
        r = map5(df.select(["pair_id", "hand_id", "pot_bb", "is_ev"]), oof); log(f"{name}: MAP@5={r:.4f}"); return r
    b = run(FE, "baseline HAND_FEATS+fam"); w = run(FE + [c for c in F.columns if c.startswith("pp_")], "with player_policy"); log(f"DELTA {w - b:+.4f}")
# Result of this run (see player_policy_eval.log): baseline MAP@5 0.5181 -> with 0.5239 (+0.0058, within 2-seed noise; soft_play -0.012,
# directed_transfer +0.016, coordinated_isolation +0.015). Univariate AUCs equal the population-policy ones (pp_surp_sum 0.892 vs surp_pair_sum 0.893,
# pp_vs_partner 0.912 vs 0.912); vs confusers the per-player versions are no sharper (pp_pmin 0.70 vs p_min_pair 0.73). Verdict: null.
