"""
HYPOTHESIS bet_grid: bots choose bet/raise sizes from a fixed grid; collusion-mode
actions may use sizes OFF the bot's usual grid.

Per-action size features (all hands, dev+eval, no labels used anywhere):
  bucket bk: 0 allin, 1/2/3 bet on flop/turn/river, 4 pf_open (first preflop raise),
             5 pf_3bet (preflop re-raise), 6 raise_post (postflop raise)
  bin      : snapped normalized size
             - bets      : pot fraction snapped to G={0.33,0.5,0.66,0.75,1.0,1.25} when the chip
                           amount is within +-1 chip of round(f*pot); else "off" (bin = -round(r,1))
             - pf_open   : amount_to / bb  (0.25 granularity)
             - pf_3bet   : amount_to / previous level (multiplier, 0.1 granularity)
             - raise_post: raise increment / (pot_before+to_call) (0.1 granularity)
  grid_dist: |r - nearest grid fraction| (bets), |to_bb - nearest of {2,2.25,2.5,3}| (pf_open),
             |mult - nearest of {2.2,2.5,3,3.5}| (pf_3bet)
  offgrid  : grid_dist above tolerance (bets: not within 1 chip; pf_open: >0.06; pf_3bet: >0.12)
  minraise : raise increment == max(last increment on street, bb)          (exact min-raise)
  allin_less: all-in that does not cover the call, or all-in raise below the min legal raise
  Per-player leave-one-HAND-out histogram of (bucket, bin):
     c_pl  = # other hands where this player used this bin in this bucket
     n_pl  = # other actions of this player in this bucket
     logp_pl = log((c_pl+0.5)/(n_pl+0.5*K))  (K = #bins in bucket, population)
     p_pop   = population frequency of bin in bucket
     lr      = logp_pl - log(p_pop)   (negative => player uses this size less than population)
     never   = (c_pl==0) & (n_pl>=10)

Per (pair_id, hand_id) aggregation over the two partners' actions -> bg_* columns (see AGG).

RE-RUN FOR EVAL PAIRS: per-action features are computed for every hand of both phases
(a player's grid uses all their hands, dev+eval, leave-one-hand-out). To compute the
pair-hand table for evaluation pairs, call
    python3 work_step3/hyp/bet_grid.py --pairs <parquet with pair_id,hand_id,player_1,player_2> --out <parquet>
(default: all positive dev pairs' 45,129 hands from dev_hand_features_v9 -> work_step3/hyp/bet_grid.parquet,
 followed by the AUC / MAP@5 test).
"""
import polars as pl, numpy as np, os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")
D="data/"; W="work_step3/"; H=W+"hyp/"
GB=np.array([0.33,0.5,0.66,0.75,1.0,1.25]); GO=np.array([2.0,2.25,2.5,3.0]); G3=np.array([2.2,2.5,3.0,3.5])

def action_size_table():
    """One row per bet/raise/all_in action of every hand (dev+eval) with size bins and
    leave-one-hand-out per-player grid statistics. Keys are integer codes (hid, pid) to keep memory low;
    the returned frame carries the string hand_id/player_id back for joining."""
    h=pl.read_parquet(D+"hands.parquet",columns=["hand_id","big_blind"]).with_row_index("hid")
    players=pl.read_parquet(D+"seats.parquet",columns=["player_id"]).unique().sort("player_id").with_row_index("pid")
    aggr=pl.col("action").is_in(["bet","raise","all_in"])
    # only aggressive actions change the street's bet level / last increment, so filter first (18.6M -> 4.1M rows)
    a=(pl.scan_parquet(D+"actions.parquet")
        .select(["hand_id","action_no","street","player_id","action","amount","amount_to","pot_before","stack_before","to_call"])
        .filter(aggr).collect(engine="streaming"))
    a=(a.join(h.lazy().collect(),on="hand_id").join(players,on="player_id").drop(["hand_id","player_id"])
        .with_columns(pl.col("street").replace_strict({"preflop":0,"flop":1,"turn":2,"river":3},return_dtype=pl.Int8).alias("st"),
                      pl.col("action").replace_strict({"bet":0,"raise":1,"all_in":2},return_dtype=pl.Int8).alias("act"),
                      pl.col("action_no").cast(pl.Int16),
                      *[pl.col(c).cast(pl.Int32) for c in ["amount","amount_to","pot_before","stack_before","to_call","big_blind"]])
        .drop(["street","action"]).sort(["hid","action_no"]))
    a=a.with_columns(
        (pl.col("to_call")+pl.col("amount_to")-pl.col("amount")).alias("lvl"),
        (pl.col("amount")-pl.col("to_call")).alias("inc"))
    # last increment on the street before this action (default bb); prior raises on the street
    a=a.with_columns(pl.col("inc").shift(1).over(["hid","st"]).alias("last_inc_raw"),
                     (pl.int_range(pl.len()).over(["hid","st"])).cast(pl.Int8).alias("n_prior_aggr"))
    a=a.with_columns(pl.max_horizontal(pl.col("last_inc_raw").fill_null(pl.col("big_blind")),pl.col("big_blind")).alias("min_inc"))
    r_pot=pl.col("amount")/pl.col("pot_before")
    to_bb=pl.col("amount_to")/pl.col("big_blind")
    mult=pl.col("amount_to")/pl.col("lvl")
    rr=(pl.col("amount")-pl.col("to_call"))/(pl.col("pot_before")+pl.col("to_call"))
    # bucket codes: 0 allin, 1-3 bet_flop/turn/river, 4 pf_open, 5 pf_3bet, 6 raise_post
    a=a.with_columns(
        pl.when(pl.col("act")==2).then(0)
          .when(pl.col("act")==0).then(pl.col("st"))
          .when((pl.col("st")==0)&(pl.col("n_prior_aggr")==0)).then(4)
          .when(pl.col("st")==0).then(5)
          .otherwise(6).cast(pl.Int8).alias("bk"),
        r_pot.cast(pl.Float32).alias("r_pot"), to_bb.cast(pl.Float32).alias("to_bb"), mult.fill_nan(None).cast(pl.Float32).alias("mult"), rr.cast(pl.Float32).alias("rr"),
        (pl.col("inc")==pl.col("min_inc")).alias("minraise"),
        ((pl.col("act")==2)&((pl.col("amount")<pl.col("to_call"))|((pl.col("amount")>pl.col("to_call"))&(pl.col("inc")<pl.col("min_inc"))))).alias("allin_less"),
        (pl.col("amount")/pl.col("stack_before")).cast(pl.Float32).alias("stack_frac"))
    # grid snapping
    amt=a["amount"].to_numpy(); pot=a["pot_before"].to_numpy().astype(float); bk=a["bk"].to_numpy()
    r=a["r_pot"].to_numpy().astype(float); tb=a["to_bb"].to_numpy().astype(float); mu=a["mult"].to_numpy().astype(float); rrv=a["rr"].to_numpy().astype(float)
    n=len(a); binv=np.full(n,np.nan,dtype=np.float32); gdist=np.zeros(n,dtype=np.float32); off=np.zeros(n,dtype=bool)
    isbet=(bk>=1)&(bk<=3)
    cand=np.rint(GB[None,:]*pot[isbet,None]); ok=np.abs(cand-amt[isbet,None])<=1
    dd=np.abs(r[isbet,None]-GB[None,:]); j=dd.argmin(1); gd=dd[np.arange(ok.shape[0]),j]
    on=ok.any(1); jj=np.where(on,ok.argmax(1),j)
    binv[isbet]=np.where(on,GB[jj],-np.round(r[isbet],1)); gdist[isbet]=np.where(on,0.0,gd); off[isbet]=~on
    m=bk==4; dd=np.abs(tb[m,None]-GO[None,:]); gd=dd.min(1); binv[m]=np.round(tb[m]*4)/4; gdist[m]=gd; off[m]=gd>0.06
    m=bk==5; muv=np.nan_to_num(mu[m],nan=9); dd=np.abs(muv[:,None]-G3[None,:]); gd=dd.min(1); binv[m]=np.round(muv*10)/10; gdist[m]=gd; off[m]=gd>0.12
    m=bk==6; binv[m]=np.round(rrv[m]*10)/10
    m=bk==0; binv[m]=np.round(np.clip(a["stack_frac"].to_numpy()[m],0,1)*4)/4
    a=a.with_columns(pl.Series("bin",binv),pl.Series("grid_dist",gdist),pl.Series("offgrid",off)).select(
        ["hid","pid","action_no","st","act","bk","bin","r_pot","grid_dist","offgrid","minraise","allin_less","stack_frac"])
    # per-player leave-one-hand-out histogram
    hb=a.group_by(["pid","bk","bin","hid"]).len().rename({"len":"c_hand"})
    pb=hb.group_by(["pid","bk","bin"]).agg(pl.col("c_hand").sum().alias("c_tot"))
    pn=a.group_by(["pid","bk","hid"]).len().rename({"len":"n_hand"})
    pnt=pn.group_by(["pid","bk"]).agg(pl.col("n_hand").sum().alias("n_tot"))
    pop=a.group_by(["bk","bin"]).len().rename({"len":"pop_c"}); popn=a.group_by("bk").agg(pl.len().alias("pop_n"),pl.col("bin").n_unique().alias("K"))
    a=(a.join(hb,on=["pid","bk","bin","hid"]).join(pb,on=["pid","bk","bin"])
        .join(pn,on=["pid","bk","hid"]).join(pnt,on=["pid","bk"])
        .join(pop,on=["bk","bin"]).join(popn,on="bk"))
    a=a.with_columns((pl.col("c_tot")-pl.col("c_hand")).alias("c_pl"),(pl.col("n_tot")-pl.col("n_hand")).alias("n_pl"),
                     (pl.col("pop_c")/pl.col("pop_n")).alias("p_pop"))
    a=a.with_columns(((pl.col("c_pl")+0.5)/(pl.col("n_pl")+0.5*pl.col("K"))).log().cast(pl.Float32).alias("logp_pl"))
    a=a.with_columns((pl.col("logp_pl")-pl.col("p_pop").log()).cast(pl.Float32).alias("lr"),
                     ((pl.col("c_pl")==0)&(pl.col("n_pl")>=10)).alias("never"),
                     (-pl.col("p_pop").log()).cast(pl.Float32).alias("pop_surp"))
    a=a.select(["hid","pid","action_no","st","act","bk","bin","r_pot","grid_dist","offgrid","minraise","allin_less","c_pl","n_pl","logp_pl","lr","never","pop_surp","stack_frac"])
    return a.join(h.select(["hid","hand_id"]),on="hid").join(players,on="pid").drop(["hid","pid"])

AGG=["bg_n_aggr","bg_n_bet","bg_offgrid_any","bg_offgrid_cnt","bg_maxdist","bg_never_any","bg_never_cnt","bg_min_logp_pl","bg_sum_logp_pl",
     "bg_min_lr","bg_sum_lr","bg_mean_lr","bg_minraise_any","bg_minraise_cnt","bg_allin_any","bg_allin_less_any","bg_small_bet_any",
     "bg_max_pop_surp","bg_sum_pop_surp","bg_both_offgrid","bg_pf_open_off","bg_post_off","bg_min_c_pl","bg_max_r_pot"]

def pair_hand_features(pairs, act):
    """pairs: DataFrame(pair_id, hand_id, player_1, player_2). act: action_size_table()."""
    ph=pl.concat([pairs.select(["pair_id","hand_id",pl.col("player_1").alias("player_id"),pl.lit(1).alias("who")]),
                  pairs.select(["pair_id","hand_id",pl.col("player_2").alias("player_id"),pl.lit(2).alias("who")])])
    x=ph.join(act,on=["hand_id","player_id"],how="inner")
    x=x.with_columns(((pl.col("bk")>=1)&(pl.col("bk")<=3)).alias("isbet"))
    g=x.group_by(["pair_id","hand_id"]).agg(
        pl.len().alias("bg_n_aggr"), pl.col("isbet").sum().alias("bg_n_bet"),
        pl.col("offgrid").any().cast(pl.Float32).alias("bg_offgrid_any"), pl.col("offgrid").sum().alias("bg_offgrid_cnt"),
        pl.col("grid_dist").max().alias("bg_maxdist"),
        pl.col("never").any().cast(pl.Float32).alias("bg_never_any"), pl.col("never").sum().alias("bg_never_cnt"),
        pl.col("logp_pl").min().alias("bg_min_logp_pl"), pl.col("logp_pl").sum().alias("bg_sum_logp_pl"),
        pl.col("lr").min().alias("bg_min_lr"), pl.col("lr").sum().alias("bg_sum_lr"), pl.col("lr").mean().alias("bg_mean_lr"),
        pl.col("minraise").any().cast(pl.Float32).alias("bg_minraise_any"), pl.col("minraise").sum().alias("bg_minraise_cnt"),
        (pl.col("act")==2).any().cast(pl.Float32).alias("bg_allin_any"),
        pl.col("allin_less").any().cast(pl.Float32).alias("bg_allin_less_any"),
        (pl.col("isbet")&(pl.col("r_pot")<0.3)).any().cast(pl.Float32).alias("bg_small_bet_any"),
        pl.col("pop_surp").max().alias("bg_max_pop_surp"), pl.col("pop_surp").sum().alias("bg_sum_pop_surp"),
        (pl.col("offgrid").filter(pl.col("who")==1).any() & pl.col("offgrid").filter(pl.col("who")==2).any()).cast(pl.Float32).alias("bg_both_offgrid"),
        (pl.col("offgrid")&(pl.col("bk")==4)).any().cast(pl.Float32).alias("bg_pf_open_off"),
        (pl.col("offgrid")&(pl.col("st")!=0)).any().cast(pl.Float32).alias("bg_post_off"),
        pl.col("c_pl").min().alias("bg_min_c_pl"), pl.col("r_pot").filter(pl.col("isbet")).max().alias("bg_max_r_pot"))
    out=pairs.select(["pair_id","hand_id"]).join(g,on=["pair_id","hand_id"],how="left")
    fill0=["bg_n_aggr","bg_n_bet","bg_offgrid_any","bg_offgrid_cnt","bg_maxdist","bg_never_any","bg_never_cnt","bg_minraise_any","bg_minraise_cnt",
           "bg_allin_any","bg_allin_less_any","bg_small_bet_any","bg_max_pop_surp","bg_sum_pop_surp","bg_both_offgrid","bg_pf_open_off","bg_post_off",
           "bg_sum_logp_pl","bg_sum_lr"]
    out=out.with_columns([pl.col(c).fill_null(0).cast(pl.Float32) for c in fill0]+[pl.col(c).cast(pl.Float32) for c in AGG if c not in fill0])
    return out

# ----------------------------------------------------------------------------- evaluation
def map5(df, score):
    # host formula: per positive pair sort by score desc, ties pot_bb desc, hand_id asc; AP@5 / min(n_rel,5)
    d=df.select(["pair_id","hand_id","pot_bb","is_ev"]).with_columns(pl.Series("s",score)).sort(["pair_id","s","pot_bb","hand_id"],descending=[False,True,True,False])
    res=[]
    for _,g in d.group_by("pair_id",maintain_order=True):
        rel=g["is_ev"].to_numpy(); nrel=rel.sum()
        if nrel==0: continue
        top=rel[:5]; hits=0; ap=0.0
        for k in range(len(top)):
            if top[k]: hits+=1; ap+=hits/(k+1)
        res.append(ap/min(nrel,5))
    return float(np.mean(res))

def run_test(feat):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    lab=pl.read_csv(D+"development_labels.csv"); ev=pl.read_csv(D+"development_evidence.csv")
    pos=lab.filter(pl.col("label")==1); ids=set(pos["pair_id"])
    src=open("step9_pipeline.py").read(); ns={}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")],ns); HF=ns["HAND_FEATS"]
    cols=["pair_id","hand_id","table_id","hand_idx","player_1","player_2"]+HF
    df=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(cols).collect() for i in range(4)])
    df=df.join(ev.select(["pair_id","hand_id",pl.lit(True).alias("is_ev")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_ev").fill_null(False)).join(pos.select(["pair_id","behavior_family"]),on="pair_id")
    df=df.join(feat,on=["pair_id","hand_id"],how="left")
    tables=sorted(df["table_id"].unique().to_list()); rng=np.random.RandomState(42); TF={t:int(v) for t,v in zip(tables,rng.permutation(len(tables))%5)}
    df=df.with_columns(pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold")).sort(["pair_id","hand_idx"])
    for f in ["directed_transfer","soft_play","coordinated_isolation"]: df=df.with_columns((pl.col("behavior_family")==f).cast(pl.Float32).alias("fam_"+f))
    FE=HF+["fam_directed_transfer","fam_soft_play","fam_coordinated_isolation"]
    y=df["is_ev"].to_numpy().astype(int); fold=df["fold"].to_numpy()
    conf=(~df["is_ev"].to_numpy())&(df["surp_pair_max"].to_numpy()>=3)
    print(f"rows {len(df)} ev {y.sum()} confusers {conf.sum()}")
    auc_all={}; auc_conf={}
    for c in AGG:
        v=df[c].to_numpy().astype(float); ok=~np.isnan(v)
        auc_all[c]=float(roc_auc_score(y[ok],v[ok]))
        m=ok&(conf|(y==1)); auc_conf[c]=float(roc_auc_score(y[m],v[m]))
        print(f"{c:22s} AUC ev-vs-all {auc_all[c]:.3f}  ev-vs-confusers {auc_conf[c]:.3f}  nonnull {ok.mean():.2f}")
    P=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=3)
    def oof_for(FEATS):
        X=df.select(FEATS).to_numpy().astype(np.float32); oof=np.zeros(len(y))
        for f in range(5):
            tr,te=fold!=f,fold==f
            for sd in (42,49): m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y[tr]),num_boost_round=400); oof[te]+=m.predict(X[te])/2
        return oof
    t=time.time(); o0=oof_for(FE); m0=map5(df,o0); print(f"baseline MAP@5 {m0:.4f} ({time.time()-t:.0f}s)")
    t=time.time(); o1=oof_for(FE+AGG); m1=map5(df,o1); print(f"with bet_grid MAP@5 {m1:.4f} ({time.time()-t:.0f}s)  delta {m1-m0:+.4f}")
    # per-family
    fam=df["behavior_family"].to_numpy()
    for f in ["directed_transfer","soft_play","coordinated_isolation"]:
        m=fam==f; sub=df.filter(pl.Series(m)); print(f"  {f:22s} base {map5(sub,o0[m]):.4f} with {map5(sub,o1[m]):.4f}")
    return auc_all,auc_conf,m0,m1

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--pairs",default=None); ap.add_argument("--out",default=H+"bet_grid.parquet"); ap.add_argument("--no-test",action="store_true")
    args=ap.parse_args()
    t=time.time(); act=action_size_table(); print("action table",act.shape,f"{time.time()-t:.0f}s")
    act.write_parquet(H+"bet_grid_actions.parquet")
    print(act.group_by("bk").agg(pl.len(),pl.col("offgrid").mean(),pl.col("minraise").mean(),pl.col("never").mean(),pl.col("allin_less").mean()).sort("bk"))
    if args.pairs: pairs=pl.read_parquet(args.pairs).select(["pair_id","hand_id","player_1","player_2"]).unique()
    else:
        lab=pl.read_csv(D+"development_labels.csv"); ids=set(lab.filter(pl.col("label")==1)["pair_id"])
        pairs=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(["pair_id","hand_id","player_1","player_2"]).collect() for i in range(4)])
    feat=pair_hand_features(pairs,act); feat.write_parquet(args.out); print("saved",args.out,feat.shape)
    if not args.no_test: run_test(feat)
