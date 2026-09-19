import polars as pl, numpy as np, lightgbm as lgb, pandas as pd, os, warnings; warnings.filterwarnings("ignore")
d="data/"; W="work_step3/"
lab=pl.read_csv(d+"development_labels.csv"); ev=pl.read_csv(d+"development_evidence.csv"); pos=lab.filter(pl.col("label")==1); ids=set(pos["pair_id"])
df=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).collect() for i in range(4)])
df=df.join(ev.select(["pair_id","hand_id",pl.lit(True).alias("is_ev")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_ev").fill_null(False)).join(pos.select(["pair_id","behavior_family"]),on="pair_id")
tp=pl.read_parquet(W+"pos_hand_templates.parquet"); TEC=[c for c in tp.columns if c.startswith("te_") or c.startswith("n_tpl")]
df=df.join(tp.select(["pair_id","hand_id"]+TEC),on=["pair_id","hand_id"])
src=open("step9_pipeline.py").read(); ns={}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")],ns); HF=ns["HAND_FEATS"]
tables=sorted(df["table_id"].unique().to_list()); rng=np.random.RandomState(42); TF={t:int(v) for t,v in zip(tables,rng.permutation(len(tables))%5)}
df=df.with_columns(pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold")).sort(["pair_id","hand_idx"])
FAM=["directed_transfer","soft_play","coordinated_isolation"]
for f in FAM: df=df.with_columns((pl.col("behavior_family")==f).cast(pl.Float32).alias("fam_"+f))
FAMC=["fam_"+f for f in FAM]
oofs=np.load(W+"tpl_oofs.npy")   # [baseline+family, +template TE] stage-1 OOF, same row order (sorted pair_id,hand_idx)
def map5(frame,col):
    dd=frame.select(["pair_id",col,"is_ev","pot_bb","hand_id"]).to_pandas().sort_values(["pair_id",col,"pot_bb","hand_id"],ascending=[True,False,False,True]); vals=[]
    for _,g in dd.groupby("pair_id",sort=False):
        r=g.is_ev.values.astype(int); n=r.sum()
        if n==0: continue
        t=r[:5]; vals.append((np.cumsum(t)/np.arange(1,6)*t).sum()/min(n,5))
    return np.mean(vals)
def pool_features(frame,K=20):
    pool=(frame.with_columns(pl.col("hand_score").rank(method="ordinal",descending=True).over("pair_id").alias("_r")).filter(pl.col("_r")<=K).sort(["pair_id","hand_idx"])
      .with_columns(pl.len().over("pair_id").cast(pl.Float32).alias("pool_size"),pl.col("hand_idx").rank(method="ordinal").over("pair_id").cast(pl.Float32).alias("t_rank_pool"),
                    (pl.col("hand_score")/pl.col("hand_score").max().over("pair_id")).cast(pl.Float32).alias("hs_rel"),(pl.col("hand_idx").cast(pl.Float32)/3000).alias("idx_rel"),
                    pl.col("hand_idx").cast(pl.Int32).diff().over("pair_id").fill_null(9999).cast(pl.Float32).alias("gap_prev_pool"),(-pl.col("hand_idx").cast(pl.Int32).diff(-1).over("pair_id")).fill_null(9999).cast(pl.Float32).alias("gap_next_pool"),
                    (pl.col("hand_score")>0.5).cast(pl.Float32).cum_sum().over("pair_id").alias("_nsb"),pl.col("hand_score").cum_sum().over("pair_id").alias("_scb"),pl.col("hand_score").sum().over("pair_id").alias("_sps"),
                    (pl.col("hand_score")>0.5).sum().over("pair_id").cast(pl.Float32).alias("n_strong_pool"),pl.col("hand_score").rank(method="ordinal",descending=True).over("pair_id").cast(pl.Float32).alias("score_rank_pool"))
      .with_columns((pl.col("t_rank_pool")/pl.col("pool_size")).alias("t_rank_pool_rel"),(pl.col("_nsb")-(pl.col("hand_score")>0.5).cast(pl.Float32)).alias("n_strong_before"),(pl.col("_scb")-pl.col("hand_score")).alias("score_cum_before"),(pl.col("_scb")/pl.col("_sps")).alias("score_cum_rel")).drop(["_nsb","_scb","_sps"]))
    near=(pool.select(["pair_id","hand_id","hand_idx"]).join(pool.select(["pair_id",pl.col("hand_idx").alias("_o")]),on="pair_id").filter((pl.col("_o")-pl.col("hand_idx")).abs()<=20).group_by(["pair_id","hand_id"]).agg(pl.len().cast(pl.Float32).alias("n_pool_within20")))
    return pool.join(near,on=["pair_id","hand_id"],how="left").with_columns(pl.col("n_pool_within20").fill_null(1.0)).drop("_r")
EXTRA=["t_rank_pool","t_rank_pool_rel","hand_score","hs_rel","idx_rel","gap_prev_pool","gap_next_pool","n_pool_within20","pool_size","n_strong_before","score_cum_before","score_cum_rel","n_strong_pool","score_rank_pool"]
P=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_data_in_leaf=40,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
def stage2(s1,feats,K=20,rounds=500,seeds=(42,45,47)):
    fr=df.with_columns(pl.Series("hand_score",s1)); pool=pool_features(fr,K).to_pandas(); pool["s2"]=0.0
    X=pool[feats].astype("float32").values; y=pool["is_ev"].astype(int).values; fold=pool["fold"].values
    for f in range(5):
        tr,te=fold!=f,fold==f
        for sd in seeds: m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y[tr]),num_boost_round=rounds); pool.loc[te,"s2"]+=m.predict(X[te])/len(seeds)
    out=fr.join(pl.from_pandas(pool[["pair_id","hand_id","s2"]]),on=["pair_id","hand_id"],how="left").with_columns(pl.when(pl.col("s2").is_not_null()).then(10.0+pl.col("s2")).otherwise(pl.col("hand_score")).alias("final"))
    return map5(out,"final"), pool, m
print(f"stage-1 baseline MAP@5={map5(df.with_columns(pl.Series('hand_score',oofs[0])),'hand_score'):.4f}; +TE {map5(df.with_columns(pl.Series('hand_score',oofs[1])),'hand_score'):.4f}",flush=True)
r0,_,_=stage2(oofs[0],HF+EXTRA+FAMC); print(f"stage-2 baseline (Step 9 config)            MAP@5={r0:.4f}",flush=True)
r1,_,_=stage2(oofs[1],HF+EXTRA+FAMC); print(f"stage-2 on TE stage-1, no TE in pool        MAP@5={r1:.4f}",flush=True)
r2,pool,m=stage2(oofs[1],HF+EXTRA+FAMC+TEC); print(f"stage-2 on TE stage-1, + TE in pool         MAP@5={r2:.4f}",flush=True)
imp=sorted(zip(m.feature_importance("gain"),HF+EXTRA+FAMC+TEC),reverse=True)[:12]; print("top stage-2 feats:",[(f,int(v)) for v,f in imp])
r3,_,_=stage2(oofs[1],HF+EXTRA+FAMC+TEC,K=25); print(f"stage-2 + TE, K=25                          MAP@5={r3:.4f}",flush=True)
