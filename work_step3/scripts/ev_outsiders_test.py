import polars as pl, numpy as np, lightgbm as lgb, pandas as pd, os, warnings; warnings.filterwarnings("ignore")
exec(open("work_step3/scripts/ev_outsiders.py").read().split("e=df.filter")[0])   # reuse: builds df with out_* features
src=open("step9_pipeline.py").read(); ns={}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")],ns); HF=ns["HAND_FEATS"]
full=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).collect() for i in range(4)])
df=full.join(df.select(["pair_id","hand_id","is_ev","behavior_family"]+[c for c in df.columns if c.startswith("out_")]),on=["pair_id","hand_id"])
tables=sorted(df["table_id"].unique().to_list()); rng=np.random.RandomState(42); TF={t:int(v) for t,v in zip(tables,rng.permutation(len(tables))%5)}
df=df.with_columns(pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold")).sort(["pair_id","hand_idx"])
FAM=["directed_transfer","soft_play","coordinated_isolation"]
for f in FAM: df=df.with_columns((pl.col("behavior_family")==f).cast(pl.Float32).alias("fam_"+f))
FAMC=["fam_"+f for f in FAM]; OUT=[c for c in df.columns if c.startswith("out_")]
# within-pair percentile ranks of outsider feats (like _prank)
df=df.with_columns([(pl.col(c).fill_null(-1).rank(method="average").over("pair_id")/pl.len().over("pair_id")).cast(pl.Float32).alias(f"{c}_prank") for c in OUT])
OUTP=OUT+[f"{c}_prank" for c in OUT]
df=df.with_columns([pl.col(c).fill_null(-1) for c in OUT])
pdf=df.to_pandas(); y=pdf.is_ev.values.astype(int); fold=pdf.fold.values; pid=pdf.pair_id.values
def map5(s):
    dd=pd.DataFrame({"p":pid,"s":s,"y":y,"pot":pdf.pot_bb.values,"h":pdf.hand_id.values}).sort_values(["p","s","pot","h"],ascending=[True,False,False,True]); vals=[]
    for _,g in dd.groupby("p",sort=False):
        r=g.y.values; n=r.sum()
        if n==0: continue
        t=r[:5]; vals.append((np.cumsum(t)/np.arange(1,6)*t).sum()/min(n,5))
    return np.mean(vals)
P=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
def run(feats,rounds=400,seeds=(42,49)):
    X=pdf[feats].values.astype(np.float32); oof=np.zeros(len(y))
    for f in range(5):
        tr,tst=fold!=f,fold==f
        for sd in seeds: m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y[tr]),num_boost_round=rounds); oof[tst]+=m.predict(X[tst])/len(seeds)
    return oof,m
o1,_=run(HF+FAMC); print(f"stage-1 baseline(+family)   MAP@5={map5(o1):.4f}",flush=True)
o2,m=run(HF+FAMC+OUTP); print(f"stage-1 + outsider feats    MAP@5={map5(o2):.4f}",flush=True)
imp=sorted(zip(m.feature_importance("gain"),HF+FAMC+OUTP),reverse=True)[:12]; print("top:",[(f,int(v)) for v,f in imp])
np.save(W+"out_oofs.npy",np.vstack([o1,o2])); pdf[["pair_id","hand_id"]+OUTP].to_parquet(W+"pos_hand_outsiders.parquet",index=False)
