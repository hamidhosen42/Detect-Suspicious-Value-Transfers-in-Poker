"""MIL v3: no within-pair prank/to_max features. Leave-table-out scoring for dev (OOF) and eval; report dev views and the dev/eval tail alignment."""
import polars as pl, numpy as np, lightgbm as lgb, pandas as pd, os, warnings, time; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
T0=time.time(); log=lambda m: print(f"[{time.time()-T0:5.0f}s] {m}",flush=True)
W="work_step3/"
src=open("step9_pipeline.py").read(); ns={}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")],ns); HF=ns["HAND_FEATS"]
HF3=[c for c in HF if not c.endswith("_prank") and not c.endswith("_to_max")]; log(f"features {len(HF)} -> {len(HF3)} (no prank/to_max)")
dev_pairs=pl.read_parquet(W+"dev_pairs.parquet"); oof=pd.read_parquet(W+"dev_oof_v9.parquet"); oof=oof[oof.window==0]; pair_oof=dict(zip(oof.pair_id_orig,oof.oof))
TABLES=sorted(dev_pairs["table_id"].unique().to_list()); rng=np.random.RandomState(42); TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%5)}
lab=dict(zip(dev_pairs["pair_id"],dev_pairs["label"].fill_null(0))); islab=dict(zip(dev_pairs["pair_id"],dev_pairs["is_labeled"]))
unl=[p for p in dev_pairs["pair_id"] if not islab[p]]; rs=np.random.RandomState(7)
clean=[p for p in unl if pair_oof.get(p,0)<0.001]; hard=[p for p in unl if 0.001<=pair_oof.get(p,0)<0.3]
keep=set(p for p in dev_pairs["pair_id"] if islab[p])|set(rs.choice(clean,12000,replace=False))|set(rs.choice(hard,min(3000,len(hard)),replace=False))
cols=["pair_id","hand_id","table_id","hand_idx"]+HF3
hs=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(keep))).select(cols).collect() for i in range(4)])
hs=pl.concat([hs.filter(pl.col("hand_idx")<2000).with_columns(pl.lit(0,dtype=pl.Int8).alias("win")),hs.filter(pl.col("hand_idx")>=1000).with_columns(pl.lit(1,dtype=pl.Int8).alias("win"))])
hs=hs.with_columns(pl.col("pair_id").replace_strict(lab,return_dtype=pl.Int8).alias("plabel"),pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold"),(pl.col("pair_id")+"_"+pl.col("win").cast(pl.Utf8)).alias("bag"))
X=hs.select(HF3).to_numpy().astype(np.float32); pl_=hs["plabel"].to_numpy(); fold=hs["fold"].to_numpy(); bag=hs["bag"].to_numpy()
P=dict(objective="binary",learning_rate=0.05,num_leaves=63,min_data_in_leaf=100,feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
y=pl_.astype(int); w=np.where(pl_==1,0.3,1.0); K=8
for it in range(3):
    s=np.zeros(len(y))
    for f in range(5):
        tr,te=fold!=f,fold==f; s[te]=lgb.train(P,lgb.Dataset(X[tr],y[tr],weight=w[tr]),num_boost_round=300).predict(X[te])
    df=pd.DataFrame({"bag":bag,"s":s,"pl":pl_}); df["rk"]=df.groupby("bag").s.rank(ascending=False,method="first")
    y=np.where((df.pl==1)&(df.rk<=K),1,0); w=np.where(df.pl==1,np.where(df.rk<=K,1.0,0.2),1.0); log(f"MIL v3 iter {it} done")
models={f:lgb.train(P,lgb.Dataset(X[fold!=f],y[fold!=f],weight=w[fold!=f]),num_boost_round=300) for f in range(5)}
imp=sorted(zip(models[0].feature_importance("gain"),HF3),reverse=True)[:12]; print("MIL v3 top feats:",[(f,int(v)) for v,f in imp],flush=True)
def agg(lf):
    parts=[]; n=lf.select(pl.len()).collect().item()
    for st in range(0,n,1_500_000):
        c=lf.slice(st,1_500_000).collect(); Xc=c.select(HF3).to_numpy().astype(np.float32); sc=np.zeros(c.height); fo=c["table_id"].replace_strict(TF,return_dtype=pl.Int8).to_numpy()
        for f in range(5):
            m=fo==f
            if m.any(): sc[m]=models[f].predict(Xc[m])
        parts.append(c.select(["pair_id"]).with_columns(pl.Series("mil",sc)))
    return pl.concat(parts).group_by("pair_id").agg(pl.col("mil").top_k(3).mean().alias("mil_top3"),pl.col("mil").top_k(8).mean().alias("mil_top8"),pl.col("mil").mean().alias("mil_mean"),(pl.col("mil")>0.5).sum().alias("mil_n50"),(pl.col("mil")>0.8).sum().alias("mil_n80"),(pl.col("mil")>0.5).mean().alias("mil_f50"),pl.col("mil").sum().alias("mil_sum"),pl.col("mil").max().alias("mil_max")).to_pandas().set_index("pair_id")
lfd=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet") for i in range(4)])
a0=agg(lfd.filter(pl.col("hand_idx")<2000)); a1=agg(lfd.filter(pl.col("hand_idx")>=1000)); ae=agg(pl.scan_parquet(W+"eval_hand_features_v9.parquet")); log("aggregated")
a0.to_parquet(W+"milv3_dev_w0.parquet"); a1.to_parquet(W+"milv3_dev_w1.parquet"); ae.to_parquet(W+"milv3_eval.parquet")
# tail alignment
dev=pd.read_parquet(W+"dev_pair_features_v9.parquet",columns=["pair_id","pair_id_orig","window","label","is_labeled"]); d0=dev[dev.window==0].join(a0,on="pair_id_orig")
r8=pd.read_csv("deliverables_step8/submission_step8.csv")[["pair_id","risk_score"]]; ev=ae.join(r8.set_index("pair_id")); ev["rk"]=ev.risk_score.rank(ascending=False)
pos=d0[d0.label==1]; un=d0[~d0.is_labeled]; top=ev[ev.rk<=400]; rest=ev[ev.rk>2000]
for c in ["mil_top3","mil_max","mil_mean","mil_n50"]:
    print(f"{c:9s} dev POS q10/50/90={np.quantile(pos[c],[.1,.5,.9]).round(3)} | eval top-400 q={np.quantile(top[c],[.1,.5,.9]).round(3)} || dev unl q50/90/99={np.quantile(un[c],[.5,.9,.99]).round(3)} | eval rank>2000 q={np.quantile(rest[c],[.5,.9,.99]).round(3)}",flush=True)
