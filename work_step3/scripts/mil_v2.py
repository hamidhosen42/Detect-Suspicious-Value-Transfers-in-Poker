"""MIL v2 ablation: cleaner/more negatives, both windows as bags, K variants, MIL score also as evidence-pool feature. Reports three dev views."""
import polars as pl, numpy as np, lightgbm as lgb, pandas as pd, os, warnings, time; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
T0=time.time(); log=lambda m: print(f"[{time.time()-T0:5.0f}s] {m}",flush=True)
W="work_step3/"
src=open("step9_pipeline.py").read(); ns={}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")],ns); HF=ns["HAND_FEATS"]
dev_pairs=pl.read_parquet(W+"dev_pairs.parquet"); oof=pd.read_parquet(W+"dev_oof_v11.parquet"); oof=oof[oof.window==0]; pair_oof=dict(zip(oof.pair_id_orig,oof.oof))
TABLES=sorted(dev_pairs["table_id"].unique().to_list()); rng=np.random.RandomState(42); TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%5)}
lab=dict(zip(dev_pairs["pair_id"],dev_pairs["label"].fill_null(0))); islab=dict(zip(dev_pairs["pair_id"],dev_pairs["is_labeled"]))
unl=[p for p in dev_pairs["pair_id"] if not islab[p]]; rs=np.random.RandomState(7)
clean=[p for p in unl if pair_oof.get(p,0)<0.001]; neg_sample=set(rs.choice(clean,12000,replace=False))
hard=[p for p in unl if 0.001<=pair_oof.get(p,0)<0.3]; hard_sample=set(rs.choice(hard,min(3000,len(hard)),replace=False))   # hard negatives (loose-bot-like, below pseudo-positive range)
keep=set(p for p in dev_pairs["pair_id"] if islab[p])|neg_sample|hard_sample
cols=["pair_id","hand_id","table_id","hand_idx"]+HF
hs=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(keep))).select(cols).collect() for i in range(4)])
# bags = (pair, window): window0 hands [0,2000), window1 hands [1000,3000)
hs=pl.concat([hs.filter(pl.col("hand_idx")<2000).with_columns(pl.lit(0,dtype=pl.Int8).alias("win")),hs.filter(pl.col("hand_idx")>=1000).with_columns(pl.lit(1,dtype=pl.Int8).alias("win"))])
hs=hs.with_columns(pl.col("pair_id").replace_strict(lab,return_dtype=pl.Int8).alias("plabel"),pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold"),(pl.col("pair_id")+"_"+pl.col("win").cast(pl.Utf8)).alias("bag"))
log(f"hand sample {hs.shape}; positive-bag hands {(hs['plabel']==1).sum():,}; bags {hs['bag'].n_unique():,}")
X=hs.select(HF).to_numpy().astype(np.float32); pl_=hs["plabel"].to_numpy(); fold=hs["fold"].to_numpy(); bag=hs["bag"].to_numpy()
P=dict(objective="binary",learning_rate=0.05,num_leaves=63,min_data_in_leaf=100,feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
def mil(K,iters=3,rounds=300):
    y=pl_.astype(int); w=np.where(pl_==1,0.3,1.0)
    for it in range(iters):
        s=np.zeros(len(y))
        for f in range(5):
            tr,te=fold!=f,fold==f; s[te]=lgb.train(P,lgb.Dataset(X[tr],y[tr],weight=w[tr]),num_boost_round=rounds).predict(X[te])
        df=pd.DataFrame({"bag":bag,"s":s,"pl":pl_}); df["rk"]=df.groupby("bag").s.rank(ascending=False,method="first")
        y=np.where((df.pl==1)&(df.rk<=K),1,0); w=np.where(df.pl==1,np.where(df.rk<=K,1.0,0.2),1.0)
    models={f:lgb.train(P,lgb.Dataset(X[fold!=f],y[fold!=f],weight=w[fold!=f]),num_boost_round=rounds) for f in range(5)}
    return models
def agg_all(models,tag):
    out={}
    for w_,(lo,hi) in enumerate([(0,2000),(1000,3000)]):
        lf=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet") for i in range(4)]).filter((pl.col("hand_idx")>=lo)&(pl.col("hand_idx")<hi)); parts=[]
        n=lf.select(pl.len()).collect().item()
        for st in range(0,n,1_500_000):
            c=lf.slice(st,1_500_000).collect(); Xc=c.select(HF).to_numpy().astype(np.float32); sc=np.zeros(c.height); fo=c["table_id"].replace_strict(TF,return_dtype=pl.Int8).to_numpy()
            for f in range(5):
                m=fo==f
                if m.any(): sc[m]=models[f].predict(Xc[m])
            parts.append(c.select(["pair_id"]).with_columns(pl.Series("mil",sc)))
        out[w_]=pl.concat(parts).group_by("pair_id").agg(pl.col("mil").top_k(3).mean().alias(f"{tag}_top3"),pl.col("mil").top_k(8).mean().alias(f"{tag}_top8"),pl.col("mil").mean().alias(f"{tag}_mean"),(pl.col("mil")>0.5).sum().alias(f"{tag}_n50"),(pl.col("mil")>0.8).sum().alias(f"{tag}_n80"),(pl.col("mil")>0.5).mean().alias(f"{tag}_f50"),pl.col("mil").sum().alias(f"{tag}_sum"),pl.col("mil").max().alias(f"{tag}_max")).to_pandas().set_index("pair_id")
    return out
dev=pd.read_parquet(W+"dev_pair_features_v9.parquet")
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
# v1 MIL columns from the Step 11 run
m0=pd.read_parquet(W+"mil_dev_w0_v11.parquet").set_index("pair_id"); m1=pd.read_parquet(W+"mil_dev_w1_v11.parquet").set_index("pair_id")
def merge(dev,m0,m1):
    vals=np.where((dev.window==0).values[:,None],m0.reindex(dev.pair_id_orig).values,m1.reindex(dev.pair_id_orig).values)
    for i,c in enumerate(m0.columns): dev[c]=vals[:,i]
    return list(m0.columns)
MC1=merge(dev,m0,m1)
y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; folds=dev.fold.values; W0=(dev.window==0).values; nsh=dev.n_shared.values
ref=pd.read_parquet(W+"hidden150_ref.parquet"); hidden=set(ref[~ref.is_labeled].sort_values("o",ascending=False).pair_id_orig[:150]); hid=dev.pair_id_orig.isin(hidden).values
PP=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=os.cpu_count())
wt=np.where(labm,np.where(y==1,5.0,1.0),0.1)
def cv(feats):
    Xp=dev[feats].astype("float32"); o=np.zeros(len(y))
    for f in range(5):
        tr,te=folds!=f,folds==f; o[te]=lgb.train(PP,lgb.Dataset(Xp[tr],y[tr],weight=wt[tr]),num_boost_round=750).predict(Xp[te])
    return o
def report(nm,o):
    yh=np.where(hid,1,y)
    for tag,mm in [("w0",W0),("w0&ns>=38",W0&(nsh>=38))]: print(f"{nm:34s} [{tag:9s}] labelled={ap(y[mm&labm],o[mm&labm]):.4f} pessimistic={ap(y[mm],o[mm]):.4f} hidden150={ap(yh[mm],o[mm]):.4f}",flush=True)
report("Step9 + MIL v1 (Step 11)",cv(F+MC1))
for K in [8,15]:
    models=mil(K); a=agg_all(models,f"milv2k{K}"); MC2=merge(dev,a[0],a[1]); log(f"MIL v2 K={K} aggregated")
    report(f"Step9 + MIL v2 K={K}",cv(F+MC2)); report(f"Step9 + MIL v1 + v2 K={K}",cv(F+MC1+MC2))
    a[0].to_parquet(W+f"milv2k{K}_dev_w0.parquet"); a[1].to_parquet(W+f"milv2k{K}_dev_w1.parquet")
