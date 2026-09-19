"""MIL hand model: learn colluder-hand vs non-colluder-hand from pair labels (top-k pseudo labels), then pair aggregates -> three dev views."""
import polars as pl, numpy as np, lightgbm as lgb, pandas as pd, os, warnings, time; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
T0=time.time(); log=lambda m: print(f"[{time.time()-T0:5.0f}s] {m}",flush=True)
d="data/"; W="work_step3/"
src=open("step9_pipeline.py").read(); ns={}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")],ns); HF=ns["HAND_FEATS"]
dev_pairs=pl.read_parquet(W+"dev_pairs.parquet"); oof=pd.read_parquet(W+"dev_oof_v9.parquet"); oof=oof[oof.window==0]
pair_oof=dict(zip(oof.pair_id_orig,oof.oof)); fold_of=dict(zip(dev_pairs["pair_id"],dev_pairs["fold"] if "fold" in dev_pairs.columns else [0]*dev_pairs.height))
# folds by table (same permutation as the pipeline)
TABLES=sorted(dev_pairs["table_id"].unique().to_list()); rng=np.random.RandomState(42); TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%5)}
lab=dict(zip(dev_pairs["pair_id"],dev_pairs["label"].fill_null(0))); islab=dict(zip(dev_pairs["pair_id"],dev_pairs["is_labeled"]))
# hand sample: all hands of labelled pairs + hands of 6000 random unlabelled pairs with low pair OOF (clean negatives) + all hands of top-300 unlabelled (pseudo-positive candidates excluded from negatives)
unl=[p for p in dev_pairs["pair_id"] if not islab[p]]
rs=np.random.RandomState(1); clean=[p for p in unl if pair_oof.get(p,0)<0.002]; neg_sample=set(rs.choice(clean,6000,replace=False))
lab_pairs=set(p for p in dev_pairs["pair_id"] if islab[p]); keep=lab_pairs|neg_sample
cols=["pair_id","hand_id","table_id","hand_idx"]+HF
hs=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(keep))&(pl.col("hand_idx")<2000)).select(cols).collect() for i in range(4)])
hs=hs.with_columns(pl.col("pair_id").replace_strict(lab,return_dtype=pl.Int8).alias("plabel"),pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold"))
log(f"hand sample {hs.shape}; positive-pair hands {(hs['plabel']==1).sum():,}")
X=hs.select(HF).to_numpy().astype(np.float32); pl_=hs["plabel"].to_numpy(); fold=hs["fold"].to_numpy(); pid=hs["pair_id"].to_numpy()
P=dict(objective="binary",learning_rate=0.05,num_leaves=63,min_data_in_leaf=100,feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
# iteration 0 pseudo labels: for positive pairs, top-k hands by the evidence-style score are unknown here -> start with 'all hands of positive pairs = weak positive (w=0.3)', negatives w=1
K=8
y_h=pl_.copy().astype(int); w=np.where(pl_==1,0.3,1.0); s=np.zeros(len(y_h))
for it in range(3):
    s=np.zeros(len(y_h))
    for f in range(5):
        tr,te=fold!=f,fold==f
        m=lgb.train(P,lgb.Dataset(X[tr],y_h[tr],weight=w[tr]),num_boost_round=300); s[te]=m.predict(X[te])
    # new pseudo labels: within positive pairs, top-K hands by score -> 1 (w=1), rest -> 0 (w=0.2); negatives stay 0 (w=1)
    df=pd.DataFrame({"pid":pid,"s":s,"pl":pl_}); df["rk"]=df.groupby("pid").s.rank(ascending=False,method="first")
    y_h=np.where((df.pl==1)&(df.rk<=K),1,0); w=np.where(df.pl==1,np.where(df.rk<=K,1.0,0.2),1.0)
    # bag-level check: pair AP on the labelled pairs from top-K mean
    agg=df.groupby("pid").apply(lambda g: pd.Series({"top3":g.s.nlargest(3).mean(),"top8":g.s.nlargest(8).mean(),"n50":(g.s>0.5).sum(),"pl":g.pl.iloc[0]}))
    la=agg[[islab[p] for p in agg.index]]
    log(f"MIL iter {it}: labelled-pair AP from top3 mean={ap(la.pl,la.top3):.4f} top8={ap(la.pl,la.top8):.4f} n50={ap(la.pl,la.n50):.4f} | all sampled pairs AP top3={ap(agg.pl,agg.top3):.4f}")
# final model on all sampled hands (per fold, for OOF scoring of ALL dev hands) -> score all dev pair-hands (window 0) and eval hands, aggregate
models={}
for f in range(5):
    tr=fold!=f; models[f]=lgb.train(P,lgb.Dataset(X[tr],y_h[tr],weight=w[tr]),num_boost_round=300)
full=lgb.train(P,lgb.Dataset(X,y_h,weight=w),num_boost_round=300)
imp=sorted(zip(full.feature_importance("gain"),HF),reverse=True)[:15]; print("MIL top feats:",[(f,int(v)) for v,f in imp])
def agg_scores(lf,fold_map,use_full):
    parts=[]
    n=lf.select(pl.len()).collect().item(); CH=1_500_000
    for st in range(0,n,CH):
        c=lf.slice(st,CH).collect(); Xc=c.select(HF).to_numpy().astype(np.float32); sc=np.zeros(c.height)
        if use_full: sc=full.predict(Xc)
        else:
            fo=c["table_id"].replace_strict(fold_map,return_dtype=pl.Int8).to_numpy()
            for f in range(5):
                m=fo==f
                if m.any(): sc[m]=models[f].predict(Xc[m])
        parts.append(c.select(["pair_id"]).with_columns(pl.Series("mil",sc)))
    a=pl.concat(parts).group_by("pair_id").agg(pl.col("mil").top_k(3).mean().alias("mil_top3"),pl.col("mil").top_k(8).mean().alias("mil_top8"),pl.col("mil").mean().alias("mil_mean"),
        (pl.col("mil")>0.5).sum().alias("mil_n50"),(pl.col("mil")>0.8).sum().alias("mil_n80"),(pl.col("mil")>0.5).mean().alias("mil_f50"),pl.col("mil").sum().alias("mil_sum"),pl.col("mil").max().alias("mil_max"))
    return a
dev_agg=agg_scores(pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet") for i in range(4)]).filter(pl.col("hand_idx")<2000),TF,False)
dev_agg.write_parquet(W+"mil_dev_w0.parquet"); log(f"dev window-0 MIL aggregates {dev_agg.shape}")
dev_agg2=agg_scores(pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet") for i in range(4)]).filter(pl.col("hand_idx")>=1000),TF,False)
dev_agg2.write_parquet(W+"mil_dev_w1.parquet"); log("dev window-1 done")
ev_agg=agg_scores(pl.scan_parquet(W+"eval_hand_features_v9.parquet"),None,True); ev_agg.write_parquet(W+"mil_eval.parquet"); log(f"eval MIL aggregates {ev_agg.shape}")
# pair-model ablation with MIL features
dev=pd.read_parquet(W+"dev_pair_features_v9.parquet")
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
m0=dev_agg.to_pandas().set_index("pair_id"); m1=dev_agg2.to_pandas().set_index("pair_id")
MC=list(m0.columns)
milf=np.where((dev.window==0).values[:,None], m0.reindex(dev.pair_id_orig).values, m1.reindex(dev.pair_id_orig).values)
for i,c in enumerate(MC): dev[c]=milf[:,i]
y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; folds=dev.fold.values; W0=(dev.window==0).values; nsh=dev.n_shared.values
ref=pd.read_parquet(W+"hidden150_ref.parquet"); hidden=set(ref[~ref.is_labeled].sort_values("o",ascending=False).pair_id_orig[:150]); hid=dev.pair_id_orig.isin(hidden).values
PP=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=os.cpu_count())
wt=np.where(labm,np.where(y==1,5.0,1.0),0.1)
def cv(feats):
    Xp=dev[feats].astype("float32"); o=np.zeros(len(y))
    for f in range(5):
        tr,te=folds!=f,folds==f; mm=lgb.train(PP,lgb.Dataset(Xp[tr],y[tr],weight=wt[tr]),num_boost_round=750); o[te]=mm.predict(Xp[te])
    return o
def report(nm,o):
    yh=np.where(hid,1,y)
    for tag,mm in [("w0",W0),("w0&ns>=38",W0&(nsh>=38))]: print(f"{nm:26s} [{tag:9s}] labelled={ap(y[mm&labm],o[mm&labm]):.4f} pessimistic={ap(y[mm],o[mm]):.4f} hidden150={ap(yh[mm],o[mm]):.4f}",flush=True)
report("baseline (Step 9 feats)",cv(F)); report("+ MIL aggregates",cv(F+MC))
