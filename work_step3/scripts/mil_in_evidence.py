"""Does a per-hand MIL score help the evidence ranker? Train MIL v2-style model (quick), get OOF hand scores for positive pairs, add to stage-1/stage-2 harness."""
import polars as pl, numpy as np, lightgbm as lgb, pandas as pd, os, warnings, time; warnings.filterwarnings("ignore")
T0=time.time(); log=lambda m: print(f"[{time.time()-T0:5.0f}s] {m}",flush=True)
W="work_step3/"; d="data/"
src=open("step9_pipeline.py").read(); ns={}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")],ns); HF=ns["HAND_FEATS"]
dev_pairs=pl.read_parquet(W+"dev_pairs.parquet"); oof=pd.read_parquet(W+"dev_oof_v13.parquet"); oof=oof[oof.window==0]; pair_oof=dict(zip(oof.pair_id_orig,oof.oof))
TABLES=sorted(dev_pairs["table_id"].unique().to_list()); rng=np.random.RandomState(42); TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%5)}
lab=dict(zip(dev_pairs["pair_id"],dev_pairs["label"].fill_null(0))); islab=dict(zip(dev_pairs["pair_id"],dev_pairs["is_labeled"]))
unl=[p for p in dev_pairs["pair_id"] if not islab[p]]; rs=np.random.RandomState(7)
clean=[p for p in unl if pair_oof.get(p,0)<0.001]; hard=[p for p in unl if 0.001<=pair_oof.get(p,0)<0.3]
keep=set(p for p in dev_pairs["pair_id"] if islab[p])|set(rs.choice(clean,8000,replace=False))|set(rs.choice(hard,min(3000,len(hard)),replace=False))
cols=["pair_id","hand_id","table_id","hand_idx"]+HF
hs=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(keep))).select(cols).collect() for i in range(4)])
hs=hs.with_columns(pl.col("pair_id").replace_strict(lab,return_dtype=pl.Int8).alias("plabel"),pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold"))
X=hs.select(HF).to_numpy().astype(np.float32); pl_=hs["plabel"].to_numpy(); fold=hs["fold"].to_numpy(); bag=hs["pair_id"].to_numpy()
P=dict(objective="binary",learning_rate=0.05,num_leaves=63,min_data_in_leaf=100,feature_fraction=0.6,bagging_fraction=0.7,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
y=pl_.astype(int); w=np.where(pl_==1,0.3,1.0); K=8
for it in range(3):
    s=np.zeros(len(y))
    for f in range(5):
        tr,te=fold!=f,fold==f; s[te]=lgb.train(P,lgb.Dataset(X[tr],y[tr],weight=w[tr]),num_boost_round=300).predict(X[te])
    df=pd.DataFrame({"bag":bag,"s":s,"pl":pl_}); df["rk"]=df.groupby("bag").s.rank(ascending=False,method="first")
    y=np.where((df.pl==1)&(df.rk<=K),1,0); w=np.where(df.pl==1,np.where(df.rk<=K,1.0,0.2),1.0); log(f"MIL iter {it} done")
# final OOF score for positive pairs' hands
s=np.zeros(len(y))
for f in range(5):
    tr,te=fold!=f,fold==f; s[te]=lgb.train(P,lgb.Dataset(X[tr],y[tr],weight=w[tr]),num_boost_round=300).predict(X[te])
pos_ids=set(p for p in dev_pairs["pair_id"] if lab[p]==1)
mil=hs.select(["pair_id","hand_id"]).with_columns(pl.Series("mil",s)).filter(pl.col("pair_id").is_in(list(pos_ids)))
mil=mil.with_columns((pl.col("mil").rank(method="average").over("pair_id")/pl.len().over("pair_id")).alias("mil_prank"),(pl.col("mil")/pl.col("mil").max().over("pair_id")).alias("mil_rel"))
mil.write_parquet(W+"pos_hand_mil.parquet"); log(f"saved per-hand MIL for positive pairs {mil.shape}")
# evidence harness
exec(open("work_step3/scripts/ev_template_s2.py").read().split("print(f\"stage-1 baseline")[0])
df=df.join(mil,on=["pair_id","hand_id"],how="left").with_columns([pl.col(c).fill_null(0.0) for c in ["mil","mil_prank","mil_rel"]])
MILC=["mil","mil_prank","mil_rel"]
pdf=df.to_pandas(); yy=pdf.is_ev.values.astype(int); fo=pdf.fold.values; pid=pdf.pair_id.values
def map5s(sc):
    dd=pd.DataFrame({"p":pid,"s":sc,"y":yy,"pot":pdf.pot_bb.values,"h":pdf.hand_id.values}).sort_values(["p","s","pot","h"],ascending=[True,False,False,True]); vals=[]
    for _,g in dd.groupby("p",sort=False):
        r=g.y.values; n=r.sum()
        if n==0: continue
        t=r[:5]; vals.append((np.cumsum(t)/np.arange(1,6)*t).sum()/min(n,5))
    return np.mean(vals)
PS=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
def run1(feats,seeds=(42,49)):
    Xf=pdf[feats].values.astype(np.float32); o=np.zeros(len(yy))
    for f in range(5):
        tr,te=fo!=f,fo==f
        for sd in seeds: o[te]+=lgb.train({**PS,"seed":sd},lgb.Dataset(Xf[tr],yy[tr]),num_boost_round=400).predict(Xf[te])/len(seeds)
    return o
o1=run1(HF+FAMC); print(f"stage-1 base(+family)      MAP@5={map5s(o1):.4f}",flush=True)
o2=run1(HF+FAMC+MILC); print(f"stage-1 + MIL hand score   MAP@5={map5s(o2):.4f}",flush=True)
r0,_,_=stage2(o1,HF+EXTRA+FAMC+TEC); print(f"stage-2 base+TE                 MAP@5={r0:.4f}",flush=True)
r1,_,_=stage2(o2,HF+EXTRA+FAMC+TEC+MILC); print(f"stage-2 +MIL stage-1, +MIL+TE   MAP@5={r1:.4f}",flush=True)
r2,_,_=stage2(o1,HF+EXTRA+FAMC+TEC+MILC); print(f"stage-2 base stage-1, +MIL+TE   MAP@5={r2:.4f}",flush=True)
