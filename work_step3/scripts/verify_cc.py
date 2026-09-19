import polars as pl, numpy as np, lightgbm as lgb, pandas as pd, os, warnings; warnings.filterwarnings("ignore")
W="work_step3/"; H=W+"hyp/"; d="data/"
lab=pl.read_csv(d+"development_labels.csv"); ev=pl.read_csv(d+"development_evidence.csv"); pos=lab.filter(pl.col("label")==1); ids=set(pos["pair_id"])
df=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).collect() for i in range(4)])
df=df.join(ev.select(["pair_id","hand_id",pl.lit(True).alias("is_ev")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_ev").fill_null(False)).join(pos.select(["pair_id","behavior_family"]),on="pair_id")
cc=pl.read_parquet(H+"category_combo.parquet"); CCR=[c for c in cc.columns if c.startswith("cc_") and not c.startswith("cc_te")]; CCT=[c for c in cc.columns if c.startswith("cc_te") or "_te" in c]
print("raw cc feats",len(CCR),"te feats",len(CCT))
df=df.join(cc.select(["pair_id","hand_id"]+CCR),on=["pair_id","hand_id"],how="left")
st=pl.read_parquet(H+"street_timing.parquet"); STC=[c for c in st.columns if c not in ("pair_id","hand_id")]; df=df.join(st,on=["pair_id","hand_id"],how="left")
pp=pl.read_parquet(H+"player_policy.parquet") if os.path.exists(H+"player_policy.parquet") else None
if pp is not None: PPC=[c for c in pp.columns if c not in ("pair_id","hand_id")]; df=df.join(pp,on=["pair_id","hand_id"],how="left")
else: PPC=[]
src=open("step9_pipeline.py").read(); ns={}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")],ns); HF=ns["HAND_FEATS"]
FAM=["directed_transfer","soft_play","coordinated_isolation"]
for f in FAM: df=df.with_columns((pl.col("behavior_family")==f).cast(pl.Float32).alias("fam_"+f))
FAMC=["fam_"+f for f in FAM]
pdf=df.to_pandas(); y=pdf.is_ev.values.astype(int); pid=pdf.pair_id.values
def map5(s):
    dd=pd.DataFrame({"p":pid,"s":s,"y":y,"pot":pdf.pot_bb.values,"h":pdf.hand_id.values}).sort_values(["p","s","pot","h"],ascending=[True,False,False,True]); vals=[]
    for _,g in dd.groupby("p",sort=False):
        r=g.y.values; n=r.sum()
        if n==0: continue
        t=r[:5]; vals.append((np.cumsum(t)/np.arange(1,6)*t).sum()/min(n,5))
    return np.mean(vals)
P=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
tables=sorted(pdf.table_id.unique())
for perm_seed in [7,123]:
    rng=np.random.RandomState(perm_seed); TF={t:int(v) for t,v in zip(tables,rng.permutation(len(tables))%5)}; fold=pdf.table_id.map(TF).values
    def run(feats,seeds=(1,2,3)):
        X=pdf[feats].fillna(0).values.astype(np.float32); o=np.zeros(len(y))
        for f in range(5):
            tr,te=fold!=f,fold==f
            for sd in seeds: o[te]+=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y[tr]),num_boost_round=400).predict(X[te])/len(seeds)
        return o
    b=map5(run(HF+FAMC)); c1=map5(run(HF+FAMC+CCR)); c2=map5(run(HF+FAMC+STC)); c3=map5(run(HF+FAMC+CCR+STC+PPC))
    print(f"perm {perm_seed}: baseline={b:.4f} +cc_raw={c1:.4f} ({c1-b:+.4f}) +street={c2:.4f} ({c2-b:+.4f}) +cc+street+policy={c3:.4f} ({c3-b:+.4f})",flush=True)
