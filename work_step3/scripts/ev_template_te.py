import polars as pl, numpy as np, lightgbm as lgb, pandas as pd, os, warnings, collections; warnings.filterwarnings("ignore")
d="data/"; W="work_step3/"
lab=pl.read_csv(d+"development_labels.csv"); ev=pl.read_csv(d+"development_evidence.csv"); pos=lab.filter(pl.col("label")==1); ids=set(pos["pair_id"])
df=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).collect() for i in range(4)])
df=df.join(ev.select(["pair_id","hand_id",pl.lit(True).alias("is_ev")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_ev").fill_null(False)).join(pos.select(["pair_id","behavior_family"]),on="pair_id")
ph=pl.read_parquet(W+"dev_pair_hands.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(["pair_id","hand_id","player_1","player_2"])
ac=pl.scan_parquet(W+"action_context.parquet").filter(pl.col("hand_id").is_in(df["hand_id"].unique().to_list())).select(["hand_id","action_no","street_no","player_id","action","to_call","last_aggr"]).collect()
a=ph.join(df.select(["pair_id","hand_id","loser_is_p1"]),on=["pair_id","hand_id"]).join(ac,on="hand_id")
a=a.with_columns(pl.when(pl.col("player_id")==pl.col("player_1")).then(pl.lit("A")).when(pl.col("player_id")==pl.col("player_2")).then(pl.lit("B")).otherwise(pl.lit("O")).alias("who"))
a=a.with_columns(pl.when(pl.col("who")=="O").then(pl.lit("O")).when(pl.col("loser_is_p1")==1).then(pl.when(pl.col("who")=="A").then(pl.lit("L")).otherwise(pl.lit("W"))).otherwise(pl.when(pl.col("who")=="A").then(pl.lit("W")).otherwise(pl.lit("L"))).alias("role"),
                  pl.when(pl.col("to_call")==0).then(pl.lit("-")).when((pl.col("last_aggr")==pl.col("player_1"))|(pl.col("last_aggr")==pl.col("player_2"))).then(pl.lit("p")).otherwise(pl.lit("o")).alias("facing"))
a=a.sort(["pair_id","hand_id","action_no"]).with_columns((pl.col("role")+pl.col("street_no").cast(pl.Utf8)+pl.col("facing")+":"+pl.col("action").str.slice(0,2)).alias("tok"))
tp=a.group_by(["pair_id","hand_id"]).agg(pl.col("tok").filter(pl.col("role")!="O").str.join(" ").alias("tpl_pair"),
                                        pl.col("tok").str.join(" ").alias("tpl_all"),
                                        pl.col("tok").filter((pl.col("role")!="O")&(pl.col("street_no")==0)).str.join(" ").alias("tpl_pf"),
                                        pl.col("tok").filter(pl.col("role")!="O").head(3).str.join(" ").alias("tpl_first3"))
df=df.join(tp,on=["pair_id","hand_id"],how="left").with_columns([pl.col(c).fill_null("") for c in ["tpl_pair","tpl_all","tpl_pf","tpl_first3"]])
src=open("step9_pipeline.py").read(); ns={}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")],ns); HF=ns["HAND_FEATS"]
tables=sorted(df["table_id"].unique().to_list()); rng=np.random.RandomState(42); TF={t:int(v) for t,v in zip(tables,rng.permutation(len(tables))%5)}
df=df.with_columns(pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold")).sort(["pair_id","hand_idx"])
FAM=["directed_transfer","soft_play","coordinated_isolation"]
for f in FAM: df=df.with_columns((pl.col("behavior_family")==f).cast(pl.Float32).alias("fam_"+f))
pdf=df.to_pandas(); y=pdf.is_ev.values.astype(int); fold=pdf.fold.values; pid=pdf.pair_id.values
def te(col,by_family=False,k=10.0):
    """fold-aware smoothed target encoding of a template column (optionally within family)"""
    out=np.zeros(len(pdf)); key=pdf[col].values if not by_family else (pdf.behavior_family.values+"|"+pdf[col].values)
    for f in range(5):
        tr,tst=fold!=f,fold==f; prior=y[tr].mean()
        s=pd.DataFrame({"k":key[tr],"y":y[tr]}).groupby("k").y.agg(["sum","count"])
        m=(s["sum"]+k*prior)/(s["count"]+k); out[tst]=pd.Series(key[tst]).map(m).fillna(prior).values
    return out
TE={}
for col in ["tpl_pair","tpl_all","tpl_pf","tpl_first3"]:
    TE[f"te_{col}"]=te(col); TE[f"te_{col}_fam"]=te(col,True)
    pdf[f"n_{col}"]=pdf[col].str.count(" ")+1
for k,v in TE.items(): pdf[k]=v
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
    return oof
FAMC=["fam_"+f for f in FAM]; TEC=list(TE.keys())+[f"n_{c}" for c in ["tpl_pair","tpl_all","tpl_pf","tpl_first3"]]
o1=run(HF+FAMC); print(f"stage-1 baseline(+family)      MAP@5={map5(o1):.4f}",flush=True)
o2=run(HF+FAMC+TEC); print(f"stage-1 + template TE          MAP@5={map5(o2):.4f}",flush=True)
print("template TE univariate MAP@5:",{k:round(map5(pdf[k].values+1e-6*o1),4) for k in ["te_tpl_pair","te_tpl_pair_fam","te_tpl_all_fam","te_tpl_first3_fam"]})
pdf[["pair_id","hand_id","is_ev","behavior_family","tpl_pair","tpl_all","tpl_pf","tpl_first3"]+TEC].to_parquet(W+"pos_hand_templates.parquet",index=False)
np.save(W+"tpl_oofs.npy",np.vstack([o1,o2]))
