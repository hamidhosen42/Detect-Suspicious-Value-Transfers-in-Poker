import polars as pl, numpy as np, lightgbm as lgb, pandas as pd, os, warnings; warnings.filterwarnings("ignore")
d="data/"; W="work_step3/"
lab=pl.read_csv(d+"development_labels.csv"); ev=pl.read_csv(d+"development_evidence.csv")
pos=lab.filter(pl.col("label")==1); ids=set(pos["pair_id"])
df=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).collect() for i in range(4)])
df=df.join(ev.select(["pair_id","hand_id",pl.lit(True).alias("is_ev")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_ev").fill_null(False)).join(pos.select(["pair_id","behavior_family"]),on="pair_id")
src=open("step9_pipeline.py").read(); ns={}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")],ns); HF=ns["HAND_FEATS"]
tables=sorted(df["table_id"].unique().to_list()); rng=np.random.RandomState(42); TF={t:int(v) for t,v in zip(tables,rng.permutation(len(tables))%5)}
df=df.with_columns(pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold")).sort(["pair_id","hand_idx"])
for f in ["directed_transfer","soft_play","coordinated_isolation"]: df=df.with_columns((pl.col("behavior_family")==f).cast(pl.Float32).alias("fam_"+f))
FE=HF+["fam_directed_transfer","fam_soft_play","fam_coordinated_isolation"]
X=df.select(FE).to_numpy().astype(np.float32); y=df["is_ev"].to_numpy().astype(int); fold=df["fold"].to_numpy()
P=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
oof=np.zeros(len(y))
for f in range(5):
    tr,te=fold!=f,fold==f
    for sd in (42,49): m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y[tr]),num_boost_round=400); oof[te]+=m.predict(X[te])/2
df=df.with_columns(pl.Series("hand_score",oof))
df.select(["pair_id","hand_id","hand_idx","behavior_family","is_ev","hand_score","fold"]).write_parquet(W+"pos_hand_oof_v9.parquet"); print("saved")
