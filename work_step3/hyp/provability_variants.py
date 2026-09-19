# Follow-up to provability.py: seed-noise check and reduced feature subsets (same folds; seeds as stated per run).
import polars as pl, numpy as np, pandas as pd, lightgbm as lgb, os, time, warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "work_step3/hyp"); from provability import map5, PUBLIC
D="data/"; W="work_step3/"; t0=time.time()
lab=pl.read_csv(D+"development_labels.csv"); ev=pl.read_csv(D+"development_evidence.csv"); pos=lab.filter(pl.col("label")==1); ids=list(set(pos["pair_id"]))
src=open("step9_pipeline.py").read(); ns={}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")],ns); HF=ns["HAND_FEATS"]
df=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(ids)).collect() for i in range(4)])
df=(df.join(ev.select(["pair_id","hand_id",pl.lit(True).alias("is_ev")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_ev").fill_null(False))
      .join(pos.select(["pair_id","behavior_family"]),on="pair_id").join(pl.read_parquet(W+"hyp/provability.parquet"),on=["pair_id","hand_id"]))
PV=[c for c in df.columns if c.startswith("pv_")]
tables=sorted(df["table_id"].unique().to_list()); rng=np.random.RandomState(42); TF={t:int(v) for t,v in zip(tables,rng.permutation(len(tables))%5)}
df=df.with_columns(pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold")).sort(["pair_id","hand_idx"])
FAM=["directed_transfer","soft_play","coordinated_isolation"]
for f in FAM: df=df.with_columns((pl.col("behavior_family")==f).cast(pl.Float32).alias("fam_"+f))
FAMC=["fam_"+f for f in FAM]; pdf=df.to_pandas(); y=pdf.is_ev.values.astype(int); fold=pdf.fold.values; pid=pdf.pair_id.values
P=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=3)
def run(F,seeds,rounds=400):
    X=pdf[F].values.astype(np.float32); oof=np.zeros(len(y))
    for f in range(5):
        tr,te=fold!=f,fold==f
        for sd in seeds: m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y[tr]),num_boost_round=rounds); oof[te]+=m.predict(X[te])/len(seeds)
    return oof
def report(name,oof):
    v=map5(pid,oof,y,pdf.pot_bb.values,pdf.hand_id.values); fam=pdf.groupby("pair_id").behavior_family.first()
    per={f:np.mean([s for p,s in v.items() if fam[p]==f]) for f in FAM}
    print(f"{name:44s} MAP@5={np.mean(list(v.values())):.4f}  DT={per['directed_transfer']:.4f} SP={per['soft_play']:.4f} CI={per['coordinated_isolation']:.4f}  [{time.time()-t0:.0f}s]",flush=True); return v
SEL=["pv_L_share_of_losses","pv_W_won_rank6","pv_L_fold_to_partner_street","pv_L_fold_to_partner_tocall_bb","pv_W_won_not_best6","pv_nosd_partner_won_after_fold_to_partner",
     "pv_L_fold_best_street","pv_prov_transfer","pv_W_won_L_lost_last_two","pv_n_out_fold_to_pair_best","pv_n_partner_sd","pv_L_fold_cat_margin"]
res={}
res["base_s2"]=report("baseline, seeds (43,50)",run(HF+FAMC,(43,50)))
res["all_s2"]=report("+ provability all, seeds (43,50)",run(HF+FAMC+PV,(43,50)))
res["sel"]=report("+ selected 12 pv feats, seeds (42,49)",run(HF+FAMC+SEL,(42,49)))
res["shl"]=report("+ pv_L_share_of_losses only, seeds (42,49)",run(HF+FAMC+["pv_L_share_of_losses"],(42,49)))
d=np.array([res["all_s2"][p]-res["base_s2"][p] for p in res["base_s2"]]); print(f"seed-swap delta all-base = {np.mean(d):+.4f} (se {d.std(ddof=1)/np.sqrt(len(d)):.4f})")
