# Robustness check for stack_fraction: compact subset + seed-noise floor (same folds as the harness).
import os, sys, time, json; os.environ.setdefault("POLARS_MAX_THREADS","3")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl, pandas as pd
from stack_fraction import run_harness, map5_within_pairs, FEATS, D, W, ROOT
os.chdir(ROOT)
lab=pl.read_csv(os.path.join(D,"development_labels.csv")); ev=pl.read_csv(os.path.join(D,"development_evidence.csv"))
pos=lab.filter(pl.col("label")==1); ids=set(pos["pair_id"])
df=pl.concat([pl.scan_parquet(os.path.join(W,f"dev_hand_features_v9_{i}.parquet")).filter(pl.col("pair_id").is_in(list(ids))).collect() for i in range(4)])
df=(df.join(ev.select(["pair_id","hand_id",pl.lit(True).alias("is_ev")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_ev").fill_null(False)).join(pos.select(["pair_id","behavior_family"]),on="pair_id"))
src=open("step9_pipeline.py").read(); ns={}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")],ns); HF=ns["HAND_FEATS"]
tables=sorted(df["table_id"].unique().to_list()); rng=np.random.RandomState(42); TF={t:int(v) for t,v in zip(tables,rng.permutation(len(tables))%5)}
df=df.with_columns(pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold")).sort(["pair_id","hand_idx"])
for f in ["directed_transfer","soft_play","coordinated_isolation"]: df=df.with_columns((pl.col("behavior_family")==f).cast(pl.Float32).alias("fam_"+f))
df=df.join(pl.read_parquet(os.path.join(W,"hyp","stack_fraction.parquet")),on=["pair_id","hand_id"],how="left")
FAM=["fam_directed_transfer","fam_soft_play","fam_coordinated_isolation"]
COMPACT=["transfer_frac_loser","transfer_frac_loser_prank","gain_frac","gain_frac_prank","contrib_frac_min","contrib_frac_min_prank",
         "tocall_frac_vs_partner_max","tocall_frac_vs_partner_max_prank","call_frac_vs_partner_prank","fold_frac_vs_partner_prank",
         "commit_frac_loser_vs_partner_prank","spr_flop_eff","spr_flop_eff_prank","max_bet_frac_winner_prank"]
pdf=df.select(["pair_id","hand_id","pot_bb","is_ev","behavior_family","fold"]).to_pandas()
def ev_(oof,name):
    pdf["hand_score"]=oof; m=map5_within_pairs(pdf); fm={f:round(map5_within_pairs(pdf[pdf.behavior_family==f]),4) for f in ["directed_transfer","soft_play","coordinated_isolation"]}
    per=[round(map5_within_pairs(pdf[pdf.fold==k]),4) for k in range(5)]
    print(f"{name:34s} MAP@5={m:.4f} fam={json.dumps(fm)} folds={per}",flush=True); return m
res={}
for name,FE,seeds in [("baseline seeds(7,11)",HF+FAM,(7,11)),("compact seeds(42,49)",HF+FAM+COMPACT,(42,49)),("compact seeds(7,11)",HF+FAM+COMPACT,(7,11))]:
    t=time.time(); oof,_=run_harness(df,FE,seeds=seeds); res[name]=ev_(oof,name); print(f"   ({time.time()-t:.0f}s)",flush=True)
print(json.dumps(res))
