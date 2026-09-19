import polars as pl, numpy as np, lightgbm as lgb, pandas as pd, os, warnings; warnings.filterwarnings("ignore")
exec(open("work_step3/scripts/ev_template_s2.py").read().split("print(f\"stage-1 baseline")[0])
out=pl.read_parquet(W+"pos_hand_outsiders.parquet"); OUTP=[c for c in out.columns if c.startswith("out_")]
df=df.join(out,on=["pair_id","hand_id"],how="left").with_columns([pl.col(c).fill_null(-1) for c in OUTP])
for seeds in [(42,45,47),(11,13,17)]:
    r0,_,_=stage2(oofs[0],HF+EXTRA+FAMC,seeds=seeds); r1,_,_=stage2(oofs[0],HF+EXTRA+FAMC+OUTP+TEC,seeds=seeds); r2,_,_=stage2(oofs[0],HF+EXTRA+FAMC+TEC,seeds=seeds); r3,_,_=stage2(oofs[0],HF+EXTRA+FAMC+OUTP,seeds=seeds)
    print(f"seeds {seeds}: Step-9 stage-1 -> pool base={r0:.4f}  +TE={r2:.4f}  +outsider={r3:.4f}  +outsider+TE={r1:.4f}",flush=True)
