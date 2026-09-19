import polars as pl, numpy as np, lightgbm as lgb, pandas as pd, os, warnings; warnings.filterwarnings("ignore")
exec(open("work_step3/scripts/ev_template_s2.py").read().split("print(f\"stage-1 baseline")[0])   # reuse df (with TE cols), pool_features, stage2, HF, EXTRA, FAMC, TEC, oofs
out=pl.read_parquet(W+"pos_hand_outsiders.parquet"); OUTP=[c for c in out.columns if c.startswith("out_")]
df=df.join(out,on=["pair_id","hand_id"],how="left").with_columns([pl.col(c).fill_null(-1) for c in OUTP])
o_out=np.load(W+"out_oofs.npy")   # [baseline, +outsider] stage-1 OOF (same row order? both sorted pair_id,hand_idx -> yes)
r_base,_,_=stage2(oofs[1],HF+EXTRA+FAMC); print(f"stage-2 on TE stage-1 (base pool feats)         MAP@5={r_base:.4f}",flush=True)
r_out,_,_=stage2(o_out[1],HF+EXTRA+FAMC+OUTP); print(f"stage-2 on outsider stage-1 + outsider in pool  MAP@5={r_out:.4f}",flush=True)
r_all,pool,m=stage2(o_out[1],HF+EXTRA+FAMC+OUTP+TEC); print(f"stage-2 outsider stage-1 + outsider + TE pool   MAP@5={r_all:.4f}",flush=True)
imp=sorted(zip(m.feature_importance("gain"),HF+EXTRA+FAMC+OUTP+TEC),reverse=True)[:12]; print("top:",[(f,int(v)) for v,f in imp])
# blend of stage-1 scores (TE + outsider) as pool entry, both feature sets
s1=(pd.Series(oofs[1]).rank().values+pd.Series(o_out[1]).rank().values)/2/len(oofs[1])
r_bl,_,_=stage2(s1,HF+EXTRA+FAMC+OUTP+TEC); print(f"stage-2 on blended stage-1 + outsider + TE pool  MAP@5={r_bl:.4f}",flush=True)
