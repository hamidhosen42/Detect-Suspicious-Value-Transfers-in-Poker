import polars as pl, numpy as np
from sklearn.metrics import roc_auc_score
d="data/"; W="work_step3/"
lab=pl.read_csv(d+"development_labels.csv"); ev=pl.read_csv(d+"development_evidence.csv")
pos=lab.filter(pl.col("label")==1); ids=set(pos["pair_id"])
hands=pl.read_parquet(d+"hands.parquet").sort(["table_id","started_at","hand_id"]).with_columns(pl.int_range(0,pl.len()).over("table_id").alias("hand_idx"))
hands=hands.with_columns((pl.col("started_at").shift(-1).over("table_id")-pl.col("started_at")).dt.total_microseconds().alias("dur_us"),
                         (pl.col("started_at")-pl.col("started_at").shift(1).over("table_id")).dt.total_microseconds().alias("gap_prev_us"))
na=pl.read_parquet(d+"actions.parquet",columns=["hand_id","action_no"]).group_by("hand_id").agg(pl.len().alias("n_actions"))
hands=hands.join(na,on="hand_id",how="left").with_columns(pl.col("n_actions").fill_null(0))
# residual duration after linear fit on n_actions (population)
h=hands.filter(pl.col("dur_us").is_not_null()).to_pandas()
A=np.c_[np.ones(len(h)),h.n_actions.values,h.players_at_showdown.values,(h.board_cards.fillna("").str.len()>0).astype(int)]
coef,*_=np.linalg.lstsq(A,h.dur_us.values,rcond=None); h["dur_res"]=h.dur_us.values-A@coef
print("duration fit: mean dur (s)",h.dur_us.mean()/1e6," resid std (s)",h.dur_res.std()/1e6," corr(dur,n_actions)",np.corrcoef(h.dur_us,h.n_actions)[0,1].round(3))
# fractional part / microsecond pattern
h["us_frac"]=(h.dur_us%1_000_000)
ph=pl.read_parquet(W+"dev_pair_hands.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(["pair_id","hand_id"])
x=ph.join(pl.from_pandas(h[["hand_id","dur_us","gap_prev_us","dur_res","n_actions","us_frac"]]),on="hand_id").join(ev.select(["pair_id","hand_id",pl.lit(True).alias("is_ev")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_ev").fill_null(False))
hf=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(["pair_id","hand_id","surp_pair_max"]).collect() for i in range(4)])
x=x.join(hf,on=["pair_id","hand_id"])
e=x.filter(pl.col("is_ev")); c=x.filter(~pl.col("is_ev")&(pl.col("surp_pair_max")>=3.0)); n=x.filter(~pl.col("is_ev"))
for col in ["dur_us","gap_prev_us","dur_res","n_actions","us_frac"]:
    a=e[col].fill_null(0).to_numpy().astype(float); b=c[col].fill_null(0).to_numpy().astype(float); bb=n[col].fill_null(0).to_numpy().astype(float)
    print(f"{col:12s} AUC ev vs confusers={roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[a,b]):.3f}  ev vs all non-ev={roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(bb))],np.r_[a,bb]):.3f}  ev median={np.median(a):.0f} conf median={np.median(b):.0f}")
# started_at second-of-day / weekday patterns
hh=hands.filter(pl.col("hand_id").is_in(e["hand_id"].to_list()))
print("evidence started_at minute-of-hour dist (top):",hh["started_at"].dt.minute().value_counts().sort("count",descending=True).head(5).to_dict())
print(hands.select(["started_at"]).head(3), hands["started_at"].dt.microsecond().head(5).to_list())
