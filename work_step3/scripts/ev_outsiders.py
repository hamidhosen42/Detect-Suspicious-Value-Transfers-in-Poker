import polars as pl, numpy as np
from sklearn.metrics import roc_auc_score
d="data/"; W="work_step3/"
lab=pl.read_csv(d+"development_labels.csv"); ev=pl.read_csv(d+"development_evidence.csv"); pos=lab.filter(pl.col("label")==1); ids=set(pos["pair_id"])
hf=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(["pair_id","hand_id","surp_pair_max","both_vpip_a","p1_chen","p2_chen","winner_chen","loser_chen","winner_rk_final","loser_rk_final","winner_cat_final","loser_cat_final"]).collect() for i in range(4)])
ph=pl.read_parquet(W+"dev_pair_hands.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(["pair_id","hand_id","player_1","player_2"])
phs=pl.scan_parquet(W+"player_hands_v9.parquet").select(["hand_id","player_id","chen","vpip_a","pfr","contrib_bb","net_bb","folded","went_to_showdown","won_share","pos","surp_sum","n_surp2","loose"]).collect()
st=pl.read_parquet(W+"player_strength.parquet").select(["hand_id","player_id","v_flop","v_final","cat_final","rk_flop","rk_final"])
o=ph.join(phs,on="hand_id").filter((pl.col("player_id")!=pl.col("player_1"))&(pl.col("player_id")!=pl.col("player_2"))).join(st,on=["hand_id","player_id"])
oa=o.group_by(["pair_id","hand_id"]).agg(
    pl.col("chen").max().alias("out_chen_max"),pl.col("chen").mean().alias("out_chen_mean"),(pl.col("chen")>=8).sum().alias("out_n_strong_pf"),
    pl.col("vpip_a").sum().alias("out_n_vpip"),pl.col("pfr").sum().alias("out_n_pfr"),pl.col("contrib_bb").sum().alias("out_contrib"),pl.col("net_bb").min().alias("out_net_min"),pl.col("net_bb").max().alias("out_net_max"),
    pl.col("won_share").sum().alias("out_won"),pl.col("went_to_showdown").sum().alias("out_n_sd"),(~pl.col("folded")).sum().alias("out_n_notfold"),
    pl.col("cat_final").max().alias("out_cat_max"),pl.col("rk_final").min().alias("out_rk_best"),pl.col("rk_flop").min().alias("out_rk_flop_best"),(pl.col("rk_final")==1).sum().alias("out_has_best"),
    pl.col("surp_sum").sum().alias("out_surp_sum"),pl.col("surp_sum").max().alias("out_surp_max"),pl.col("n_surp2").sum().alias("out_n_surp2"),pl.col("loose").sum().alias("out_loose_sum"),
    pl.col("pos").min().alias("out_pos_min"))
df=hf.join(oa,on=["pair_id","hand_id"],how="left").join(ev.select(["pair_id","hand_id",pl.lit(True).alias("is_ev")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_ev").fill_null(False)).join(pos.select(["pair_id","behavior_family"]),on="pair_id")
e=df.filter(pl.col("is_ev")); c=df.filter(~pl.col("is_ev")&(pl.col("surp_pair_max")>=3.0))
print("evidence",e.height,"confusers",c.height)
cols=[x for x in df.columns if x.startswith("out_")]
for fam in ["ALL","directed_transfer","soft_play","coordinated_isolation"]:
    ef=e if fam=="ALL" else e.filter(pl.col("behavior_family")==fam); cf=c if fam=="ALL" else c.filter(pl.col("behavior_family")==fam)
    res=[]
    for col in cols:
        a=ef[col].fill_null(-1).to_numpy().astype(float); b=cf[col].fill_null(-1).to_numpy().astype(float)
        res.append((round(roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[a,b]),3),col,round(float(np.median(a)),2),round(float(np.median(b)),2)))
    res=sorted(res,key=lambda t:abs(t[0]-0.5),reverse=True)
    print(f"\n{fam}: top outsider features (AUC ev vs confusers, ev median, conf median):",res[:8])
