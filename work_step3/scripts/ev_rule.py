import polars as pl, numpy as np
from scipy.stats import spearmanr
d="data/"; W="work_step3/"
lab=pl.read_csv(d+"development_labels.csv"); ev=pl.read_csv(d+"development_evidence.csv")
pos=lab.filter(pl.col("label")==1); ids=set(pos["pair_id"])
cols=["pair_id","hand_id","hand_idx","pot_bb","transfer_any","surp_pair_max","surp_pair_sum","both_vpip_a","one_folded","fold_to_partner","both_showdown","outsider_fold_to_pair","dump_better_hand","fold_better_to_partner","surp_vs_partner","last_street","players_at_showdown","partner_won_other_lost"]
df=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(cols).collect() for i in range(4)])
df=df.join(ev,on=["pair_id","hand_id"],how="left").with_columns(pl.col("evidence_rank").is_not_null().alias("is_ev")).join(pos.select(["pair_id","behavior_family"]),on="pair_id").sort(["pair_id","hand_idx"])
e=df.filter(pl.col("is_ev")).with_columns(pl.col("hand_idx").rank().over("pair_id").alias("chrono"))
# 1. escalation: within labelled evidence, chronological order vs size
for c in ["transfer_any","pot_bb","surp_pair_max","surp_pair_sum","surp_vs_partner"]:
    rs=[spearmanr(g["chrono"],g[c]).correlation for _,g in e.group_by("pair_id") if g.height>=4 and g[c].n_unique()>1]
    print(f"within-pair spearman(chrono, {c:16s}) = {np.nanmean(rs):+.3f}")
# 2. what is evidence_rank when it is not chronological? rank vs chrono mismatch structure
mm=e.with_columns((pl.col("evidence_rank")-pl.col("chrono")).alias("diff"))
print("rank-chrono diff distribution:",mm["diff"].value_counts().sort("diff").to_dict())
# for mismatched pairs: does rank follow pot / transfer / surprise better than chrono?
bad=set(mm.filter(pl.col("diff")!=0)["pair_id"]); print("pairs with non-chronological rank:",len(bad),"of",e["pair_id"].n_unique())
for c in ["chrono","pot_bb","transfer_any","surp_pair_max","both_showdown","one_folded","last_street","players_at_showdown"]:
    rs=[spearmanr(g["evidence_rank"],g[c]).correlation for _,g in e.filter(pl.col("pair_id").is_in(list(bad))).group_by("pair_id") if g.height>=4 and g[c].n_unique()>1]
    print(f"  mismatched pairs: spearman(rank,{c:18s}) = {np.nanmean(rs):+.3f}")
# 3. in mismatched pairs, which hand is out of order: the one with rank 5? print a few examples
ex=mm.filter(pl.col("pair_id").is_in(list(bad))).select(["pair_id","behavior_family","evidence_rank","chrono","hand_idx","pot_bb","transfer_any","surp_pair_max","both_showdown","one_folded","last_street"]).sort(["pair_id","evidence_rank"])
print(ex.head(30))
