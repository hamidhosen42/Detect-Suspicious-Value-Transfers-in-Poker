import polars as pl, numpy as np, itertools
d="data/"; W="work_step3/"
lab=pl.read_csv(d+"development_labels.csv"); ev=pl.read_csv(d+"development_evidence.csv")
pos=lab.filter(pl.col("label")==1); ids=set(pos["pair_id"])
cols=["pair_id","hand_id","hand_idx","pot_bb","transfer_any","both_showdown","one_folded","last_street","players_at_showdown","loser_is_p1","fold_to_partner","p1_folded","p2_folded","p1_won_share","p2_won_share","p1_net_bb","p2_net_bb","p1_seat_no","p2_seat_no","p1_pos","p2_pos","partner_won_other_lost","big_blind"]
df=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(cols).collect() for i in range(4)])
e=df.join(ev,on=["pair_id","hand_id"]).join(pos.select(["pair_id","behavior_family"]),on="pair_id")
e=e.with_columns((pl.col("p1_won_share")>0).cast(pl.Int8).alias("p1_won"),(pl.col("p2_won_share")>0).cast(pl.Int8).alias("p2_won"),
                 ((pl.col("p1_won_share")==0)&(pl.col("p2_won_share")==0)).cast(pl.Int8).alias("neither_won"),
                 (pl.col("p1_net_bb")*pl.col("big_blind")).alias("p1_net"),(pl.col("p2_net_bb")*pl.col("big_blind")).alias("p2_net"),
                 (pl.col("pot_bb")*pl.col("big_blind")).alias("pot"))
keys={"chrono":["hand_idx"],"showdown,chrono":["both_showdown","hand_idx"],"one_folded desc,chrono":["-one_folded","hand_idx"],"loser_is_p1,chrono":["loser_is_p1","hand_idx"],
      "p1_won,chrono":["p1_won","hand_idx"],"neither_won,chrono":["neither_won","hand_idx"],"last_street,chrono":["last_street","hand_idx"],"players_sd,chrono":["players_at_showdown","hand_idx"],
      "pot asc":["pot"],"pot desc":["-pot"],"transfer asc":["transfer_any"],"transfer desc":["-transfer_any"],"p1_net asc":["p1_net"],"p2_net asc":["p2_net"],"abs net diff asc":["absdiff"],"abs net diff desc":["-absdiff"],
      "fold_to_partner desc,chrono":["-fold_to_partner","hand_idx"],"hand_id":["hand_id"]}
e=e.with_columns((pl.col("p1_net")-pl.col("p2_net")).abs().alias("absdiff"))
res={}
for nm,ks in keys.items():
    ok=[]
    for pid,g in e.group_by("pair_id"):
        cols_=[k.lstrip("-") for k in ks]; desc=[k.startswith("-") for k in ks]
        s=g.sort(cols_,descending=desc)["evidence_rank"].to_numpy()
        ok.append((s==np.arange(1,len(s)+1)).all())
    res[nm]=np.mean(ok)
for k,v in sorted(res.items(),key=lambda t:-t[1]): print(f"{k:28s} rank matches in {v:.3f} of pairs")
# per family
for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
    ef=e.filter(pl.col("behavior_family")==fam); out={}
    for nm,ks in keys.items():
        ok=[]
        for pid,g in ef.group_by("pair_id"):
            cols_=[k.lstrip("-") for k in ks]; desc=[k.startswith("-") for k in ks]
            s=g.sort(cols_,descending=desc)["evidence_rank"].to_numpy(); ok.append((s==np.arange(1,len(s)+1)).all())
        out[nm]=np.mean(ok)
    print(fam, sorted(out.items(),key=lambda t:-t[1])[:4])
