import polars as pl, numpy as np
from sklearn.metrics import roc_auc_score
d="data/"; W="work_step3/"
lab=pl.read_csv(d+"development_labels.csv"); ev=pl.read_csv(d+"development_evidence.csv")
pos=lab.filter(pl.col("label")==1); ids=set(pos["pair_id"])
cols=["pair_id","hand_id","hand_idx","p1_pos","p2_pos","p1_seat_no","p2_seat_no","p1_chen","p2_chen","p1_pfr","p2_pfr","p1_vpip_a","p2_vpip_a","p1_net_bb","p2_net_bb","big_blind","surp_pair_max","both_vpip_a","pot_bb","players_at_showdown"]
df=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(cols).collect() for i in range(4)])
seats=pl.read_parquet(d+"seats.parquet",columns=["hand_id","player_id","starting_stack"])
df=(df.join(seats.rename({"player_id":"player_1","starting_stack":"p1_stack"}),on=["hand_id","player_1"],how="left") if "player_1" in df.columns else df)
ph=pl.read_parquet(W+"dev_pair_hands.parquet").select(["pair_id","hand_id","player_1","player_2"])
df=df.join(ph,on=["pair_id","hand_id"]).join(seats.rename({"player_id":"player_1","starting_stack":"p1_stack"}),on=["hand_id","player_1"]).join(seats.rename({"player_id":"player_2","starting_stack":"p2_stack"}),on=["hand_id","player_2"])
hands=pl.read_parquet(d+"hands.parquet",columns=["hand_id","players_dealt"])
df=df.join(hands,on="hand_id").join(ev.select(["pair_id","hand_id",pl.lit(True).alias("is_ev")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_ev").fill_null(False)).join(pos.select(["pair_id","behavior_family"]),on="pair_id")
df=df.with_columns((pl.col("p1_stack")/pl.col("big_blind")).alias("p1_stack_bb"),(pl.col("p2_stack")/pl.col("big_blind")).alias("p2_stack_bb"),
                   ((pl.col("p1_seat_no")-pl.col("p2_seat_no")+6)%6).alias("seat_dist"),
                   pl.min_horizontal(pl.col("p1_pos"),pl.col("p2_pos")).alias("pos_min"),pl.max_horizontal(pl.col("p1_pos"),pl.col("p2_pos")).alias("pos_max"))
df=df.with_columns((pl.col("p1_stack_bb")/pl.col("p2_stack_bb")).alias("stack_ratio"),(pl.col("p1_stack_bb")+pl.col("p2_stack_bb")).alias("stack_sum"),pl.min_horizontal("p1_stack_bb","p2_stack_bb").alias("stack_min"),pl.max_horizontal("p1_stack_bb","p2_stack_bb").alias("stack_max"),
                   (pl.col("p1_stack_bb")-pl.col("p2_stack_bb")).abs().alias("stack_gap"))
# who is the aggressor/raiser; position of raiser vs partner
df=df.with_columns(pl.when(pl.col("p1_pfr")&~pl.col("p2_pfr")).then(pl.col("p1_pos")).when(pl.col("p2_pfr")&~pl.col("p1_pfr")).then(pl.col("p2_pos")).otherwise(-1).alias("raiser_pos"),
                   pl.when(pl.col("p1_pfr")&~pl.col("p2_pfr")).then(pl.col("p2_pos")).when(pl.col("p2_pfr")&~pl.col("p1_pfr")).then(pl.col("p1_pos")).otherwise(-1).alias("partner_of_raiser_pos"))
# collusion-mode confusers: non-evidence hands with high surprise
conf=df.filter((~pl.col("is_ev"))&(pl.col("surp_pair_max")>=3.0))
e=df.filter(pl.col("is_ev"))
print("evidence:",e.height," confusers (non-ev, surp>=3):",conf.height)
for c in ["p1_stack_bb","p2_stack_bb","stack_min","stack_max","stack_gap","stack_ratio","stack_sum","seat_dist","pos_min","pos_max","players_dealt","raiser_pos","partner_of_raiser_pos","pot_bb"]:
    x=pl.concat([e.select(c),conf.select(c)])[c].fill_null(-9).to_numpy().astype(float); y=np.r_[np.ones(e.height),np.zeros(conf.height)]
    print(f"{c:22s} AUC(ev vs confusers)={roc_auc_score(y,x):.3f}   ev median={e[c].median():.2f}  conf median={conf[c].median():.2f}   ev q10/q90={np.quantile(e[c].fill_null(-9).to_numpy(),[.1,.9]).round(1)}  conf q10/q90={np.quantile(conf[c].fill_null(-9).to_numpy(),[.1,.9]).round(1)}")
print("\nseat_dist distribution: evidence", e["seat_dist"].value_counts().sort("seat_dist").to_dict()["count"], " confusers", conf["seat_dist"].value_counts().sort("seat_dist").to_dict()["count"])
print("players_dealt: evidence", e["players_dealt"].value_counts().sort("players_dealt").to_dict(), " confusers", conf["players_dealt"].value_counts().sort("players_dealt").to_dict())
for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
    ef=e.filter(pl.col("behavior_family")==fam); cf=conf.filter(pl.col("behavior_family")==fam)
    print(f"\n{fam}: stack_min ev q={np.quantile(ef['stack_min'],[.1,.5,.9]).round(0)} conf q={np.quantile(cf['stack_min'],[.1,.5,.9]).round(0)} | stack_ratio ev q={np.quantile(ef['stack_ratio'],[.1,.5,.9]).round(2)} conf q={np.quantile(cf['stack_ratio'],[.1,.5,.9]).round(2)} | seat_dist ev {ef['seat_dist'].value_counts().sort('seat_dist')['count'].to_list()} conf {cf['seat_dist'].value_counts().sort('seat_dist')['count'].to_list()}")
