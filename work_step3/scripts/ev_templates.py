import polars as pl, numpy as np, collections
d="data/"; W="work_step3/"
lab=pl.read_csv(d+"development_labels.csv"); ev=pl.read_csv(d+"development_evidence.csv"); pos=lab.filter(pl.col("label")==1); ids=set(pos["pair_id"])
hf=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(["pair_id","hand_id","surp_pair_max","loser_is_p1"]).collect() for i in range(4)])
ph=pl.read_parquet(W+"dev_pair_hands.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(["pair_id","hand_id","player_1","player_2"])
x=ph.join(hf,on=["pair_id","hand_id"]).join(ev.select(["pair_id","hand_id",pl.lit(True).alias("is_ev")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_ev").fill_null(False)).join(pos.select(["pair_id","behavior_family"]),on="pair_id")
ac=pl.scan_parquet(W+"action_context.parquet").filter(pl.col("hand_id").is_in(x["hand_id"].unique().to_list())).select(["hand_id","action_no","street_no","player_id","action","to_call","last_aggr"]).collect()
# template: sequence of partner actions labelled by role (W=winner/loser? use L=loser (lower net), O=other partner), with street and whether facing partner (p) or outsider (o) or nothing (-)
a=x.join(ac,on="hand_id").with_columns(
    pl.when(pl.col("player_id")==pl.col("player_1")).then(pl.lit("A")).when(pl.col("player_id")==pl.col("player_2")).then(pl.lit("B")).otherwise(pl.lit("")).alias("who"))
a=a.filter(pl.col("who")!="").with_columns(
    pl.when(pl.col("loser_is_p1")==1).then(pl.when(pl.col("who")=="A").then(pl.lit("L")).otherwise(pl.lit("W"))).otherwise(pl.when(pl.col("who")=="A").then(pl.lit("W")).otherwise(pl.lit("L"))).alias("role"),
    pl.when(pl.col("to_call")==0).then(pl.lit("-")).when((pl.col("last_aggr")==pl.col("player_1"))|(pl.col("last_aggr")==pl.col("player_2"))).then(pl.lit("p")).otherwise(pl.lit("o")).alias("facing"))
a=a.sort(["pair_id","hand_id","action_no"]).with_columns((pl.col("role")+pl.col("street_no").cast(pl.Utf8)+pl.col("facing")+":"+pl.col("action").str.slice(0,2)).alias("tok"))
t=a.group_by(["pair_id","hand_id","is_ev","behavior_family","surp_pair_max"]).agg(pl.col("tok").str.join(" ").alias("template"))
conf=t.filter(~pl.col("is_ev")&(pl.col("surp_pair_max")>=3.0)); e=t.filter(pl.col("is_ev"))
for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
    ef=e.filter(pl.col("behavior_family")==fam); cf=conf.filter(pl.col("behavior_family")==fam)
    ce=collections.Counter(ef["template"]); cc=collections.Counter(cf["template"])
    top=ce.most_common(12); cov=sum(v for _,v in top)/ef.height
    print(f"\n== {fam}: evidence {ef.height} hands, {len(ce)} distinct templates; top-12 cover {cov:.2f} of evidence | confusers {cf.height} hands, {len(cc)} distinct, top-12 cover {sum(v for _,v in cc.most_common(12))/max(1,cf.height):.2f}")
    for tpl,v in top: print(f"   {v:4d} ev  {cc.get(tpl,0):4d} conf   {tpl}")
    # template-level precision: P(ev | template) over evidence+confusers
    allc=ce+cc; prec=sorted([(ce[k]/allc[k],allc[k],k) for k in allc if allc[k]>=8],reverse=True)[:8]
    print("   highest-precision templates (n>=8):",[(round(p,2),n) for p,n,k in prec])
