"""Player-degree / partner-exclusivity features at the pair level (dev OOF-safe: degrees from the stage-1 OOF scores of OTHER pairs), tested on Step 8 features."""
import pandas as pd, numpy as np, lightgbm as lgb, os, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
W="work_step3/"
dev=pd.read_parquet(W+"dev_pair_features_v8.parquet"); ev=pd.read_parquet(W+"eval_pair_features_v16.parquet")
oof=pd.read_parquet(W+"dev_oof_v16.parquet"); assert (oof.pair_id.values==dev.pair_id.values).all()
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; folds=dev.fold.values; W0=(dev.window==0).values
ref=pd.read_parquet(W+"hidden150_ref.parquet"); hidden=set(ref[~ref.is_labeled].sort_values("o",ascending=False).pair_id_orig[:150]); hid=dev.pair_id_orig.isin(hidden).values
def degree_feats(pairs, score, p1, p2):
    """for each pair: best score among OTHER pairs of player_1 (and player_2), count of other pairs above thresholds, this pair's rank among each player's pairs."""
    df=pd.DataFrame({"pid":pairs,"s":score,"p1":p1,"p2":p2})
    long=pd.concat([df[["pid","s","p1"]].rename(columns={"p1":"pl"}),df[["pid","s","p2"]].rename(columns={"p2":"pl"})])
    g=long.groupby("pl")["s"]
    long["max_all"]=g.transform("max"); long["n_pairs"]=g.transform("size"); long["rk"]=g.rank(ascending=False,method="min")
    long["sum_all"]=g.transform("sum"); long["n_hi"]=long.groupby("pl")["s"].transform(lambda s:(s>0.05).sum())
    # best OTHER pair score: if this pair is the player's max, use the second max
    long["max2"]=long.groupby("pl")["s"].transform(lambda s: s.nlargest(2).iloc[-1] if len(s)>1 else 0.0)
    long["best_other"]=np.where(long.rk==1,long.max2,long.max_all); long["n_hi_other"]=long.n_hi-(long.s>0.05).astype(int)
    long["sum_other"]=long.sum_all-long.s
    a=long.groupby("pid").agg(bo_max=("best_other","max"),bo_min=("best_other","min"),nhi_o_sum=("n_hi_other","sum"),nhi_o_max=("n_hi_other","max"),rk_max=("rk","max"),rk_min=("rk","min"),so_max=("sum_other","max"),so_min=("sum_other","min"),npairs_min=("n_pairs","min"))
    a["excl"]=a.rk_max==1   # this pair is the top pair for BOTH players
    return a
# dev: use OOF stage-3 score of Step 16 (per window); eval: Step 16 risk
DF=[]
for w in [0,1]:
    m=(dev.window==w).values; a=degree_feats(dev.pair_id.values[m],oof.oof.values[m],dev.player_1.values[m],dev.player_2.values[m]); DF.append(a)
A=pd.concat(DF).reindex(dev.pair_id); DC=list(A.columns)
for c in DC: dev[c]=A[c].values.astype("float32")
er=pd.read_parquet(W+"eval_risk_v16.parquet").set_index("pair_id").reindex(ev.pair_id)
Ae=degree_feats(ev.pair_id.values,er.risk.values,ev.player_1.values,ev.player_2.values).reindex(ev.pair_id)
for c in DC: ev[c]=Ae[c].values.astype("float32")
# add the pair's own score too (stacking) and rank-relative versions
dev["own"]=oof.oof.values; ev["own"]=er.risk.values
dev["own_minus_bo"]=dev.own-dev.bo_max; ev["own_minus_bo"]=ev.own-ev.bo_max
DC2=DC+["own","own_minus_bo"]
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=os.cpu_count())
w=np.where(labm,np.where(y==1,5.0,1.0),0.1)
def cv(feats,rounds=750):
    X=dev[feats].astype("float32"); o=np.zeros(len(y))
    for f in range(5):
        tr,te=folds!=f,folds==f; o[te]=lgb.train(P,lgb.Dataset(X[tr],y[tr],weight=w[tr]),num_boost_round=rounds).predict(X[te])
    return o
def views(nm,o):
    m=W0; yh=np.where(hid,1,y); print(f"{nm:34s} labelled={ap(y[m&labm],o[m&labm]):.4f} pessimistic={ap(y[m],o[m]):.4f} hidden150={ap(yh[m],o[m]):.4f}",flush=True)
views("Step 16 OOF (reference)",oof.oof.values)
o1=cv(F+DC); views("Step 8 feats + degree feats",o1)
o2=cv(DC2,300); views("stacker: own + degree only",o2)
o3=cv(F+DC2); views("Step 8 feats + degree + own",o3)
np.save(W+"degree_oofs.npy",np.vstack([o1,o2,o3])); dev[["pair_id"]+DC2].to_parquet(W+"degree_dev.parquet",index=False); ev[["pair_id"]+DC2].to_parquet(W+"degree_eval.parquet",index=False)
