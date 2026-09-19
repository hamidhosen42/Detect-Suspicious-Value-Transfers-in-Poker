"""Step 16 pair model: 10-seed full-data retrain on Step 8 features -> rank-blend with the Step 16 CV risk -> submission with Step 16 evidence/families."""
import pandas as pd, numpy as np, lightgbm as lgb, os, warnings; warnings.filterwarnings("ignore")
W="work_step3/"
dev=pd.read_parquet(W+"dev_pair_features_v8.parquet"); ev=pd.read_parquet(W+"eval_pair_features_v16.parquet"); oof=pd.read_parquet(W+"dev_oof_v16.parquet")
assert (dev.pair_id.values==oof.pair_id.values).all()
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]; assert all(c in ev.columns for c in F)
y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; o=oof.oof.values
pos_oof=o[labm&(y==1)]; t_amb,t_ps=np.quantile(pos_oof,0.05),np.quantile(pos_oof,0.25)
unl=~labm; pseudo=unl&(o>=t_ps); amb=unl&(o>=t_amb)&~pseudo
y2=np.where(pseudo,1,y); w=np.where(labm,np.where(y==1,5.0,1.0),np.where(pseudo,1.0,np.where(amb,0.2,0.1)))
print("pseudo",pseudo.sum(),"amb",amb.sum(),"feats",len(F),flush=True)
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
X=dev[F].astype("float32"); Xe=ev[F].astype("float32"); risk=np.zeros(len(ev))
SEEDS=[42,53,65,79,88,101,113,127,139,151]
for i,sd in enumerate(SEEDS):
    risk+=lgb.train({**P,"seed":sd},lgb.Dataset(X,y2,weight=w),num_boost_round=750).predict(Xe)/len(SEEDS); print("seed",i+1,flush=True)
pd.DataFrame({"pair_id":ev.pair_id,"risk10":risk}).to_parquet(W+"eval_risk_v16_10seeds.parquet",index=False)
s=pd.read_csv("deliverables_step16/submission_step16.csv"); m=s.merge(pd.DataFrame({"pair_id":ev.pair_id,"risk10":risk}),on="pair_id")
from scipy.stats import spearmanr; print("spearman CV vs 10-seed:",round(spearmanr(m.risk_score,m.risk10).correlation,4)," top-500 overlap:",len(set(m.nlargest(500,'risk_score').pair_id)&set(m.nlargest(500,'risk10').pair_id)))
r=((m.risk_score.rank()+m.risk10.rank())/(2*len(m))).round(9); sec=(m.risk_score+m.risk10)/2
order=pd.DataFrame({"r":r,"sec":sec}).sort_values(["r","sec"],ascending=[True,False],kind="mergesort"); dup=order.groupby("r").cumcount().reindex(m.index)
m["risk_new"]=np.clip(r+1e-13*dup.values,0,1); assert m.risk_new.is_unique
fp=pd.read_parquet(W+"eval_risk_v16.parquet"); m=m.merge(fp[["pair_id","p_directed_transfer","p_soft_play","p_coordinated_isolation"]],on="pair_id")
prior={"directed_transfer":148,"soft_play":132,"coordinated_isolation":92}; mat=np.column_stack([m[f"p_{f}"]/prior[f] for f in prior]); fam=np.array(list(prior))[mat.argmax(1)]
pct=m.risk_new.rank(ascending=False,method="first")/len(m); m["pb"]=np.where(pct<=0.05,fam,"none")
out=m[["pair_id","risk_new","pb"]+[f"evidence_hand_{i}" for i in range(1,6)]].rename(columns={"risk_new":"risk_score","pb":"predicted_behavior"})
ss=pd.read_csv("data/sample_submission.csv")[["pair_id"]].merge(out,on="pair_id",how="left"); assert ss.isna().sum().sum()==0 and len(ss)==112540
ss.to_csv("deliverables_step16/submission_step16_bag15.csv",index=False); print("wrote deliverables_step16/submission_step16_bag15.csv",ss.predicted_behavior.value_counts().to_dict())
