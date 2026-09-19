"""Step 9 stage-3 pair model retrained with 10 seeds on cached features -> eval risk (for blending)."""
import pandas as pd, numpy as np, lightgbm as lgb, os, warnings; warnings.filterwarnings("ignore")
W="work_step3/"
dev=pd.read_parquet(W+"dev_pair_features_v9.parquet"); ev=pd.read_parquet(W+"eval_pair_features_v9.parquet"); oof=pd.read_parquet(W+"dev_oof_v9.parquet")
assert (dev.pair_id.values==oof.pair_id.values).all()
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; o=oof.oof.values
pos_oof=o[labm&(y==1)]; t_amb,t_ps=np.quantile(pos_oof,0.05),np.quantile(pos_oof,0.25)
unl=~labm; pseudo=unl&(o>=t_ps); amb=unl&(o>=t_amb)&~pseudo
y2=np.where(pseudo,1,y); w=np.where(labm,np.where(y==1,5.0,1.0),np.where(pseudo,1.0,np.where(amb,0.2,0.1)))
print("pseudo",pseudo.sum(),"amb",amb.sum(),flush=True)
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
X=dev[F].astype("float32"); Xe=ev[F].astype("float32"); risk=np.zeros(len(ev))
SEEDS=[42,53,65,79,88,101,113,127,139,151]
for i,sd in enumerate(SEEDS):
    m=lgb.train({**P,"seed":sd},lgb.Dataset(X,y2,weight=w),num_boost_round=750); risk+=m.predict(Xe)/len(SEEDS); print("seed",i+1,"done",flush=True)
pd.DataFrame({"pair_id":ev.pair_id,"risk10":risk}).to_parquet(W+"eval_risk_v9_10seeds.parquet",index=False); print("saved")
