import pandas as pd, numpy as np, lightgbm as lgb, os, warnings; warnings.filterwarnings("ignore")
W="work_step3/"
def bag(dev_file, ev_file, oof_file, out, seeds):
    dev=pd.read_parquet(W+dev_file); ev=pd.read_parquet(W+ev_file); oof=pd.read_parquet(W+oof_file).set_index("pair_id").reindex(dev.pair_id)
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
    F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
    y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; o=oof.oof.values
    pos_oof=o[labm&(y==1)]; t_amb,t_ps=np.quantile(pos_oof,0.05),np.quantile(pos_oof,0.25)
    unl=~labm; pseudo=unl&(o>=t_ps); amb=unl&(o>=t_amb)&~pseudo
    y2=np.where(pseudo,1,y); w=np.where(labm,np.where(y==1,5.0,1.0),np.where(pseudo,1.0,np.where(amb,0.2,0.1)))
    P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
    X=dev[F].astype("float32"); Xe=ev[F].astype("float32"); risk=np.zeros(len(ev))
    for i,sd in enumerate(seeds):
        risk+=lgb.train({**P,"seed":sd},lgb.Dataset(X,y2,weight=w),num_boost_round=750).predict(Xe)/len(seeds); print(out,"seed",i+1,flush=True)
    pd.DataFrame({"pair_id":ev.pair_id,"risk":risk}).to_parquet(W+out,index=False)
S2=[211,223,229,233,239,241,251,257,263,269]
bag("dev_pair_features_v8.parquet","eval_pair_features_v16.parquet","dev_oof_v16.parquet","eval_risk_v8_seeds2.parquet",S2)
bag("dev_pair_features_v9.parquet","eval_pair_features_v9.parquet","dev_oof_v9.parquet","eval_risk_v9_seeds2.parquet",S2)
print("done")
