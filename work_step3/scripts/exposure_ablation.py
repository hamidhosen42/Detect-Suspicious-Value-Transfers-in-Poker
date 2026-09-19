import pandas as pd, numpy as np, lightgbm as lgb, os, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
W="work_step3/"
dev=pd.read_parquet(W+"dev_pair_features_v9.parquet")
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; folds=dev.fold.values; ns=dev.n_shared.values; W0=(dev.window==0).values
print("rows:",len(dev)," rows with n_shared<38:",(ns<38).sum()," positives with n_shared<38:",((ns<38)&(y==1)&labm).sum(),"of",((y==1)&labm).sum())
ref=pd.read_parquet(W+"hidden150_ref.parquet"); hidden=set(ref[~ref.is_labeled].sort_values("o",ascending=False).pair_id_orig[:150]); hid=dev.pair_id_orig.isin(hidden).values
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=os.cpu_count())
X=dev[F].astype("float32")
def cv(w,rounds=750):
    oof=np.zeros(len(y))
    for f in range(5):
        tr=(folds!=f)&(w>0); te=folds==f; m=lgb.train(P,lgb.Dataset(X[tr],y[tr],weight=w[tr]),num_boost_round=rounds); oof[te]=m.predict(X[te])
    return oof
def report(nm,o):
    for tag,m in [("all w0",W0),("w0 & n_shared>=38",W0&(ns>=38))]:
        yh=np.where(hid,1,y)
        print(f"{nm:34s} [{tag:18s}] labelled={ap(y[m&labm],o[m&labm]):.4f} pessimistic={ap(y[m],o[m]):.4f} hidden150={ap(yh[m],o[m]):.4f}",flush=True)
base=np.where(labm,np.where(y==1,5.0,1.0),0.1)
o0=cv(base); report("stage-1 PU0.1 (all rows)",o0)
o1=cv(np.where(ns<38,0.0,base)); report("drop rows n_shared<38",o1)
o2=cv(base*np.clip(ns/60.0,0.15,1.0)); report("weight ∝ min(1,n_shared/60)",o2)
o3=cv(np.where((ns<38)&(y==1),0.0,base)); report("drop only positives n_shared<38",o3)
np.save(W+"exposure_oofs.npy",np.vstack([o0,o1,o2,o3]))
