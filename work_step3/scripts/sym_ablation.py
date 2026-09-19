import pandas as pd, numpy as np, lightgbm as lgb, os, warnings, re; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
W="work_step3/"
dev=pd.read_parquet(W+"dev_pair_features_v9.parquet")
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
p1=[c for c in F if c.startswith("p1_")]; p2=[c.replace("p1_","p2_",1) for c in p1]; assert all(c in F for c in p2)
print("p1/p2 feature pairs:",len(p1), p1[:8])
y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; folds=dev.fold.values; W0=(dev.window==0).values; ns=dev.n_shared.values
ref=pd.read_parquet(W+"hidden150_ref.parquet"); hidden=set(ref[~ref.is_labeled].sort_values("o",ascending=False).pair_id_orig[:150]); hid=dev.pair_id_orig.isin(hidden).values
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=os.cpu_count())
X=dev[F].astype("float32"); Xs=X.copy(); Xs[p1]=X[p2].values; Xs[p2]=X[p1].values   # swapped view
w=np.where(labm,np.where(y==1,5.0,1.0),0.1)
def report(nm,o):
    yh=np.where(hid,1,y)
    for tag,m in [("w0",W0),("w0&ns>=38",W0&(ns>=38))]:
        print(f"{nm:30s} [{tag:9s}] labelled={ap(y[m&labm],o[m&labm]):.4f} pessimistic={ap(y[m],o[m]):.4f} hidden150={ap(yh[m],o[m]):.4f}",flush=True)
oof0=np.zeros(len(y)); oof1=np.zeros(len(y)); oof2=np.zeros(len(y))
for f in range(5):
    tr=folds!=f; te=folds==f
    m0=lgb.train(P,lgb.Dataset(X[tr],y[tr],weight=w[tr]),num_boost_round=750); oof0[te]=m0.predict(X[te])
    # symmetric augmentation: original + swapped rows
    Xa=pd.concat([X[tr],Xs[tr]],ignore_index=True); ya=np.r_[y[tr],y[tr]]; wa=np.r_[w[tr],w[tr]]
    m1=lgb.train(P,lgb.Dataset(Xa,ya,weight=wa),num_boost_round=750); oof1[te]=m1.predict(X[te]); oof2[te]=(m1.predict(X[te])+m1.predict(Xs[te]))/2
report("baseline",oof0); report("sym-augmented",oof1); report("sym-augmented + TTA",oof2)
report("blend baseline+TTA (rank avg)",pd.Series(oof0).rank().values+pd.Series(oof2).rank().values)
np.save(W+"sym_oofs.npy",np.vstack([oof0,oof1,oof2]))
