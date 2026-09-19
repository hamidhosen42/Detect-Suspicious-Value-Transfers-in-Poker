"""Treat the top-ranked unlabelled dev pairs as label noise: exclude (w=0) or pseudo-label them, with the set defined OOF-safely from the Step 16 stage-1 OOF. Three views."""
import pandas as pd, numpy as np, lightgbm as lgb, os, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
W="work_step3/"
dev=pd.read_parquet(W+"dev_pair_features_v8.parquet"); oof=pd.read_parquet(W+"dev_oof_v16.parquet"); assert (oof.pair_id.values==dev.pair_id.values).all()
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; folds=dev.fold.values; W0=(dev.window==0).values; o16=oof.oof.values
ref=pd.read_parquet(W+"hidden150_ref.parquet"); hidden=set(ref[~ref.is_labeled].sort_values("o",ascending=False).pair_id_orig[:150]); hid=dev.pair_id_orig.isin(hidden).values
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=os.cpu_count())
X=dev[F].astype("float32")
def cv(yf,w,rounds=750):
    o=np.zeros(len(y))
    for f in range(5):
        tr,te=(folds!=f)&(w>0),folds==f; o[te]=lgb.train(P,lgb.Dataset(X[tr],yf[tr],weight=w[tr]),num_boost_round=rounds).predict(X[te])
    return o
def views(nm,o):
    m=W0; yh=np.where(hid,1,y); print(f"{nm:44s} labelled={ap(y[m&labm],o[m&labm]):.4f} pessimistic={ap(y[m],o[m]):.4f} hidden150={ap(yh[m],o[m]):.4f}",flush=True)
views("Step 16 (3-stage PU, reference)",o16)
# rank of each unlabelled pair by the Step 16 OOF (per window); 'hidden set' = top-K unlabelled
rk=pd.Series(o16).groupby(dev.window.values).rank(ascending=False).values
pos_oof=o16[labm&(y==1)]
for K,label in [(300,"exclude"),(600,"exclude"),(1000,"exclude"),(300,"pseudo"),(600,"pseudo")]:
    top=(~labm)&(rk<=K)
    if label=="exclude": yf=y.copy(); w=np.where(labm,np.where(y==1,5.0,1.0),np.where(top,0.0,0.1))
    else: yf=np.where(top,1,y); w=np.where(labm,np.where(y==1,5.0,1.0),np.where(top,0.5,0.1))
    views(f"top-{K} unlabelled {label}d (single stage)",cv(yf,w))
# reference single-stage PU 0.1 for fairness
views("single-stage PU0.1 (no pseudo, reference)",cv(y,np.where(labm,np.where(y==1,5.0,1.0),0.1)))
