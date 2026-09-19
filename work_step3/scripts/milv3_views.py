import pandas as pd, numpy as np, lightgbm as lgb, os, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
W="work_step3/"
dev=pd.read_parquet(W+"dev_pair_features_v8.parquet")
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
m0=pd.read_parquet(W+"milv3_dev_w0.parquet"); m1=pd.read_parquet(W+"milv3_dev_w1.parquet")
vals=np.where((dev.window==0).values[:,None],m0.reindex(dev.pair_id_orig).values,m1.reindex(dev.pair_id_orig).values); MC=list(m0.columns)
for i,c in enumerate(MC): dev[c]=vals[:,i]
y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; folds=dev.fold.values; W0=(dev.window==0).values
ref=pd.read_parquet(W+"hidden150_ref.parquet"); hidden=set(ref[~ref.is_labeled].sort_values("o",ascending=False).pair_id_orig[:150]); hid=dev.pair_id_orig.isin(hidden).values
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=os.cpu_count())
w=np.where(labm,np.where(y==1,5.0,1.0),0.1)
def cv(feats):
    X=dev[feats].astype("float32"); o=np.zeros(len(y))
    for f in range(5):
        tr,te=folds!=f,folds==f; o[te]=lgb.train(P,lgb.Dataset(X[tr],y[tr],weight=w[tr]),num_boost_round=750).predict(X[te])
    return o
for nm,fs in [("Step 8 feats",F),("Step 8 + MIL v3 (no prank)",F+MC),("Step 8 + MIL v3 top3/mean/max only",F+["mil_top3","mil_mean","mil_max"])]:
    o=cv(fs); yh=np.where(hid,1,y); m=W0
    print(f"{nm:36s} labelled={ap(y[m&labm],o[m&labm]):.4f} pessimistic={ap(y[m],o[m]):.4f} hidden150={ap(yh[m],o[m]):.4f}",flush=True)
