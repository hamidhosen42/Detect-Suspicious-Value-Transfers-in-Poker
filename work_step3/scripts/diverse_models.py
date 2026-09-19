import pandas as pd, numpy as np, lightgbm as lgb, os, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
W="work_step3/"
dev=pd.read_parquet(W+"dev_pair_features_v8.parquet"); oof=pd.read_parquet(W+"dev_oof_v16.parquet"); assert (oof.pair_id.values==dev.pair_id.values).all()
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; folds=dev.fold.values; W0=(dev.window==0).values
ref=pd.read_parquet(W+"hidden150_ref.parquet"); hidden=set(ref[~ref.is_labeled].sort_values("o",ascending=False).pair_id_orig[:150]); hid=dev.pair_id_orig.isin(hidden).values
def views(nm,o):
    m=W0; yh=np.where(hid,1,y); print(f"{nm:38s} labelled={ap(y[m&labm],o[m&labm]):.4f} pessimistic={ap(y[m],o[m]):.4f} hidden150={ap(yh[m],o[m]):.4f}",flush=True)
r=lambda a: pd.Series(a).rank(pct=True).values
o16=oof.oof.values; views("Step 16 LGBM (reference)",o16)
X=dev[F].fillna(0).astype("float32").values; w=np.where(labm,np.where(y==1,5.0,1.0),0.1)
# 1) ExtraTrees (different inductive bias)
oet=np.zeros(len(y))
for f in range(5):
    tr,te=folds!=f,folds==f; m=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=5,max_features=0.3,n_jobs=-1,random_state=42).fit(X[tr],y[tr],sample_weight=w[tr]); oet[te]=m.predict_proba(X[te])[:,1]
views("ExtraTrees",oet); views("rank avg LGBM+ET",(r(o16)+r(oet))/2); views("rank 0.75 LGBM + 0.25 ET",0.75*r(o16)+0.25*r(oet))
# 2) Logistic on quantile-transformed features
olr=np.zeros(len(y))
for f in range(5):
    tr,te=folds!=f,folds==f; qt=QuantileTransformer(n_quantiles=200,output_distribution="normal",subsample=200000,random_state=0).fit(X[tr]); m=LogisticRegression(C=0.05,max_iter=2000).fit(qt.transform(X[tr]),y[tr],sample_weight=w[tr]); olr[te]=m.predict_proba(qt.transform(X[te]))[:,1]
views("Logistic (quantile)",olr); views("rank avg LGBM+LR",(r(o16)+r(olr))/2); views("rank 0.8 LGBM + 0.2 LR",0.8*r(o16)+0.2*r(olr))
# 3) LGBM with different params (deeper, dart)
od=np.zeros(len(y)); P=dict(objective="binary",boosting="dart",learning_rate=0.05,num_leaves=63,min_data_in_leaf=50,feature_fraction=0.4,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,drop_rate=0.1,verbose=-1,seed=7,num_threads=os.cpu_count())
for f in range(5):
    tr,te=folds!=f,folds==f; od[te]=lgb.train(P,lgb.Dataset(X[tr],y[tr],weight=w[tr]),num_boost_round=500).predict(X[te])
views("LGBM dart",od); views("rank avg LGBM+dart",(r(o16)+r(od))/2); views("rank avg LGBM+ET+dart",(r(o16)+r(oet)+r(od))/3)
np.save(W+"diverse_oofs.npy",np.vstack([oet,olr,od]))
