"""Three dev views for a pair-model OOF file: labelled AP, pessimistic AP, AP with a FIXED hidden-150 set (top-150 unlabelled under the Step-6-feature stage-1 model)."""
import pandas as pd, numpy as np, lightgbm as lgb, os, sys, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
W="work_step3/"; REF=W+"hidden150_ref.parquet"
if not os.path.exists(REF):
    dev=pd.read_parquet(W+"dev_pair_features_v6.parquet")
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
    F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
    y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; folds=dev.fold.values
    w=np.where(labm,np.where(y==1,5.0,1.0),0.1); X=dev[F].astype("float32"); oof=np.zeros(len(y))
    P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=os.cpu_count())
    for f in range(5):
        tr=(folds!=f); te=folds==f; m=lgb.train(P,lgb.Dataset(X[tr],y[tr],weight=w[tr]),num_boost_round=750); oof[te]=m.predict(X[te])
    d=dev[dev.window==0][["pair_id_orig","is_labeled"]].copy(); d["o"]=oof[dev.window.values==0]
    d.to_parquet(REF,index=False)
d=pd.read_parquet(REF); hidden=set(d[~d.is_labeled].sort_values("o",ascending=False).pair_id_orig[:150])
def views(o):
    o=o[o.window==0]; y=o.label.values; yh=np.where(o.pair_id_orig.isin(hidden),1,y)
    return dict(labelled=ap(y[o.is_labeled],o.oof[o.is_labeled]),pessimistic=ap(y,o.oof),hidden150=ap(yh,o.oof))
if __name__=="__main__":
    for f in sys.argv[1:]:
        print(f, {k:round(v,4) for k,v in views(pd.read_parquet(f)).items()})
