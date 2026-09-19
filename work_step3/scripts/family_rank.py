"""Family-aware pair ranking on Step 8 features: (a) per-family one-vs-all rankers -> max/sum combination with the global risk; (b) risk re-weighting by family confidence. Three dev views + eval file."""
import pandas as pd, numpy as np, lightgbm as lgb, os, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
W="work_step3/"
dev=pd.read_parquet(W+"dev_pair_features_v8.parquet"); ev=pd.read_parquet(W+"eval_pair_features_v16.parquet"); oof=pd.read_parquet(W+"dev_oof_v16.parquet")
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; folds=dev.fold.values; W0=(dev.window==0).values; fam=dev.behavior_family.fillna("").values; o16=oof.oof.values
ref=pd.read_parquet(W+"hidden150_ref.parquet"); hidden=set(ref[~ref.is_labeled].sort_values("o",ascending=False).pair_id_orig[:150]); hid=dev.pair_id_orig.isin(hidden).values
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=os.cpu_count())
X=dev[F].astype("float32"); Xe=ev[F].astype("float32")
def views(nm,o):
    m=W0; yh=np.where(hid,1,y); print(f"{nm:40s} labelled={ap(y[m&labm],o[m&labm]):.4f} pessimistic={ap(y[m],o[m]):.4f} hidden150={ap(yh[m],o[m]):.4f}",flush=True)
views("Step 16 global (reference)",o16)
FAMS=["directed_transfer","soft_play","coordinated_isolation"]
# per-family rankers: positives of that family vs everything else (other-family positives as negatives w=1, confirmed neg 1, unlabelled 0.1)
oo={}; ee={}
for f_ in FAMS:
    yf=((y==1)&(fam==f_)).astype(int); w=np.where(labm,np.where(y==1,np.where(fam==f_,5.0,1.0),1.0),0.1)
    o=np.zeros(len(y)); e=np.zeros(len(ev))
    for k in range(5):
        tr,te=folds!=k,folds==k; m=lgb.train(P,lgb.Dataset(X[tr],yf[tr],weight=w[tr]),num_boost_round=600); o[te]=m.predict(X[te]); e+=m.predict(Xe)/5
    oo[f_]=o; ee[f_]=e; print(f"  {f_:22s} own-family AP (labelled, w0) = {ap(yf[W0&labm],o[W0&labm]):.4f}",flush=True)
S=np.column_stack([oo[f] for f in FAMS]); Se=np.column_stack([ee[f] for f in FAMS])
views("family rankers: max",S.max(1)); views("family rankers: sum",S.sum(1)); views("family rankers: 1-prod(1-p)",1-np.prod(1-S,1))
r=lambda a: pd.Series(a).rank(pct=True).values
views("rank avg: global + max-family",(r(o16)+r(S.max(1)))/2); views("rank avg: global + sum-family",(r(o16)+r(S.sum(1)))/2)
views("global * (1-prod)^0.5",o16*np.sqrt(1-np.prod(1-S,1)))
np.save(W+"family_rank_dev.npy",S); np.save(W+"family_rank_eval.npy",Se)
