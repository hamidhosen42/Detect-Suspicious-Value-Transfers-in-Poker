import pandas as pd, numpy as np, lightgbm as lgb, os, sys, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
W="work_step3/"
ref=pd.read_parquet(W+"hidden150_ref.parquet"); hidden=set(ref[~ref.is_labeled].sort_values("o",ascending=False).pair_id_orig[:150]); hidden300=set(ref[~ref.is_labeled].sort_values("o",ascending=False).pair_id_orig[:300])
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=42,num_threads=os.cpu_count())
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
def run(name,dev,extra_cols=None):
    F=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
    y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; folds=dev.fold.values; W0=(dev.window==0).values; ns=dev.n_shared.values
    hid=dev.pair_id_orig.isin(hidden).values; hid3=dev.pair_id_orig.isin(hidden300).values
    w=np.where(labm,np.where(y==1,5.0,1.0),0.1); X=dev[F].astype("float32"); o=np.zeros(len(y))
    for f in range(5):
        tr,te=folds!=f,folds==f; o[te]=lgb.train(P,lgb.Dataset(X[tr],y[tr],weight=w[tr]),num_boost_round=750).predict(X[te])
    m=W0; yh=np.where(hid,1,y); yh3=np.where(hid3,1,y)
    # hidden-only AP: rank of hidden positives among unlabelled pairs (excluding labelled pairs entirely)
    mu=W0&~labm
    print(f"{name:22s} feats={len(F):3d} labelled={ap(y[m&labm],o[m&labm]):.4f} pessimistic={ap(y[m],o[m]):.4f} hidden150={ap(yh[m],o[m]):.4f} hidden300={ap(yh3[m],o[m]):.4f} | hidden150 among unlabelled only={ap(hid[mu].astype(int),o[mu]):.4f}",flush=True)
    return o
for tag in sys.argv[1:]:
    dev=pd.read_parquet(W+f"dev_pair_features_{tag}.parquet")
    if tag=="v9mil":
        dev=pd.read_parquet(W+"dev_pair_features_v9.parquet"); m0=pd.read_parquet(W+"mil_dev_w0_v11.parquet").set_index("pair_id"); m1=pd.read_parquet(W+"mil_dev_w1_v11.parquet").set_index("pair_id")
        vals=np.where((dev.window==0).values[:,None],m0.reindex(dev.pair_id_orig).values,m1.reindex(dev.pair_id_orig).values)
        for i,c in enumerate(m0.columns): dev[c]=vals[:,i]
    run(tag,dev)
