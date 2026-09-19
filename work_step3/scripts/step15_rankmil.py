"""Step 15: Step 14 features with MIL aggregates (and stacked hs_* aggregates) replaced by within-phase percentile ranks -> pair model -> submission."""
import pandas as pd, numpy as np, lightgbm as lgb, os, sys, warnings, json; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
W="work_step3/"; TAG=sys.argv[1] if len(sys.argv)>1 else "v14"
dev=pd.read_parquet(W+"dev_pair_features_v9.parquet"); ev=pd.read_parquet(W+f"eval_pair_features_{TAG}.parquet")
m0=pd.read_parquet(W+f"mil_dev_w0_{TAG}.parquet").set_index("pair_id"); m1=pd.read_parquet(W+f"mil_dev_w1_{TAG}.parquet").set_index("pair_id"); me=pd.read_parquet(W+f"mil_eval_{TAG}.parquet").set_index("pair_id")
MC=list(m0.columns)
vals=np.where((dev.window==0).values[:,None],m0.reindex(dev.pair_id_orig).values,m1.reindex(dev.pair_id_orig).values)
for i,c in enumerate(MC): dev[c]=vals[:,i]
for c in MC: ev[c]=me.reindex(ev.pair_id)[c].values
STACK=MC+["hs_mean","hs_max","hs_top3","hs_top5","hs_p90","hs_n95","hs_n99","hs_f95","ev_n10","ev_n50","ev_f10","hs_top10","hs_sum","hs_burst_max","hs_burst_excess"]
STACK=[c for c in STACK if c in dev.columns and c in ev.columns]
# within-phase percentile ranks: dev per window population, eval over all eval pairs
for c in STACK:
    dev[c+"_pr"]=dev.groupby("window")[c].rank(pct=True,method="average").astype("float32"); ev[c+"_pr"]=ev[c].rank(pct=True,method="average").astype("float32")
META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","window","pair_id_orig"}
F_all=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
F_rank=[c for c in F_all if c not in STACK]            # raw stacked aggregates removed, percentile versions kept
y=dev.label.fillna(0).astype(int).values; labm=dev.is_labeled.values; folds=dev.fold.values; W0=(dev.window==0).values
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
def fit_cv(F,yf,w,seeds):
    X=dev[F].astype("float32"); Xe=ev[F].astype("float32"); oof=np.zeros(len(y)); risk=np.zeros(len(ev)); n=0
    for f in range(5):
        tr,te=(folds!=f)&(w>0),folds==f
        for sd in seeds:
            m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],yf[tr],weight=w[tr]),num_boost_round=750); oof[te]+=m.predict(X[te])/len(seeds); risk+=m.predict(Xe); n+=1
    return oof,risk/n
def views(o):
    return f"labelled={ap(y[W0&labm],o[W0&labm]):.4f} pessimistic={ap(y[W0],o[W0]):.4f}"
w1=np.where(labm,np.where(y==1,5.0,1.0),0.1)
o1,_=fit_cv(F_rank,y,w1,[42]); print("stage-1 (rank-MIL feats):",views(o1),flush=True)
pos_oof=o1[labm&(y==1)]; ta,tp=np.quantile(pos_oof,0.05),np.quantile(pos_oof,0.50); unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp)
y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.2,0.1)))
o2,_=fit_cv(F_rank,y2,w2,[42]); print("stage-2:",views(o2),f"pseudo={ps.sum()} amb={amb.sum()}",flush=True)
pos_oof=o2[labm&(y==1)]; ta,tp=np.quantile(pos_oof,0.05),np.quantile(pos_oof,0.25); amb=unl&(o2>=ta)&(o2<tp); ps=unl&(o2>=tp)
y3=np.where(ps,1,y); w3=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.2,0.1)))
o3,risk=fit_cv(F_rank,y3,w3,[42,53,65,79,88]); print("stage-3:",views(o3),f"pseudo={ps.sum()} amb={amb.sum()}",flush=True)
pd.DataFrame({"pair_id":dev.pair_id,"pair_id_orig":dev.pair_id_orig,"window":dev.window,"label":y,"is_labeled":labm,"oof":o3}).to_parquet(W+"dev_oof_v15.parquet",index=False)
pd.DataFrame({"pair_id":ev.pair_id,"risk":risk}).to_parquet(W+"eval_risk_v15.parquet",index=False)
# submission: Step 14 evidence + families (family heads from eval_risk_v14 p_*), risk from this model, rho=0.05 gate
base=pd.read_csv(sys.argv[2] if len(sys.argv)>2 else "deliverables_step14/submission_step14_lto.csv"); fp=pd.read_parquet(W+f"eval_risk_{TAG}.parquet")
m=base.merge(pd.DataFrame({"pair_id":ev.pair_id,"risk_new":risk}),on="pair_id").merge(fp[["pair_id","p_directed_transfer","p_soft_play","p_coordinated_isolation"]],on="pair_id")
r=m.risk_new.round(9); order=pd.DataFrame({"r":r,"sec":m.risk_score}).sort_values(["r","sec"],ascending=[True,False],kind="mergesort"); dup=order.groupby("r").cumcount().reindex(m.index)
m["risk_score"]=np.clip(r+1e-13*dup.values,0,1); assert m.risk_score.is_unique
prior={"directed_transfer":148,"soft_play":132,"coordinated_isolation":92}; mat=np.column_stack([m[f"p_{f}"]/prior[f] for f in prior]); fam=np.array(list(prior))[mat.argmax(1)]
pct=m.risk_score.rank(ascending=False,method="first")/len(m); m["predicted_behavior"]=np.where(pct<=0.05,fam,"none")
out=m[["pair_id","risk_score","predicted_behavior"]+[f"evidence_hand_{i}" for i in range(1,6)]]
ss=pd.read_csv("data/sample_submission.csv")[["pair_id"]].merge(out,on="pair_id",how="left"); assert ss.isna().sum().sum()==0 and len(ss)==112540
os.makedirs("deliverables_step15",exist_ok=True); ss.to_csv("deliverables_step15/submission_step15_rankmil.csv",index=False); print("wrote deliverables_step15/submission_step15_rankmil.csv")
# shift check: eval percentile features are by construction aligned; report raw-vs-rank top-500 overlap with Step 8
r8=pd.read_csv("deliverables_step8/submission_step8.csv")[["pair_id","risk_score"]].merge(ss[["pair_id","risk_score"]],on="pair_id",suffixes=("_8","_15"))
print("top-500 overlap with Step 8:",len(set(r8.nlargest(500,'risk_score_8').pair_id)&set(r8.nlargest(500,'risk_score_15').pair_id)))
