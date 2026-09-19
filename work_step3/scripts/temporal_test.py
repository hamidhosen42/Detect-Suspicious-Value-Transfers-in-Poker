import polars as pl, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
W="work_step3/"; d="data/"
lab=pd.read_csv(d+"development_labels.csv"); pos=set(lab[lab.label==1].pair_id); neg=set(lab[lab.label==0].pair_id)
print("check ids:",list(pos)[:2])
o=pd.read_parquet(W+"dev_oof_v16.parquet"); o=o[o.window==0].copy(); o["rank"]=o.oof.rank(ascending=False)
hn=set(o[(o.label==0)&(o["rank"]<=600)].pair_id_orig)      # hard negatives / hidden colluders
rnd=set(o[(~o.is_labeled)&(o["rank"]>5000)].sample(3000,random_state=0).pair_id_orig)
ids=pos|neg|hn|rnd
hf=pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v8_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))&(pl.col("hand_idx")<2000)).select(["pair_id","hand_id","hand_idx","surp_pair_sum","surp_pair_min","both_surp2","both_vpip_a","surp_vs_partner"]).collect() for i in range(4)]).sort(["pair_id","hand_idx"])
def feats(g):
    idx=g["hand_idx"].to_numpy().astype(float); s=g["surp_pair_sum"].to_numpy(); n=len(idx)
    if n<10: return None
    hot=(g["both_surp2"].to_numpy()==1)
    k=hot.sum()
    out=dict(n=n,k=int(k))
    if k>=2:
        hi=idx[hot]; span=(hi.max()-hi.min())/(idx.max()-idx.min()+1)
        out["hot_span"]=span                                   # fraction of the pair's window covered by hot hands
        out["hot_std_rel"]=hi.std()/(idx.std()+1e-6)           # spread of hot hands relative to all hands
        # expected span for k random hands among n: use simulation-free approx via ranks
        r=np.flatnonzero(hot); out["hot_rank_span"]=(r.max()-r.min())/(n-1)
        # burstiness: max number of hot hands within any 20% window of the pair's hands
        q=int(max(2,0.2*n)); cs=np.cumsum(hot); best=max(cs[i+q-1]-(cs[i-1] if i>0 else 0) for i in range(0,n-q+1)); out["hot_burst20"]=best/k
        # onset: rank position of first hot hand (relative)
        out["hot_first_rel"]=r.min()/(n-1); out["hot_last_rel"]=r.max()/(n-1)
    else:
        out.update(hot_span=np.nan,hot_std_rel=np.nan,hot_rank_span=np.nan,hot_burst20=np.nan,hot_first_rel=np.nan,hot_last_rel=np.nan)
    # surprise time-profile: correlation of surprise with time, first-half vs second-half means
    h=n//2; out["surp_half_diff"]=s[:h].mean()-s[h:].mean(); out["surp_time_corr"]=np.corrcoef(np.arange(n),s)[0,1] if s.std()>0 else 0
    return out
rows=[]
for (pid,),g in hf.group_by(["pair_id"],maintain_order=True):
    f=feats(g)
    if f is None: continue
    f["pair_id"]=pid; f["grp"]="pos" if pid in pos else ("neg" if pid in neg else ("hardneg" if pid in hn else "rnd")); rows.append(f)
df=pd.DataFrame(rows); print(df.groupby("grp").size())
cols=["k","hot_span","hot_std_rel","hot_rank_span","hot_burst20","hot_first_rel","hot_last_rel","surp_half_diff","surp_time_corr"]
print(df.groupby("grp")[cols].median().round(3).to_string())
# AUC pos vs hardneg (the boundary that matters), and pos vs rnd
for a,b in [("pos","hardneg"),("pos","rnd"),("pos","neg")]:
    x=df[df.grp.isin([a,b])].dropna(subset=["hot_span"]); yy=(x.grp==a).astype(int)
    print(f"{a} vs {b} (n={len(x)}):",{c:round(roc_auc_score(yy,x[c].fillna(x[c].median())),3) for c in cols})
