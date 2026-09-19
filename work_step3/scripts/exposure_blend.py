"""Where do the three risk sources disagree, and does exposure-dependent weighting help? Uses Step 8 CV OOF (v8), Step 9 OOF (v9), Step 16 OOF (=v8 features) on dev; eval from submissions."""
import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
W="work_step3/"
o8=pd.read_parquet(W+"dev_oof_v16.parquet")   # Step 16 pair model == Step 8 features/model (CV OOF)
o9=pd.read_parquet(W+"dev_oof_v9.parquet").set_index("pair_id").reindex(o8.pair_id).reset_index()
d=pd.read_parquet(W+"dev_pair_features_v8.parquet",columns=["pair_id","n_shared"]).set_index("pair_id").reindex(o8.pair_id).reset_index()
assert o9.oof.notna().all() and d.n_shared.notna().all()
y=o8.label.values; lab=o8.is_labeled.values; w0=(o8.window==0).values; ns=d.n_shared.values
ref=pd.read_parquet(W+"hidden150_ref.parquet"); hidden=set(ref[~ref.is_labeled].sort_values("o",ascending=False).pair_id_orig[:150]); hid=o8.pair_id_orig.isin(hidden).values; yh=np.where(hid,1,y)
r=lambda a: pd.Series(a).rank(pct=True).values
def views(nm,o,m=w0): print(f"{nm:34s} labelled={ap(y[m&lab],o[m&lab]):.4f} pessimistic={ap(y[m],o[m]):.4f} hidden150={ap(yh[m],o[m]):.4f}",flush=True)
views("Step 8/16 OOF",o8.oof.values); views("Step 9 OOF",o9.oof.values); views("rank avg 8+9",(r(o8.oof)+r(o9.oof))/2)
for lo,hi in [(38,60),(60,90),(90,400)]:
    m=w0&(ns>=lo)&(ns<hi); print(f"-- exposure [{lo},{hi}) pairs={m.sum()} positives={int(y[m&lab].sum())}"); views("   Step 8",o8.oof.values,m); views("   Step 9",o9.oof.values,m); views("   avg",(r(o8.oof)+r(o9.oof))/2,m)
