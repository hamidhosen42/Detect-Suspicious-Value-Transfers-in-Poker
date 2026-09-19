"""Support check for street_timing: baseline vs structural-subset delta under a given seed pair, per-pair AP saved for bootstrap.
usage: python3 street_timing_check.py <seed1> <seed2>"""
import sys, os, numpy as np, pandas as pd, polars as pl, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from street_timing import ROOT, D, W, H, FEATURES
S1, S2 = int(sys.argv[1]), int(sys.argv[2])
DEV = [c for c in FEATURES if c.startswith("surp") or c.startswith("dev") or c in ("max_surp_street", "last_dev_street", "n_dev_streets", "both_dev_same_street", "loser_dev_street", "loser_surp_end")]
STRUCT = [c for c in FEATURES if c not in DEV]
lab = pl.read_csv(D + "development_labels.csv"); ev = pl.read_csv(D + "development_evidence.csv"); pos = lab.filter(pl.col("label") == 1); ids = pos["pair_id"].to_list()
src = open(f"{ROOT}/step9_pipeline.py").read(); ns = {}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")], ns); HF = ns["HAND_FEATS"]
feats = pl.read_parquet(H + "street_timing.parquet")
df = pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(ids)).collect() for i in range(4)])
df = (df.join(ev.select(["pair_id", "hand_id", pl.lit(True).alias("is_ev")]), on=["pair_id", "hand_id"], how="left").with_columns(pl.col("is_ev").fill_null(False))
        .join(pos.select(["pair_id", "behavior_family"]), on="pair_id").join(feats, on=["pair_id", "hand_id"], how="left"))
tables = sorted(df["table_id"].unique().to_list()); rng = np.random.RandomState(42); TF = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % 5)}
df = df.with_columns(pl.col("table_id").replace_strict(TF, return_dtype=pl.Int8).alias("fold")).sort(["pair_id", "hand_idx"])
FAM = []
for f in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    df = df.with_columns((pl.col("behavior_family") == f).cast(pl.Float32).alias("fam_" + f)); FAM.append("fam_" + f)
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=3)
y = df["is_ev"].to_numpy().astype(int); fold = df["fold"].to_numpy()
pdf = df.select(["pair_id", "hand_id", "pot_bb", "is_ev", "behavior_family"]).to_pandas()

def oof_for(fe):
    X = df.select(fe).to_numpy().astype(np.float32); oof = np.zeros(len(y))
    for f in range(5):
        tr, te = fold != f, fold == f
        for sd in (S1, S2):
            m = lgb.train({**P, "seed": sd}, lgb.Dataset(X[tr], y[tr]), num_boost_round=400); oof[te] += m.predict(X[te]) / 2
    return oof

def per_pair_ap(oof):
    pdf["hand_score"] = oof; out = {}
    d = pdf.sort_values(["pair_id", "hand_score", "pot_bb", "hand_id"], ascending=[True, False, False, True], kind="mergesort")
    for pid, g in d.groupby("pair_id", sort=False):
        rel = g.is_ev.to_numpy(); top = rel[:5]; hits = np.cumsum(top); out[pid] = float(np.sum((hits / np.arange(1, 6)) * top) / min(int(rel.sum()), 5))
    return pd.Series(out)

res = {}
for tag, fe in (("base", HF + FAM), ("struct", HF + FAM + STRUCT)):
    res[tag] = per_pair_ap(oof_for(fe)); print(f"seeds {S1},{S2} {tag}: MAP@5={res[tag].mean():.4f}", flush=True)
fam = pos.to_pandas().set_index("pair_id")["behavior_family"]
r = pd.DataFrame(res); r["family"] = fam.reindex(r.index).values; r["delta"] = r["struct"] - r["base"]
r.to_csv(H + f"street_timing_check_{S1}_{S2}.csv")
rs = np.random.RandomState(0)
for name, sub in [("all", r)] + [(k, g) for k, g in r.groupby("family")]:
    dv = sub["delta"].to_numpy(); boots = np.array([dv[rs.randint(0, len(dv), len(dv))].mean() for _ in range(4000)])
    print(f"seeds {S1},{S2} {name:22s} n={len(dv):3d} base={sub['base'].mean():.4f} struct={sub['struct'].mean():.4f} delta={dv.mean():+.4f} CI95=[{np.percentile(boots, 2.5):+.4f},{np.percentile(boots, 97.5):+.4f}] P(<=0)={np.mean(boots <= 0):.3f}", flush=True)
