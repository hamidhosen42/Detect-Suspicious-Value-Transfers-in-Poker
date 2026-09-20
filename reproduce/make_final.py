"""Step B of the final assembly: build the selected submission (final_J) from

  work/final_bag20_step8.parquet, work/final_bag20_step9.parquet   (reproduce/bag20.py)
  work/eval_risk_v18.parquet                                        (family-head probabilities, written by step18_pipeline.py)
  work/submission.csv                                               (Step 18 evidence lists, written by step18_pipeline.py)

risk_score       = rank-average of the two bagged rankers (ties broken by the mean probability, then a 1e-13 epsilon; no two pairs share a score)
predicted_behavior = argmax of the three family heads divided by the family prior, for the top 5 % of pairs by risk; "none" otherwise
evidence_hand_1..5 = Step 18 evidence re-ranker (unchanged)
Usage:  POKER_WORK_DIR=work_step3 POKER_DATA_DIR=data python3 reproduce/make_final.py  ->  final_submission.csv
"""
import os, pandas as pd, numpy as np
W = os.environ.get("POKER_WORK_DIR", "work_step3").rstrip("/") + "/"; D = os.environ.get("POKER_DATA_DIR", "data").rstrip("/") + "/"
ev = pd.read_csv(W + "submission.csv")                      # Step 18 output: pair_id, risk (unused), behaviour (unused), evidence_hand_1..5
b8 = pd.read_parquet(W + "final_bag20_step8.parquet").rename(columns={"risk": "r8"})
b9 = pd.read_parquet(W + "final_bag20_step9.parquet").rename(columns={"risk": "r9"})
fp = pd.read_parquet(W + "eval_risk_v18.parquet")[["pair_id", "p_directed_transfer", "p_soft_play", "p_coordinated_isolation"]]
m = ev.merge(b8, on="pair_id").merge(b9, on="pair_id").merge(fp, on="pair_id"); n = len(m); assert n == 112_540
r = ((m.r8.rank() + m.r9.rank()) / (2 * n)).round(9); sec = (m.r8 + m.r9) / 2
order = pd.DataFrame({"r": r, "sec": sec}).sort_values(["r", "sec"], ascending=[True, False], kind="mergesort"); dup = order.groupby("r").cumcount().reindex(m.index)
risk = np.clip(r.values + 1e-13 * dup.values, 0, 1); assert len(np.unique(risk)) == n, "risk ties"
prior = {"directed_transfer": 148, "soft_play": 132, "coordinated_isolation": 92}      # labelled family counts in development_labels.csv
mat = np.column_stack([m[f"p_{f}"] / prior[f] for f in prior]); fam = np.array(list(prior))[mat.argmax(1)]
pct = pd.Series(risk).rank(ascending=False, method="first").values / n
out = pd.DataFrame({"pair_id": m.pair_id, "risk_score": risk, "predicted_behavior": np.where(pct <= 0.05, fam, "none")})
for i in range(1, 6): out[f"evidence_hand_{i}"] = m[f"evidence_hand_{i}"]
sub = pd.read_csv(D + "sample_submission.csv")[["pair_id"]].merge(out, on="pair_id", how="left"); assert sub.isna().sum().sum() == 0
sub.to_csv("final_submission.csv", index=False); print("wrote final_submission.csv", sub.shape, sub.predicted_behavior.value_counts().to_dict())
