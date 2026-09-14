# Reproducible initial baseline

From the repository root:

```sh
python3 -m pip install -r baseline/requirements.txt
python3 -m unittest discover -s baseline -p 'test_*.py'
python3 baseline/run.py
```

Use `--rebuild` to regenerate feature caches. The pipeline reads the competition files in the root and writes only to `artifacts/baseline/`. This initial version uses hands and seats, not actions or hole cards. It runs on CPU with four LightGBM threads; use column-selected data on a machine with sufficient RAM (the final validation builds a 12-million-row membership index).

Outputs: `submission.csv`, `REPORT.md`, `metrics.json`, `oof_predictions.csv`, `folds.csv`, models, cached features/evidence, and `manifest.json` with data/code hashes and environment versions. Artifacts and competition data are gitignored.

The binary risk classifier trains only on confirmed positives/negatives. A separate classifier trains on the three known positive families. Hyperparameters are fixed at 250 trees, depth 5, 15 leaves and learning rate 0.03; there is no validation-based stopping or threshold search. Entire tables are held out, ensuring players cannot appear on both sides. Unknown development pairs receive predictions from the model holding out their entire table and are used only for a clearly labeled stress diagnostic.

Feature construction uses symmetric seat outcomes, contributions in big blinds, exposure, folding/showdown rates and directional flow imbalance. IDs only index joins, groups and evidence; their numeric values do not enter the model. Evidence retrieval is a fixed heuristic over shared hands, independent of labels and predicted risk. Consequently, this version has no learned evidence stacking and needs no nested evidence-model cross-fitting. Introduce nested cross-fitting before adding a learned evidence model to pair features.

The confirmed-label composite uses the supplied host code in `official_metric.py`, including exact tie handling. The broader stress AP uses sklearn's threshold-grouped definition: unlabeled development pairs lack original pair IDs and have unknown truth. Stress behavior/composite is deliberately omitted. Stress AP and labeled composite are not leaderboard estimates.

Validation reads the exported CSV back and checks schema through the host scorer, complete pair coverage, finite valid scores, unique evidence per pair, evaluation phase, both-player membership, and exact shared-hand exposure. It does not upload to Kaggle.

The host scorer is copied from the supplied `slash-poker-competition-metric.ipynb`, attributed to https://www.kaggle.com/code/florianderoofr/slash-poker-competition-metric.
