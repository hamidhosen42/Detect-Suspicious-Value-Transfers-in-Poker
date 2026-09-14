"""Create the self-contained Kaggle deliverable from reviewed local modules."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
cells=[md('''# Poker: reproducible baseline with fit diagnostics

**Kaggle instructions:** Add the competition data, select CPU, and Run All. No GPU or internet is required when the checked dependencies are present. Download `/kaggle/working/submission.csv` after all validation cells pass.

This is a **seat-based reference baseline**, not a high-score claim. It uses 5 table/player-disjoint folds, fixed LightGBM hyperparameters, official scoring, and an unlearned evidence heuristic. It never trains on evaluation labels or tunes on outer-fold results.

Overfitting/underfitting cannot be guaranteed absent. Train–validation gaps, frozen learning-curve diagnostics, a simpler model comparator, and a table bootstrap help identify concerns. If these results motivate tuning, use inner grouped CV and preserve a fresh outer assessment. The unlabeled-background score is contaminated, not a leaderboard forecast.
'''),code('''from pathlib import Path
import os, sys, importlib.util, json, subprocess, shutil

required_modules = ['numpy', 'pandas', 'pyarrow', 'sklearn', 'lightgbm']
missing = [m for m in required_modules if importlib.util.find_spec(m) is None]
if missing:
    raise RuntimeError(f"Missing dependencies: {missing}. Install these before Run All.")

START = Path.cwd()
KAGGLE = Path('/kaggle/input').exists()
required_files = ['hands.parquet', 'seats.parquet', 'development_labels.csv',
                  'development_evidence.csv', 'evaluation_pairs.csv', 'sample_submission.csv']
if KAGGLE:
    candidates = sorted({p.parent for p in Path('/kaggle/input').rglob('development_labels.csv')})
else:
    candidates = [START]
valid = [p for p in candidates if all((p / f).exists() for f in required_files)]
if len(valid) != 1:
    raise RuntimeError(f"Expected one competition folder, found {valid}. Add the competition data or set candidates explicitly.")
DATA_DIR = valid[0].resolve()
WORK = Path('/kaggle/working') if KAGGLE else START / 'artifacts/kaggle_notebook_validation'
WORK.mkdir(parents=True, exist_ok=True)
CODE_DIR = WORK / 'baseline_module'
CODE_DIR.mkdir(exist_ok=True)
OUT = WORK / 'artifacts/baseline'
os.chdir(WORK)
print('Data:', DATA_DIR, '\\nWorking:', WORK)
'''),md('''## Included source

The next cells write the complete modules locally; no repository download is needed. The official metric comes from the supplied host notebook:
https://www.kaggle.com/code/florianderoofr/slash-poker-competition-metric

IDs are join/group keys only. Fixed features are symmetric seat outcomes/contributions in big blinds, fold/showdown rates, exposure, and flow imbalance. There are no learned evidence features, so this baseline has no evidence-stacking leakage path.
''')]
for name in ['official_metric.py','run.py','test_baseline.py','diagnostics.py']:
    cells.append(code(f'%%writefile baseline_module/{name}\n'+(root/'baseline'/name).read_text()))
cells.extend([
md('''## Run metric/retrieval checks, then full pipeline

Fixed model: 250 trees, learning rate 0.03, depth 5, 15 leaves, L2 regularization 5. No early stopping against the outer fold. The behavior classifier uses only training-fold positive labels; the reported score never uses oracle behavior assignments.
'''),code('''subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(CODE_DIR), '-p', 'test_*.py'], check=True)
subprocess.run([sys.executable, str(CODE_DIR / 'run.py'),
                '--data-dir', str(DATA_DIR), '--output-dir', str(OUT)], check=True)
'''),md('''## Inspect actual out-of-fold results

The confirmed-label composite uses the exact host scorer. Broader background AP treats unknown relationships as negatives and uses threshold-grouped AP because original IDs for unknown development pairs are unavailable. Do not compare its value directly with the official leaderboard.
'''),code('''import pandas as pd
from IPython.display import display, Markdown, FileLink
metrics = json.loads((OUT / 'metrics.json').read_text())
display(pd.DataFrame([metrics['confirmed_labels']]))
display(pd.DataFrame(metrics['folds']))
print('Unlabeled-background diagnostic:', metrics['unlabeled_as_negative_stress'])
display(Markdown((OUT / 'REPORT.md').read_text()))
'''),md('''## Overfitting and underfitting diagnostics

Refit the **same frozen** outer models and inspect their predictions after 50, 125, and 250 trees. These results do not select a replacement model. Training and validation AP/log-loss are reported separately. A regularized logistic regression serves as a simpler comparator.

Large training/validation gaps suggest overfitting or population differences. Poor training and validation performance can indicate weak features or insufficient capacity, but AP depends on prevalence and neither pattern proves a unique cause. Log-loss is descriptive here because risk is trained on curated labels rather than evaluation prevalence.

The confidence interval resamples tables from fixed OOF predictions; it is not a private-leaderboard uncertainty estimate. Repeated tuning against these outer results would overfit the validation set.
'''),code('''sys.path.insert(0, str(CODE_DIR))
from diagnostics import diagnose
fit_rows, diagnostics = diagnose(OUT)
display(pd.DataFrame(diagnostics['summary']))
print('Fixed-OOF table-bootstrap 95% interval:', diagnostics['fixed_oof_pair_ap_table_bootstrap_95_percentile'])
print('Player/table overlap:', diagnostics['table_and_player_overlap'])
for alert in diagnostics['alerts']:
    print('Diagnostic:', alert)
'''),md('''## Export only after all checks pass

The pipeline has checked every pair ID, finite score, allowed behavior, duplicate evidence, both-player membership, evaluation phase, and recomputed exposure. The file contains five shared evaluation hands for every pair.

Fit diagnostics may flag genuine limitations; they do not silently tune the submission or certify that it will generalize. This initial baseline lacks action-context, learned evidence retrieval, and a fourth-family detector.
'''),code('''submission_path = WORK / 'submission.csv'
shutil.copyfile(OUT / 'submission.csv', submission_path)
from run import digest
manifest = json.loads((OUT / 'manifest.json').read_text())
assert digest(submission_path) == manifest['submission_sha256']
print('Ready:', submission_path)
print('Rows:', len(pd.read_csv(submission_path)))
print('SHA256:', manifest['submission_sha256'])
display(FileLink('submission.csv'))
display(FileLink('artifacts/baseline/metrics.json'))
display(FileLink('artifacts/baseline/validation_diagnostics.json'))
display(FileLink('artifacts/baseline/fit_diagnostics.csv'))
''')])
nb=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},
    'kaggle':{'isInternetEnabled':False,'isGpuEnabled':False,
              'dataSources':[{'sourceType':'competition','sourceId':163730}]}})
nbf.validate(nb)
path=root/'kaggle_trustworthy_baseline.ipynb'
nbf.write(nb,path)
print(path)
