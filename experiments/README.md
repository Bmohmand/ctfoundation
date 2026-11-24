# Decoder evaluation on CT-FM embeddings: tuning, data scaling, cross-validation

Three scripts, run in order. Every step is resumable: rerun the same command after a crash.

```bat
:: Windows (researchenv); use plain `python` if the env is activated
set PY=C:\Users\bahra\anaconda3\envs\researchenv\python.exe
%PY% -m pip install "numpy<2.5" pandas pyarrow scipy scikit-learn xgboost matplotlib optuna
git lfs pull

:: 1) tune hyperparameters on the development pool only (test set untouched)
%PY% experiments\tune.py --dataset sino --device cuda
%PY% experiments\tune.py --dataset ich  --device cuda

:: 2) scaling + 3/5/10-fold CV for every model (default: all models, 5 seeds at sizes <= 10%)
%PY% experiments\scaling_cv.py --dataset sino --device cuda
%PY% experiments\scaling_cv.py --dataset ich  --device cuda

:: 3) tables + figures -> results\<dataset>\report\
%PY% experiments\make_report.py --dataset sino
%PY% experiments\make_report.py --dataset ich
```

Useful flags: `--models LR KNN MLP-focal` (subset of models), `--ratios 0.01 0.02`,
`--seeds 0` (single draw), `--n-boot 200` (faster CIs), `tune.py --models MLP --force` (retune one model).

## Models

| Name | What it is |
|---|---|
| `LR` | Logistic regression (linear probe). C re-selected on each training subset |
| `SVM` | ICH: linear SVM; SinoCT: RBF SVM (as in the deck). C re-selected on each training subset |
| `KNN` | Cosine k-NN, distance-weighted; k re-selected inside every fit (3-fold CV) |
| `RF`, `XGB` | Random forest, XGBoost (grids from `NewICHModels.py`) |
| `MLP` | MLP with architecture/lr from a fresh Optuna search on the dev pool (same search space as `MLP_Optimized.ipynb`) |
| `MLP-focal`, `MLP-bce`, `MLP-wbce` | Same MLP, different loss: focal (gamma=2), plain BCE, BCE with `pos_weight` |
| `MLP5`, `MLP7` (ICH) / `MLP3`, `MLP5` (SinoCT) | **Legacy** architectures exactly as in the deck, for continuity |

Any MLP name takes a loss suffix (e.g. `MLP7-focal`). Default loss: weighted BCE for ICH, plain BCE for
SinoCT (balanced classes).

## Protocol

- **De-duplication.** SinoCT: 100 rows repeat an earlier URL (rows 5700-5799 again as 5800-5899) -> dropped
  (6,830 -> 6,730). ICH: exact-duplicate embeddings dropped; 32 embeddings present in both the hemorrhage and
  control files with conflicting labels are removed entirely (16,125 -> 16,058).
- **SinoCT label.** Positive class = `"1,0"` (abnormal), the same mapping as `NewSinoCTModels.py`. An old
  comment in `OIdSinoCTModels/sino_ct_model.ipynb` called `"1,0"` normal; worth confirming once against the
  AIMI label file, since sensitivity/specificity/F1/AP depend on the direction (AUC does not).
- **Test set.** One stratified 20% hold-out (`random_state=42`; stratified on `any` / `abnormal`). Never used
  for training, tuning, early stopping, model selection or thresholds.
- **Nested training subsets.** For each seed the development pool is ordered once (stratified, so each prefix
  keeps the class balance); subset r% = the first r% of that order. 1% is inside 2%, inside 5%, ... inside
  100%, identical for every model. Sizes: 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100% of the dev pool.
- **Repeated draws.** Seeds 0-4 at sizes <= 10% (where the particular scans drawn matter); seed 0 only above
  that. The test set is the same for every seed.
- **Patient grouping.** The CSVs have scan URLs but no patient IDs, so splits are at the scan level and
  patient overlap cannot be checked. If a URL -> patient mapping exists, splits should be grouped by patient.
- **Hyperparameters.** `tune.py` tunes on the development pool only (3-fold randomized search, ROC AUC on
  `any` / `abnormal`; Optuna 50 trials for the MLP, scored on a 15% dev-pool validation split). The old
  tuned values (`ich_results.txt`, `model_hyperparameters.json`) are **not** reused: they were tuned on a
  different split that overlaps 81% of the new test set, and the 7-layer search scored trials on the test set.
  Size-sensitive settings are re-selected per training subset: C for LR/SVM (one-standard-error rule, which
  favours stronger regularization when CV can't tell C values apart) and k for k-NN.
- **CV.** 3-, 5- and 10-fold `StratifiedKFold` inside each subset; scaling fit on training folds only. Each
  fold's threshold metrics use a threshold chosen on the *other* folds' out-of-fold predictions. A scheme is
  skipped (`NA`) when there are fewer positives than folds.
- **Test evaluation.** Refit on the whole subset. Threshold = mean Youden-J threshold from the 3/5/10-fold
  out-of-fold predictions (per label for ICH), same rule as the ASFNR slides; F1 is also reported at the
  F1-maximizing out-of-fold threshold. 95% CIs from 1,000 stratified bootstrap resamples. Model comparisons
  are paired bootstrap delta-AUC vs logistic regression on identical resamples, Holm-adjusted.
- **Scores.** AUC/AP always use continuous scores (`predict_proba` / `decision_function`), never hard labels.
- **MLP training.** Early stopping on a 10% split of the training data. Budget is size-aware (>= 3,000
  gradient steps, no early stop in the first 20% of the budget) because at 1% an epoch is only ~4 steps and
  "50 epochs, patience 5" stopped training after ~25 updates (validation loss 1.137 -> 1.094 with the
  longer budget). At full size this is the original 50 epochs / patience 5.
- **Metrics.** ROC AUC, PR AUC (AP), prevalence-normalized PR skill = (AP - prevalence)/(1 - prevalence),
  accuracy, sensitivity, specificity, precision, F1. ICH metrics are macro averages over the six labels.

## What changed vs. the deck's experiments (numbers will not match exactly)

1. De-duplicated data and a new stratified test split (not the old `train_test_split(random_state=42)`).
2. (SinoCT positive class unchanged: `"1,0"`.)
3. All hyperparameters retuned on the development pool; LR/SVM C and k-NN k re-selected per training size.
4. AUC from continuous scores everywhere (the SinoCT notebooks computed AUC from hard 0/1 predictions).
5. Thresholds chosen on out-of-fold predictions instead of a fixed 0.5.
6. MLPs early-stop on training data instead of training a fixed 50 epochs or stopping on the test set.
7. Linear SVM for ICH uses tuned C (the old C=1 barely converged at 1,408 dimensions).
