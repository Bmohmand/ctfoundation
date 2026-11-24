# Decoder evaluation: scaling + cross-validation

`scaling_cv.py` runs every decoder on ICH and SinoCT with one shared protocol;
`make_report.py` turns the results into slide tables and figures.

```bash
pip install numpy pandas scikit-learn xgboost torch matplotlib
# CSVs come from Git LFS: git lfs pull

python experiments/scaling_cv.py --dataset sino --models LR SVM RF XGB MLP3 MLP5
python experiments/scaling_cv.py --dataset ich  --models LR SVM MLP5 MLP7
python experiments/scaling_cv.py --dataset ich  --models RF XGB --device cuda   # slow on CPU (XGB ≈ a day on 2 cores)
python experiments/make_report.py --dataset ich
python experiments/make_report.py --dataset sino
```

Runs are resumable: rerun the same command after a crash. Each finished (model, training size)
is written to `results/<dataset>/<model>.jsonl` in one flushed write and skipped next time; a size
that was interrupted mid-CV is discarded and redone from scratch. Checkpoints are per training
size, not per fold, so at most one size's work is lost.

## Protocol

- **De-duplication.** SinoCT: 100 rows repeat an earlier URL (rows 5700–5799 again as
  5800–5899) → dropped (6,830 → 6,730). ICH: exact-duplicate embeddings dropped; 32 embeddings
  that appear in both the hemorrhage and control files with conflicting labels are removed
  entirely (16,125 → 16,058).
- **Test set.** One stratified 20% hold-out (`random_state=42`; stratified on `any` for ICH,
  `label` for SinoCT). Never used for training, tuning, early stopping or thresholds.
- **Nested training subsets.** The development pool is ordered once (seeded, stratified so every
  prefix keeps the class balance). Subset r% = the first r% of that order, so the 1% subset is
  inside the 2% subset, which is inside 5%, and so on, identically for every model. Sizes:
  1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100% of the development pool.
  The datasets have no patient ID, only scan URLs, so subsets are nested at the *scan* level and
  patient overlap between train and test cannot be checked.
- **CV.** At every size: 3-, 5- and 10-fold `StratifiedKFold` inside the subset. Scaling is fit
  on training folds only. A scheme is skipped (`NA`) when there are fewer positives than folds.
- **Test evaluation.** Refit on the full subset; threshold = mean of the Youden-J thresholds
  from the 3/5/10-fold out-of-fold predictions (per label for ICH). 95% CIs from 1,000
  stratified bootstrap resamples of the test set.
- **Scores.** AUC/AP always use continuous scores (`predict_proba` or `decision_function`),
  never hard labels.
- **MLPs.** Early stopping (patience 5, max 50 epochs) on a 10% split carved from the
  *training* data. Architectures are the ones already in the repo:
  - ICH MLP (5-layer): 1408→128→64→6, dropout 0.5, lr 1e-3 (`MLP.ipynb`)
  - ICH MLP (7-layer): the Optuna architecture in `model_hyperparameters.json`.
    *Caveat:* that search scored trials on the test split, so the architecture choice is still
    test-informed even though training no longer touches the test set.
  - SinoCT MLP (3-layer): 1408→1536→896→128→1 (`KFoldMLP.ipynb`)
  - SinoCT MLP (5-layer): 1024(+residual)→512→256→128→1 (`NewSinoCTModels.py`)
- **Classical models.** ICH LR/RF/XGB use the tuned values in `ich_results.txt`; ICH linear SVM
  uses C=0.001, chosen by 3-fold CV on the 20% development subset (C=1 barely converges).
  SinoCT models use library defaults as in `NewSinoCTModels.py`; the RBF SVM is scored with
  `decision_function`.
- **ICH metrics** are macro averages over the six labels (`any` + five subtypes).
