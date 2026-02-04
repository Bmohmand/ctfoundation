import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from scipy.stats import ttest_rel
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import os
import json
import time

# --- 1. MLP Model Definition (from MLP_Optimized.ipynb) ---
class HemorrhageMLP(nn.Module):
    def __init__(self, input_size, output_size, layer_sizes, dropout_rates):
        super(HemorrhageMLP, self).__init__()
        
        layers = []
        prev_size = input_size

        for i, size in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rates[i]))
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class PyTorchMLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=None, output_size=6, epochs=50, batch_size=32, lr=0.000187, device=None):
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Best params from MLP_Optimized.ipynb
        self.layer_sizes = [807, 467, 258, 180, 126, 85, 43]
        self.dropout_rates = [0.192, 0.252, 0.398, 0.441, 0.411, 0.411, 0.175]

    def fit(self, X, y):
        if hasattr(X, 'values'): X = X.values
        if hasattr(y, 'values'): y = y.values
            
        if self.input_size is None:
            self.input_size = X.shape[1]
        
        # Re-initialize model to ensure fresh start
        self.model = HemorrhageMLP(self.input_size, self.output_size, self.layer_sizes, self.dropout_rates).to(self.device)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        # Calculate class weights for BCEWithLogitsLoss (handling imbalance)
        pos_weights = []
        for i in range(y.shape[1]):
            num_pos = np.sum(y[:, i])
            num_neg = len(y) - num_pos
            weight = num_neg / (num_pos + 1e-6)
            pos_weights.append(weight)
        pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                output = self.model(xb)
                loss = criterion(output, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
        return self

    def predict(self, X):
        if hasattr(X, 'values'): X = X.values
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(X_tensor)
            preds = (torch.sigmoid(output) > 0.5).float()
        return preds.cpu().numpy()

    def predict_proba(self, X):
        if hasattr(X, 'values'): X = X.values
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(X_tensor)
            probs = torch.sigmoid(output).cpu().numpy()
        return probs

# --- 2. Multi-Label Wrapper for Sklearn Models ---
class MultiLabelSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    Wraps single-output classifiers (RF, SVM, XGB) to handle multi-label classification
    by training one model per label.
    """
    def __init__(self, model_class, init_params=None, custom_weight_logic=False):
        self.model_class = model_class
        self.init_params = init_params if init_params else {}
        self.custom_weight_logic = custom_weight_logic
        self.models = []

    def fit(self, X, y):
        self.models = []
        n_outputs = y.shape[1]
        
        for i in range(n_outputs):
            params = self.init_params.copy()
            
            # Apply custom class weighting logic only if requested AND not overridden by params
            if self.custom_weight_logic and 'class_weight' not in params:
                positives = np.sum(y[:, i])
                negatives = len(y) - positives
                ratio = negatives / max(positives, 1)
                weight = {0: 1, 1: max(10, int(ratio))}
                params['class_weight'] = weight
            
            model = self.model_class(**params)
            model.fit(X, y[:, i])
            self.models.append(model)
        return self

    def predict(self, X):
        preds = []
        for model in self.models:
            preds.append(model.predict(X))
        return np.column_stack(preds)

    def predict_proba(self, X):
        probs = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)
                # If the model was trained on only one class, predict_proba returns 1 column
                if y_prob.shape[1] > 1:
                    p = y_prob[:, 1]
                else:
                    # If only class 1 was present, prob is 1.0; if only class 0, prob is 0.0
                    p = np.full(X.shape[0], float(model.classes_[0]))
            else:
                p = model.decision_function(X)
            probs.append(p)
        return np.column_stack(probs)

# --- 3. Main Experiment Class ---
class NewICHModels:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_full = None
        self.y_full = None
        self.models = {}
        self.results = {}
        self.cv_scores = {} # Store raw CV scores for statistical testing
        self.target_columns = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

    def load_and_split_data(self, hemo_csv_path, control_csv_path, test_size=0.2, train_subset_ratio=1.0, random_state=42, use_pca=False, pca_variance=0.95):
        """
        Loads ICH data, combines hemo and control, parses embeddings, splits, and optionally applies PCA.
        """
        if self.X_full is None:
            print("Loading data from CSV files...")
            if not os.path.exists(hemo_csv_path):
                raise FileNotFoundError(f"Hemo CSV not found at: {os.path.abspath(hemo_csv_path)}")
            if not os.path.exists(control_csv_path):
                raise FileNotFoundError(f"Control CSV not found at: {os.path.abspath(control_csv_path)}")
                
            df_hemo = pd.read_csv(hemo_csv_path)
            df_control = pd.read_csv(control_csv_path)
            df = pd.concat([df_hemo, df_control], ignore_index=True)
            
            # Parse embeddings
            df['embedding'] = df['embedding'].apply(eval)
            
            self.X_full = np.array(df['embedding'].tolist(), dtype=np.float32)
            self.y_full = df[self.target_columns].values.astype(np.float32)
        
        X, y = self.X_full, self.y_full

        # 1. Create Fixed Test Set
        X_train_full, self.X_test, y_train_full, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 2. Optionally subsample the training set
        if train_subset_ratio < 1.0:
            self.X_train, _, self.y_train, _ = train_test_split(
                X_train_full, y_train_full, train_size=train_subset_ratio, random_state=random_state
            )
        else:
            self.X_train = X_train_full
            self.y_train = y_train_full
            
        # Scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        # 3. Optional PCA
        if use_pca:
            print(f"Applying PCA to retain {pca_variance*100}% variance...")
            pca = PCA(n_components=pca_variance, random_state=random_state)
            self.X_train = pca.fit_transform(self.X_train)
            self.X_test = pca.transform(self.X_test)
            print(f"PCA complete. New feature dimension: {self.X_train.shape[1]}")

        print(f"Data loaded. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")

    def tune_baselines(self, n_iter=10):
        """
        Tunes non-DL models using RandomizedSearchCV on the 'any' label (index 0).
        Returns a dictionary of optimized parameters.
        """
        print("\n--- Tuning Baselines (on 'any' label) ---")
        
        grids = {
            'RandomForest': (RandomForestClassifier(random_state=42, n_jobs=-1), {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }),
            # 'SVM': (LinearSVC(random_state=42, dual=False, max_iter=5000), {
            #     'C': [0.1, 1, 10, 100],
            #     'class_weight': ['balanced', None],
            #     'penalty': ['l1', 'l2']
            #  }),
            'XGBoost': (XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'), {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 1.0],
                'scale_pos_weight': [1, 5, 10]
            }),
            'LogisticRegression': (LogisticRegression(random_state=42, max_iter=1000), {
                'C': [0.1, 1, 10, 100],
                'solver': ['lbfgs', 'liblinear'],
                'class_weight': ['balanced', None]
            })
        }
        
        best_params = {}
        # Tune on 'any' label (index 0) as a proxy for general hemorrhage detection features
        y_tune = self.y_train[:, 0]
        
        for name, (est, grid) in grids.items():
            print(f"Tuning {name}...")
            start_time = time.time()
            search = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
            search.fit(self.X_train, y_tune)
            elapsed_time = time.time() - start_time
            best_params[name] = search.best_params_
            print(f"  Best {name} params: {search.best_params_}")
            print(f"  Time taken: {elapsed_time:.2f} seconds")
            
        return best_params

    def initialize_models(self, tuned_params=None):
        """
        Initializes models. Uses tuned_params if provided, otherwise defaults.
        """
        # Default Params
        rf_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        svm_params = {'C': 1.0, 'class_weight': 'balanced', 'random_state': 42, 'dual': False, 'max_iter': 5000}
        xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1}
        lr_params = {'solver': 'lbfgs', 'class_weight': 'balanced', 'max_iter': 1000, 'random_state': 42}

        # Override with tuned params if available
        if tuned_params:
            if 'RandomForest' in tuned_params: rf_params.update(tuned_params['RandomForest'])
            if 'SVM' in tuned_params: svm_params.update(tuned_params['SVM'])
            if 'XGBoost' in tuned_params: xgb_params.update(tuned_params['XGBoost'])
            if 'LogisticRegression' in tuned_params: lr_params.update(tuned_params['LogisticRegression'])

        # Note: custom_weight_logic for RF is disabled if we are using tuned params that might include class_weight
        rf_custom_logic = True if tuned_params is None else False

        self.models = {
            'RandomForest': MultiLabelSklearnWrapper(RandomForestClassifier, rf_params, custom_weight_logic=rf_custom_logic),
            'SVM': MultiLabelSklearnWrapper(LinearSVC, svm_params),
            'XGBoost': MultiLabelSklearnWrapper(XGBClassifier, xgb_params),
            'LogisticRegression': MultiLabelSklearnWrapper(LogisticRegression, lr_params),
            'MLP': PyTorchMLPWrapper(epochs=50, batch_size=32, lr=0.000187)
        }

    def _calculate_metrics(self, y_true, y_pred, y_prob=None):
        """Calculates average metrics across all labels."""
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        aucs = []

        for i in range(y_true.shape[1]):
            accuracies.append(accuracy_score(y_true[:, i], y_pred[:, i]))
            precisions.append(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
            recalls.append(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
            f1s.append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
            if y_prob is not None:
                try:
                    aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
                except ValueError:
                    aucs.append(0.5)

        return {
            'accuracy': np.mean(accuracies),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1_score': np.mean(f1s),
            'roc_auc': np.mean(aucs) if y_prob is not None else 0.0
        }

    def train_and_evaluate(self):
        """
        Performs K-Fold CV and Fixed Test Set evaluation.
        """
        if self.X_train is None:
            raise ValueError("Data not loaded.")

        fold_options = [10] # Focus on 10-fold for statistical robustness

        for name, model in self.models.items():
            print(f"\n--- Processing Model: {name} ---")
            self.results[name] = {}
            self.cv_scores[name] = [] # Reset scores

            # 1. K-Fold Cross Validation
            for k in fold_options:
                kf = KFold(n_splits=k, shuffle=True, random_state=42)
                fold_accuracies = []
                
                for train_idx, val_idx in kf.split(self.X_train):
                    X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
                    y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
                    
                    model.fit(X_fold_train, y_fold_train)
                    val_pred = model.predict(X_fold_val)
                    
                    # Calculate average accuracy for this fold across all labels
                    fold_accs = [accuracy_score(y_fold_val[:, i], val_pred[:, i]) for i in range(y_fold_val.shape[1])]
                    avg_fold_acc = np.mean(fold_accs)
                    fold_accuracies.append(avg_fold_acc)

                self.cv_scores[name] = fold_accuracies # Store for t-test
                mean_cv = np.mean(fold_accuracies)
                self.results[name][f'{k}_fold_cv_mean'] = mean_cv
                print(f"  {k}-Fold CV Average Accuracy: {mean_cv:.4f}")

            # 2. Retrain on full Training Set and Evaluate on Fixed Test Set
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - start_time
            
            test_pred = model.predict(self.X_test)
            test_prob = model.predict_proba(self.X_test) if hasattr(model, "predict_proba") else None
            
            metrics = self._calculate_metrics(self.y_test, test_pred, test_prob)
            
            self.results[name].update({
                'fixed_test_accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'train_time': train_time
            })
            
            # Store predictions for plotting
            self.results[name]['y_true'] = self.y_test
            self.results[name]['y_pred'] = test_pred
            self.results[name]['y_prob'] = test_prob
            
            print(f"  Fixed Test Set Avg Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Avg ROC AUC: {metrics['roc_auc']:.4f}")

    def perform_statistical_tests(self, baseline_model='MLP'):
        """
        Performs paired t-tests comparing baseline_model against all others using CV scores.
        """
        print(f"\n--- Statistical Significance Tests (Baseline: {baseline_model}) ---")
        if baseline_model not in self.cv_scores:
            print(f"Baseline model {baseline_model} not found in results.")
            return

        baseline_scores = self.cv_scores[baseline_model]
        
        for name, scores in self.cv_scores.items():
            if name == baseline_model:
                continue
            
            if len(scores) != len(baseline_scores):
                print(f"Skipping {name}: mismatched fold counts.")
                continue

            t_stat, p_val = ttest_rel(baseline_scores, scores)
            print(f"  {baseline_model} vs {name}: t-stat={t_stat:.4f}, p-value={p_val:.4f}")
            if p_val < 0.05:
                print(f"    -> Significant difference (p < 0.05)")
            else:
                print(f"    -> No significant difference")

    def save_text_report(self, output_file="experiment_results.txt", tuned_params=None):
        """Saves all results and statistical tests to a text file."""
        with open(output_file, "w") as f:
            f.write("=== ICH Hemorrhage Detection Experiment Results ===\n\n")
            
            if tuned_params:
                f.write("--- Tuned Hyperparameters ---\n")
                for model, params in tuned_params.items():
                    f.write(f"{model}: {params}\n")
                f.write("\n")
            
            f.write("--- Model Performance ---\n")
            for name, metrics in self.results.items():
                f.write(f"Model: {name}\n")
                for k, v in metrics.items():
                    if k not in ['y_true', 'y_pred', 'y_prob']:
                        if isinstance(v, float):
                            f.write(f"  {k}: {v:.4f}\n")
                        else:
                            f.write(f"  {k}: {v}\n")
                f.write("\n")
            
            f.write("--- Statistical Significance (Paired t-test vs MLP) ---\n")
            if 'MLP' in self.cv_scores:
                baseline_scores = self.cv_scores['MLP']
                for name, scores in self.cv_scores.items():
                    if name == 'MLP': continue
                    if len(scores) == len(baseline_scores):
                        t_stat, p_val = ttest_rel(baseline_scores, scores)
                        sig = "Significant" if p_val < 0.05 else "Not Significant"
                        f.write(f"MLP vs {name}: t={t_stat:.4f}, p={p_val:.4f} ({sig})\n")
            f.write("\n")
        print(f"Results saved to {output_file}")

    def save_detailed_plots(self, output_dir="full_plots_ich"):
        """Generates plots. For multi-label, we plot average ROC or per-class ROC."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for name, metrics in self.results.items():
            y_true = metrics.get('y_true')
            y_prob = metrics.get('y_prob')
            
            if y_true is None or y_prob is None:
                continue

            # Plot ROC Curve (Micro-average or per class)
            plt.figure(figsize=(10, 7))
            
            # Plot curve for each class
            for i, col in enumerate(self.target_columns):
                fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                plt.plot(fpr, tpr, lw=1, label=f'{col} (AUC = {auc:.2f})')
                
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{name} ROC Curves per Hemorrhage Type')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_roc_curve.png"))
            plt.close()

            # Metrics Bar Plot
            metric_names = ['fixed_test_accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            plot_metrics = {k: v for k, v in metrics.items() if k in metric_names}
            
            if plot_metrics:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=list(plot_metrics.keys()), y=list(plot_metrics.values()))
                plt.title(f'{name} Average Performance Metrics')
                plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_metrics_bar.png"))
                plt.close()

    def run_learning_curve_experiment(self, hemo_path, control_path, ratios, use_pca=False, tune=False, output_file="ich_learning_results.txt", plot_file="ich_learning_curve.png"):
        """
        Cycles through training set sizes, trains models, saves results to txt, and plots learning curves.
        """
        history = {}

        try:
            with open(output_file, "w") as f:
                for r in ratios:
                    header = f"\n{'='*30}\nTraining with {r*100}% of data\n{'='*30}\n"
                    print(header)
                    f.write(header)

                    self.load_and_split_data(hemo_path, control_path, train_subset_ratio=r, use_pca=use_pca)
                    
                    tuned_params = None
                    if tune:
                        tuned_params = self.tune_baselines()
                        
                    self.initialize_models(tuned_params)
                    self.train_and_evaluate()
                    self.perform_statistical_tests()

                    for name, metrics in self.results.items():
                        if name not in history:
                            history[name] = {'ratios': [], 'accuracy': []}
                        
                        acc = metrics.get('fixed_test_accuracy', 0.0)
                        history[name]['ratios'].append(r)
                        history[name]['accuracy'].append(acc)

                        f.write(f"Model: {name}\n")
                        for k, v in metrics.items():
                            if k in ['y_true', 'y_pred', 'y_prob']:
                                continue
                            if isinstance(v, float):
                                f.write(f"  {k}: {v:.4f}\n")
                            else:
                                f.write(f"  {k}: {v}\n")
                        f.write("-" * 20 + "\n")
        except PermissionError:
            print(f"Permission Error: Could not write to {output_file}. Please ensure the file is closed in other programs (like Excel) and you have write access to the directory.")
            return

        plt.figure(figsize=(10, 6))
        for name, data in history.items():
            plt.plot(data['ratios'], data['accuracy'], marker='o', label=name)
        
        plt.title('Model Performance vs Training Set Size (ICH)')
        plt.xlabel('Training Set Ratio')
        plt.ylabel('Test Set Average Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_file)
        plt.close()
        print(f"\nExperiment finished. Results saved to {output_file}, plot saved to {plot_file}")
        
        # Save detailed plots for the final run
        self.save_detailed_plots()

    def run_experiment(self, hemo_path, control_path, use_pca=False, tune=False, output_file="full_ich_results.txt"):
        """
        Runs the full pipeline: Load -> (Tune) -> Train/Eval -> Stats -> Plot -> Save Text
        """
        self.load_and_split_data(hemo_path, control_path, use_pca=use_pca)
        
        tuned_params = None
        if tune:
            tuned_params = self.tune_baselines()
            
        self.initialize_models(tuned_params)
        self.train_and_evaluate()
        self.perform_statistical_tests()
        self.save_detailed_plots()
        self.save_text_report(output_file, tuned_params)

if __name__ == "__main__":
    pipeline = NewICHModels()
    
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Paths to your CSV files
    hemo_path = "Rsna_ICH_Hemm.csv"
    control_path = "Rsna_ICH_Control.csv"
    hemo_path = os.path.join(parent_dir, "Rsna_ICH_Hemm.csv")
    control_path = os.path.join(parent_dir, "Rsna_ICH_Control.csv")
    
    # Define ratios to test
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    
    # Run with PCA and Tuning enabled for research-grade comparison
    if os.path.exists(hemo_path) and os.path.exists(control_path):
        # pipeline.run_experiment(hemo_path, control_path, use_pca=True, tune=True)
        pipeline.run_learning_curve_experiment(hemo_path, control_path, ratios, use_pca=True, tune=True)
    else:
        print(f"Error: Data files not found at {hemo_path} or {control_path}")
