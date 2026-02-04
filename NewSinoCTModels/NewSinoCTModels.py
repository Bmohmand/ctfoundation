import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import os

class SinoMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(SinoMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

        # Dropout layers with increasing dropout rates
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)

        # Residual Layers
        self.identity_1024 = nn.Linear(input_size, 1024)

    def forward(self, x):
        # First layer with minimal dropout
        res1 = self.identity_1024(x)
        x1 = self.relu(self.bn1(self.fc1(x)))
        x1 = self.dropout1(x1)
        x1 = x1 + res1
        
        # Hidden layers with increasing dropout
        x2 = self.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout2(x2)
        
        x3 = self.relu(self.bn3(self.fc3(x2)))
        x3 = self.dropout3(x3)
        
        x4 = self.relu(self.bn4(self.fc4(x3)))
        x4 = self.dropout4(x4)
        
        # Output layer
        x = self.fc5(x4)
        return x

class PyTorchMLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=None, output_size=1, epochs=50, batch_size=32, lr=1e-3, device=None):
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        if self.input_size is None:
            self.input_size = X.shape[1]
            
        self.model = SinoMLP(self.input_size, self.output_size).to(self.device)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                output = self.model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(X_tensor)
            preds = (torch.sigmoid(output) > 0.5).float()
        return preds.cpu().numpy().flatten()

    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(X_tensor)
            prob = torch.sigmoid(output).cpu().numpy()
        return np.hstack((1 - prob, prob))

class NewSinoCTModels:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

    def load_and_split_data(self, csv_path, target_column='label', test_size=0.2, train_subset_ratio=1.0, random_state=42):
        """
        Loads data from a single CSV, creates a fixed test set, and optionally subsamples the training set.
        
        Args:
            csv_path: Path to the dataset CSV.
            target_column: Name of the target variable.
            test_size: Fraction of data to reserve for the fixed test set.
            train_subset_ratio: Fraction of the *training* portion to use (0.0 < ratio <= 1.0).
                                Use this to simulate smaller training sets (e.g., 10%, 5%) while keeping the test set constant.
            random_state: Seed for reproducibility.
        """
        df = pd.read_csv(csv_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Preprocessing specific to Sino_CT dataset structure
        if 'embedding' in df.columns:
            # Convert string representation of list to actual list, then to numpy array
            X = np.array(df['embedding'].apply(eval).tolist(), dtype=np.float32)
        else:
            X = df.drop(columns=[target_column]).values

        if df[target_column].dtype == object:
             # Map '1,0' to 1 and '0,1' to 0 if necessary
             y = df[target_column].apply(lambda x: 1 if x == '1,0' else 0).values
        else:
             y = df[target_column].values

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
            
        # Scale the features using StandardScaler
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        print(f"Data loaded. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")

    def initialize_models(self):
        """
        Initializes the models (RandomForest, SVM, XGBoost, MLP).
        """
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=20),
            'SVM': SVC(kernel='rbf', probability=True, random_state=20),
            'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=20),
            'MLP': PyTorchMLPWrapper(epochs=50, batch_size=32, lr=1e-3)
        }

    def train_and_evaluate(self):
        """
        Performs 3, 5, and 10-fold CV on the training set, then retrains on the full
        training set to evaluate against the fixed test set.
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_and_split_data() first.")

        fold_options = [3, 5, 10]

        for name, model in self.models.items():
            print(f"\n--- Processing Model: {name} ---")
            self.results[name] = {}

            # 1. K-Fold Cross Validation (3, 5, 10)
            for k in fold_options:
                kf = KFold(n_splits=k, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=kf, scoring='accuracy')
                mean_cv = np.mean(cv_scores)
                self.results[name][f'{k}_fold_cv_mean'] = mean_cv
                print(f"  {k}-Fold CV Accuracy: {mean_cv:.4f}")

            # 2. Retrain on full Training Set and Evaluate on Fixed Test Set
            model.fit(self.X_train, self.y_train)
            test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            test_acc = accuracy_score(self.y_test, test_pred)
            test_prec = precision_score(self.y_test, test_pred, zero_division=0)
            test_rec = recall_score(self.y_test, test_pred, zero_division=0)
            test_f1 = f1_score(self.y_test, test_pred, zero_division=0)
            
            if hasattr(model, "predict_proba"):
                test_prob = model.predict_proba(self.X_test)[:, 1]
                test_roc_auc = roc_auc_score(self.y_test, test_prob)
            else:
                test_prob = None
                test_roc_auc = 0.0
            
            self.results[name]['fixed_test_accuracy'] = test_acc
            self.results[name]['precision'] = test_prec
            self.results[name]['recall'] = test_rec
            self.results[name]['f1_score'] = test_f1
            self.results[name]['roc_auc'] = test_roc_auc
            
            # Store predictions for plotting
            self.results[name]['y_true'] = self.y_test
            self.results[name]['y_pred'] = test_pred
            self.results[name]['y_prob'] = test_prob
            
            print(f"  Fixed Test Set Accuracy: {test_acc:.4f}")
            print(f"  Precision: {test_prec:.4f}")
            print(f"  Recall: {test_rec:.4f}")
            print(f"  F1 Score: {test_f1:.4f}")
            print(f"  ROC AUC: {test_roc_auc:.4f}")

    def save_detailed_plots(self, output_dir="plots"):
        """Generates and saves Confusion Matrix, ROC Curve, and Metrics Bar Plot for each model."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for name, metrics in self.results.items():
            y_true = metrics.get('y_true')
            y_pred = metrics.get('y_pred')
            y_prob = metrics.get('y_prob')
            
            if y_true is None or y_pred is None:
                continue

            # 1. Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
            plt.close()

            # 2. ROC Curve
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                plt.figure(figsize=(10, 7))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics.get("roc_auc", 0):.4f})')
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{name} ROC Curve')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_roc_curve.png"))
                plt.close()

            # 3. Metrics Bar Plot
            metric_names = ['fixed_test_accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            plot_metrics = {k: v for k, v in metrics.items() if k in metric_names}
            
            if plot_metrics:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=list(plot_metrics.keys()), y=list(plot_metrics.values()))
                plt.title(f'{name} Performance Metrics')
                plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_metrics_bar.png"))
                plt.close()

    def run_learning_curve_experiment(self, csv_path, ratios, target_column='label', output_file="results.txt", plot_file="learning_curve.png"):
        """
        Cycles through training set sizes, trains models, saves results to txt, and plots learning curves.
        """
        history = {}

        with open(output_file, "w") as f:
            for r in ratios:
                header = f"\n{'='*30}\nTraining with {r*100}% of data\n{'='*30}\n"
                print(header)
                f.write(header)

                self.load_and_split_data(csv_path, train_subset_ratio=r)
                self.load_and_split_data(csv_path, target_column=target_column, train_subset_ratio=r)
                self.initialize_models()
                self.train_and_evaluate()

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
                        f.write(f"  {k}: {v:.4f}\n")
                    f.write("-" * 20 + "\n")

        plt.figure(figsize=(10, 6))
        for name, data in history.items():
            plt.plot(data['ratios'], data['accuracy'], marker='o', label=name)
        
        plt.title('Model Performance vs Training Set Size')
        plt.xlabel('Training Set Ratio')
        plt.ylabel('Test Set Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_file)
        print(f"\nExperiment finished. Results saved to {output_file}, plot saved to {plot_file}")
        
        # Save detailed plots for the final run
        self.save_detailed_plots()
        print(f"Detailed plots saved to 'plots/' directory.")

if __name__ == "__main__":
    # Example usage
    pipeline = NewSinoCTModels()
    
    # Define ratios to test (e.g., 1%, 10%, 50%, 100% of training data)
    ratios = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    
    # Run the experiment
    pipeline.run_learning_curve_experiment(r"ctfoundation\Sino_CT.csv", ratios, target_column='label')