import os
import argparse
import joblib
import pandas as pd
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
from cuml.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import cupy as cp

# ==== Torch MLPs ====
class TorchMLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid() if output_dim == 1 else nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

class TorchMLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# ==== Utility Functions ====
def train_torch_classifier(X_train, y_train, input_dim, output_dim, model_path):
    model = TorchMLPClassifier(input_dim, output_dim).to("cuda")
    criterion = nn.BCELoss() if output_dim == 1 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to("cuda")
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32 if output_dim == 1 else torch.long).to("cuda")
    if output_dim == 1:
        y_train_tensor = y_train_tensor.unsqueeze(1)

    model.train()
    for _ in range(20):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), model_path)
    return model

def train_torch_regressor(X_train, y_train, input_dim, model_path):
    model = TorchMLPRegressor(input_dim).to("cuda")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to("cuda")
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to("cuda")

    model.train()
    for _ in range(20):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), model_path)
    return model

def evaluate_classification(y_true, y_pred, y_proba=None):
    average = 'weighted' if len(set(y_true)) > 2 else 'binary'
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class='ovr' if len(set(y_true)) > 2 else 'raise')
        except:
            metrics["roc_auc"] = np.nan
    return metrics

def evaluate_regression(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

# ==== Main Training Script ====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, required=True, help="Path to features parquet file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save trained models")
    args = parser.parse_args()

    df = pd.read_parquet(args.features_path)
    df = df.dropna()

    features = {
        "descriptors": [col for col in df.columns if col.startswith("desc_")],
        "morgan": [col for col in df.columns if col.startswith("morgan_")],
        "torsion": [col for col in df.columns if col.startswith("torsion_")],
        "maccs": [col for col in df.columns if col.startswith("maccs_")],
        "topo": [col for col in df.columns if col.startswith("topo_")]
    }

    models_cls = {
        "RF": RandomForestClassifier,
        "SVM": lambda: SVC(probability=True, class_weight="balanced", kernel="rbf"),
        "XGBoost": lambda: XGBClassifier(tree_method="hist", device="cuda", eval_metric='mlogloss'),
        "MLP": "torch"
    }

    models_reg = {
        "RF": RandomForestRegressor,
        "SVM": SVR,
        "XGBoost": lambda: XGBRegressor(tree_method="hist", device="cuda"),
        "MLP": "torch"
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for task in ["classification", "regression"]:
        for model_name in models_cls.keys():
            for feat_type, feat_cols in features.items():
                print(f"[Training] Task={task}, Model={model_name}, Features={feat_type}")

                X = df[feat_cols].copy().clip(-1e6, 1e6).values.astype(np.float32)

                if task == "classification":
                    label_map = {-1: 0, 0: 1, 1: 2}
                    y = df["Category"].map(label_map).values
                    output_dim = 3
                    stratify_col = y
                else:
                    y = df["Toxicity Value"].values
                    output_dim = 1
                    stratify_col = y > np.median(y)

                results = []
                for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify_col)):

                    out_dir = os.path.join(args.output_dir, task, model_name, feat_type)
                    os.makedirs(out_dir, exist_ok=True)
                    model_path = os.path.join(out_dir, f"fold_{fold}.pkl") if model_name != "MLP" else os.path.join(out_dir, f"fold_{fold}.pt")
                    scaler_path = os.path.join(out_dir, f"fold_{fold}_scaler.pkl")

                    if os.path.exists(model_path):
                        print(f"[Skip] Found {model_path}")
                        continue
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    normalize = model_name in ["SVM", "MLP"]
                    if normalize:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_val = scaler.transform(X_val)            

                    if model_name == "MLP":
                        if task == "classification":
                            model = train_torch_classifier(X_train, y_train, X.shape[1], output_dim, model_path)
                        else:
                            model = train_torch_regressor(X_train, y_train, X.shape[1], model_path)
                    else:
                        model_class = models_cls if task == "classification" else models_reg
                        model = model_class[model_name]() if callable(model_class[model_name]) else model_class[model_name]()
                        model.fit(X_train, y_train)
                        joblib.dump(model, model_path)
                        if normalize:
                            joblib.dump(scaler, scaler_path)

                    # Predict and evaluate
                    if model_name == "MLP":
                        model.eval()
                        with torch.no_grad():
                            logits = model(torch.tensor(X_val, dtype=torch.float32).to("cuda"))
                            preds = logits.cpu().numpy()
                            if output_dim > 1:
                                y_pred = preds.argmax(axis=1)
                                y_proba = preds
                            else:
                                y_pred = (preds > 0.5).astype(int).flatten()
                                y_proba = preds.flatten()
                    else:
                        y_pred = model.predict(X_val)
                        y_proba = None
                        if task == "classification" and hasattr(model, "predict_proba"):
                            try:
                                y_proba = model.predict_proba(X_val)
                                if isinstance(y_proba, cp.ndarray):
                                    y_proba = cp.asnumpy(y_proba)
                                y_pred = y_proba.argmax(axis=1) if y_proba.shape[1] > 1 else (y_proba[:, 0] > 0.5).astype(int)
                            except:
                                pass

                    metrics = evaluate_classification(y_val, y_pred, y_proba) if task == "classification" else evaluate_regression(y_val, y_pred)
                    metrics["fold"] = fold
                    results.append(metrics)

                    # === GPU Memory Cleanup ===
                    del model
                    if normalize:
                        del scaler
                    gc.collect()
                    torch.cuda.empty_cache()

                if results:
                    avg = pd.DataFrame(results).mean(numeric_only=True)
                    avg.to_frame().T.to_csv(os.path.join(out_dir, "avg_metrics.csv"), index=False)

if __name__ == "__main__":
    main()
