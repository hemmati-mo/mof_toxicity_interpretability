import os
import joblib
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
from cuml.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
import cupy as cp
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    average_precision_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, r2_score
)

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

# ==== Metrics ====
def eval_classification(y_true, y_pred, y_proba=None):
    average = 'weighted' if len(set(y_true)) > 2 else 'binary'
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "pr_auc": np.nan
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class='ovr' if len(set(y_true)) > 2 else 'raise')
        except:
            metrics["roc_auc"] = np.nan

        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_proba if y_proba.ndim == 1 or y_proba.shape[1] == 1 else y_proba[:,1])
        except:
            metrics["pr_auc"] = np.nan

    return metrics

def eval_regression(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

# ==== Main Prediction Function ====
def predict_and_save(models_base, results_base, test_data_base):
    featurizer_prefix_map = {
        "morgan": "morgan_",
        "torsion": "torsion_",
        "maccs": "maccs_",
        "topo": "topo_",
        "descriptors": "desc_"
    }

    datasets = [d for d in os.listdir(models_base) if os.path.isdir(os.path.join(models_base, d))]

    for dataset in datasets:
        dataset_dir = os.path.join(models_base, dataset)

        test_file = os.path.join(test_data_base, dataset, "test.csv")
        if not os.path.exists(test_file):
            print(f"⚠️ Test file not found: {test_file}, skipping {dataset}.")
            continue

        df_test = pd.read_csv(test_file)

        tasks = os.listdir(dataset_dir)

        for task in tasks:
            task_dir = os.path.join(dataset_dir, task)
            models = os.listdir(task_dir)

            for model_name in models:
                model_dir = os.path.join(task_dir, model_name)
                featurizers = os.listdir(model_dir)

                for featurizer in featurizers:
                    feat_dir = os.path.join(model_dir, featurizer)

                    # === Check if metrics already exist ===
                    res_dir = os.path.join(results_base, dataset, task, model_name, featurizer)
                    out_file = os.path.join(res_dir, "test_metrics.csv")
                    if os.path.exists(out_file):
                        print(f"✅ Skipping {dataset}/{task}/{model_name}/{featurizer} (already evaluated).")
                        continue

                    # === Only include model files, exclude scalers ===
                    fold_files = [f for f in os.listdir(feat_dir)
                                  if f.startswith("fold_") and (f.endswith(".pkl") or f.endswith(".pt")) and "scaler" not in f]

                    prefix = featurizer_prefix_map.get(featurizer, featurizer+"_")
                    feat_cols = [col for col in df_test.columns if col.startswith(prefix)]

                    if len(feat_cols) == 0:
                        print(f"⚠️ No features found for {featurizer} in {dataset}. Skipping.")
                        continue

                    X_test = df_test[feat_cols].clip(-1e6, 1e6).values.astype(np.float32)

                    if task == "classification":
                        label_map = {-1: 0, 0: 1, 1: 2}
                        y_test = df_test["Category"].map(label_map).values
                        output_dim = 3
                    else:
                        y_test = df_test["Toxicity Value"].values
                        output_dim = 1

                    records = []

                    for fold_file in tqdm(fold_files, desc=f"{dataset}/{task}/{model_name}/{featurizer}"):
                        fold_num = int(fold_file.split('_')[1].split('.')[0])
                        model_path = os.path.join(feat_dir, fold_file)

                        # === Load scaler if exists ===
                        scaler_path = os.path.join(feat_dir, f"fold_{fold_num}_scaler.pkl")
                        if os.path.exists(scaler_path):
                            scaler = joblib.load(scaler_path)
                            X_scaled = scaler.transform(X_test)
                        else:
                            X_scaled = X_test

                        # === Load model ===
                        if model_name == "RF":
                            model = joblib.load(model_path)

                        elif model_name == "SVM":
                            model = joblib.load(model_path)

                        elif model_name == "XGBoost":
                            model = joblib.load(model_path)

                        elif model_name == "MLP":
                            input_dim = X_test.shape[1]
                            if task == "classification":
                                model = TorchMLPClassifier(input_dim, output_dim).cuda()
                            else:
                                model = TorchMLPRegressor(input_dim).cuda()
                            model.load_state_dict(torch.load(model_path))
                            model.eval()

                        # === Predict ===
                        if model_name == "MLP":
                            with torch.no_grad():
                                inputs = torch.tensor(X_scaled, dtype=torch.float32).cuda()
                                outputs = model(inputs).cpu().numpy()
                            if task == "classification":
                                if output_dim > 1:
                                    y_pred = np.argmax(outputs, axis=1)
                                    y_proba = outputs
                                else:
                                    y_pred = (outputs > 0.5).astype(int).flatten()
                                    y_proba = outputs.flatten()
                                metrics = eval_classification(y_test, y_pred, y_proba)
                            else:
                                y_pred = outputs.flatten()
                                metrics = eval_regression(y_test, y_pred)

                        else:
                            if task == "classification":
                                if hasattr(model, "predict_proba"):
                                    y_proba = model.predict_proba(X_scaled)
                                    if isinstance(y_proba, cp.ndarray):
                                        y_proba = cp.asnumpy(y_proba)
                                    y_pred = np.argmax(y_proba, axis=1) if y_proba.shape[1] > 1 else (y_proba[:,0] > 0.5).astype(int)
                                else:
                                    y_proba = None
                                    y_pred = model.predict(X_scaled)
                                metrics = eval_classification(y_test, y_pred, y_proba)
                            else:
                                y_pred = model.predict(X_scaled)
                                metrics = eval_regression(y_test, y_pred)

                        metrics["fold"] = fold_num
                        records.append(metrics)

                    # === Save per-featurizer CSV ===
                    os.makedirs(res_dir, exist_ok=True)
                    df_out = pd.DataFrame(records)
                    df_out.to_csv(out_file, index=False)

    print(f"\n✅ All predictions complete. Results saved in {results_base}")

# ==== Run ====
if __name__ == "__main__":
    models_base = "/content/mof_biocompatibility/models"
    results_base = "/content/mof_biocompatibility/results"
    test_data_base = "/content/mof_biocompatibility/folds"

    predict_and_save(models_base, results_base, test_data_base)
