import pandas as pd
import os
import argparse
from sklearn.model_selection import StratifiedKFold, train_test_split

def create_splits(parquet_path, output_dir):
    df = pd.read_parquet(parquet_path)

    os.makedirs(output_dir, exist_ok=True)

    # Global 10% test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.10,
        stratify=df["Category"],
        random_state=42
    )

    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # 10-fold CV on 90%
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df["Category"])):
        fold_train = train_val_df.iloc[train_idx]
        fold_val = train_val_df.iloc[val_idx]

        fold_train.to_csv(os.path.join(output_dir, f"fold_{fold}_train.csv"), index=False)
        fold_val.to_csv(os.path.join(output_dir, f"fold_{fold}_val.csv"), index=False)

    print(f"✅ Splits saved to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Create 10-fold CV and test split from parquet")
    parser.add_argument("--parquet_path", type=str, required=True, help="Input parquet file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for folds")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    create_splits(args.parquet_path, args.output_dir)
