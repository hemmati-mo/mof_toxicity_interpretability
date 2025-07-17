import sys
import os
import argparse
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.featurization import compute_all_features

def parse_args():
    parser = argparse.ArgumentParser(description="Compute molecular features and save as Parquet.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to input CSV file containing SMILES.")
    parser.add_argument('--output_path', type=str, required=True, help="Output Parquet file path.")
    return parser.parse_args()

def main():
    args = parse_args()

    # === Check if output already exists ===
    if os.path.exists(args.output_path):
        print(f"⚠️ {args.output_path} already exists. Skipping feature computation.")
        return

    # === Load input ===
    df = pd.read_csv(args.csv_path)
    smiles_list = df['Canonical SMILES'].tolist()

    # === Compute features ===
    print(f"Computing features for {len(smiles_list)} molecules using {multiprocessing.cpu_count()} threads...\n")

    features = Parallel(n_jobs=-1, backend="threading")(
        delayed(compute_all_features)(smiles, idx)
        for idx, smiles in enumerate(tqdm(smiles_list, desc="Computing features"))
    )

    # === Merge and save ===
    features_df = pd.DataFrame(features)
    result = pd.concat([df, features_df], axis=1)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    result.to_parquet(args.output_path)

    print("\n✅ Feature computation complete!")
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()
