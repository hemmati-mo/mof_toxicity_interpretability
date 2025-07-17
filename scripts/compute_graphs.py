import sys
import os
import argparse
from rdkit import Chem
from torch_geometric.data import Data
import torch
from tqdm import tqdm
import pandas as pd
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Allow import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.featurization import mol_to_graph

def generate_graphs(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    valid = 0
    skipped = 0
    already_exists = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating graphs"):
        graph_path = os.path.join(output_dir, f"{idx}.pt")

        if os.path.exists(graph_path):
            already_exists += 1
            continue

        smiles = row['Canonical SMILES']
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(f"[Warning] Invalid SMILES at row {idx}: {smiles}")
            skipped += 1
            continue

        try:
            graph = mol_to_graph(mol)

            # Validate graph
            if not isinstance(graph, Data):
                raise ValueError("mol_to_graph did not return a PyG Data object.")

            if graph.x is None or graph.edge_index is None or graph.x.size(0) == 0:
                raise ValueError("Graph has empty features or edges.")

            torch.save(graph, graph_path)
            valid += 1

        except Exception as e:
            print(f"[Error] Failed at row {idx}: {e}")
            skipped += 1

    print(f"\n[Done] {valid} graphs saved. {skipped} molecules skipped. {already_exists} already existed.")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate molecular graphs from SMILES")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV with 'Canonical SMILES' column")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save graphs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_graphs(args.csv_path, args.output_dir)
