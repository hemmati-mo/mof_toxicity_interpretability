# MOF Toxicity Interpretability

A cheminformatics and machine-learning workflow for predicting intraperitoneal and oral toxicity of organic linkers of Metal-Organic Frameworks.

- `ip_data_cleaned.csv`: LD50 of intraperitoneal toxicity measurements.
- `oral_data_cleaned.csv`: LD50 oral toxicity measurements.

The framework takes labeled SMILES strings as input, canonicalizes them, featurizes the resulting molecular structures, and predicts the target output.

## Feature Types

Features are computed with RDKit in `utils/featurization.py`:

- RDKit descriptors: physicochemical properties such as mass, polarity,
  hydrogen-bonding terms, ring counts, and related molecular properties.
- Morgan fingerprints: circular substructure fingerprints similar to ECFP.
- MACCS keys: predefined structural keys for common chemical fragments.
- RDKit topological fingerprints: path-based molecular connectivity patterns.
- Topological torsion fingerprints: atom-sequence patterns that encode local
  bonded topology.
- Molecular graphs: atoms are nodes and bonds are edges, with atom/bond features
  suitable for graph neural networks.

These features are useful because toxicity often depends on structural alerts,
lipophilicity, polarity, molecular size, ionization-related patterns, aromatic
systems, halogenation, and reactive functional groups.

## Repository Layout

```text
data/       Cleaned oral and intraperitoneal toxicity CSV files
features/   Precomputed RDKit feature tables in Parquet format
results/    Test metrics for trained model/feature combinations
scripts/    Feature generation, graph generation, splitting, training, evaluation
utils/      Shared featurization code
```

## Main Scripts

- `scripts/compute_features.py`: reads a CSV with `Canonical SMILES`, computes
  descriptors and fingerprints, and writes a Parquet feature table.
- `scripts/compute_graphs.py`: converts SMILES molecules into PyTorch Geometric
  graph files.
- `scripts/split_folds.py`: creates a stratified 10% test set and 10-fold
  cross-validation splits.
- `scripts/train_classical_models.py`: trains Random Forest, SVM, XGBoost, and
  MLP models for classification and regression.
- `scripts/evaluation_classical_models.py`: evaluates saved fold models on held
  out test data and writes `test_metrics.csv` files.

## Example Usage

Compute features:

```bash
python scripts/compute_features.py \
  --csv_path data/oral_data_cleaned.csv \
  --output_path features/oral_data_cleaned.parquet
```

Create folds:

```bash
python scripts/split_folds.py \
  --parquet_path features/oral_data_cleaned.parquet \
  --output_dir folds/oral_data_cleaned
```

Train models:

```bash
python scripts/train_classical_models.py \
  --features_path features/oral_data_cleaned.parquet \
  --output_dir models/oral_data_cleaned
```

## Dependencies

The code expects a Python environment with RDKit, pandas, NumPy, scikit-learn,
PyTorch, PyTorch Geometric, XGBoost, joblib, tqdm, and RAPIDS cuML/CuPy. The
training scripts use CUDA-oriented libraries and call `.cuda()`, so a GPU
environment is likely required for the full workflow.

## Current Data Summary

- Oral dataset: 22,692 molecules.
- Intraperitoneal dataset: 35,118 molecules.
- Result tables cover 2 datasets, 2 tasks, 4 model families, and 5 feature
  families, for 80 test metric files.

## Notes

This repository is strongest as a baseline cheminformatics benchmark for
route-specific molecular toxicity. For MOF toxicity interpretability, the next
step would be to connect predictions back to chemically meaningful fragments or
descriptors, and to add MOF-specific information such as metal identity,
coordination environment, linker release, particle properties, and stability in
biological media.
