# Terrain Dynamics

Physics-informed learning for robot-terrain interaction prediction.

## Setup

```bash
git clone https://github.com/neu-autonomy/F1-Vault.git
cd F1-Vault
pip install -r requirements.txt
```

## Quick Start

Train model from scratch:

```bash
python train.py --hdf5_path "path_to_hdf5_file" --device cuda --simulator dphysics --amp --dt 0.1 --batch_size 512 --num_workers 8 --epochs 100 --lr 1e-3
```

Using existing model:

```bash
python train.py --hdf5_path "path_to_hdf5_file"  --device cuda --simulator dphysics --amp --dt 0.1 --batch_size 512 --num_workers 8 --epochs 200 --lr 5e-4  --weights path_to_model
```
