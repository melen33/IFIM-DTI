# IFIM-DTI

Implementation of IFIM-DTI for drug-target interaction prediction. 

## Project Structure

```text
IFIM-DTI/
├── configs/                 # Training and model configuration
├── dataloader/              # DTI dataset and batching logic
├── datasets/                # Dataset CSV files, split files, cached features
├── models/                  # IFIM-DTI model modules 
│   ├── drug/molformer/      # MolFormer tokenizer/model files
│   └── protein/esm2_model/  # ESM2 model files
│   └── LLM/Qwen/            # LLM model files
├── utils/                   # Shared utilities and project path helpers
├── main.py              # Training and cross-validation entry point
├── preparation.py           # Protein feature and confounder dictionary generation
├── test.py                  # Checkpoint evaluation entry point
├── trainer.py           # Training loop and metric saving
└── requirements.txt
```


## Environment

Python 3.9+ is recommended. Install PyTorch according to your CUDA version first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The model expects local copies of:

- MolFormer under `models/drug/molformer/`
- ESM2 under `models/protein/esm2_model/`
- Qwen cache under `models--Qwen` or another path configured by `configs/model_config.yaml`

## Data Preparation

Each dataset should contain a raw `{dataset}.csv` and an ID-annotated `{dataset}_with_id.csv`. Split folders use files such as:

- `train_fold{n}.csv`, `val_fold{n}.csv`, `test_fold{n}.csv` for five-fold splits
- `source_train_with_id.csv`, `target_train_with_id.csv`, `target_test_with_id.csv` for cluster transfer splits
- `train_with_id.csv`, `val_with_id.csv`, `test_with_id.csv` for single random/cold splits

Protein features and confounder dictionaries are generated automatically if missing:

- `pr_f_1280_2000.pkl`
- `C_1280_2000_8.pkl`

## Training

Run a single fold:

```bash
python main.py --data biosnap --split split_random --fold 1
```

Run all five folds:

```bash
python main.py --data biosnap --split split_random
```

Useful options:

```bash
python main.py \
  --data bindingdb \
  --split split_double_cold \
  --fold 1 \
  --device cuda:0 \
  --train_config configs/train_config.yaml \
  --model_config configs/model_config.yaml
```

Outputs are written to `results/{dataset}/{split}/fold{n}/seed_{seed}/`.

## Testing

Evaluate a trained checkpoint:

```bash
python test.py \
  --data biosnap \
  --split split_random \
  --fold 1 \
  --model_path 
```

Test metrics and predictions are saved under `results/test_results/`.

## Reproducibility

1. Use the same dataset split files under `datasets/{dataset}/{split}/`.
2. Keep `TRAIN.SEED`, batch size, learning rate, and augmentation settings fixed in `configs/train_config.yaml`.
3. Use the same local MolFormer, ESM2, and LLM checkpoints.
4. Run each fold with `--fold 1` through `--fold 5`, or omit `--fold` to run all folds.
5. Report the saved `cv_results_seed{seed}.txt` file together with per-fold checkpoints and metric files.

## Naming Notes

The released configuration uses paper-facing names:

- `DrugEncoder`
- `ProteinEncoder`
- `InteractionDecoder`
- IF-LLM settings: `use_llm`, `llm_model_path`, `lora_*`


