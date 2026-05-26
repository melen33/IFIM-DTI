import argparse
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pickle
import warnings
from pathlib import Path
from time import time
import pandas as pd
import torch
from omegaconf import OmegaConf
from dataloader.dataloader import DTIDataset, get_dataLoader
from transformers import AutoTokenizer

from models.IFIM import IFIM

from trainer import Trainer
from utils.utils import set_seed, mkdir, load_config_file
from utils.paths import CONFIG_DIR, DATA_DIR, MODEL_DIR, RESULT_DIR
from preparation import generate_esm2_feature, kmeans_for_c
import numpy as np

parser = argparse.ArgumentParser(description="DTI prediction")
parser.add_argument('--data', default='biosnap', type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='cluster', type=str, metavar='S', help="split task",
                    choices=['random', 'cold', 'cluster', 'augmented','split_double_cold','split_drug_cold','split_protein_cold','split_random','cluster_new'])
parser.add_argument('--fold', default=None, type=int,
                    help="which fold to run, None means run all 5 folds")
parser.add_argument('--device', default='cuda:0', type=str,
                    help="training device, e.g. cuda:0 or cpu")
parser.add_argument('--train_config', default=None, type=str,
                    help="path to the training config; if omitted, use configs/{data}/{split}/train_config.yaml")
parser.add_argument('--model_config', default=None, type=str,
                    help="path to the model config; if omitted, use configs/{data}/{split}/model_config.yaml")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
print(f"Running on: {device}", end="\n\n")

def get_fold_paths(data_folder, fold):
    train_path = data_folder / f'train_fold{fold}.csv'
    val_path = data_folder / f'val_fold{fold}.csv'
    test_path = data_folder / f'test_fold{fold}.csv'
    return train_path, val_path, test_path

def resolve_config_path(config_arg, config_name):
    if config_arg is not None:
        return Path(config_arg)

    data_config_dir = CONFIG_DIR / args.data / args.split
    config_path = data_config_dir / config_name
    if config_path.is_file():
        return config_path

    fallback_path = CONFIG_DIR / config_name
    if fallback_path.is_file():
        print(
            f"[Warning] Missing {config_path}; "
            f"fallback to {fallback_path}"
        )
        return fallback_path

    raise FileNotFoundError(
        f"Cannot find {config_name}. Tried {config_path} and {fallback_path}."
    )

def run_one_fold(fold, config, model_configs):
    print(f"\n{'='*25} Running Fold {fold} {'='*25}")

    torch.cuda.empty_cache()
    # print(config)
    set_seed(seed=config.TRAIN.SEED + fold)  # 推荐：不同 fold 不同 seed
    # set_seed(seed=config.TRAIN.SEED)


    dataset_root = DATA_DIR / args.data
    split_dir = dataset_root / args.split
    molformer_path = MODEL_DIR / 'drug' / 'molformer'

    # ===== dataset paths =====
    if args.split == 'cluster':
        train_path = split_dir / 'source_train_with_id.csv'
        val_path = split_dir / 'target_train_with_id.csv'
        test_path = split_dir / 'target_test_with_id.csv'
    elif args.split in [
        'split_double_cold', 'split_drug_cold',
        'split_protein_cold', 'split_random','cluster_new'
    ]:
        train_path, val_path, test_path = get_fold_paths(split_dir, fold)
    else:
        train_path = split_dir / 'train_with_id.csv'
        val_path = split_dir / 'val_with_id.csv'
        test_path = split_dir / 'test_with_id.csv'

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)
    df_test  = pd.read_csv(test_path)

    # ===== feature paths =====
    if args.split in [
        'split_double_cold', 'split_drug_cold',
        'split_protein_cold', 'split_random', 'cluster_new'
    ]:
        protein_path = dataset_root / config.TRAIN.PR_PATH
        c_path = dataset_root / config.TRAIN.C_PATH
    else:
        protein_path = split_dir / config.TRAIN.PR_PATH
        c_path = split_dir / config.TRAIN.C_PATH

    # ===== generate features (once) =====
    if not protein_path.is_file():
        generate_esm2_feature(config, args.data, args.split)

    if not c_path.is_file():
        if args.split in [
            'split_double_cold', 'split_drug_cold',
            'split_protein_cold', 'split_random', 'cluster_new'
        ]:
            kmeans_for_c(config, df_train, dataset_root)
        else:
            kmeans_for_c(config, df_train, split_dir)

    # ===== load confounder =====
    C = pickle.load(open(c_path, 'rb'))

    c = torch.from_numpy(C['cluster_centers']).to(device)\
             .permute(1, 0).float()
    p_ci = C['prior'].to(device).float()
    aa_dict = C['aa'].to(device).float()

     # ===== model =====
    model = IFIM(model_configs=model_configs, c=c, p_ci=p_ci).to(device)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.TRAIN.LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY
    )

    # ===== datasets =====
    pr_f = pickle.load(open(protein_path, 'rb'))

    train_dataset = DTIDataset(df_train.index.values, df_train, pr_f)
    val_dataset   = DTIDataset(df_val.index.values, df_val, pr_f)
    test_dataset  = DTIDataset(df_test.index.values, df_test, pr_f)

    drug_tokenizer = AutoTokenizer.from_pretrained(
        molformer_path, trust_remote_code=True
    )

    bz  = config.TRAIN.BATCH_SIZE
    MLM = config.TRAIN.MLM

    train_loader = get_dataLoader(
        bz, train_dataset, drug_tokenizer,
        aa=aa_dict,
        shuffle=True,
        MLM=MLM,
        mask_rate=config.TRAIN.MASK_PROBABILITY,
        target_random_deletion_ratio=config.TRAIN.TARGET_RANDOM_DROP_RATIO,
        mutation_rate=config.TRAIN.MUTAION
    )
    val_loader  = get_dataLoader(bz, val_dataset, drug_tokenizer)
    test_loader = get_dataLoader(bz, test_dataset, drug_tokenizer)

    # ===== output dir (per fold) =====
    output_path = RESULT_DIR / args.data / args.split / f"fold{fold}" / f"{config.TRAIN.OUTPUT_DIR}{config.TRAIN.SEED}"
    mkdir(output_path)

    trainer = Trainer(
        model, opt, device,
        train_loader, val_loader, test_loader,
        output_path, config
    )

    result, best_epoch = trainer.train()



    return result


# =========================
# Main (CV controller)
# =========================
def main():
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    train_config_path = resolve_config_path(args.train_config, 'train_config.yaml')
    model_config_path = resolve_config_path(args.model_config, 'model_config.yaml')
    print(f"[Config] train_config: {train_config_path}")
    print(f"[Config] model_config: {model_config_path}")

    train_config = load_config_file(train_config_path)
    model_config = load_config_file(model_config_path)
    config = OmegaConf.merge(train_config, model_config)
    model_configs = dict(model_config)

    all_results = []

    if args.fold is not None:
        # ---- single fold ----
        assert 1 <= args.fold <= 5
        result = run_one_fold(args.fold, config, model_configs)
        all_results.append(result)
    else:
        # ---- 5-fold CV ----
        for fold in range(1, 6):
            result = run_one_fold(fold, config, model_configs)
            all_results.append(result)

    # ===== summarize =====
    print("\n" + "=" * 60)
    print("5-Fold Cross Validation Results")
    print("=" * 60)

    metrics = [k for k in all_results[0].keys() if k != "best_epoch"]
    for m in metrics:
        values = [r[m] for r in all_results]
        print(f"{m.upper():12s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # ===== save CV results to txt =====
    save_dir = RESULT_DIR / args.data / args.split
    mkdir(save_dir)

    save_path = save_dir / f"cv_results_seed{config.TRAIN.SEED}.txt"

    with open(save_path, "w") as f:
        f.write("5-Fold Cross Validation Results\n")
        f.write("=" * 60 + "\n")
        for m in metrics:
            values = [r[m] for r in all_results]
            mean = np.mean(values)
            std = np.std(values)
            line = f"{m.upper():12s}: {mean:.4f} ± {std:.4f}\n"
            f.write(line)

    print(f"\n[Saved] CV results saved to: {save_path}")

    return all_results


if __name__ == '__main__':
    s = time()
    main()
    e = time()
    print(f"\nTotal running time: {round(e - s, 2)}s")
