import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
from transformers import AutoTokenizer

from dataloader.dataloader import DTIDataset, get_dataLoader
from models.IFIM import IFIM
from utils.paths import CONFIG_DIR, DATA_DIR, MODEL_DIR, RESULT_DIR
from utils.utils import load_config_file, mkdir


parser = argparse.ArgumentParser(description="DTI five-fold checkpoint testing")
parser.add_argument('--data', default='bindingdb', type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='split_random', type=str, metavar='S',
                    help="split task",
                    choices=[
                        'random', 'cold', 'cluster', 'augmented',
                        'split_double_cold', 'split_drug_cold',
                        'split_protein_cold', 'split_random', 'cluster_new'
                    ])
parser.add_argument('--model_paths', nargs=5, required=True, type=str,
                    help='five checkpoint paths ordered from fold1 to fold5')
parser.add_argument('--output_dir', default=str(RESULT_DIR / 'test_5fold_results'), type=str,
                    help='directory to save five-fold test results')
parser.add_argument('--device', default='cuda:0', type=str,
                    help="testing device, e.g. cuda:0 or cpu")
parser.add_argument('--train_config', default=None, type=str,
                    help="path to the training config; if omitted, use configs/{data}/{split}/train_config.yaml")
parser.add_argument('--model_config', default=None, type=str,
                    help="path to the model config; if omitted, use configs/{data}/{split}/model_config.yaml")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')


def resolve_config_path(config_arg, config_name):
    if config_arg is not None:
        return Path(config_arg)

    data_config_path = CONFIG_DIR / args.data / args.split / config_name
    if data_config_path.is_file():
        return data_config_path

    fallback_path = CONFIG_DIR / config_name
    if fallback_path.is_file():
        print(f"[Warning] Missing {data_config_path}; fallback to {fallback_path}")
        return fallback_path

    raise FileNotFoundError(
        f"Cannot find {config_name}. Tried {data_config_path} and {fallback_path}."
    )


def get_test_path(split_dir, fold):
    if args.split == 'cluster':
        return split_dir / 'target_test_with_id.csv'
    if args.split in [
        'split_double_cold', 'split_drug_cold',
        'split_protein_cold', 'split_random', 'cluster_new'
    ]:
        return split_dir / f'test_fold{fold}.csv'
    return split_dir / 'test_with_id.csv'


def load_model(model_path, model_configs, c, p_ci):
    model = IFIM(c=c, p_ci=p_ci, model_configs=model_configs).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']

    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def test_model(model, test_loader, fold):
    y_label, y_pred = [], []

    with torch.no_grad():
        loop = tqdm(test_loader, colour='#aaaaaa', file=sys.stdout)
        loop.set_description(f'Testing Fold {fold}')

        for batch in loop:
            labels = batch['labels']
            input_proteins = batch['batch_inputs_pr']['input_ids'].to(device)
            input_drugs = batch['batch_inputs_drug'].to(device)
            pr_mask = batch['batch_inputs_pr']['attention_mask'].to(device)

            output = model(input_drugs, input_proteins, pr_mask=pr_mask)
            probs = torch.softmax(output['logits'], dim=-1)
            scores = probs[:, 1]

            y_label.extend(labels)
            y_pred.extend(scores.detach().cpu().tolist())

    auroc = roc_auc_score(y_label, y_pred)
    auprc = average_precision_score(y_label, y_pred)

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_bin = (np.array(y_pred) >= optimal_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_label, y_pred_bin).ravel()

    metrics = {
        'auroc': auroc,
        'auprc': auprc,
        'f1': f1_score(y_label, y_pred_bin),
        'recall': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'precision': precision_score(y_label, y_pred_bin),
        'accuracy': accuracy_score(y_label, y_pred_bin),
        'mcc': matthews_corrcoef(y_label, y_pred_bin),
        'threshold': optimal_threshold,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }

    return metrics, y_label, y_pred


def print_metrics(title, metrics):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"Optimal Threshold: {metrics['threshold']:.4f}")
    print("Confusion Matrix:")
    print(f"  TP: {metrics['tp']}, TN: {metrics['tn']}")
    print(f"  FP: {metrics['fp']}, FN: {metrics['fn']}")


def save_fold_results(metrics, y_label, y_pred, output_dir):
    mkdir(output_dir)

    pd.DataFrame({
        'label': y_label,
        'prediction': y_pred
    }).to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Test Results\n")
        f.write("=" * 60 + "\n")
        for key in ['auroc', 'auprc', 'f1', 'recall', 'specificity', 'precision', 'accuracy', 'mcc', 'threshold']:
            f.write(f"{key.upper():12s}: {metrics[key]:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TP: {metrics['tp']}, TN: {metrics['tn']}\n")
        f.write(f"  FP: {metrics['fp']}, FN: {metrics['fn']}\n")


def save_summary(all_results, output_dir):
    mkdir(output_dir)

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(os.path.join(output_dir, 'fold_results.csv'), index=False)

    metric_keys = ['auroc', 'auprc', 'f1', 'recall', 'specificity', 'precision', 'accuracy', 'mcc', 'threshold']
    summary_path = os.path.join(output_dir, 'fivefold_results.txt')

    with open(summary_path, 'w') as f:
        f.write("Five-Fold Test Results\n")
        f.write("=" * 60 + "\n")
        for _, row in result_df.iterrows():
            f.write(
                f"Fold {int(row['fold'])}: "
                f"AUROC={row['auroc']:.4f}, AUPRC={row['auprc']:.4f}, "
                f"F1={row['f1']:.4f}, ACC={row['accuracy']:.4f}, MCC={row['mcc']:.4f}\n"
            )

        f.write("\nMean +/- Std\n")
        f.write("=" * 60 + "\n")
        for key in metric_keys:
            values = result_df[key].to_numpy(dtype=float)
            f.write(f"{key.upper():12s}: {np.mean(values):.4f} +/- {np.std(values):.4f}\n")

    print(f"\n[Saved] Five-fold summary saved to: {summary_path}")
    print(f"[Saved] Per-fold CSV saved to: {os.path.join(output_dir, 'fold_results.csv')}")


def main():
    warnings.filterwarnings("ignore")
    print(f"Running on: {device}\n")

    train_config_path = resolve_config_path(args.train_config, 'train_config.yaml')
    model_config_path = resolve_config_path(args.model_config, 'model_config.yaml')
    print(f"[Config] train_config: {train_config_path}")
    print(f"[Config] model_config: {model_config_path}\n")

    train_config = load_config_file(train_config_path)
    model_config = load_config_file(model_config_path)
    config = OmegaConf.merge(train_config, model_config)
    model_configs = dict(model_config)

    dataset_root = DATA_DIR / args.data
    split_dir = dataset_root / args.split
    molformer_path = MODEL_DIR / 'drug' / 'molformer'

    if args.split in [
        'split_double_cold', 'split_drug_cold',
        'split_protein_cold', 'split_random', 'cluster_new'
    ]:
        protein_path = dataset_root / config.TRAIN.PR_PATH
        c_path = dataset_root / config.TRAIN.C_PATH
    else:
        protein_path = split_dir / config.TRAIN.PR_PATH
        c_path = split_dir / config.TRAIN.C_PATH

    if not protein_path.is_file():
        raise FileNotFoundError(f"Protein feature file not found: {protein_path}")
    if not c_path.is_file():
        raise FileNotFoundError(f"Confounder file not found: {c_path}")

    C = pickle.load(open(c_path, 'rb'))
    c = torch.from_numpy(C['cluster_centers']).to(device).permute(1, 0).float()
    p_ci = C['prior'].to(device).float()

    pr_f = pickle.load(open(protein_path, 'rb'))
    drug_tokenizer = AutoTokenizer.from_pretrained(molformer_path, trust_remote_code=True)
    batch_size = config.TRAIN.BATCH_SIZE

    all_results = []
    base_output_dir = os.path.join(args.output_dir, args.data, args.split)

    for fold, model_path in enumerate(args.model_paths, start=1):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Fold {fold} checkpoint not found: {model_path}")

        print(f"\n{'=' * 25} Fold {fold} {'=' * 25}")
        print(f"Loading model from: {model_path}")

        test_path = get_test_path(split_dir, fold)
        if not test_path.is_file():
            raise FileNotFoundError(f"Fold {fold} test file not found: {test_path}")

        df_test = pd.read_csv(test_path)
        print(f"Test dataset size: {len(df_test)}")

        test_dataset = DTIDataset(df_test.index.values, df_test, pr_f)
        test_loader = get_dataLoader(batch_size, test_dataset, drug_tokenizer)

        model = load_model(model_path, model_configs, c, p_ci)
        metrics, y_label, y_pred = test_model(model, test_loader, fold)
        print_metrics(f"Fold {fold} Test Results", metrics)

        fold_output_dir = os.path.join(base_output_dir, f"fold{fold}")
        save_fold_results(metrics, y_label, y_pred, fold_output_dir)

        all_results.append({
            'fold': fold,
            'model_path': str(model_path),
            **metrics,
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Five-Fold Mean +/- Std")
    print("=" * 60)
    result_df = pd.DataFrame(all_results)
    for key in ['auroc', 'auprc', 'f1', 'recall', 'specificity', 'precision', 'accuracy', 'mcc', 'threshold']:
        values = result_df[key].to_numpy(dtype=float)
        print(f"{key.upper():12s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    save_summary(all_results, base_output_dir)
    return all_results


if __name__ == '__main__':
    main()
