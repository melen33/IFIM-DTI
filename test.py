import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse
import pickle
import warnings
import pandas as pd
import torch
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, accuracy_score, \
    f1_score, precision_score, matthews_corrcoef
from tqdm import tqdm

from dataloader.dataloader import DTIDataset, get_dataLoader
from transformers import AutoTokenizer
from models.IFIM import IFIM
from utils.utils import set_seed, mkdir, load_config_file
from utils.paths import CONFIG_DIR, DATA_DIR, MODEL_DIR, RESULT_DIR

parser = argparse.ArgumentParser(description="DTI prediction testing")
parser.add_argument('--data', default='bindingdb', type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='split_random', type=str, metavar='S', help="split task",
                    choices=['random', 'cold', 'cluster', 'augmented', 'split_double_cold', 'split_drug_cold',
                             'split_protein_cold', 'split_random', 'cluster_new'])
parser.add_argument('--fold', default=1, type=int,
                    help="which fold to test")
parser.add_argument('--model_path', type=str,
                    required ='true',
                    help='path to the saved model checkpoint')
parser.add_argument('--output_dir', default=str(RESULT_DIR / 'test_results'), type=str,
                    help='directory to save test results')
parser.add_argument('--device', default='cuda:0', type=str,
                    help="testing device, e.g. cuda:0 or cpu")
parser.add_argument('--train_config', default=str(CONFIG_DIR / 'train_config.yaml'), type=str,
                    help="path to the training config")
parser.add_argument('--model_config', default=str(CONFIG_DIR / 'model_config.yaml'), type=str,
                    help="path to the model config")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')


def get_fold_paths(data_folder, fold):
    test_path = data_folder / f'test_fold{fold}.csv'
    return test_path


def load_model(model_path, model_configs, c, p_ci):
    model = IFIM(c=c, p_ci=p_ci, model_configs=model_configs).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)


    model.eval()
    return model


def test_model(model, test_loader, device):
    y_label, y_pred = [], []

    with torch.no_grad():
        loop = tqdm(test_loader, colour='#aaaaaa', file=sys.stdout)
        loop.set_description(f'Testing')

        for step, batch in enumerate(loop):
            labels = batch['labels']
            input_proteins = batch['batch_inputs_pr']['input_ids'].to(device)
            input_drugs = batch['batch_inputs_drug'].to(device)
            pr_mask = batch['batch_inputs_pr']['attention_mask'].to(device)


            output = model(input_drugs, input_proteins, pr_mask=pr_mask)
            # scores = output['logits'][:, 1]
            logits = output['logits']  # (B, 2)
            probs = torch.softmax(logits, dim=-1)  # (B, 2)
            scores = probs[:, 1]  # P(y=1)

            y_label.extend(labels)
            # y_pred.extend(scores.tolist())
            y_pred.extend(scores.detach().cpu().tolist())

    auroc = roc_auc_score(y_label, y_pred)
    auprc = average_precision_score(y_label, y_pred)

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    y_pred_bin = (np.array(y_pred) >= optimal_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_label, y_pred_bin).ravel()

    acc = accuracy_score(y_label, y_pred_bin)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = precision_score(y_label, y_pred_bin)
    f1 = f1_score(y_label, y_pred_bin)
    mcc = matthews_corrcoef(y_label, y_pred_bin)

    metrics = {
        'auroc': auroc,
        'auprc': auprc,
        'f1': f1,
        'recall': recall,
        'specificity': specificity,
        'precision': precision,
        'accuracy': acc,
        'mcc': mcc,
        'threshold': optimal_threshold,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

    return metrics, y_label, y_pred


def save_results(metrics, y_label, y_pred, output_dir):
    mkdir(output_dir)

    results_path = os.path.join(output_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Test Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"AUROC: {metrics['auroc']:.4f}\n")
        f.write(f"AUPRC: {metrics['auprc']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"Specificity: {metrics['specificity']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"MCC: {metrics['mcc']:.4f}\n")
        f.write(f"Optimal Threshold: {metrics['threshold']:.4f}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TP: {metrics['tp']}, TN: {metrics['tn']}\n")
        f.write(f"  FP: {metrics['fp']}, FN: {metrics['fn']}\n")

    predictions_path = os.path.join(output_dir, 'predictions.csv')
    df = pd.DataFrame({
        'label': y_label,
        'prediction': y_pred
    })
    df.to_csv(predictions_path, index=False)

    print(f"\n[Saved] Test results saved to: {results_path}")
    print(f"[Saved] Predictions saved to: {predictions_path}")


def main():
    warnings.filterwarnings("ignore")
    print(f"Running on: {device}\n")

    train_config = load_config_file(args.train_config)
    model_config = load_config_file(args.model_config)
    config = OmegaConf.merge(train_config, model_config)
    model_configs = dict(model_config)


    if args.model_path is None:
        raise ValueError("Please provide --model_path pointing to a trained checkpoint.")

    print(f"Loading model from: {args.model_path}")

    dataset_root = DATA_DIR / args.data
    split_dir = dataset_root / args.split
    molformer_path = MODEL_DIR / 'drug' / 'molformer'

    if args.split == 'cluster':
        test_path = split_dir / 'target_test_with_id.csv'
    elif args.split in [
        'split_double_cold', 'split_drug_cold',
        'split_protein_cold', 'split_random', 'cluster_new'
    ]:
        test_path = get_fold_paths(split_dir, args.fold)
    else:
        test_path = split_dir / 'test_with_id.csv'

    df_test = pd.read_csv(test_path)
    print(f"Test dataset size: {len(df_test)}\n")

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
    aa_dict = C['aa'].to(device).float()

    model = load_model(args.model_path, model_configs, c, p_ci)
    print("Model loaded successfully!\n")

    pr_f = pickle.load(open(protein_path, 'rb'))
    test_dataset = DTIDataset(df_test.index.values, df_test, pr_f)

    drug_tokenizer = AutoTokenizer.from_pretrained(molformer_path, trust_remote_code=True)
    bz = config.TRAIN.BATCH_SIZE
    test_loader = get_dataLoader(bz, test_dataset, drug_tokenizer)

    print("Starting testing...")
    metrics, y_label, y_pred = test_model(model, test_loader, device)

    print("\n" + "=" * 60)
    print("Test Results")
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
    print(f"Confusion Matrix:")
    print(f"  TP: {metrics['tp']}, TN: {metrics['tn']}")
    print(f"  FP: {metrics['fp']}, FN: {metrics['fn']}")

    output_dir = os.path.join(args.output_dir, args.data, args.split, f"fold{args.fold}")
    save_results(metrics, y_label, y_pred, output_dir)

    return metrics


if __name__ == '__main__':
    main()
