import sys
import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, precision_recall_curve,
    accuracy_score, f1_score, precision_score,
    matthews_corrcoef
)
from prettytable import PrettyTable
from tqdm import tqdm
# from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast




class Trainer(object):
    def __init__(self, model, optim, device,
                 train_dataloader, val_dataloader, test_dataloader,
                 output_path, config):

        self.model = model
        self.optim = optim
        self.device = device
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.epochs = config.TRAIN.MAX_EPOCH
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

        self.current_epoch = 0
        self.step = 0

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.nb_training = len(self.train_dataloader)

        self.best_state_dict = None
        self.best_epoch = None
        self.best_auroc = 0.0

        # ===== Early Stopping =====
        self.early_stop = getattr(config.TRAIN, "EARLY_STOP", True)
        self.early_stop_patience = getattr(config.TRAIN, "EARLY_STOP_PATIENCE", 5)
        self.early_stop_delta = getattr(config.TRAIN, "EARLY_STOP_DELTA", 1e-4)

        self.no_improve_epochs = 0
        self.best_val_metric = -float("inf")

        self.train_loss_epoch = []
        self.val_loss_epoch = []
        self.val_auroc_epoch = []
        self.test_metrics = {}

        self.output_dir = output_path

        self.train_table = PrettyTable(["# Epoch", "Train_loss"])
        self.val_table = PrettyTable(["# Epoch", "AUROC", "AUPRC"])
        self.test_table = PrettyTable([
            "# Best Epoch", "AUROC", "AUPRC",
            "Precision", "Recall", "F1",
            "Specificity", "Accuracy", "MCC", "Threshold"
        ])

    # =========================
    # 核心：保存完整模型状态
    # =========================
    @staticmethod
    def get_trainable_state_dict(model):
        return {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        }

    def train(self):
        float2str = lambda x: '%0.4f' % x

        for _ in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()

            self.train_loss_epoch.append(train_loss)
            self.train_table.add_row(
                [f"epoch {self.current_epoch}", float2str(train_loss)]
            )

            auroc, auprc = self.test(dataloader="val")
            self.val_auroc_epoch.append(auroc)

            self.val_table.add_row(
                [f"epoch {self.current_epoch}",
                 float2str(auroc), float2str(auprc)]
            )

            # ===== best model =====
            if auroc > self.best_auroc:
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
                self.best_state_dict = self.get_trainable_state_dict(self.model)

            # ===== Early Stopping =====
            if self.early_stop:
                if auroc > self.best_val_metric + self.early_stop_delta:
                    self.best_val_metric = auroc
                    self.no_improve_epochs = 0
                else:
                    self.no_improve_epochs += 1

                if self.no_improve_epochs >= self.early_stop_patience:
                    print(
                        f"\n Early Stopping Triggered! "
                        f"No improvement for {self.early_stop_patience} epochs.\n"
                    )
                    break

            print(
                f"Validation at Epoch {self.current_epoch} | "
                f"AUROC {auroc:.4f} | AUPRC {auprc:.4f}"
            )

        # =========================
        # Test with best weights
        # =========================
        self.model.load_state_dict(self.best_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        (auroc, auprc, f1, precision, recall,
         specificity, accuracy, mcc, thred_optim) = self.test(dataloader="test")

        self.test_table.add_row(list(map(float2str, [
            self.best_epoch, auroc, auprc,
            precision, recall, f1,
            specificity, accuracy, mcc,
            thred_optim
        ])))

        print(
            f"Test at Best Model of Epoch {self.best_epoch} | "
            f"AUROC {auroc:.4f} | AUPRC {auprc:.4f} | "
            f"Recall {recall:.4f} | Specificity {specificity:.4f} | "
            f"Accuracy {accuracy:.4f} | Threshold {thred_optim:.4f}"
        )

        self.test_metrics = {
            "auroc": auroc,
            "auprc": auprc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "accuracy": accuracy,
            "mcc": mcc,
            "threshold": thred_optim,
            "best_epoch": self.best_epoch
        }

        self.save_result()
        return self.test_metrics, self.best_epoch

    def save_result(self):
        os.makedirs(self.output_dir, exist_ok=True)

        #  只保存 LoRA / 可训练参数
        if self.config.TRAIN.SAVE_MODEL:

            torch.save(
                self.model.state_dict(),
                os.path.join(self.output_dir, f"best_epoch_{self.best_epoch}_full.pth")
            )
        # metrics
        torch.save({
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_auroc": self.val_auroc_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }, os.path.join(self.output_dir, "result_metrics.pt"))

        # tables
        with open(os.path.join(self.output_dir, "train_metrics.txt"), "w") as f:
            f.write(self.train_table.get_string())

        with open(os.path.join(self.output_dir, "valid_metrics.txt"), "w") as f:
            f.write(self.val_table.get_string())

        with open(os.path.join(self.output_dir, "test_metrics.txt"), "w") as f:
            f.write(self.test_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0.0
        # scaler = GradScaler()
        scaler = GradScaler(device='cuda')
        loop = tqdm(self.train_dataloader, file=sys.stdout, colour="#bdbdbd")
        loop.set_description(f"Train Epoch [{self.current_epoch}/{self.epochs}]")

        for step, batch in enumerate(loop):
            self.optim.zero_grad()
            self.step += 1

            # with autocast():
            with autocast(device_type='cuda'):
                input_drugs = batch['batch_inputs_drug'].to(self.device)
                input_proteins = batch['batch_inputs_pr']['input_ids'].to(self.device)
                pr_mask = batch['batch_inputs_pr']['attention_mask'].to(self.device)
                # labels = torch.tensor(batch['labels']).to(self.device)
                labels = torch.tensor(batch['labels'], dtype=torch.long).to(self.device)

                drug_labels = batch['masked_drug_labels']
                if drug_labels is not None:
                    inputs_drugs_m = batch['batch_inputs_drug_m'].to(self.device)
                    drug_labels = drug_labels.to(self.device)

                    output = self.model(
                        input_drugs, input_proteins,
                        pr_mask=pr_mask, masked_drugs=inputs_drugs_m
                    )

                    b_loss = self.criterion(output['logits'], labels)

                    mlm_loss = nn.CrossEntropyLoss(ignore_index=-1)(
                        output['drug_mlm_logits'], drug_labels
                    )
                    loss = b_loss + mlm_loss

                else:
                    output = self.model(input_drugs, input_proteins, pr_mask=pr_mask)
                    loss = self.criterion(output['logits'], labels)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            scaler.step(self.optim)
            scaler.update()


            loss_epoch += loss.item()
            loop.set_postfix(avg_loss=loss_epoch / (step + 1))

        return loss_epoch / len(self.train_dataloader)

    def test(self, dataloader="test"):
        y_label, y_pred = [], []

        loader = self.test_dataloader if dataloader == "test" else self.val_dataloader
        loop = tqdm(loader, file=sys.stdout, colour="#bdbdbd")

        with torch.no_grad():
            self.model.eval()
            loop.set_description("Test" if dataloader == "test" else "Validation")

            for batch in loop:
                labels = batch['labels']
                input_proteins = batch['batch_inputs_pr']['input_ids'].to(self.device)
                input_drugs = batch['batch_inputs_drug'].to(self.device)
                pr_mask = batch['batch_inputs_pr']['attention_mask'].to(self.device)

                output = self.model(input_drugs, input_proteins, pr_mask=pr_mask)
                # scores = output['logits'][:, 1]
                logits = output['logits']  # (B, 2)
                probs = torch.softmax(logits, dim=-1)  # (B, 2)
                scores = probs[:, 1]  # P(y=1)

                y_label.extend(labels)
                # y_pred.extend(scores.tolist())
                y_pred.extend(scores.detach().cpu().tolist())

        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)

        if dataloader != "test":
            return auroc, auprc

        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        y_bin = (np.array(y_pred) >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_label, y_bin).ravel()

        return (
            auroc,
            auprc,
            f1_score(y_label, y_bin),
            precision_score(y_label, y_bin),
            tp / (tp + fn),
            tn / (tn + fp),
            accuracy_score(y_label, y_bin),
            matthews_corrcoef(y_label, y_bin),
            optimal_threshold
        )
