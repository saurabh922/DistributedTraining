import torch
import os
import logging
from dotenv import load_dotenv
from collections import namedtuple
import inspect

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset
import torchmetrics

import mlflow
import mlflow.pytorch

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    logger.info("HF Token loaded.")
else:
    logger.error("HF Token not found in environment variables.")
    exit(1)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("IMDB-sentiment-analysis")


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logger.info("Using device: %s", device)


tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased", use_auth_token=HF_TOKEN, local_files_only=True)
bert_model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2, local_files_only=True)
bert_model = bert_model.to(device)

with open("tokenizer.txt", "w") as f:
    f.write(str(tokenizer))

with open("bert_model_function.txt", "w") as f:
    f.write(inspect.getsource(bert_model.forward))

with open("bert_model.txt", "w") as f:
    f.write(str(bert_model))

imdb_dataset = load_dataset("stanfordnlp/imdb")
train_dataset = imdb_dataset["train"]
test_dataset = imdb_dataset["test"]
logger.info("Length of train dataset: %d", train_dataset.num_rows)

split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"].select(range(100))
valid_dataset = split_dataset["test"].select(range(10))

sample_output = tokenizer("This is a test sentence", padding=True, truncation=True, max_length=256, return_tensors="pt")
print(sample_output)


fields = ["input_ids", "attention_mask"]
class SeqPair(namedtuple("SeqPair", fields)):
    def to(self, device):
        return SeqPair(
            self.input_ids.to(device),
            self.attention_mask.to(device)
        )


def collate_fn(batch):
    src_txt = [item["text"] for item in batch]
    tgt_labels = [item["label"] for item in batch]
    src_encoding = tokenizer(src_txt, padding=True, truncation=True, max_length=256, return_tensors="pt")
    src_token_ids = src_encoding["input_ids"]
    src_attention_mask = src_encoding["attention_mask"]
    inputs = SeqPair(src_token_ids, src_attention_mask)
    tgt_labels = torch.tensor(tgt_labels, dtype=torch.long)
    return inputs, tgt_labels

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=4, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)

def evaluate_fn(model, metric, data_loader):
    model.eval()
    metric.reset()
    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            output = model(
                input_ids=X_batch.input_ids, 
                attention_mask=X_batch.attention_mask, 
                labels=y_batch)
            y_pred = output.logits.argmax(dim=1)
            metric.update(y_pred, y_batch)
    return metric.compute()

def train_fn(model, optimizer, metric, train_data_loader, valid_data_loader, n_epochs=5, patience=5, factor=0.5):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=patience, factor=factor)
    history = {"train_loss": [], "train_metric": [], "valid_metric": []}
    for epoch in range(n_epochs):
        train_loss = 0
        metric.reset()
        model.train()
        for X_batch, y_batch in train_data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(
                input_ids=X_batch.input_ids, 
                attention_mask=X_batch.attention_mask, 
                labels=y_batch)
            y_pred = output.logits.argmax(dim=1)
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            metric.update(y_pred, y_batch)
        # logger.info("This is the Learning rate: %.8f", optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss / len(train_data_loader))
        mlflow.log_metric("train_loss", history["train_loss"][-1], step=epoch)
        history["train_metric"].append(metric.compute().item())
        mlflow.log_metric("train_metrics", history["train_metric"][-1], step=epoch)
        valid_metrics = evaluate_fn(model, metric, valid_data_loader).item()
        history["valid_metric"].append(valid_metrics)
        mlflow.log_metric("valid_metrics", history["valid_metric"][-1], step=epoch)
        scheduler.step(valid_metrics)
        mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        "checkpoint.pt")

        logger.info("Epoch %d | Train Loss: %.4f | Train Metric: %.8f | Valid Metric: %.8f", epoch + 1, history["train_loss"][-1], history["train_metric"][-1], history["valid_metric"][-1])
    return history


if __name__ == "__main__":
    optimizer = torch.optim.NAdam(bert_model.parameters(), lr=1e-5)
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)

    with mlflow.start_run():
        mlflow.log_params({
            "model": "google/mobilebert-uncased",
            "optimizer": "NAdam",
            "learning_Rate": 1e-5,
            "batch_size": 4,
            "n_epochs": 5,
            "patience": 5,
            "factor": 0.5,
        })
        history = train_fn(bert_model, optimizer, metric, train_loader, valid_loader)


        with open("history.json", "w") as f:
            import json
            json.dump(history, f)

        mlflow.pytorch.log_model(bert_model, artifact_path="mobilebert-uncased", serialization_format="pickle")
        mlflow.log_artifact("checkpoint.pt")
        mlflow.log_artifact("history.json")

        


