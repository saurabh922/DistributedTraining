import torch
import os
import logging
from dotenv import load_dotenv
from collections import namedtuple
import inspect
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
import torchmetrics

import mlflow
import mlflow.pytorch

load_dotenv()

# --- DDP Initialization ---
# torchrun/Kubeflow sets these environment variables automatically
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
IS_MAIN_PROCESS = (RANK == 0)

if WORLD_SIZE > 1:
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(LOCAL_RANK)
    device = f"cuda:{LOCAL_RANK}"
else:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Suppress logging on worker processes to keep terminal outputs tidy
logging.basicConfig(
    level=logging.INFO if IS_MAIN_PROCESS else logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

HF_TOKEN = os.getenv("HF_TOKEN")
if IS_MAIN_PROCESS:
    if HF_TOKEN:
        logger.info("HF Token loaded.")
    else:
        logger.error("HF Token not found in environment variables.")
        exit(1)
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("IMDB-sentiment-analysis")

logger.info("Rank %d using device: %s", RANK, device)

# Load tokenizer and base model onto the local GPU target
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased", use_auth_token=HF_TOKEN, local_files_only=True)
raw_bert_model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2, local_files_only=True)
raw_bert_model = raw_bert_model.to(device)

# Wrap model with DDP
if WORLD_SIZE > 1:
    bert_model = DDP(raw_bert_model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
else:
    bert_model = raw_bert_model

# Only write structure text logs on the main node
if IS_MAIN_PROCESS:
    with open("tokenizer.txt", "w") as f:
        f.write(str(tokenizer))
    with open("bert_model_function.txt", "w") as f:
        f.write(inspect.getsource(raw_bert_model.forward))
    with open("bert_model.txt", "w") as f:
        f.write(str(raw_bert_model))

imdb_dataset = load_dataset("stanfordnlp/imdb")
train_dataset = imdb_dataset["train"]
test_dataset = imdb_dataset["test"]

split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"].select(range(100))
valid_dataset = split_dataset["test"].select(range(10))

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

# Create Samplers for distributed data splitting
train_sampler = DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True) if WORLD_SIZE > 1 else None
valid_sampler = DistributedSampler(valid_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False) if WORLD_SIZE > 1 else None
test_sampler = DistributedSampler(test_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False) if WORLD_SIZE > 1 else None

# Set shuffle=False when a DistributedSampler handles text partitioning
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=4, sampler=valid_sampler, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, sampler=test_sampler, collate_fn=collate_fn)

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
            
    val_acc = metric.compute()
    if WORLD_SIZE > 1:
        dist.all_reduce(val_acc, op=dist.ReduceOp.SUM)
        val_acc = val_acc / WORLD_SIZE
    return val_acc

def train_fn(model, optimizer, metric, train_data_loader, valid_data_loader, n_epochs=5, patience=5, factor=0.5):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=patience, factor=factor)
    history = {"train_loss": [], "train_metric": [], "valid_metric": []}
    
    for epoch in range(n_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        train_loss_tensor = torch.tensor(0.0).to(device)
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
            train_loss_tensor += loss.detach()
            metric.update(y_pred, y_batch)
            
        # Synchronize losses and training metrics across cluster ranks
        if WORLD_SIZE > 1:
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (train_loss_tensor / (len(train_data_loader) * WORLD_SIZE)).item()
        else:
            avg_loss = (train_loss_tensor / len(train_data_loader)).item()

        train_acc = metric.compute()
        if WORLD_SIZE > 1:
            dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
            train_acc = train_acc / WORLD_SIZE

        valid_metrics = evaluate_fn(model, metric, valid_data_loader).item()
        
        # Scheduler requires adjustments across all ranks to maintain step sync
        scheduler.step(valid_metrics)
        
        if IS_MAIN_PROCESS:
            history["train_loss"].append(avg_loss)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            history["train_metric"].append(train_acc.item())
            mlflow.log_metric("train_metrics", train_acc.item(), step=epoch)
            history["valid_metric"].append(valid_metrics)
            mlflow.log_metric("valid_metrics", valid_metrics, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch)

            # Strip the structural 'module.' DDP prefix before serialization
            unwrap_model = model.module if WORLD_SIZE > 1 else model
            torch.save({
                "epoch": epoch,
                "model_state_dict": unwrap_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, "checkpoint.pt")

            logger.info("Epoch %d | Train Loss: %.4f | Train Metric: %.8f | Valid Metric: %.8f", 
                        epoch + 1, avg_loss, train_acc.item(), valid_metrics)
            
    return history


if __name__ == "__main__":
    optimizer = torch.optim.NAdam(bert_model.parameters(), lr=1e-5)
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)

    if IS_MAIN_PROCESS:
        mlflow.start_run()
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

    if IS_MAIN_PROCESS:
        with open("history.json", "w") as f:
            json.dump(history, f)

        unwrap_model = bert_model.module if WORLD_SIZE > 1 else bert_model
        mlflow.pytorch.log_model(unwrap_model, artifact_path="mobilebert-uncased", serialization_format="pickle")
        mlflow.log_artifact("checkpoint.pt")
        mlflow.log_artifact("history.json")
        mlflow.end_run()

    if WORLD_SIZE > 1:
        dist.destroy_process_group()

# torchrun --nproc_per_node=2 your_script_name.py

# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=0 \
#     --master_addr="10.0.0.1" \
#     --master_port=29500 \
#     your_script.py