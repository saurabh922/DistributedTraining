import torch
import os
import logging
from dotenv import load_dotenv
from collections import namedtuple
import inspect
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
# REMOVED: DistributedSampler (TP processes the same batch across the mesh)
import torch.distributed as dist
# NEW: Modern PyTorch Parallelism imports
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, shard_module
from datasets import load_dataset
import torchmetrics

import mlflow
import mlflow.pytorch

load_dotenv()

# --- TP Environment & Mesh Initialization ---
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
IS_MAIN_PROCESS = (RANK == 0)

if WORLD_SIZE > 1:
    # Initialize standard distributed group first
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(LOCAL_RANK)
    device = f"cuda:{LOCAL_RANK}"
    
    # Create a 1D Device Mesh across all available GPUs for Tensor Parallelism
    tp_mesh = init_device_mesh("cuda", (WORLD_SIZE,), mesh_dim_names=("tp",))
else:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tp_mesh = None

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

tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased", use_auth_token=HF_TOKEN, local_files_only=True)
raw_bert_model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2, local_files_only=True)
raw_bert_model = raw_bert_model.to(device)

# --- NEW: Shard Model Layers for Tensor Parallelism ---
if WORLD_SIZE > 1 and tp_mesh is not None:
    logger.info("Sharding MobileBERT layers using Tensor Parallelism...")
    
    # Define how MobileBERT linear layers shard across the GPU mesh
    # (WORLD_SIZE = 3)
    # Layer 1 (Input layer): (20) neurons (Weight matrix (W_{0}) shape: ([1, 20]))
    # Layer 2 (Hidden layer): (10) neurons (Weight matrix (W_{1}) shape: ([20, 10]))
    # Layer 3 (Output layer): (4) neurons (Weight matrix (W_{2}) shape: ([10, 4]))
    
    # Column-Wise Sharding: Slicing Layer 2 (dim=1 is 10, so 10 collumns we have)
    ## PyTorch's ColwiseParallel pads the matrix to the next multiple of (3, as we have 3 gpu). 
    ## It treats Layer 2 as if it has (12) neurons ((12 div 3 = 4) columns per GPU). 
    ## The final (2) columns are filled with dummy zeros that get ignored later.
    ## GPU 0 gets Columns 1–4 (Shape: ([20, 4])), GPU2 get 5-8, so on
    ## Every GPU receives the exact same input vector of size ([1, 20]).
    ## GPU 0 multiplies ([1, 20] times [20, 4]) to get a partial output of ([1, 4]).
    
    # Row-Wise Sharding: Slicing Layer 3
    ## Because Layer 2's outputs are scattered across 3 GPUs, Layer 3 must accept these split inputs. We apply Row-wise sharding, 
    ## Original Matrix ((W_{2})): Shape is ([10, 4]). Here dim=0 means we have 10 rows
    ## With Padding: Because Layer 2 was padded to (12) outputs, Layer 3's rows are padded to (12) inputs. Its padded shape becomes ([12, 4])
    ## The Execution: The (12) rows are divided cleanly by (3) GPUs ((12 div 3 = 4) rows per GPU).
    ## GPU 0 gets Rows 1–4 (Shape: ([4, 4])), GPU2 get 5-8, so on 
    ## GPU 0 multiplies its hidden chunk ([1, 4] times [4, 4]) to get an output slice of ([1, 4]). 
    ## Now, the actual values of Layer 3 are incomplete because each GPU only multiplied a piece of the rows. 
    ## To complete the matrix multiplication, these three matrix outputs must be added together. 
    ## PyTorch automatically triggers an All-Reduce (SUM) collective communication.
    
    # Build the explicit surgery map (Plan)
    ## tp_plan = {"layer2": ColwiseParallel(), "layer3": RowwiseParallel()}
    
    tp_plan = {
        "classifier": ColwiseParallel(), # Slices the output dimension across GPUs
    }
    
    # Programmatically find and add attention/intermediate layers to the plan if present
    for name, module in raw_bert_model.named_modules():
        if "query" in name and isinstance(module, torch.nn.Linear):
            tp_plan[name] = ColwiseParallel()
        elif "key" in name and isinstance(module, torch.nn.Linear):
            tp_plan[name] = ColwiseParallel()
        elif "value" in name and isinstance(module, torch.nn.Linear):
            tp_plan[name] = ColwiseParallel()
        elif "output.dense" in name and isinstance(module, torch.nn.Linear):
            tp_plan[name] = RowwiseParallel() # Sums up activations across the row
            
    # Apply the sharding plan to the model weights dynamically
    bert_model = shard_module(raw_bert_model, tp_mesh, plan=tp_plan)
else:
    bert_model = raw_bert_model

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

# CRITICAL CHANGED: shuffle=True for train, no DistributedSamplers are used in pure TP
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
            
    # In pure TP, all GPUs see the same batch, so metric.compute() is already identical
    val_acc = metric.compute()
    return val_acc

def train_fn(model, optimizer, metric, train_data_loader, valid_data_loader, n_epochs=5, patience=5, factor=0.5):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=patience, factor=factor)
    history = {"train_loss": [], "train_metric": [], "valid_metric": []}
    
    for epoch in range(n_epochs):
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
            
        # Losses are naturally uniform since data replication happens across the TP mesh
        avg_loss = (train_loss_tensor / len(train_data_loader)).item()
        train_acc = metric.compute()
        valid_metrics = evaluate_fn(model, metric, valid_data_loader).item()
        
        scheduler.step(valid_metrics)
        
        if IS_MAIN_PROCESS:
            history["train_loss"].append(avg_loss)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            history["train_metric"].append(train_acc.item())
            mlflow.log_metric("train_metrics", train_acc.item(), step=epoch)
            history["valid_metric"].append(valid_metrics)
            mlflow.log_metric("valid_metrics", valid_metrics, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(), # DTensor state_dict contains global shape metadata
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
            "parallelism": "TensorParallelism"
        })

    history = train_fn(bert_model, optimizer, metric, train_loader, valid_loader)

    if IS_MAIN_PROCESS:
        with open("history.json", "w") as f:
            json.dump(history, f)

        # Log un-sharded weights back to MLflow safely
        mlflow.pytorch.log_model(bert_model, artifact_path="mobilebert-uncased", serialization_format="pickle")
        mlflow.log_artifact("checkpoint.pt")
        mlflow.log_artifact("history.json")
        mlflow.end_run()

    if WORLD_SIZE > 1:
        dist.destroy_process_group()

# torchrun --nproc_per_node=2 your_tp_script.py

# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=0 \
#     --master_addr="10.0.0.1" \
#     --master_port=29500 \
#     your_script.py