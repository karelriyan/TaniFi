"""
DiLoCo (Distributed Low-Communication) Simulation for TaniFi

This script implements a simplified DiLoCo federated learning simulation
for bandwidth-constrained agricultural networks.

Key Features:
- Local training with configurable steps before synchronization
- LoRA adapters for efficient parameter updates
- Evaluation metrics: loss, accuracy, macro-F1
- Centralized baseline for comparison
- Simulates farmer nodes with varying connectivity

Paper: "Simulation of Bandwidth-Efficient Federated Learning Architectures
        for Resource-Constrained Agricultural Networks in Indonesia"

Usage:
    # Federated training with real data
    python diloco_trainer.py --real-data --config ../../experiments/config.yaml

    # Centralized baseline
    python diloco_trainer.py --real-data --centralized --config ../../experiments/config.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class YOLOv11ClassificationModel(nn.Module):
    """
    YOLOv11 Nano classification model wrapper for federated learning.
    Uses pretrained ImageNet backbone with custom classification head.
    Exposes .features and .classifier for LoRA adapter insertion.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        from ultralytics import YOLO

        yolo = YOLO('yolo11n-cls.pt')
        yolo_model = yolo.model.model  # nn.Sequential of YOLO layers

        # Backbone: layers 0-9 (Conv, C3k2, C2PSA blocks)
        backbone = nn.Sequential(*list(yolo_model.children())[:-1])

        # Classify layer has: conv(256→1280) + pool + drop + linear(1280→1000)
        classify_layer = yolo_model[-1]

        # Features = backbone + classify.conv + classify.pool → [batch, 1280, 1, 1]
        self.features = nn.Sequential(
            backbone,
            classify_layer.conv,
            classify_layer.pool,
        )

        # Feature dimension after pooling
        self.feature_dim = 1280

        # New classification head for our num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LoRAAdapter(nn.Module):
    """
    LoRA (Low-Rank Adaptation) adapter for efficient fine-tuning.
    Represents the "Shard" that each farmer owns and trains locally.
    Only these small adapters are transmitted, not the full model.
    """
    def __init__(self, base_model, rank=4):
        super().__init__()
        self.base_model = base_model
        self.rank = rank

        for param in self.base_model.parameters():
            param.requires_grad = False

        # Auto-detect feature dimension from base model
        feature_dim = getattr(base_model, 'feature_dim', 128)

        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, rank),
            nn.ReLU(),
            nn.Linear(rank, feature_dim)
        )

    def forward(self, x):
        features = self.base_model.features(x)
        adapted_features = features + self.adapter(features.flatten(1)).view_as(features)
        output = self.base_model.classifier(adapted_features)
        return output

    def get_adapter_params(self):
        """Get only the adapter parameters (the 'shard')."""
        return {name: param.clone().detach()
                for name, param in self.adapter.named_parameters()}

    def set_adapter_params(self, params):
        """Set adapter parameters from received 'shard'."""
        with torch.no_grad():
            for name, param in self.adapter.named_parameters():
                param.copy_(params[name])


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on a dataset, returning loss, accuracy, and macro-F1."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1

            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'loss': float(avg_loss),
        'accuracy': float(accuracy),
        'f1_macro': float(f1)
    }


# =============================================================================
# CLASS WEIGHTING
# =============================================================================

def compute_class_weights(dataset):
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    For imbalanced datasets, this produces higher weights for rare classes
    so the model is penalized more for misclassifying minority samples.

    Example: distribution {0:77, 1:21, 2:6} -> weights ~[0.45, 1.65, 5.78]
    """
    from collections import Counter
    labels = [dataset[i][1] for i in range(len(dataset))]
    if isinstance(labels[0], torch.Tensor):
        labels = [l.item() for l in labels]
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = torch.tensor(
        [total / (num_classes * counts[c]) for c in sorted(counts.keys())],
        dtype=torch.float32
    )
    print(f"  Class weights: {dict(sorted(counts.items()))} -> {weights.tolist()}")
    return weights


# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================
FAST_MODE = True


# =============================================================================
# FARMER NODE
# =============================================================================

class FarmerNode:
    """Simulates a single farmer/agent node with local training."""
    def __init__(self, node_id, base_model, data_subset, device='cpu',
                 total_rounds=20, warmup_rounds=5, class_weights=None):
        self.node_id = node_id
        self.device = device
        self.model = LoRAAdapter(base_model, rank=4).to(device)
        self.total_rounds = total_rounds
        self.warmup_rounds = warmup_rounds
        self.current_round = 0

        if FAST_MODE:
            batch_size = 64
            num_workers = 4
            pin_memory = device != 'cpu'
        else:
            batch_size = 8
            num_workers = 0
            pin_memory = False

        self.dataloader = DataLoader(
            data_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        self.base_lr = 0.001
        self.optimizer = optim.AdamW(self.model.adapter.parameters(), lr=self.base_lr, weight_decay=0.0001)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.use_amp = FAST_MODE and device == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        self.local_losses = []
        self.sync_count = 0

    def _update_lr(self):
        """Update learning rate with warmup + cosine decay."""
        if self.current_round < self.warmup_rounds:
            # Linear warmup
            lr = self.base_lr * (self.current_round + 1) / self.warmup_rounds
        else:
            # Cosine decay
            progress = (self.current_round - self.warmup_rounds) / max(1, self.total_rounds - self.warmup_rounds)
            lr = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def local_training(self, num_steps=500):
        """Perform local training for specified number of steps."""
        self._update_lr()
        self.model.train()
        step = 0
        epoch_losses = []

        while step < num_steps:
            for data, target in self.dataloader:
                if step >= num_steps:
                    break
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()

                epoch_losses.append(loss.item())
                step += 1

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        self.local_losses.append(avg_loss)
        self.current_round += 1
        return avg_loss

    def get_shard(self):
        """Extract the trained 'shard' (LoRA adapter parameters)."""
        return self.model.get_adapter_params()

    def update_shard(self, aggregated_params):
        """Receive and apply aggregated parameters from global update."""
        self.model.set_adapter_params(aggregated_params)
        self.sync_count += 1


# =============================================================================
# DILOCO COORDINATOR
# =============================================================================

class DiLoCoCoordinator:
    """DiLoCo Coordinator - manages federated learning simulation."""

    def __init__(self, base_model, num_farmers=10, local_steps=100, device='cpu'):
        if num_farmers < 1:
            raise ValueError(f"num_farmers must be >= 1, got {num_farmers}")
        if local_steps < 1:
            raise ValueError(f"local_steps must be >= 1, got {local_steps}")
        self.base_model = base_model
        self.num_farmers = num_farmers
        self.local_steps = local_steps
        self.device = device
        self.farmer_nodes = []
        self.global_model = base_model.to(device)
        self.val_loader = None
        self.test_loader = None
        self.test_metrics = None

        self.global_metrics = {
            'rounds': [],
            'avg_loss': [],
            'bandwidth_saved': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1_macro': [],
        }

    def _make_progress_bar(self, current, total, bar_length=20):
        filled = int(bar_length * current / total)
        bar = '\u2588' * filled + '\u2591' * (bar_length - filled)
        return f"[{bar}]"

    def initialize_farmers(self, dataset, non_iid=True, val_dataset=None, test_dataset=None,
                            total_rounds=20, warmup_rounds=5):
        """Create farmer nodes with partitioned data."""
        self._total_rounds = total_rounds
        self._warmup_rounds = warmup_rounds
        print(f"\n Initializing {self.num_farmers} farmer nodes...")

        # Compute class weights from training data for weighted loss
        self._class_weights = compute_class_weights(dataset)

        # Set up evaluation loaders
        if val_dataset is not None:
            self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        if test_dataset is not None:
            self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

        total_size = len(dataset)
        min_samples_per_farmer = 1
        if total_size < self.num_farmers * min_samples_per_farmer:
            use_replacement = True
        else:
            use_replacement = False

        if non_iid:
            sizes = np.random.randint(
                min_samples_per_farmer,
                max(min_samples_per_farmer + 1, total_size // self.num_farmers * 3),
                self.num_farmers
            )
            if use_replacement:
                sizes = np.maximum(sizes, min_samples_per_farmer)
            else:
                sizes = (sizes / sizes.sum() * total_size).astype(int)
                sizes = np.maximum(sizes, min_samples_per_farmer)
        else:
            samples_per_farmer = max(min_samples_per_farmer, total_size // self.num_farmers)
            sizes = [samples_per_farmer] * self.num_farmers

        if use_replacement:
            for i in range(self.num_farmers):
                subset_indices = np.random.choice(total_size, size=sizes[i], replace=True)
                subset = Subset(dataset, subset_indices.tolist())
                farmer = FarmerNode(node_id=i, base_model=self.base_model, data_subset=subset,
                                    device=self.device, total_rounds=self._total_rounds,
                                    warmup_rounds=self._warmup_rounds,
                                    class_weights=self._class_weights)
                self.farmer_nodes.append(farmer)
        else:
            indices = np.random.permutation(total_size)
            start_idx = 0
            for i in range(self.num_farmers):
                end_idx = min(start_idx + sizes[i], total_size)
                if start_idx >= total_size:
                    start_idx = 0
                    end_idx = sizes[i]
                subset_indices = indices[start_idx:end_idx]
                subset = Subset(dataset, subset_indices.tolist())
                farmer = FarmerNode(node_id=i, base_model=self.base_model, data_subset=subset,
                                    device=self.device, total_rounds=self._total_rounds,
                                    warmup_rounds=self._warmup_rounds,
                                    class_weights=self._class_weights)
                self.farmer_nodes.append(farmer)
                start_idx = end_idx

        print(f"   Farmers initialized with {min(sizes)}-{max(sizes)} samples each")

    def aggregate_shards(self, shards):
        """Aggregate shards from multiple farmers (FedAvg)."""
        aggregated = {}
        for param_name in shards[0].keys():
            stacked = torch.stack([shard[param_name] for shard in shards])
            aggregated[param_name] = stacked.mean(dim=0)
        return aggregated

    def federated_round(self, round_num):
        """Execute one round of DiLoCo federated learning."""
        print(f"\n Round {round_num + 1}")
        print("   Local training phase...")

        local_losses = []
        num_farmers = len(self.farmer_nodes)
        for i, farmer in enumerate(self.farmer_nodes):
            if num_farmers <= 20 or (i + 1) % 10 == 0 or i == 0 or i == num_farmers - 1:
                pct = ((i + 1) / num_farmers) * 100
                bar = self._make_progress_bar(i + 1, num_farmers, bar_length=15)
                print(f"\r   Training farmers: {bar} {pct:5.1f}% ({i + 1}/{num_farmers})", end="", flush=True)
            loss = farmer.local_training(self.local_steps)
            local_losses.append(loss)
        print()

        avg_loss = np.mean(local_losses)
        print(f"   Average local loss: {avg_loss:.4f}")

        # Collect and aggregate shards
        shards = [farmer.get_shard() for farmer in self.farmer_nodes]

        full_model_size = sum(p.numel() for p in self.base_model.parameters())
        shard_size = sum(p.numel() for p in shards[0].values())
        bandwidth_ratio = shard_size / full_model_size

        print(f"   Bandwidth savings: {(1-bandwidth_ratio)*100:.1f}% ({shard_size} vs {full_model_size} params)")

        aggregated_params = self.aggregate_shards(shards)
        for farmer in self.farmer_nodes:
            farmer.update_shard(aggregated_params)

        # Track metrics
        self.global_metrics['rounds'].append(round_num)
        self.global_metrics['avg_loss'].append(avg_loss)
        self.global_metrics['bandwidth_saved'].append((1-bandwidth_ratio)*100)

        # Evaluate on validation set (average metrics across all farmers)
        if self.val_loader is not None:
            criterion = nn.CrossEntropyLoss()
            all_val = [evaluate_model(f.model, self.val_loader, criterion, self.device)
                       for f in self.farmer_nodes]
            val_metrics = {
                'loss': float(np.mean([v['loss'] for v in all_val])),
                'accuracy': float(np.mean([v['accuracy'] for v in all_val])),
                'f1_macro': float(np.mean([v['f1_macro'] for v in all_val])),
            }
            self.global_metrics['val_loss'].append(val_metrics['loss'])
            self.global_metrics['val_accuracy'].append(val_metrics['accuracy'])
            self.global_metrics['val_f1_macro'].append(val_metrics['f1_macro'])
            print(f"   Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}")

        return avg_loss

    def train(self, num_rounds=10, config=None, checkpoint_rounds=None):
        """Run full federated learning training."""
        training_start_time = time.time()

        if checkpoint_rounds is None:
            checkpoint_rounds = []
        checkpoint_rounds = list(checkpoint_rounds)
        if num_rounds not in checkpoint_rounds:
            checkpoint_rounds.append(num_rounds)
        checkpoint_rounds = sorted(set(checkpoint_rounds))

        print(f"\n{'='*60}")
        print(f"DiLoCo Federated Learning Simulation")
        print(f"{'='*60}")
        print(f"Farmers: {self.num_farmers}")
        print(f"Local steps per round: {self.local_steps}")
        print(f"Total rounds: {num_rounds}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        round_times = []

        for round_num in range(num_rounds):
            round_start = time.time()
            progress_pct = (round_num / num_rounds) * 100
            progress_bar = self._make_progress_bar(round_num, num_rounds)
            print(f"\r{progress_bar} {progress_pct:5.1f}% | Round {round_num + 1}/{num_rounds}", end="", flush=True)

            loss = self.federated_round(round_num)

            round_time = time.time() - round_start
            round_times.append(round_time)
            avg_round_time = np.mean(round_times) if round_times else 0
            remaining_time = avg_round_time * (num_rounds - round_num - 1)

            print(f"   Round time: {round_time:.1f}s | Est. remaining: {remaining_time:.0f}s")

            if round_num + 1 in checkpoint_rounds:
                self.save_checkpoint(round_num + 1, config)

        total_time = time.time() - training_start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Average round time: {np.mean(round_times):.1f}s")
        print(f"{'='*60}")

        return self.global_metrics

    def save_checkpoint(self, round_num, config=None):
        """Save checkpoint including metrics and model."""
        checkpoint_dir = Path(__file__).parent.parent / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diloco_f{self.num_farmers}_r{self.total_rounds}_s{self.local_steps}_{timestamp}.pt"
        checkpoint_path = checkpoint_dir / filename

        checkpoint = {
            'global_metrics': self.global_metrics,
            'config': config,
            'num_farmers': self.num_farmers,
            'local_steps': self.local_steps,
            'total_rounds': self.total_rounds,
            'timestamp': timestamp,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    def evaluate_final(self, test_loader=None):
        """Evaluate all farmer models on test set and compute average."""
        if test_loader is None:
            if self.test_loader is None:
                raise ValueError("No test loader provided or initialized")
            test_loader = self.test_loader

        criterion = nn.CrossEntropyLoss()
        all_metrics = [evaluate_model(f.model, test_loader, criterion, self.device)
                       for f in self.farmer_nodes]

        self.test_metrics = {
            'avg_loss': float(np.mean([m['loss'] for m in all_metrics])),
            'avg_accuracy': float(np.mean([m['accuracy'] for m in all_metrics])),
            'avg_f1_macro': float(np.mean([m['f1_macro'] for m in all_metrics])),
            'std_accuracy': float(np.std([m['accuracy'] for m in all_metrics])),
            'std_f1_macro': float(np.std([m['f1_macro'] for m in all_metrics])),
            'num_farmers': len(self.farmer_nodes),
        }

        print(f"\nFinal Test Results (avg across {len(self.farmer_nodes)} farmers):")
        print(f"  Loss:       {self.test_metrics['avg_loss']:.4f}")
        print(f"  Accuracy:   {self.test_metrics['avg_accuracy']:.4f} ± {self.test_metrics['std_accuracy']:.4f}")
        print(f"  F1-Macro:   {self.test_metrics['avg_f1_macro']:.4f} ± {self.test_metrics['std_f1_macro']:.4f}")

        return self.test_metrics


# =============================================================================
# CENTRALIZED BASELINE
# =============================================================================

def train_centralized_baseline(model, train_dataset, val_dataset, test_dataset,
                               num_epochs=10, batch_size=32, device='cpu'):
    """Centralized baseline training for comparison."""
    print(f"\n{'='*60}")
    print(f"Centralized Baseline Training")
    print(f"{'='*60}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute class weights for weighted loss
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_preds, train_labels = [], []
        epoch_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_preds.extend(output.argmax(dim=1).cpu().numpy())
            train_labels.extend(target.cpu().numpy())

        train_loss = epoch_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_preds.extend(output.argmax(dim=1).cpu().numpy())
                val_labels.extend(target.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Final test evaluation
    model.eval()
    test_preds, test_labels = [], []
    test_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            test_preds.extend(output.argmax(dim=1).cpu().numpy())
            test_labels.extend(target.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)

    print(f"\nFinal Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1-Macro: {test_f1:.4f}")

    return history, {'loss': test_loss, 'accuracy': test_acc, 'f1_macro': test_f1}


# =============================================================================
# DATASET CREATION
# =============================================================================

def create_weedsgalore_dataset(img_size=64, split='train'):
    """Create WeedsGalore dataset with real labels from semantic masks."""
    from torchvision import transforms
    try:
        from .weedsgalore_loader import WeedsGaloreDataset
    except ImportError:
        from weedsgalore_loader import WeedsGaloreDataset

    # Training gets augmentation; val/test get clean transforms
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    project_root = Path(__file__).parent.parent.parent
    dataset_root = project_root / 'data/weedsgalore/weedsgalore-dataset'

    if not dataset_root.exists():
        raise FileNotFoundError(f"WeedsGalore dataset not found at: {dataset_root}")

    dataset = WeedsGaloreDataset(root_dir=dataset_root, split=split, transform=transform)
    return dataset


def create_dataset(use_real_data=None, num_samples=10000, img_size=64, num_classes=3, split='train'):
    """Create dataset - either synthetic or WeedsGalore real data."""
    from torchvision import transforms
    from torchvision.datasets import FakeData

    if use_real_data is None:
        # Try to detect if real data exists
        project_root = Path(__file__).parent.parent.parent
        dataset_root = project_root / 'data/weedsgalore/weedsgalore-dataset'
        if dataset_root.exists():
            use_real_data = True
            print(f"✅ WeedsGalore dataset found at: {dataset_root}")
        else:
            use_real_data = False
            print(f"⚠️  WeedsGalore dataset not found. Using synthetic data.")

    if use_real_data:
        return create_weedsgalore_dataset(img_size=img_size, split=split)
    else:
        # Synthetic data for quick testing
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = FakeData(size=num_samples, image_size=(3, img_size, img_size),
                           num_classes=num_classes, transform=transform)
        return dataset


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main_training(config_file=None, centralized=False, real_data=True, save_plots=True):
    """Main training function."""
    import yaml
    import argparse

    # Load config if provided
    if config_file:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'num_farmers': 10,
            'local_steps': 100,
            'total_rounds': 20,
            'warmup_rounds': 5,
            'img_size': 64,
            'batch_size': 32,
            'num_epochs_baseline': 10,
        }

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print(f"\n{'='*60}")
    print("Loading datasets...")
    train_dataset = create_dataset(use_real_data=real_data, img_size=config['img_size'], split='train')
    val_dataset = create_dataset(use_real_data=real_data, img_size=config['img_size'], split='val')
    test_dataset = create_dataset(use_real_data=real_data, img_size=config['img_size'], split='test')

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    if centralized:
        # Centralized baseline training
        print(f"\n{'='*60}")
        print("Running centralized baseline...")
        model = YOLOv11ClassificationModel(num_classes=3).to(device)
        history, test_metrics = train_centralized_baseline(
            model, train_dataset, val_dataset, test_dataset,
            num_epochs=config['num_epochs_baseline'],
            batch_size=config['batch_size'],
            device=device
        )

        # Save baseline results
        results_dir = Path(__file__).parent.parent / 'experiments' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            'config': config,
            'history': history,
            'test_metrics': test_metrics,
            'timestamp': timestamp,
            'training_type': 'centralized_baseline',
        }

        json_path = results_dir / f'centralized_baseline_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Centralized baseline results saved to: {json_path}")

        # Plot results
        if save_plots:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.plot(history['train_loss'], label='Train')
            plt.plot(history['val_loss'], label='Val')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.plot(history['train_acc'], label='Train')
            plt.plot(history['val_acc'], label='Val')
            plt.title('Accuracy')
            plt.xlabel('Epoch')

            plt.subplot(1, 3, 3)
            plt.plot(history['train_f1'], label='Train')
            plt.plot(history['val_f1'], label='Val')
            plt.title('F1-Score (Macro)')
            plt.xlabel('Epoch')

            plt.tight_layout()
            plot_path = results_dir / f'centralized_baseline_{timestamp}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📊 Plot saved to: {plot_path}")

    else:
        # Federated learning with DiLoCo
        print(f"\n{'='*60}")
        print("Running DiLoCo federated learning...")

        base_model = YOLOv11ClassificationModel(num_classes=3).to(device)
        coordinator = DiLoCoCoordinator(
            base_model,
            num_farmers=config['num_farmers'],
            local_steps=config['local_steps'],
            device=device
        )

        coordinator.initialize_farmers(
            train_dataset,
            non_iid=True,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            total_rounds=config['total_rounds'],
            warmup_rounds=config['warmup_rounds']
        )

        global_metrics = coordinator.train(
            num_rounds=config['total_rounds'],
            config=config
        )

        test_metrics = coordinator.evaluate_final()

        # Save results
        results_dir = Path(__file__).parent.parent / 'experiments' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            'config': config,
            'global_metrics': global_metrics,
            'test_metrics': test_metrics,
            'timestamp': timestamp,
            'training_type': 'diloco_federated',
            'num_farmers': config['num_farmers'],
            'local_steps': config['local_steps'],
            'total_rounds': config['total_rounds'],
        }

        json_path = results_dir / f'diloco_{config["num_farmers"]}f_{config["total_rounds"]}r_{config["local_steps"]}s_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ DiLoCo results saved to: {json_path}")

        # Plot results
        if save_plots:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(global_metrics['rounds'], global_metrics['avg_loss'])
            plt.title('Average Local Loss per Round')
            plt.xlabel('Round')
            plt.ylabel('Loss')

            plt.subplot(1, 3, 2)
            plt.plot(global_metrics['rounds'], global_metrics['val_accuracy'])
            plt.title('Validation Accuracy per Round')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')

            plt.subplot(1, 3, 3)
            plt.plot(global_metrics['rounds'], global_metrics['val_f1_macro'])
            plt.title('Validation F1-Macro per Round')
            plt.xlabel('Round')
            plt.ylabel('F1-Macro')

            plt.tight_layout()
            plot_path = results_dir / f'diloco_{config["num_farmers"]}f_{config["total_rounds"]}r_{config["local_steps"]}s_{timestamp}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📊 Plot saved to: {plot_path}")

    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"{'='*60}")

    return True


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DiLoCo Federated Learning Simulation')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--centralized', action='store_true',
                        help='Run centralized baseline instead of federated')
    parser.add_argument('--real-data', action='store_true',
                        help='Use real WeedsGalore data instead of synthetic')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable saving plots')
    parser.add_argument('--num-farmers', type=int, default=None,
                        help='Override number of farmers in config')
    parser.add_argument('--local-steps', type=int, default=None,
                        help='Override local steps per round in config')
    parser.add_argument('--total-rounds', type=int, default=None,
                        help='Override total rounds in config')

    args = parser.parse_args()

    # Override config if command-line arguments are provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    if args.num_farmers is not None:
        if config is None:
            config = {}
        config['num_farmers'] = args.num_farmers

    if args.local_steps is not None:
        if config is None:
            config = {}
        config['local_steps'] = args.local_steps

    if args.total_rounds is not None:
        if config is None:
            config = {}
        config['total_rounds'] = args.total_rounds

    success = main_training(
        config_file=args.config,
        centralized=args.centralized,
        real_data=args.real_data,
        save_plots=not args.no_plots
    )

    if not success:
        exit(1)