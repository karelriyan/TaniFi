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

            if round_times:
                avg_round_time = np.mean(round_times)
                remaining = num_rounds - round_num
                eta_seconds = avg_round_time * remaining
                eta_str = f"ETA: {eta_seconds/60:.1f} min" if eta_seconds > 60 else f"ETA: {eta_seconds:.0f}s"
            else:
                eta_str = "ETA: calculating..."

            print(f"\n{'='*60}")
            print(f"PROGRESS: {progress_bar} {progress_pct:.0f}% | Round {round_num + 1}/{num_rounds} | {eta_str}")
            print(f"{'='*60}")

            self.federated_round(round_num)

            round_time = time.time() - round_start
            round_times.append(round_time)
            print(f"   Round completed in {round_time:.1f}s")

            current_round = round_num + 1
            if current_round in checkpoint_rounds:
                self.execution_duration = time.time() - training_start_time
                self.save_results(config=config, checkpoint_round=current_round, total_rounds=num_rounds)
                self.plot_metrics(checkpoint_round=current_round, total_rounds=num_rounds)

        # Final test evaluation (average across all farmers)
        if self.test_loader is not None:
            criterion = nn.CrossEntropyLoss()
            all_test = [evaluate_model(f.model, self.test_loader, criterion, self.device)
                        for f in self.farmer_nodes]
            self.test_metrics = {
                'loss': float(np.mean([t['loss'] for t in all_test])),
                'accuracy': float(np.mean([t['accuracy'] for t in all_test])),
                'f1_macro': float(np.mean([t['f1_macro'] for t in all_test])),
            }
            print(f"\n{'='*60}")
            print(f"TEST RESULTS")
            print(f"{'='*60}")
            print(f"   Test Loss: {self.test_metrics['loss']:.4f}")
            print(f"   Test Accuracy: {self.test_metrics['accuracy']:.4f}")
            print(f"   Test F1 (macro): {self.test_metrics['f1_macro']:.4f}")
            print(f"{'='*60}")

        self.execution_duration = time.time() - training_start_time
        print(f"\nTraining Complete! ({self.execution_duration/60:.2f} minutes)")

        # Final save with test metrics included
        self.save_results(config=config, checkpoint_round=num_rounds, total_rounds=num_rounds)

    def save_results(self, config=None, checkpoint_round=None, total_rounds=None):
        """Save training results with comprehensive metadata."""
        results_dir = Path(__file__).parent.parent.parent / 'experiments' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_rounds = checkpoint_round if checkpoint_round else len(self.global_metrics['rounds'])

        comprehensive_results = {
            "experiment_info": {
                "experiment_id": f"diloco_{self.num_farmers}f_{self.local_steps}steps_{timestamp}",
                "timestamp": datetime.now().isoformat(),
                "description": "DiLoCo federated learning simulation for TaniFi research",
                "is_federated": True,
                "checkpoint_round": checkpoint_round,
                "total_rounds": total_rounds
            },
            "configuration": {
                "num_farmers": self.num_farmers,
                "local_steps": self.local_steps,
                "num_rounds": current_rounds,
                "total_planned_rounds": total_rounds,
                "device": str(self.device),
            },
            "execution_time_seconds": getattr(self, 'execution_duration', None),
            "execution_time_minutes": getattr(self, 'execution_duration', 0) / 60 if hasattr(self, 'execution_duration') else None,
            "environment": {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            },
            "results": {
                "metrics": {
                    "rounds": self.global_metrics['rounds'][:current_rounds],
                    "avg_loss": self.global_metrics['avg_loss'][:current_rounds],
                    "bandwidth_saved": self.global_metrics['bandwidth_saved'][:current_rounds],
                    "val_loss": self.global_metrics['val_loss'][:current_rounds],
                    "val_accuracy": self.global_metrics['val_accuracy'][:current_rounds],
                    "val_f1_macro": self.global_metrics['val_f1_macro'][:current_rounds],
                },
                "summary": {
                    "initial_loss": self.global_metrics['avg_loss'][0],
                    "final_loss": self.global_metrics['avg_loss'][current_rounds - 1],
                    "loss_reduction_percent": (
                        (self.global_metrics['avg_loss'][0] - self.global_metrics['avg_loss'][current_rounds - 1]) /
                        self.global_metrics['avg_loss'][0] * 100
                    ),
                    "avg_bandwidth_saved_percent": np.mean(self.global_metrics['bandwidth_saved'][:current_rounds]),
                    "final_val_accuracy": self.global_metrics['val_accuracy'][-1] if self.global_metrics.get('val_accuracy') else None,
                    "final_val_f1": self.global_metrics['val_f1_macro'][-1] if self.global_metrics.get('val_f1_macro') else None,
                },
                "test_metrics": self.test_metrics,
            }
        }

        if config is not None:
            comprehensive_results["full_config"] = config

        filename = f'diloco_{self.num_farmers}f_{current_rounds}r_{self.local_steps}s_{timestamp}.json'
        results_file = results_dir / filename

        # If we already saved for this round, overwrite that file
        if hasattr(self, '_last_save_round') and self._last_save_round == current_rounds and hasattr(self, '_last_save_path'):
            results_file = Path(self._last_save_path)

        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)

        self._last_save_round = current_rounds
        self._last_save_path = str(results_file)
        print(f"Results saved to: {results_file}")

    def plot_metrics(self, checkpoint_round=None, total_rounds=None):
        """Generate plots for paper."""
        current_rounds = checkpoint_round if checkpoint_round else len(self.global_metrics['rounds'])
        rounds_data = self.global_metrics['rounds'][:current_rounds]
        loss_data = self.global_metrics['avg_loss'][:current_rounds]
        bandwidth_data = self.global_metrics['bandwidth_saved'][:current_rounds]

        has_val = len(self.global_metrics.get('val_accuracy', [])) >= current_rounds

        ncols = 3 if has_val else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

        # Loss curve
        axes[0].plot(rounds_data, loss_data, marker='o', linewidth=2)
        axes[0].set_xlabel('Federated Round')
        axes[0].set_ylabel('Average Loss')
        axes[0].set_title('Training Loss ({} farmers, {} steps)'.format(self.num_farmers, self.local_steps))
        axes[0].grid(True, alpha=0.3)

        # Bandwidth savings
        axes[1].plot(rounds_data, bandwidth_data, marker='s', color='green', linewidth=2)
        axes[1].set_xlabel('Federated Round')
        axes[1].set_ylabel('Bandwidth Saved (%)')
        axes[1].set_title('Communication Efficiency')
        axes[1].grid(True, alpha=0.3)

        # Accuracy curve
        if has_val:
            acc_data = self.global_metrics['val_accuracy'][:current_rounds]
            f1_data = self.global_metrics['val_f1_macro'][:current_rounds]
            axes[2].plot(rounds_data, acc_data, marker='o', linewidth=2, label='Accuracy')
            axes[2].plot(rounds_data, f1_data, marker='s', linewidth=2, label='F1 (macro)')
            axes[2].set_xlabel('Federated Round')
            axes[2].set_ylabel('Score')
            axes[2].set_title('Validation Metrics')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        results_dir = Path(__file__).parent.parent.parent / 'experiments' / 'results'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = 'diloco_{}f_{}r_{}s_{}.png'.format(self.num_farmers, current_rounds, self.local_steps, timestamp)
        plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()


# =============================================================================
# CENTRALIZED BASELINE
# =============================================================================

def train_centralized_baseline(train_dataset, val_dataset, test_dataset,
                                num_classes=3, num_epochs=10, device='cpu',
                                config=None):
    """Train a centralized baseline (no federation) for comparison.

    Trains YOLOv11ClassificationModel on ALL training data.
    This serves as the upper-bound performance reference.
    """
    print(f"\n{'='*60}")
    print(f"Centralized Baseline Training")
    print(f"{'='*60}")
    print(f"Classes: {num_classes}, Epochs: {num_epochs}, Device: {device}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"{'='*60}\n")

    model = YOLOv11ClassificationModel(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    if use_amp:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    metrics = {
        'epochs': [], 'train_loss': [],
        'val_loss': [], 'val_accuracy': [], 'val_f1_macro': []
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses)
        val_results = evaluate_model(model, val_loader, criterion, device)

        metrics['epochs'].append(epoch)
        metrics['train_loss'].append(float(avg_train_loss))
        metrics['val_loss'].append(val_results['loss'])
        metrics['val_accuracy'].append(val_results['accuracy'])
        metrics['val_f1_macro'].append(val_results['f1_macro'])

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
              f"Val Acc: {val_results['accuracy']:.4f}, Val F1: {val_results['f1_macro']:.4f}")

    execution_time = time.time() - start_time

    # Final test evaluation
    test_results = evaluate_model(model, test_loader, criterion, device)
    print(f"\nTest - Loss: {test_results['loss']:.4f}, Acc: {test_results['accuracy']:.4f}, F1: {test_results['f1_macro']:.4f}")

    # Save results
    results = {
        "experiment_info": {
            "experiment_id": f"centralized_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "description": "Centralized baseline (no federation, no LoRA)",
            "is_federated": False,
        },
        "configuration": {
            "num_farmers": 0,
            "local_steps": 0,
            "num_rounds": num_epochs,
            "num_classes": num_classes,
            "device": str(device),
        },
        "execution_time_seconds": execution_time,
        "execution_time_minutes": execution_time / 60,
        "environment": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        },
        "results": {
            "metrics": metrics,
            "summary": {
                "initial_loss": metrics['train_loss'][0],
                "final_loss": metrics['train_loss'][-1],
                "loss_reduction_percent": (metrics['train_loss'][0] - metrics['train_loss'][-1]) / metrics['train_loss'][0] * 100,
                "avg_bandwidth_saved_percent": 0.0,
                "final_val_accuracy": metrics['val_accuracy'][-1],
                "final_val_f1": metrics['val_f1_macro'][-1],
            },
            "test_metrics": test_results,
        }
    }

    if config is not None:
        results["full_config"] = config

    results_dir = Path(__file__).parent.parent.parent / 'experiments' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'centralized_baseline_{timestamp}.json'
    with open(results_dir / filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_dir / filename}")

    return results


# =============================================================================
# DATA LOADING
# =============================================================================

USE_REAL_DATA = True

def create_dummy_dataset(num_samples=10000, img_size=64, num_classes=3):
    """Create dummy dataset for pipeline testing."""
    print("Using DUMMY dataset (random data)")
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    return torch.utils.data.TensorDataset(images, labels)


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
    dataset_root = project_root / 'data/raw/weedsgalore/weedsgalore-dataset'

    if not dataset_root.exists():
        raise FileNotFoundError(f"WeedsGalore dataset not found at: {dataset_root}")

    dataset = WeedsGaloreDataset(root_dir=dataset_root, split=split, transform=transform)
    return dataset


def create_dataset(use_real_data=None, num_samples=10000, img_size=64, num_classes=3, split='train'):
    """Main dataset creation function."""
    should_use_real = use_real_data if use_real_data is not None else USE_REAL_DATA

    if should_use_real:
        dataset = create_weedsgalore_dataset(img_size=img_size, split=split)
    else:
        dataset = create_dummy_dataset(num_samples=num_samples, img_size=img_size, num_classes=num_classes)

    return dataset


def load_config(config_path='../../experiments/config.yaml'):
    """Load configuration from YAML file."""
    import yaml
    config_file = Path(__file__).parent / config_path
    if not config_file.exists():
        print(f"Config file not found: {config_file}, using defaults")
        return None
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from: {config_file}")
    return config


# =============================================================================
# MAIN
# =============================================================================

def main(config_path='../../experiments/config.yaml', use_real_data=None, centralized=False):
    """Main simulation entry point."""
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    config = load_config(config_path)

    if config is None:
        NUM_FARMERS = 10
        LOCAL_STEPS = 100
        NUM_ROUNDS = 10
        NUM_CLASSES = 3
        IMG_SIZE = 224
        CHECKPOINT_ROUNDS = None
    else:
        NUM_FARMERS = config['federated']['num_farmers']
        LOCAL_STEPS = config['federated']['local_steps']
        NUM_ROUNDS = config['federated']['num_rounds']
        NUM_CLASSES = config['model']['num_classes']
        IMG_SIZE = config['dataset']['image_size']
        CHECKPOINT_ROUNDS = config['federated'].get('checkpoint_rounds', None)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")

    # Determine if using real data
    should_use_real = use_real_data if use_real_data is not None else USE_REAL_DATA

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = create_dataset(use_real_data=use_real_data, img_size=IMG_SIZE, num_classes=NUM_CLASSES, split='train')
    print(f"Train: {len(train_dataset)} samples")

    val_dataset = None
    test_dataset = None
    if should_use_real:
        val_dataset = create_dataset(use_real_data=True, img_size=IMG_SIZE, num_classes=NUM_CLASSES, split='val')
        test_dataset = create_dataset(use_real_data=True, img_size=IMG_SIZE, num_classes=NUM_CLASSES, split='test')
        print(f"Val: {len(val_dataset)} samples, Test: {len(test_dataset)} samples")

    # Centralized baseline mode
    if centralized:
        if val_dataset is None or test_dataset is None:
            print("ERROR: Centralized baseline requires --real-data flag")
            return
        train_centralized_baseline(
            train_dataset, val_dataset, test_dataset,
            num_classes=NUM_CLASSES, num_epochs=NUM_ROUNDS,
            device=DEVICE, config=config
        )
        return

    # Federated training
    base_model = YOLOv11ClassificationModel(num_classes=NUM_CLASSES)

    coordinator = DiLoCoCoordinator(
        base_model=base_model,
        num_farmers=NUM_FARMERS,
        local_steps=LOCAL_STEPS,
        device=DEVICE
    )

    coordinator.initialize_farmers(train_dataset, non_iid=True,
                                    val_dataset=val_dataset, test_dataset=test_dataset,
                                    total_rounds=NUM_ROUNDS, warmup_rounds=max(1, NUM_ROUNDS // 10))

    coordinator.train(num_rounds=NUM_ROUNDS, config=config, checkpoint_rounds=CHECKPOINT_ROUNDS)

    print("\nSimulation complete!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TaniFi DiLoCo Federated Learning Trainer')
    parser.add_argument('--config', type=str, default='../../experiments/config.yaml')
    parser.add_argument('--centralized', action='store_true', help='Run centralized baseline')

    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--real-data', action='store_true')
    data_group.add_argument('--dummy-data', action='store_true')

    args = parser.parse_args()

    if args.real_data:
        use_real_data = True
    elif args.dummy_data:
        use_real_data = False
    else:
        use_real_data = None

    main(config_path=args.config, use_real_data=use_real_data, centralized=args.centralized)
