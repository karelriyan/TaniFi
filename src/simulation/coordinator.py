# src/simulation/coordinator.py
"""Coordinator for DiLoCo federated learning simulation.
Extracted from the original `diloco_trainer.py`.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from .adapters import AdapterFactory

from .farmer import FarmerNode
from .evaluation import compute_class_weights, evaluate_model
from .utils import FAST_MODE

class DiLoCoCoordinator:
    """DiLoCo Coordinator - manages federated learning simulation."""

    def __init__(self, base_model, num_farmers=10, local_steps=100, device='cpu',
                 adapter_type="lora", adapter_config=None,
                 outer_lr: float = 1.0, mu: float = 0.9):
        if num_farmers < 1:
            raise ValueError(f"num_farmers must be >= 1, got {num_farmers}")
        if local_steps < 1:
            raise ValueError(f"local_steps must be >= 1, got {local_steps}")
        self.base_model = base_model
        self.num_farmers = num_farmers
        self.local_steps = local_steps
        self.device = device
        self.farmer_nodes = []
        # Store adapter configuration before creating global model
        self.adapter_type = adapter_type
        self.adapter_config = adapter_config or {}
        # Initialize global model with adapter to hold shared parameters
        self.global_model = AdapterFactory.create_adapter(
            base_model,
            adapter_type=self.adapter_type,
            config=self.adapter_config
        ).to(device)
        self.val_loader = None
        self.test_loader = None
        self.test_metrics = None
        # DiLoCo outer optimizer (Nesterov momentum)
        self.outer_lr = outer_lr
        self.mu = mu
        self.momentum_buffer: dict[str, torch.Tensor] = {}
        self.prev_params: dict[str, torch.Tensor] = {}

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
        """Create farmer nodes with partitioned data.
        
        When non_iid=True and the dataset has >= 4 classes (combined dataset),
        uses **Extreme Non-IID** label-based partitioning:
            - First half of farmers → weed classes only (labels 0-2)
            - Second half of farmers → disease classes only (labels 3-40)
        This creates a realistic stress-test where farmer groups have completely
        disjoint label distributions.
        """
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
        num_classes = getattr(dataset, 'num_classes', 3)

        # ── Extreme Non-IID: label-based split ──────────────────────────
        if non_iid and num_classes > 3:
            print("   📊 Using EXTREME Non-IID (label-based) data distribution")
            print("      Farmer 0-{}: Weed classes (0-2)".format(self.num_farmers // 2 - 1))
            print("      Farmer {}-{}: Disease classes (3-40)".format(self.num_farmers // 2, self.num_farmers - 1))

            # Extract all labels from the dataset
            print("   Extracting labels for partitioning...")
            all_labels = []
            for i in range(total_size):
                _, label = dataset[i]
                if isinstance(label, (list, tuple, np.ndarray)):
                    label = int(label[0]) if len(label) > 0 else 0
                elif hasattr(label, 'item'):
                    label = int(label.item()) if hasattr(label, 'numel') and label.numel() == 1 else int(label)
                else:
                    label = int(label)
                all_labels.append(label)
            all_labels = np.array(all_labels)

            # Separate indices by domain
            weed_indices = np.where(all_labels < 3)[0]       # WeedsGalore: classes 0, 1, 2
            disease_indices = np.where(all_labels >= 3)[0]    # PlantVillage: classes 3-40

            np.random.shuffle(weed_indices)
            np.random.shuffle(disease_indices)

            print(f"      Weed samples: {len(weed_indices)}, Disease samples: {len(disease_indices)}")

            # Split farmers: first half → weeds, second half → diseases
            num_weed_farmers = self.num_farmers // 2
            num_disease_farmers = self.num_farmers - num_weed_farmers

            weed_splits = np.array_split(weed_indices, num_weed_farmers)
            disease_splits = np.array_split(disease_indices, num_disease_farmers)

            farmer_indices = list(weed_splits) + list(disease_splits)

            for i in range(self.num_farmers):
                subset = Subset(dataset, farmer_indices[i].tolist())
                domain = "Weed" if i < num_weed_farmers else "Disease"
                farmer = FarmerNode(node_id=i, base_model=self.base_model, data_subset=subset,
                                    device=self.device, total_rounds=self._total_rounds,
                                    warmup_rounds=self._warmup_rounds,
                                    class_weights=self._class_weights,
                                    adapter_type=self.adapter_type,
                                    adapter_config=self.adapter_config)
                self.farmer_nodes.append(farmer)
                print(f"      Farmer {i} ({domain}): {len(subset)} samples")

        # ── Standard Non-IID or IID split ───────────────────────────────
        else:
            if non_iid:
                print("   📊 Using standard random Non-IID data distribution")
            else:
                print("   📊 Using IID (uniform) data distribution")

            min_samples_per_farmer = 1
            use_replacement = total_size < self.num_farmers * min_samples_per_farmer

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
                                        class_weights=self._class_weights,
                                        adapter_type=self.adapter_type,
                                        adapter_config=self.adapter_config)
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
                                        class_weights=self._class_weights,
                                        adapter_type=self.adapter_type,
                                        adapter_config=self.adapter_config)
                    self.farmer_nodes.append(farmer)
                    start_idx = end_idx

            print(f"   Farmers initialized with {min(sizes)}-{max(sizes)} samples each")

    def aggregate_shards(self, shards):
        """Aggregate shards from multiple farmers using weighted averaging based on each farmer's data size."""
        aggregated = {}
        total_samples = sum(len(farmer.data_subset) for farmer in self.farmer_nodes)
        for param_name in shards[0].keys():
            weighted_sum = sum(
                shards[i][param_name] * len(self.farmer_nodes[i].data_subset)
                for i in range(len(shards))
            )
            aggregated[param_name] = weighted_sum / total_samples
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

        # Aggregate shards from farmers (FedAvg)
        aggregated_params = self.aggregate_shards(shards)
        # Broadcast aggregated adapter parameters to all farmers
        for farmer in self.farmer_nodes:
            farmer.update_shard(aggregated_params)

        # Track metrics
        self.global_metrics['rounds'].append(round_num)
        self.global_metrics['avg_loss'].append(avg_loss)
        self.global_metrics['bandwidth_saved'].append((1-bandwidth_ratio)*100)
        # VRAM usage logging (in MB) after each round if CUDA is available
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            self.global_metrics.setdefault('vram_mb', []).append(vram_mb)
            torch.cuda.reset_peak_memory_stats()

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

        if 'vram_mb' in self.global_metrics:
            try:
                results_dir = Path(__file__).parent.parent.parent / 'experiments' / 'results'
                results_dir.mkdir(parents=True, exist_ok=True)
                vram_path = results_dir / 'vram_usage.json'
                with open(vram_path, 'w') as f:
                    json.dump({'vram_mb': self.global_metrics['vram_mb']}, f, indent=2)
                print(f"[INFO] VRAM usage saved to {vram_path}")
            except Exception as e:
                print(f"[WARN] Failed to save VRAM usage: {e}")

        return self.global_metrics

    def save_checkpoint(self, round_num, config=None):
        """Save checkpoint including metrics and model."""
        checkpoint_dir = Path(__file__).parent.parent / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use stored total rounds attribute (_total_rounds) for filename and metadata
        total_rounds = getattr(self, "_total_rounds", getattr(self, "total_rounds", None))
        filename = f"diloco_f{self.num_farmers}_r{total_rounds}_s{self.local_steps}_{timestamp}.pt"
        checkpoint_path = checkpoint_dir / filename

        checkpoint = {
            'global_metrics': self.global_metrics,
            'config': config,
            'num_farmers': self.num_farmers,
            'local_steps': self.local_steps,
            'total_rounds': total_rounds,
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
