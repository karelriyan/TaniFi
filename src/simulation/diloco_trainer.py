"""
DiLoCo (Distributed Low-Communication) Simulation for TaniFi

This script implements a simplified DiLoCo federated learning simulation
for bandwidth-constrained agricultural networks.

Key Features:
- Local training for 500 steps before synchronization
- LoRA adapters for efficient parameter updates
- Simulates 100 farmer nodes with varying connectivity

Paper: "Simulation of Bandwidth-Efficient Federated Learning Architectures 
        for Resource-Constrained Agricultural Networks in Indonesia"

Usage:
    python diloco_trainer.py --num-farmers 100 --local-steps 500
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


class SimpleCropDiseaseModel(nn.Module):
    """
    Simplified crop disease detection model for simulation
    
    In production, this would be replaced with YOLOv11 + LoRA adapters
    For simulation, we use a simple CNN to demonstrate DiLoCo principles
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Simple CNN backbone (representing YOLOv11 backbone)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LoRAAdapter(nn.Module):
    """
    LoRA (Low-Rank Adaptation) adapter for efficient fine-tuning
    
    This represents the "Shard" that each farmer owns and trains locally
    Only these small adapters are transmitted, not the full model
    """
    def __init__(self, base_model, rank=4):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Create low-rank adaptation matrices
        # In practice, these would be applied to specific layers
        # For simulation, we add a simple adapter layer
        self.adapter = nn.Sequential(
            nn.Linear(128, rank),
            nn.ReLU(),
            nn.Linear(rank, 128)
        )
    
    def forward(self, x):
        # Forward through base model
        features = self.base_model.features(x)
        
        # Apply LoRA adapter
        adapted_features = features + self.adapter(features.flatten(1)).view_as(features)
        
        # Final classification
        output = self.base_model.classifier(adapted_features)
        return output
    
    def get_adapter_params(self):
        """Get only the adapter parameters (the 'shard')"""
        return {name: param.clone().detach() 
                for name, param in self.adapter.named_parameters()}
    
    def set_adapter_params(self, params):
        """Set adapter parameters from received 'shard'"""
        with torch.no_grad():
            for name, param in self.adapter.named_parameters():
                param.copy_(params[name])


# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================
# These settings control training speed. Adjust based on your hardware.
#
# FAST_MODE: Enable all speed optimizations (recommended for testing)
# - Larger batch size (64 vs 8)
# - More data loading workers (4 vs 0)
# - Mixed precision training (FP16) on GPU
# - cuDNN auto-tuning on GPU
# - TF32 for Ampere+ GPUs
# - Pin memory for faster GPU transfer
#
# Set FAST_MODE = False to simulate real low-end device constraints
# =============================================================================
FAST_MODE = True  # <-- Set to False for realistic simulation


class FarmerNode:
    """
    Simulates a single farmer/agent node

    Represents a farmer with:
    - Low-end Android phone (simulated resource constraints)
    - Limited internet connectivity (intermittent syncs)
    - Local dataset (unique crop varieties/conditions)
    """
    def __init__(self, node_id, base_model, data_subset, device='cpu'):
        self.node_id = node_id
        self.device = device

        # Create LoRA adapter for this farmer
        self.model = LoRAAdapter(base_model, rank=4).to(device)

        # Performance settings based on FAST_MODE
        if FAST_MODE:
            batch_size = 64      # Larger batch = faster (uses more GPU memory)
            num_workers = 4      # Parallel data loading
            pin_memory = device != 'cpu'  # Faster GPU transfer
        else:
            batch_size = 8       # Small batch size for 2GB RAM constraint
            num_workers = 0      # No parallel loading (phone constraint)
            pin_memory = False

        # Local dataset (non-IID distribution)
        self.dataloader = DataLoader(
            data_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # Local optimizer
        self.optimizer = optim.Adam(
            self.model.adapter.parameters(),
            lr=0.001
        )

        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision scaler for FAST_MODE
        self.use_amp = FAST_MODE and device == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        # Training metrics
        self.local_losses = []
        self.sync_count = 0

    def local_training(self, num_steps=500):
        """
        Perform local training for specified number of steps

        This simulates the farmer's phone training in the background
        without internet connection (offline-first architecture)
        """
        self.model.train()
        step = 0
        epoch_losses = []

        pbar = tqdm(total=num_steps, desc=f"Node {self.node_id} training",
                   leave=False, disable=True)
        
        while step < num_steps:
            for batch_idx, (data, target) in enumerate(self.dataloader):
                if step >= num_steps:
                    break

                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

                if self.use_amp:
                    # Mixed precision training (faster on modern GPUs)
                    with torch.amp.autocast('cuda'):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision training
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()

                epoch_losses.append(loss.item())
                step += 1
                pbar.update(1)

        pbar.close()

        avg_loss = np.mean(epoch_losses)
        self.local_losses.append(avg_loss)

        return avg_loss
    
    def get_shard(self):
        """
        Extract the trained 'shard' (LoRA adapter parameters)
        
        This is what gets transmitted to the network (very small, ~KB size)
        instead of the full model (would be MBs)
        """
        return self.model.get_adapter_params()
    
    def update_shard(self, aggregated_params):
        """Receive and apply aggregated parameters from global update"""
        self.model.set_adapter_params(aggregated_params)
        self.sync_count += 1


class DiLoCoCoordinator:
    """
    DiLoCo Coordinator - manages federated learning simulation
    
    Responsibilities:
    - Orchestrate local training rounds
    - Aggregate shards from farmers
    - Validate contributions (Proof of Learning)
    - Distribute updated global model
    """
    def __init__(self, base_model, num_farmers=100, local_steps=500, device='cpu'):
        self.base_model = base_model
        self.num_farmers = num_farmers
        self.local_steps = local_steps
        self.device = device
        
        self.farmer_nodes = []
        self.global_model = base_model.to(device)
        
        # Tracking metrics
        self.global_metrics = {
            'rounds': [],
            'avg_loss': [],
            'bandwidth_saved': []
        }

    def _make_progress_bar(self, current, total, bar_length=20):
        """Create a text-based progress bar."""
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        return f"[{bar}]"

    def initialize_farmers(self, dataset, non_iid=True):
        """
        Create farmer nodes with partitioned data

        non_iid: If True, each farmer gets different data distribution
                 (simulates different crop types/regions)

        Note: If dataset is smaller than num_farmers, samples will be shared
              (with replacement) to ensure each farmer has data.
        """
        print(f"\n🌾 Initializing {self.num_farmers} farmer nodes...")

        # Partition dataset among farmers
        total_size = len(dataset)

        # Handle small datasets: ensure minimum samples per farmer
        min_samples_per_farmer = 1
        if total_size < self.num_farmers * min_samples_per_farmer:
            print(f"   ⚠️  Small dataset ({total_size} samples) for {self.num_farmers} farmers")
            print(f"   ⚠️  Using sampling WITH REPLACEMENT (samples shared between farmers)")
            use_replacement = True
        else:
            use_replacement = False

        if non_iid:
            # Non-IID: Random sizes, simulating real-world conditions
            # Ensure each farmer gets at least min_samples_per_farmer
            sizes = np.random.randint(
                min_samples_per_farmer,
                max(min_samples_per_farmer + 1, total_size // self.num_farmers * 3),
                self.num_farmers
            )
            # Normalize to total dataset size (or allow overlap for small datasets)
            if use_replacement:
                # For small datasets, just use the random sizes as-is (will sample with replacement)
                sizes = np.maximum(sizes, min_samples_per_farmer)
            else:
                sizes = (sizes / sizes.sum() * total_size).astype(int)
                sizes = np.maximum(sizes, min_samples_per_farmer)  # Ensure minimum
        else:
            # IID: Equal distribution
            samples_per_farmer = max(min_samples_per_farmer, total_size // self.num_farmers)
            sizes = [samples_per_farmer] * self.num_farmers

        # Create subsets
        if use_replacement:
            # Small dataset: sample with replacement
            for i in range(self.num_farmers):
                # Randomly sample indices (with replacement)
                subset_indices = np.random.choice(total_size, size=sizes[i], replace=True)
                subset = Subset(dataset, subset_indices.tolist())

                farmer = FarmerNode(
                    node_id=i,
                    base_model=self.base_model,
                    data_subset=subset,
                    device=self.device
                )
                self.farmer_nodes.append(farmer)
        else:
            # Normal case: partition without replacement
            indices = np.random.permutation(total_size)
            start_idx = 0

            for i in range(self.num_farmers):
                end_idx = min(start_idx + sizes[i], total_size)
                if start_idx >= total_size:
                    # Wrap around if we run out of indices
                    start_idx = 0
                    end_idx = sizes[i]
                subset_indices = indices[start_idx:end_idx]
                subset = Subset(dataset, subset_indices.tolist())

                farmer = FarmerNode(
                    node_id=i,
                    base_model=self.base_model,
                    data_subset=subset,
                    device=self.device
                )
                self.farmer_nodes.append(farmer)
                start_idx = end_idx

        print(f"   ✅ Farmers initialized with {min(sizes)}-{max(sizes)} samples each")
        if use_replacement:
            print(f"   ℹ️  Note: Samples are shared between farmers due to small dataset")
    
    def aggregate_shards(self, shards):
        """
        Aggregate shards from multiple farmers
        
        Simple averaging for now - can be replaced with:
        - Weighted averaging (based on data size)
        - FedProx (proximal term)
        - Byzantine-robust aggregation
        """
        aggregated = {}
        
        # Average all parameters
        for param_name in shards[0].keys():
            stacked = torch.stack([shard[param_name] for shard in shards])
            aggregated[param_name] = stacked.mean(dim=0)
        
        return aggregated
    
    def federated_round(self, round_num):
        """
        Execute one round of DiLoCo federated learning
        
        Steps:
        1. Each farmer trains locally for `local_steps`
        2. Farmers transmit their shards
        3. Coordinator aggregates shards
        4. Updated parameters distributed back to farmers
        """
        print(f"\n🔄 Round {round_num + 1}")
        print("   📱 Local training phase...")

        # Step 1: Local training (parallel in real system)
        local_losses = []
        num_farmers = len(self.farmer_nodes)
        for i, farmer in enumerate(self.farmer_nodes):
            # Show progress every 10 farmers or for small numbers
            if num_farmers <= 20 or (i + 1) % 10 == 0 or i == 0 or i == num_farmers - 1:
                pct = ((i + 1) / num_farmers) * 100
                bar = self._make_progress_bar(i + 1, num_farmers, bar_length=15)
                print(f"\r   Training farmers: {bar} {pct:5.1f}% ({i + 1}/{num_farmers})", end="", flush=True)
            loss = farmer.local_training(self.local_steps)
            local_losses.append(loss)
        print()  # New line after progress
        
        avg_loss = np.mean(local_losses)
        print(f"   📊 Average local loss: {avg_loss:.4f}")
        
        # Step 2: Collect shards
        print("   📡 Collecting shards from farmers...")
        shards = [farmer.get_shard() for farmer in self.farmer_nodes]
        
        # Calculate bandwidth savings
        # Full model size vs shard size
        full_model_size = sum(p.numel() for p in self.base_model.parameters())
        shard_size = sum(p.numel() for p in shards[0].values())
        bandwidth_ratio = shard_size / full_model_size
        
        print(f"   💾 Bandwidth savings: {(1-bandwidth_ratio)*100:.1f}% "
              f"({shard_size} vs {full_model_size} parameters)")
        
        # Step 3: Aggregate
        print("   🔗 Aggregating shards...")
        aggregated_params = self.aggregate_shards(shards)
        
        # Step 4: Distribute
        print("   📤 Distributing updated model...")
        for farmer in self.farmer_nodes:
            farmer.update_shard(aggregated_params)
        
        # Track metrics
        self.global_metrics['rounds'].append(round_num)
        self.global_metrics['avg_loss'].append(avg_loss)
        self.global_metrics['bandwidth_saved'].append((1-bandwidth_ratio)*100)
        
        return avg_loss
    
    def train(self, num_rounds=10, config=None, checkpoint_rounds=None):
        """Run full federated learning training with time tracking and checkpoints

        Args:
            num_rounds: Total number of federated rounds to run
            config: Configuration dictionary for metadata
            checkpoint_rounds: List of rounds at which to save intermediate results (e.g., [5, 10])
        """

        # Start time tracking
        training_start_time = time.time()

        # Ensure checkpoint_rounds is a list and includes final round
        if checkpoint_rounds is None:
            checkpoint_rounds = []
        checkpoint_rounds = list(checkpoint_rounds)  # Make a copy
        if num_rounds not in checkpoint_rounds:
            checkpoint_rounds.append(num_rounds)
        checkpoint_rounds = sorted(set(checkpoint_rounds))  # Remove duplicates and sort

        print(f"\n{'='*60}")
        print(f"DiLoCo Federated Learning Simulation")
        print(f"{'='*60}")
        print(f"Farmers: {self.num_farmers}")
        print(f"Local steps per round: {self.local_steps}")
        print(f"Total rounds: {num_rounds}")
        print(f"Checkpoint rounds: {checkpoint_rounds}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Track round times for ETA estimation
        round_times = []

        for round_num in range(num_rounds):
            round_start = time.time()

            # Calculate and display progress
            progress_pct = (round_num / num_rounds) * 100
            progress_bar = self._make_progress_bar(round_num, num_rounds)

            # Estimate remaining time
            if round_times:
                avg_round_time = np.mean(round_times)
                remaining_rounds = num_rounds - round_num
                eta_seconds = avg_round_time * remaining_rounds
                eta_str = f"ETA: {eta_seconds/60:.1f} min" if eta_seconds > 60 else f"ETA: {eta_seconds:.0f}s"
            else:
                eta_str = "ETA: calculating..."

            print(f"\n{'='*60}")
            print(f"📊 PROGRESS: {progress_bar} {progress_pct:.0f}% | Round {round_num + 1}/{num_rounds} | {eta_str}")
            print(f"{'='*60}")

            self.federated_round(round_num)

            # Track round time
            round_time = time.time() - round_start
            round_times.append(round_time)
            print(f"   ⏱️  Round completed in {round_time:.1f}s")

            # Check if we should save a checkpoint (round_num is 0-indexed, so +1)
            current_round = round_num + 1
            if current_round in checkpoint_rounds:
                checkpoint_time = time.time()
                checkpoint_duration = checkpoint_time - training_start_time
                self.execution_duration = checkpoint_duration

                print(f"\n💾 Saving checkpoint at round {current_round}...")
                self.save_results(config=config, checkpoint_round=current_round, total_rounds=num_rounds)
                self.plot_metrics(checkpoint_round=current_round, total_rounds=num_rounds)

        # Final progress
        print(f"\n{'='*60}")
        print(f"📊 PROGRESS: {self._make_progress_bar(num_rounds, num_rounds)} 100% | COMPLETE!")
        print(f"{'='*60}")

        # End time tracking
        training_end_time = time.time()
        execution_duration = training_end_time - training_start_time

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Total execution time: {execution_duration:.2f} seconds ({execution_duration/60:.2f} minutes)")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Checkpoints saved: {checkpoint_rounds}")
        print(f"{'='*60}\n")
    
    def save_results(self, config=None, checkpoint_round=None, total_rounds=None):
        """Save training results with comprehensive metadata for paper

        Args:
            config: Configuration dictionary for metadata
            checkpoint_round: Current checkpoint round (for intermediate saves)
            total_rounds: Total number of rounds in the experiment
        """
        results_dir = Path('../../experiments/results')  # Fixed: Added one more ../
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine the round number for the filename
        current_rounds = checkpoint_round if checkpoint_round else len(self.global_metrics['rounds'])
        is_checkpoint = checkpoint_round is not None and total_rounds is not None and checkpoint_round < total_rounds

        # Prepare comprehensive results with metadata
        comprehensive_results = {
            "experiment_info": {
                "experiment_id": f"diloco_{self.num_farmers}f_{self.local_steps}steps_{timestamp}",
                "timestamp": datetime.now().isoformat(),
                "description": "DiLoCo federated learning simulation for TaniFi research",
                "is_checkpoint": is_checkpoint,
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
                    "bandwidth_saved": self.global_metrics['bandwidth_saved'][:current_rounds]
                },
                "summary": {
                    "initial_loss": self.global_metrics['avg_loss'][0],
                    "final_loss": self.global_metrics['avg_loss'][current_rounds - 1],
                    "loss_reduction_percent": (
                        (self.global_metrics['avg_loss'][0] - self.global_metrics['avg_loss'][current_rounds - 1]) /
                        self.global_metrics['avg_loss'][0] * 100
                    ),
                    "avg_bandwidth_saved_percent": sum(self.global_metrics['bandwidth_saved'][:current_rounds]) / current_rounds
                }
            }
        }

        # Include full config if provided
        if config is not None:
            comprehensive_results["full_config"] = config

        # Generate descriptive filename with farmers, rounds, and steps
        # Format: diloco_100f_5r_500s_20250128_105318.json (for checkpoint at round 5)
        filename = f'diloco_{self.num_farmers}f_{current_rounds}r_{self.local_steps}s_{timestamp}.json'
        results_file = results_dir / filename

        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)

        checkpoint_label = f" (checkpoint {checkpoint_round}/{total_rounds})" if is_checkpoint else ""
        print(f"📄 Results saved to: {results_file}{checkpoint_label}")
        print(f"   Experiment ID: {comprehensive_results['experiment_info']['experiment_id']}")
        print(f"   Duration: {comprehensive_results.get('execution_time_minutes', 'N/A'):.2f} minutes" if comprehensive_results.get('execution_time_minutes') else "   Duration: N/A")
    
    def plot_metrics(self, checkpoint_round=None, total_rounds=None):
        """Generate plots for paper with configuration info

        Args:
            checkpoint_round: Current checkpoint round (for intermediate saves)
            total_rounds: Total number of rounds in the experiment
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Determine the number of rounds to plot
        current_rounds = checkpoint_round if checkpoint_round else len(self.global_metrics['rounds'])

        # Get data up to current checkpoint
        rounds_data = self.global_metrics['rounds'][:current_rounds]
        loss_data = self.global_metrics['avg_loss'][:current_rounds]
        bandwidth_data = self.global_metrics['bandwidth_saved'][:current_rounds]

        # Loss curve
        axes[0].plot(rounds_data, loss_data, marker='o', linewidth=2)
        axes[0].set_xlabel('Federated Round')
        axes[0].set_ylabel('Average Loss')
        # Use .format() to avoid f-string interpolation issues
        axes[0].set_title('DiLoCo Training Loss\n({} farmers, {} rounds, {} steps)'.format(
            self.num_farmers, current_rounds, self.local_steps))
        axes[0].grid(True, alpha=0.3)

        # Bandwidth savings
        axes[1].plot(rounds_data, bandwidth_data, marker='s', color='green', linewidth=2)
        axes[1].set_xlabel('Federated Round')
        axes[1].set_ylabel('Bandwidth Saved (%)')
        # Use .format() to avoid f-string interpolation issues
        axes[1].set_title('Communication Efficiency\n({} farmers, {} rounds, {} steps)'.format(
            self.num_farmers, current_rounds, self.local_steps))
        axes[1].grid(True, alpha=0.3)

        avg_bw = np.mean(bandwidth_data)
        axes[1].axhline(y=avg_bw,
                       color='red', linestyle='--',
                       label='Avg: {:.1f}%'.format(avg_bw))
        axes[1].legend()

        plt.tight_layout()

        results_dir = Path('../../experiments/results')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Include configuration in filename
        filename = 'diloco_{}f_{}r_{}s_{}.png'.format(
            self.num_farmers, current_rounds, self.local_steps, timestamp)
        plot_file = results_dir / filename

        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to: {plot_file}")
        plt.close()


# =============================================================================
# DATA LOADING SECTION
# =============================================================================
# This section handles dataset loading for the DiLoCo federated learning simulation.
#
# HOW TO USE:
# -----------
# 1. To use DUMMY data (for quick testing without real dataset):
#    - Set USE_REAL_DATA = False
#    - Dummy data generates random images, useful for testing the training pipeline
#
# 2. To use WEEDSGALORE data (for actual research experiments):
#    - Set USE_REAL_DATA = True
#    - Ensure WeedsGalore dataset is downloaded to:
#      data/raw/weedsgalore/weedsgalore-dataset/
#    - The dataset will be loaded with proper multispectral image handling (R,G,B channels)
#
# DATASET STRUCTURE (WeedsGalore):
# --------------------------------
# data/raw/weedsgalore/weedsgalore-dataset/
# ├── 2023-05-25/
# │   ├── images/
# │   │   ├── {image_id}_R.png  (Red channel)
# │   │   ├── {image_id}_G.png  (Green channel)
# │   │   ├── {image_id}_B.png  (Blue channel)
# │   │   ├── {image_id}_NIR.png (Near-Infrared - not used for RGB)
# │   │   └── {image_id}_RE.png  (Red Edge - not used for RGB)
# │   ├── instances/
# │   └── semantics/
# ├── 2023-05-30/
# ├── 2023-06-06/
# ├── 2023-06-15/
# └── splits/
#     ├── train.txt
#     ├── val.txt
#     └── test.txt
#
# TO CHANGE DATASET:
# ------------------
# If you want to use a different dataset:
# 1. Create a new loader file in src/simulation/ (e.g., my_dataset_loader.py)
# 2. Implement a Dataset class with __len__ and __getitem__ methods
# 3. Import it below and modify create_dataset() function
# 4. Ensure your dataset returns (image_tensor, label) tuples
#
# REPRODUCIBILITY NOTES:
# ----------------------
# - Random seed is set in main() for reproducibility
# - Non-IID data partitioning uses numpy random, seeded for consistency
# - Image transforms are deterministic (resize, normalize)
# =============================================================================

# Configuration flag: Set to True to use WeedsGalore, False for dummy data
USE_REAL_DATA = False  # <-- CHANGE THIS TO SWITCH BETWEEN DUMMY/REAL DATA


def create_dummy_dataset(num_samples=10000, img_size=64, num_classes=10):
    """
    Create dummy dataset for initial testing and pipeline validation.

    This function generates random image tensors and labels for testing
    the federated learning pipeline without requiring the actual dataset.

    Args:
        num_samples: Number of random samples to generate (default: 10000)
        img_size: Image dimensions (img_size x img_size) (default: 64)
        num_classes: Number of classification classes (default: 10)

    Returns:
        torch.utils.data.TensorDataset: Dataset with random images and labels

    Usage:
        dataset = create_dummy_dataset(num_samples=5000, img_size=64, num_classes=10)
    """
    print("⚠️  Using DUMMY dataset (random data for testing)")
    print(f"   Samples: {num_samples}, Image size: {img_size}x{img_size}, Classes: {num_classes}")

    # Generate random images (batch, channels, height, width)
    images = torch.randn(num_samples, 3, img_size, img_size)

    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))

    dataset = torch.utils.data.TensorDataset(images, labels)
    return dataset


def create_weedsgalore_dataset(img_size=64, split='train'):
    """
    Create WeedsGalore dataset for actual research experiments.

    This function loads the WeedsGalore agricultural dataset which contains
    multispectral drone images of crops and weeds. The loader combines
    R, G, B channels into standard RGB images for the CNN model.

    Args:
        img_size: Target image size after resize (default: 64)
        split: Dataset split to load - 'train', 'val', or 'test' (default: 'train')

    Returns:
        WeedsGaloreDataset: Dataset with real agricultural images

    Requires:
        - WeedsGalore dataset downloaded to data/raw/weedsgalore/weedsgalore-dataset/
        - weedsgalore_loader.py in the same directory

    Usage:
        dataset = create_weedsgalore_dataset(img_size=64, split='train')
    """
    from torchvision import transforms
    # Import WeedsGalore loader - handles both direct execution and module import
    try:
        from .weedsgalore_loader import WeedsGaloreDataset
    except ImportError:
        from weedsgalore_loader import WeedsGaloreDataset

    print(f"✅ Using WEEDSGALORE dataset (real agricultural data)")
    print(f"   Split: {split}, Image size: {img_size}x{img_size}")

    # Define image transforms
    # These transforms ensure consistent input to the neural network
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize to target size
        transforms.ToTensor(),                     # Convert PIL to tensor [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization (standard)
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Path to dataset (relative to this file)
    project_root = Path(__file__).parent.parent.parent
    dataset_root = project_root / 'data/raw/weedsgalore/weedsgalore-dataset'

    # Verify dataset exists
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"WeedsGalore dataset not found at: {dataset_root}\n"
            f"Please download the dataset or set USE_REAL_DATA = False for dummy data."
        )

    # Create and return dataset
    dataset = WeedsGaloreDataset(
        root_dir=dataset_root,
        split=split,
        transform=transform
    )

    return dataset


def create_dataset(use_real_data=None, num_samples=10000, img_size=64, num_classes=10):
    """
    Main dataset creation function - routes to dummy or real data based on config.

    This is the primary function called by the training pipeline. It checks
    the USE_REAL_DATA flag (or override parameter) and returns the appropriate
    dataset.

    Args:
        use_real_data: Override for USE_REAL_DATA flag (None uses global setting)
        num_samples: Number of samples for dummy dataset (ignored for real data)
        img_size: Target image size (used for both dummy and real)
        num_classes: Number of classes for dummy dataset (ignored for real data)

    Returns:
        Dataset: Either TensorDataset (dummy) or WeedsGaloreDataset (real)

    Example:
        # Use global setting (USE_REAL_DATA flag)
        dataset = create_dataset(img_size=64)

        # Force dummy data regardless of flag
        dataset = create_dataset(use_real_data=False, num_samples=5000)

        # Force real data regardless of flag
        dataset = create_dataset(use_real_data=True)
    """
    # Determine which dataset to use
    should_use_real = use_real_data if use_real_data is not None else USE_REAL_DATA

    print("\n" + "="*60)
    print("DATASET LOADING")
    print("="*60)

    if should_use_real:
        # Load WeedsGalore dataset
        dataset = create_weedsgalore_dataset(img_size=img_size, split='train')
    else:
        # Use dummy dataset for testing
        dataset = create_dummy_dataset(
            num_samples=num_samples,
            img_size=img_size,
            num_classes=num_classes
        )

    print(f"   Total samples: {len(dataset)}")
    print("="*60 + "\n")

    return dataset


def load_config(config_path='../../experiments/config.yaml'):
    """Load configuration from YAML file"""
    import yaml
    from pathlib import Path
    
    config_file = Path(__file__).parent / config_path
    
    if not config_file.exists():
        print(f"⚠️  Config file not found: {config_file}")
        print(f"   Using default parameters...")
        return None
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Loaded config from: {config_file}")
    return config


def main(config_path='../../experiments/config.yaml', use_real_data=None):
    """
    Main simulation entry point.

    Args:
        config_path: Path to YAML configuration file
        use_real_data: Override for dataset selection (None uses USE_REAL_DATA flag)
    """
    # =========================================================================
    # REPRODUCIBILITY
    # =========================================================================
    # Set random seeds for reproducible experiments
    # Change RANDOM_SEED for different random initializations
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"🎲 Random seed set to: {RANDOM_SEED}")

    # =========================================================================
    # PERFORMANCE OPTIMIZATIONS
    # =========================================================================
    if FAST_MODE:
        print(f"⚡ FAST_MODE enabled - using performance optimizations")
        print(f"   - Batch size: 64")
        print(f"   - Data loading workers: 4")
        if torch.cuda.is_available():
            # Enable cuDNN auto-tuning (finds fastest algorithms)
            torch.backends.cudnn.benchmark = True
            # Enable TF32 for faster matrix operations on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"   - cuDNN benchmark: ON")
            print(f"   - TF32 (Ampere+ GPUs): ON")
            print(f"   - Mixed precision (FP16): ON")
            print(f"   - Pin memory: ON")
        else:
            print(f"   - GPU optimizations: OFF (no CUDA)")
    else:
        print(f"🐢 FAST_MODE disabled - simulating low-end device constraints")
        print(f"   - Batch size: 8")
        print(f"   - Data loading workers: 0")

    # Load configuration
    config = load_config(config_path)
    
    if config is None:
        # Fallback to default values
        NUM_FARMERS = 100
        LOCAL_STEPS = 500
        NUM_ROUNDS = 5
        NUM_CLASSES = 10
        BATCH_SIZE = 8
        LEARNING_RATE = 0.001
        IMG_SIZE = 64
        CHECKPOINT_ROUNDS = None  # No checkpoints by default
    else:
        # Extract parameters from config
        NUM_FARMERS = config['federated']['num_farmers']
        LOCAL_STEPS = config['federated']['local_steps']
        NUM_ROUNDS = config['federated']['num_rounds']
        NUM_CLASSES = config['model']['num_classes']
        BATCH_SIZE = config['training']['batch_size']
        LEARNING_RATE = config['training']['learning_rate']
        IMG_SIZE = config['dataset']['image_size']
        # Get checkpoint_rounds from config (optional)
        CHECKPOINT_ROUNDS = config['federated'].get('checkpoint_rounds', None)

    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"🖥️  Using device: {DEVICE}")
    print(f"📊 Configuration:")
    print(f"   Farmers: {NUM_FARMERS}")
    print(f"   Local Steps: {LOCAL_STEPS}")
    print(f"   Rounds: {NUM_ROUNDS}")
    print(f"   Checkpoint Rounds: {CHECKPOINT_ROUNDS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    
    # Create base model (Global model that farmers will adapt)
    print("\n🧠 Creating base model...")
    base_model = SimpleCropDiseaseModel(num_classes=NUM_CLASSES)
    
    # =========================================================================
    # DATASET LOADING
    # =========================================================================
    # This section loads the dataset for training. The dataset can be:
    # - Dummy data: Random tensors for testing (USE_REAL_DATA = False)
    # - WeedsGalore: Real agricultural images (USE_REAL_DATA = True)
    #
    # To switch between datasets, modify USE_REAL_DATA at the top of this file.
    # =========================================================================
    print("\n📦 Loading dataset...")
    dataset = create_dataset(
        use_real_data=use_real_data,  # None = use global USE_REAL_DATA flag
        num_samples=10000,             # Only used for dummy data
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES
    )
    
    # Initialize DiLoCo coordinator
    coordinator = DiLoCoCoordinator(
        base_model=base_model,
        num_farmers=NUM_FARMERS,
        local_steps=LOCAL_STEPS,
        device=DEVICE
    )
    
    # Initialize farmer nodes
    coordinator.initialize_farmers(dataset, non_iid=True)
    
    # Run federated training with config metadata and checkpoints
    coordinator.train(num_rounds=NUM_ROUNDS, config=config, checkpoint_rounds=CHECKPOINT_ROUNDS)
    
    print("\n✅ Simulation complete!")
    print("\nNext steps:")
    print("1. To use WeedsGalore data: Set USE_REAL_DATA = True at top of this file")
    print("2. Integrate YOLOv11 + LoRA adapters")
    print("3. Add blockchain integration (Base L2)")
    print("4. Analyze results for paper")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='TaniFi DiLoCo Federated Learning Trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with dummy data (default)
  python diloco_trainer.py --dummy-data

  # Run with WeedsGalore real data
  python diloco_trainer.py --real-data

  # Run with specific config file
  python diloco_trainer.py --config path/to/config.yaml

Dataset Selection:
  The dataset can be selected in two ways:
  1. Command line: --real-data or --dummy-data flags
  2. Code: Set USE_REAL_DATA = True/False at top of this file

  Command line flags override the code setting.
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../../experiments/config.yaml',
        help='Path to config YAML file'
    )

    # Dataset selection arguments (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        '--real-data',
        action='store_true',
        help='Use WeedsGalore real agricultural dataset'
    )
    data_group.add_argument(
        '--dummy-data',
        action='store_true',
        help='Use dummy random data for testing (default)'
    )

    args = parser.parse_args()

    # Determine use_real_data value
    # Priority: command line flag > USE_REAL_DATA global variable
    if args.real_data:
        use_real_data = True
    elif args.dummy_data:
        use_real_data = False
    else:
        use_real_data = None  # Use global USE_REAL_DATA flag

    main(config_path=args.config, use_real_data=use_real_data)