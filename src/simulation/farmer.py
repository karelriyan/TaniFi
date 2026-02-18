# src/simulation/farmer.py
"""Farmer node implementation for DiLoCo simulation.
Extracted from the original `diloco_trainer.py`.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .adapters import AdapterFactory
from .utils import FAST_MODE

class FarmerNode:
    """Simulates a single farmer/agent node with local training."""

    def __init__(self, node_id, base_model, data_subset, device='cpu',
                 total_rounds=20, warmup_rounds=5, class_weights=None,
                 adapter_type="lora", adapter_config=None):
        self.node_id = node_id
        self.device = device
        self.model = AdapterFactory.create_adapter(
            base_model,
            adapter_type=adapter_type,
            config=adapter_config
        ).to(device)
        # Log QLoRA activation status
        if adapter_type == "qlora":
            print("[INFO] QLoRA adapter selected. If bitsandbytes is available, model will be loaded in 4-bit mode.")
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
        # Store the data subset for coordinator aggregation
        self.data_subset = data_subset
        self.base_lr = 0.001
        # Determine trainable parameters based on adapter type
        if hasattr(self.model, "adapter"):
            trainable_params = self.model.adapter.parameters()
        else:
            # QLoRA model returned by PEFT wraps the base model; its trainable params are accessible via .parameters()
            trainable_params = self.model.parameters()
        self.optimizer = optim.AdamW(trainable_params, lr=self.base_lr, weight_decay=0.0001)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.use_amp = FAST_MODE and device == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
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

__all__ = ["FarmerNode"]
