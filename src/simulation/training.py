# src/simulation/training.py
"""Training utilities for TaniFi simulation.
Contains the centralized baseline training and the main training entry point.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from .model import YOLOv11ClassificationModel
from .data import create_dataset
from .evaluation import compute_class_weights, evaluate_model
from .farmer import FarmerNode
from .coordinator import DiLoCoCoordinator
from .utils import FAST_MODE


from .adapters import AdapterFactory

def train_centralized_baseline(model, train_dataset, val_dataset, test_dataset,
                               num_epochs=10, batch_size=32, device='cpu',
                               adapter_type=None, adapter_config=None):
    """Centralized baseline training for comparison."""
    print(f"\n{'='*60}")
    print(f"Centralized Baseline Training")
    if adapter_type:
        print(f"Adapter Type: {adapter_type}")
    print(f"{'='*60}")

    if adapter_type:
        print(f"Wrapping model with {adapter_type} adapter...")
        model = AdapterFactory.create_adapter(model, adapter_type=adapter_type, config=adapter_config).to(device)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute class weights for weighted loss
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimize only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=0.001, weight_decay=0.0001)
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
        for data, target in train_loader:
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
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

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
    print(f"\nFinal Test Results:\n  Loss: {test_loss:.4f}\n  Accuracy: {test_acc:.4f}\n  F1-Macro: {test_f1:.4f}")

    return history, {'loss': test_loss, 'accuracy': test_acc, 'f1_macro': test_f1}

def main_training(config_file=None, centralized=False, real_data=True, save_plots=True, adapter_type=None):
    """Main training function – orchestrates dataset creation, model init, and training."""
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
        # Support nested federated configuration
        federated_cfg = config.get('federated', {})
        config['num_farmers'] = federated_cfg.get('num_farmers', config['num_farmers'])
        config['local_steps'] = federated_cfg.get('local_steps', config['local_steps'])
        config['total_rounds'] = federated_cfg.get('num_rounds', config['total_rounds'])
        config['warmup_rounds'] = federated_cfg.get('warmup_rounds', config['warmup_rounds'])

    # Extract federated settings, supporting both flat and nested configuration structures
    federated_cfg = config.get('federated', {})
    config['num_farmers'] = federated_cfg.get('num_farmers', config.get('num_farmers', 10))
    config['local_steps'] = federated_cfg.get('local_steps', config.get('local_steps', 100))
    config['total_rounds'] = federated_cfg.get('num_rounds', config.get('total_rounds', 20))
    config['warmup_rounds'] = federated_cfg.get('warmup_rounds', config.get('warmup_rounds', 5))
    
    # Determine adapter type: CLI arg > config > default 'lora'
    if adapter_type is None:
        adapter_type = config.get('adapter_type', 'lora')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Create datasets
    print("\n" + "="*60)
    print("Loading datasets...")
    # Determine image size: prefer explicit 'img_size' if present, otherwise use the dataset configuration
    img_size = config.get('img_size')
    dataset_config = config.get('dataset', {})
    if img_size is None:
        img_size = dataset_config.get('image_size', 224)
    
    # Determine dataset name
    dataset_name = dataset_config.get('name')
    if dataset_name is None:
        if real_data:
            dataset_name = 'weedsgalore'
        else:
            dataset_name = 'synthetic'

    print(f"Dataset: {dataset_name}, Image Size: {img_size}")

    train_dataset = create_dataset(dataset_name=dataset_name, img_size=img_size, split='train')
    val_dataset = create_dataset(dataset_name=dataset_name, img_size=img_size, split='val')
    test_dataset = create_dataset(dataset_name=dataset_name, img_size=img_size, split='test')
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    # Determine number of classes
    if hasattr(train_dataset, 'num_classes'):
        num_classes = train_dataset.num_classes
    else:
        num_classes = config.get('num_classes', 3)
    print(f"Num Classes: {num_classes}")

    if centralized:
        # Centralized baseline
        model = YOLOv11ClassificationModel(num_classes=num_classes).to(device)
        history, test_metrics = train_centralized_baseline(
            model, train_dataset, val_dataset, test_dataset,
            num_epochs=config.get('num_epochs_baseline', 10),
            batch_size=config.get('batch_size', 32),
            device=device,
            adapter_type=adapter_type,
            adapter_config=config.get('adapter_config', {})
        )
        # Save results
        results_dir = Path(__file__).parent.parent / 'experiments' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'config': config,
            'history': history,
            'test_metrics': test_metrics,
            'timestamp': timestamp,
            'training_type': 'centralized_baseline',
            'adapter_type': adapter_type,
        }
        json_path = results_dir / f'centralized_baseline_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Centralized baseline results saved to: {json_path}")
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
        # Federated DiLoCo training
        model = YOLOv11ClassificationModel(num_classes=num_classes).to(device)
        coordinator = DiLoCoCoordinator(
            base_model=model,
            num_farmers=config.get('num_farmers', 10),
            local_steps=config.get('local_steps', 100),
            device=device,
            adapter_type=adapter_type,
            adapter_config=config.get('adapter_config', {}),
            outer_lr=1.0,
            mu=0.9,
        )
        coordinator.initialize_farmers(
            train_dataset,
            non_iid=True,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            total_rounds=config.get('total_rounds', 20),
            warmup_rounds=config.get('warmup_rounds', 5),
        )
        metrics = coordinator.train(num_rounds=config.get('total_rounds', 20), config=config)
        # Save federated results
        results_dir = Path(__file__).parent.parent / 'experiments' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'config': config,
            'metrics': metrics,
            'timestamp': timestamp,
            'training_type': 'federated_diLoCo',
            'adapter_type': adapter_type,
        }
        json_path = results_dir / f'federated_diLoCo_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Federated training results saved to: {json_path}")
        if save_plots:
            # Plotting of global metrics can be added here if desired
            pass
