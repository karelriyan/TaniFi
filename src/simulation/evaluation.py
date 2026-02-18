# src/simulation/evaluation.py
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on a dataset, returning loss, accuracy, and macro‑F1."""
    model.eval()
    all_preds, all_labels, total_loss, num_batches = [], [], 0.0, 0
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
    return {'loss': float(avg_loss), 'accuracy': float(accuracy), 'f1_macro': float(f1)}

def compute_class_weights(dataset):
    """Compute inverse‑frequency class weights for CrossEntropyLoss.
    Handles empty datasets by returning uniform weights.
    """
    from collections import Counter
    # If dataset is empty, return uniform weights based on known number of classes
    if len(dataset) == 0:
        # Attempt to infer number of classes from dataset attribute or default to 3
        num_classes = getattr(dataset, "num_classes", 3)
        weights = torch.ones(num_classes, dtype=torch.float32)
        print(f"  Dataset empty: returning uniform class weights {weights.tolist()}")
        return weights

    labels = [dataset[i][1] for i in range(len(dataset))]
    # Ensure labels are plain ints for counting
    if isinstance(labels[0], torch.Tensor):
        labels = [l.item() for l in labels]
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(counts)
    # Compute inverse frequency weights
    weights = torch.tensor(
        [total / (num_classes * counts[c]) for c in sorted(counts.keys())],
        dtype=torch.float32
    )
    print(f"  Class weights: {dict(sorted(counts.items()))} -> {weights.tolist()}")
    return weights

__all__ = ["evaluate_model", "compute_class_weights"]
