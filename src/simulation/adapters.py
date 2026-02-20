try:
    import torch
except ImportError:  # pragma: no cover
    import types
    torch = types.SimpleNamespace()
    class _Module:
        def __init__(self, *args, **kwargs):
            pass
        def parameters(self):
            return []
        def to(self, *args, **kwargs):
            return self
        def train(self, mode=True):
            pass
        def eval(self):
            pass
    torch.nn = types.SimpleNamespace(Module=_Module)
    torch.device = lambda *args, **kwargs: None
# Import nn from torch; works with both real torch and stub implementation
from torch import nn
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

class AdapterFactory:
    """Factory to create LoRA or QLoRA adapters"""
    
    @staticmethod
    def create_adapter(base_model, adapter_type="lora", config=None):
        if config is None:
            config = {}
            
        if adapter_type == "lora":
            return AdapterFactory._create_lora_adapter(base_model, config)
        elif adapter_type == "qlora":
            return AdapterFactory._create_qlora_adapter(base_model, config)
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")
    
    @staticmethod
    def _create_lora_adapter(base_model, config):
        """Create standard LoRA adapter"""
        rank = config.get("rank", 4)
        return LoRAAdapter(base_model, rank=rank)
    
    @staticmethod
    def _create_qlora_adapter(base_model, config):
        """Create a QLoRA adapter.

        If *bitsandbytes* is unavailable, the function falls back to a standard
        LoRA adapter without quantization, ensuring the code runs in minimal
        environments (e.g., during CI or lightweight tests).

        Important: the base_model is deep-copied before quantization so that
        multiple farmers can each get their own independent 4-bit model without
        mutating the shared original.
        """
        import copy
        import warnings
        # If bitsandbytes is not installed, skip quantization.
        if bnb is None:
            warnings.warn(
                "bitsandbytes not available – creating a standard LoRA adapter instead of QLoRA.",
                RuntimeWarning,
            )
            # Reuse the LoRA creation logic.
            return AdapterFactory._create_lora_adapter(base_model, config)

        # Deep copy so each farmer gets fresh float32 weights to quantize
        model_copy = copy.deepcopy(base_model)

        # ------------------------------------------------------------------
        # Helper: recursively replace Linear layers with 4‑bit equivalents.
        # ------------------------------------------------------------------
        def _replace_linear(module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    new_linear = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.float16,
                        quant_type="nf4",
                    )
                    new_linear.weight = bnb.nn.Params4bit(
                        child.weight.data,
                        requires_grad=False,
                        quant_type="nf4"
                    )
                    if child.bias is not None:
                        new_linear.bias = child.bias
                    setattr(module, name, new_linear)
                else:
                    _replace_linear(child)
        
        _replace_linear(model_copy)
        
        peft_config = LoraConfig(
            r=config.get("r", 4),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=config.get("target_modules", ["classifier.2"]),
            lora_dropout=config.get("lora_dropout", 0.05),
        )
        peft_model = get_peft_model(model_copy, peft_config)
        return QLoRAAdapter(peft_model)


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


class QLoRAAdapter(nn.Module):
    """Wrapper for a PEFT‑generated QLoRA model.

    Provides ``forward``, ``get_adapter_params`` and ``set_adapter_params``
    methods compatible with the federated learning workflow.
    """
    def __init__(self, peft_model):
        super().__init__()
        self.model = peft_model

    def forward(self, x):
        return self.model(x)

    def get_adapter_params(self):
        """Return a dict of trainable LoRA parameters.

        PEFT marks LoRA weights with ``requires_grad=True`` while freezing the
        rest of the base model. We collect only those parameters.
        """
        return {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def set_adapter_params(self, params):
        """Load adapter parameters into the PEFT model.

        ``params`` is a mapping from parameter name to a tensor. Only matching
        names are updated.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(params[name])