"""
authentication_network.py

The Authentication Network is the central intelligence of the transaction
authentication system.

It receives a FeatureVector (identity, voice biometrics, behavior, vehicle
context, history, and transaction features -- all pre-validated, normalized,
and encoded) and predicts, in a single forward pass:

    - trust_score      : how authentic the user appears to be   (0-1)
    - risk_score        : how risky the transaction is           (0-1)
    - decision_logits   : unnormalized scores over 4 actions
    - confidence        : how confident the model is in itself   (0-1)
    - embedding          : the 128-dim shared representation, exposed for
                            explainability / downstream anomaly detection

This module has a single responsibility: learn from features and produce
predictions. It does NOT perform feature extraction, rule checking, or
policy enforcement -- that is the job of the downstream Risk / Policy /
Decision engines.

Architecture
------------
    Feature Vector
          |
          v
    Feature Attention Layer     (learns per-feature importance gates)
          |
          v
    Feature Projection Layer
          |
          v
    Shared Residual Encoder (GELU + LayerNorm)
          |
          v
    Shared Embedding (128-dim)
      |         |
      v         v
    Trust     (embedding)
      |          |
      +--> Risk Head (consumes embedding + trust_score)
      |
      +--> Decision Head (deeper: embedding -> hidden -> hidden/2 -> logits)
      |
      +--> Confidence Head (consumes embedding + trust + risk + decision_logits)

All hyperparameters live in ModelConfig / config.yaml -- nothing is
hardcoded in the modules below.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("authentication_network")
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Decision mapping
# --------------------------------------------------------------------------- #


class Decision(IntEnum):
    """Discrete authentication actions predicted by the Decision Head."""

    ALLOW = 0
    VOICE_CHALLENGE = 1
    VOICE_AND_OTP = 2
    REJECT = 3


DECISION_LABELS = {
    Decision.ALLOW: "ALLOW",
    Decision.VOICE_CHALLENGE: "VOICE_CHALLENGE",
    Decision.VOICE_AND_OTP: "VOICE_AND_OTP",
    Decision.REJECT: "REJECT",
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    """
    All architecture, loss, optimizer, scheduler, training, and inference
    hyperparameters in one place. Load from config.yaml with
    ModelConfig.from_yaml("config.yaml") rather than hardcoding values.
    """

    # -- identity / versioning --
    version: str = "1.0.0"

    # -- architecture --
    num_features: int = 31
    attention_dim: int = 31
    proj_dim: int = 256
    embedding_dim: int = 128
    head_hidden_dim: int = 64
    decision_hidden_dim: int = 96
    num_decision_classes: int = 4
    dropout: float = 0.3
    layer_norm_eps: float = 1e-5

    # -- loss --
    label_smoothing: float = 0.05
    use_uncertainty_weighting: bool = True
    trust_weight: float = 0.25
    risk_weight: float = 0.25
    decision_weight: float = 0.40
    confidence_weight: float = 0.10

    # -- optimizer --
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # -- scheduler --
    scheduler_type: str = "cosine_annealing"
    t_max: int = 100
    eta_min: float = 0.0

    # -- training --
    max_epochs: int = 100
    batch_size: int = 256
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./runs"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "auth-network"
    mixed_precision: bool = True

    # -- inference --
    mc_dropout_samples: int = 20
    deep_ensemble_size: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load hyperparameters from a config.yaml file (flat-merged)."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load config.yaml (pip install pyyaml)"
            ) from exc

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        flat: Dict = {}
        for section in ("model", "loss", "optimizer", "scheduler", "training", "inference"):
            flat.update(raw.get(section, {}))

        # a couple of keys are renamed between the YAML layout and the
        # flat dataclass for clarity
        if "type" in raw.get("scheduler", {}):
            flat["scheduler_type"] = raw["scheduler"]["type"]
            flat.pop("type", None)

        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in flat.items() if k in known_fields}
        return cls(**filtered)

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)


# --------------------------------------------------------------------------- #
# Output containers
# --------------------------------------------------------------------------- #


@dataclass
class Prediction:
    """
    Typed output of the Authentication Network.

    All tensors carry a leading batch dimension, e.g. shape (batch_size,)
    for scalar heads, (batch_size, 4) for decision logits, and
    (batch_size, embedding_dim) for the shared embedding.
    """

    trust_score: torch.Tensor          # (batch_size,)
    risk_score: torch.Tensor           # (batch_size,)
    decision_logits: torch.Tensor      # (batch_size, num_classes)
    confidence: torch.Tensor           # (batch_size,)
    embedding: torch.Tensor            # (batch_size, embedding_dim) -- explainability
    feature_attention: Optional[torch.Tensor] = None  # (batch_size, num_features)

    @property
    def decision_probability(self) -> torch.Tensor:
        """Softmax over decision_logits, computed lazily (not stored)."""
        return F.softmax(self.decision_logits, dim=-1)

    @property
    def decision(self) -> torch.Tensor:
        """Argmax decision index per sample, shape (batch_size,)."""
        return torch.argmax(self.decision_logits, dim=-1)

    def to(self, device: torch.device) -> "Prediction":
        return Prediction(
            trust_score=self.trust_score.to(device),
            risk_score=self.risk_score.to(device),
            decision_logits=self.decision_logits.to(device),
            confidence=self.confidence.to(device),
            embedding=self.embedding.to(device),
            feature_attention=(
                self.feature_attention.to(device) if self.feature_attention is not None else None
            ),
        )


@dataclass
class UncertainPrediction:
    """
    Output of predict_with_uncertainty: mean prediction plus per-head
    epistemic uncertainty (std) estimated via MC Dropout and/or a Deep
    Ensemble.
    """

    mean: Prediction
    trust_std: torch.Tensor
    risk_std: torch.Tensor
    decision_probability_std: torch.Tensor
    confidence_std: torch.Tensor
    num_samples: int


@dataclass
class AuthenticationLabels:
    """Ground-truth labels required to train the network end to end."""

    trust_score: torch.Tensor    # (batch_size,) float in [0, 1]
    risk_score: torch.Tensor     # (batch_size,) float in [0, 1]
    decision: torch.Tensor       # (batch_size,) long, class index 0-3
    confidence: torch.Tensor     # (batch_size,) float in [0, 1]

    def to(self, device: torch.device) -> "AuthenticationLabels":
        return AuthenticationLabels(
            trust_score=self.trust_score.to(device),
            risk_score=self.risk_score.to(device),
            decision=self.decision.to(device),
            confidence=self.confidence.to(device),
        )


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class FeatureAttention(nn.Module):
    """
    Learns a per-feature importance gate before projection, so the network
    can down-weight noisy or less-informative features and up-weight
    discriminative ones. Implemented as a lightweight squeeze-and-excite
    style gate (cheaper and more stable at small batch sizes than full
    self-attention over individual scalar features):

        x -> Linear -> GELU -> Linear -> Sigmoid -> gate
        out = x * gate

    The gate is also returned so it can be inspected for explainability
    (which features the model attended to for a given sample).
    """

    def __init__(self, num_features: int, attention_dim: Optional[int] = None):
        super().__init__()
        attention_dim = attention_dim or num_features
        self.net = nn.Sequential(
            nn.Linear(num_features, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, num_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = self.net(x)
        return x * gate, gate


class ProjectionLayer(nn.Module):
    """
    Stage 1: projects the raw (attention-gated) feature vector into a
    common embedding space.

        Input -> Linear -> LayerNorm -> GELU -> Dropout

    LayerNorm + GELU (instead of BatchNorm + ReLU) gives stable statistics
    regardless of batch size, including batch_size == 1 at inference.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float, eps: float):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features, eps=eps)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ResidualBlock(nn.Module):
    """
    A single residual block used by the Shared Feature Encoder.

        x -> Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm
          -> (+x) -> GELU
    """

    def __init__(self, dim: int, dropout: float, eps: float):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.act_out = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = out + residual
        out = self.act_out(out)
        return out


class ResidualEncoder(nn.Module):
    """
    Stage 2: Shared Feature Encoder made of stacked residual blocks,
    producing the "Transaction Authenticity Representation" embedding
    that every head consumes.
    """

    def __init__(self, dim: int, num_blocks: int, dropout: float, eps: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualBlock(dim, dropout=dropout, eps=eps) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TrustHead(nn.Module):
    """
    Embedding -> Linear -> GELU -> Dropout -> Linear -> Sigmoid

    Produces a single scalar trust score in [0, 1] per sample. Trust is
    computed first because the Risk Head consumes it downstream.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float, eps: float):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = self.linear1(embedding)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        return torch.sigmoid(x).squeeze(-1)


class RiskHead(nn.Module):
    """
    Risk is correlated with trust, so this head consumes the shared
    embedding *concatenated with the predicted trust score*:

        [Embedding ; trust_score] -> Linear -> GELU -> Dropout -> Linear -> Sigmoid
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float, eps: float):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim + 1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, embedding: torch.Tensor, trust_score: torch.Tensor) -> torch.Tensor:
        x = torch.cat([embedding, trust_score.unsqueeze(-1)], dim=-1)
        x = self.linear1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        return torch.sigmoid(x).squeeze(-1)


class DecisionHead(nn.Module):
    """
    Deepened decision head (one extra hidden layer versus a plain 2-layer
    MLP), giving the model more capacity to separate the four discrete
    authentication actions:

        Embedding -> Linear -> LayerNorm -> GELU -> Dropout
                  -> Linear -> LayerNorm -> GELU -> Dropout
                  -> Linear -> num_classes logits

    Returns raw logits; softmax is applied outside the forward pass
    (exposed via Prediction.decision_probability).
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        eps: float,
    ):
        super().__init__()
        hidden_dim_2 = max(hidden_dim // 2, num_classes * 2)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim_2)
        self.norm2 = nn.LayerNorm(hidden_dim_2, eps=eps)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.linear_out = nn.Linear(hidden_dim_2, num_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = self.linear1(embedding)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.linear2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop2(x)

        return self.linear_out(x)


class ConfidenceHead(nn.Module):
    """
    Confidence should reflect not just the embedding but how the other
    three heads behaved (e.g. a near-uniform decision distribution or a
    trust/risk disagreement should lower confidence). Consumes:

        [Embedding ; trust_score ; risk_score ; decision_logits]
        -> Linear -> GELU -> Dropout -> Linear -> Sigmoid
    """

    def __init__(
        self, embedding_dim: int, num_decision_classes: int, hidden_dim: int,
        dropout: float, eps: float,
    ):
        super().__init__()
        in_dim = embedding_dim + 1 + 1 + num_decision_classes
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        embedding: torch.Tensor,
        trust_score: torch.Tensor,
        risk_score: torch.Tensor,
        decision_logits: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [embedding, trust_score.unsqueeze(-1), risk_score.unsqueeze(-1), decision_logits],
            dim=-1,
        )
        x = self.linear1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        return torch.sigmoid(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# Weight initialization
# --------------------------------------------------------------------------- #


def _init_weights(module: nn.Module) -> None:
    """Kaiming Normal init for Linear layers (fan-in, relu-family gain), zero bias."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# --------------------------------------------------------------------------- #
# Authentication Network
# --------------------------------------------------------------------------- #


class AuthenticationNetwork(nn.Module):
    """
    Multi-task network mapping a FeatureVector to (trust, risk, decision,
    confidence, embedding) predictions. All dimensions come from a
    ModelConfig instance -- nothing is hardcoded.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Feature Attention Layer -- learns which input features matter most.
        self.feature_attention = FeatureAttention(
            config.num_features, attention_dim=config.attention_dim
        )

        # Stage 1: Feature Projection Layer
        self.projection = ProjectionLayer(
            config.num_features, config.proj_dim, dropout=config.dropout, eps=config.layer_norm_eps
        )

        # Bridge from projection width down to embedding width before the
        # residual encoder (Input -> proj_dim -> proj_dim -> embedding_dim).
        self.encoder_in = nn.Sequential(
            nn.Linear(config.proj_dim, config.proj_dim),
            nn.LayerNorm(config.proj_dim, eps=config.layer_norm_eps),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.to_embedding = nn.Sequential(
            nn.Linear(config.proj_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps),
            nn.GELU(),
        )

        # Stage 2: Shared Residual Encoder, operating at proj_dim width.
        self.residual_encoder = ResidualEncoder(
            config.proj_dim, num_blocks=2, dropout=config.dropout, eps=config.layer_norm_eps
        )

        # Heads (Trust is computed first; Risk and Confidence depend on it).
        self.trust_head = TrustHead(
            config.embedding_dim, config.head_hidden_dim, config.dropout, config.layer_norm_eps
        )
        self.risk_head = RiskHead(
            config.embedding_dim, config.head_hidden_dim, config.dropout, config.layer_norm_eps
        )
        self.decision_head = DecisionHead(
            config.embedding_dim,
            config.decision_hidden_dim,
            config.num_decision_classes,
            config.dropout,
            config.layer_norm_eps,
        )
        self.confidence_head = ConfidenceHead(
            config.embedding_dim,
            config.num_decision_classes,
            config.head_hidden_dim,
            config.dropout,
            config.layer_norm_eps,
        )

        self.apply(_init_weights)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the shared trunk and return (embedding, feature_attention_gate)."""
        x, attention_gate = self.feature_attention(x)
        x = self.projection(x)
        x = self.encoder_in(x)
        x = self.residual_encoder(x)
        embedding = self.to_embedding(x)
        return embedding, attention_gate

    def forward(self, x: torch.Tensor) -> Prediction:
        """
        Args:
            x: Tensor of shape (batch_size, num_features). Features must
               already be validated, normalized, and encoded.

        Returns:
            Prediction dataclass with trust_score, risk_score,
            decision_logits, confidence, embedding, and feature_attention.
        """
        embedding, attention_gate = self.encode(x)

        trust_score = self.trust_head(embedding)
        risk_score = self.risk_head(embedding, trust_score)
        decision_logits = self.decision_head(embedding)
        confidence = self.confidence_head(embedding, trust_score, risk_score, decision_logits)

        return Prediction(
            trust_score=trust_score,
            risk_score=risk_score,
            decision_logits=decision_logits,
            confidence=confidence,
            embedding=embedding,
            feature_attention=attention_gate,
        )


# --------------------------------------------------------------------------- #
# Loss -- fixed weights or learnable (Kendall et al.) uncertainty weighting
# --------------------------------------------------------------------------- #


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task loss with learnable homoscedastic uncertainty weighting
    (Kendall, Gal & Cipolla, 2018 -- "Multi-Task Learning Using
    Uncertainty to Weigh Losses"):

        L = sum_i [ exp(-log_var_i) * L_i + log_var_i ]

    Each task learns its own log-variance, so tasks the model is
    intrinsically less certain about are automatically down-weighted,
    removing the need to hand-tune fixed loss weights. Falls back to
    fixed, configurable weights when config.use_uncertainty_weighting is
    False.
    """

    NUM_TASKS = 4  # trust, risk, decision, confidence

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.use_uncertainty_weighting:
            # log_var initialized to 0 => initial precision of 1.0 per task
            self.log_vars = nn.Parameter(torch.zeros(self.NUM_TASKS))
        else:
            self.register_parameter("log_vars", None)

    def forward(
        self, prediction: Prediction, labels: AuthenticationLabels
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        trust_loss = F.binary_cross_entropy(prediction.trust_score, labels.trust_score)
        risk_loss = F.binary_cross_entropy(prediction.risk_score, labels.risk_score)
        decision_loss = F.cross_entropy(
            prediction.decision_logits,
            labels.decision,
            label_smoothing=self.config.label_smoothing,
        )
        confidence_loss = F.binary_cross_entropy(prediction.confidence, labels.confidence)

        task_losses = [trust_loss, risk_loss, decision_loss, confidence_loss]

        if self.config.use_uncertainty_weighting:
            total = torch.zeros((), device=trust_loss.device, dtype=trust_loss.dtype)
            for log_var, task_loss in zip(self.log_vars, task_losses):
                precision = torch.exp(-log_var)
                total = total + precision * task_loss + log_var
        else:
            weights = [
                self.config.trust_weight,
                self.config.risk_weight,
                self.config.decision_weight,
                self.config.confidence_weight,
            ]
            total = sum(w * l for w, l in zip(weights, task_losses))

        breakdown = {
            "trust_loss": trust_loss.item(),
            "risk_loss": risk_loss.item(),
            "decision_loss": decision_loss.item(),
            "confidence_loss": confidence_loss.item(),
            "total_loss": total.item(),
        }
        if self.config.use_uncertainty_weighting:
            breakdown["log_vars"] = [v.item() for v in self.log_vars.detach()]

        return total, breakdown


# --------------------------------------------------------------------------- #
# Optimizer / scheduler factory
# --------------------------------------------------------------------------- #


def build_optimizer(
    model: nn.Module, loss_fn: nn.Module, config: ModelConfig
) -> torch.optim.AdamW:
    """
    AdamW optimizer over both the model parameters and (if enabled) the
    learnable uncertainty-weighting log-variances.
    """
    params: List[torch.nn.Parameter] = list(model.parameters())
    if isinstance(loss_fn, UncertaintyWeightedLoss) and loss_fn.log_vars is not None:
        params += [loss_fn.log_vars]
    return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: ModelConfig
) -> torch.optim.lr_scheduler._LRScheduler:
    """Scheduler factory, currently supports CosineAnnealingLR (per spec)."""
    if config.scheduler_type == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.t_max, eta_min=config.eta_min
        )
    raise ValueError(f"Unsupported scheduler_type: {config.scheduler_type}")


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #


class EarlyStopping:
    """Stops training when validation loss stops improving by min_delta."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# --------------------------------------------------------------------------- #
# Checkpointing / model versioning
# --------------------------------------------------------------------------- #


def save_checkpoint(
    model: AuthenticationNetwork,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    is_best: bool = False,
) -> str:
    """
    Saves model + optimizer + loss-fn state, config, epoch, and metrics.
    Filenames are versioned by config.version and epoch, e.g.:
        checkpoints/auth_net_v1.0.0_epoch012.pt
        checkpoints/auth_net_v1.0.0_best.pt
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    version = model.config.version
    filename = f"auth_net_v{version}_epoch{epoch:03d}.pt"
    path = str(Path(checkpoint_dir) / filename)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_fn_state_dict": loss_fn.state_dict(),
        "config": dataclasses.asdict(model.config),
        "metrics": metrics,
        "timestamp": time.time(),
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint: %s", path)

    if is_best:
        best_path = str(Path(checkpoint_dir) / f"auth_net_v{version}_best.pt")
        torch.save(payload, best_path)
        logger.info("Saved new best checkpoint: %s", best_path)

    return path


def load_checkpoint(
    path: str, device: Optional[torch.device] = None
) -> Tuple[AuthenticationNetwork, Dict]:
    """Loads a checkpoint and reconstructs the model with its original config."""
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device)
    config = ModelConfig(**payload["config"])
    model = AuthenticationNetwork(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, payload


# --------------------------------------------------------------------------- #
# Experiment logging (TensorBoard / Weights & Biases)
# --------------------------------------------------------------------------- #


class ExperimentLogger:
    """
    Thin wrapper around TensorBoard and/or Weights & Biases so the training
    loop stays agnostic to which backend(s) are enabled. Both are optional
    dependencies; missing packages degrade to a no-op with a warning.
    """

    def __init__(self, config: ModelConfig, run_name: Optional[str] = None):
        self.config = config
        self.run_name = run_name or f"auth_net_v{config.version}_{int(time.time())}"
        self._tb_writer = None
        self._wandb = None

        if config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                log_dir = str(Path(config.log_dir) / self.run_name)
                self._tb_writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                logger.warning("TensorBoard not available; skipping TB logging.")

        if config.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=config.wandb_project,
                    name=self.run_name,
                    config=dataclasses.asdict(config),
                )
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not available; skipping W&B logging.")

    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        for key, value in metrics.items():
            if isinstance(value, (list, tuple)):
                continue  # e.g. per-task log_vars; skip scalar-only backends
            tag = f"{prefix}/{key}" if prefix else key
            if self._tb_writer is not None:
                self._tb_writer.add_scalar(tag, value, step)
            if self._wandb is not None:
                self._wandb.log({tag: value}, step=step)

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb is not None:
            self._wandb.finish()


# --------------------------------------------------------------------------- #
# Training step (mixed precision + gradient clipping)
# --------------------------------------------------------------------------- #


def train_step(
    model: AuthenticationNetwork,
    loss_fn: UncertaintyWeightedLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    features: torch.Tensor,
    labels: AuthenticationLabels,
    device: torch.device,
    grad_clip_norm: Optional[float] = 1.0,
) -> Dict[str, float]:
    """
    Single training step: forward -> multi-task loss -> backward ->
    gradient clipping -> optimizer step, under torch.cuda.amp mixed
    precision.

    Returns the loss breakdown dict for logging.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    features = features.to(device)
    labels = labels.to(device)

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        prediction = model(features)
        loss, breakdown = loss_fn(prediction, labels)

    scaler.scale(loss).backward()

    if grad_clip_norm is not None:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

    scaler.step(optimizer)
    scaler.update()

    return breakdown


@torch.no_grad()
def eval_step(
    model: AuthenticationNetwork,
    loss_fn: UncertaintyWeightedLoss,
    features: torch.Tensor,
    labels: AuthenticationLabels,
    device: torch.device,
) -> Dict[str, float]:
    """Single validation step (no gradient updates)."""
    model.eval()
    features = features.to(device)
    labels = labels.to(device)
    prediction = model(features)
    _, breakdown = loss_fn(prediction, labels)
    return breakdown


# --------------------------------------------------------------------------- #
# Full training pipeline
# --------------------------------------------------------------------------- #


def train(
    model: AuthenticationNetwork,
    config: ModelConfig,
    train_batches: Sequence[Tuple[torch.Tensor, AuthenticationLabels]],
    val_batches: Sequence[Tuple[torch.Tensor, AuthenticationLabels]],
    device: Optional[torch.device] = None,
) -> AuthenticationNetwork:
    """
    Production-style training loop wiring together: AdamW + cosine
    annealing, mixed precision, gradient clipping, early stopping,
    checkpointing (including a running "best" checkpoint), and
    TensorBoard/W&B logging. Batches are pre-built (features, labels)
    pairs so this function stays agnostic to the Dataset/DataLoader
    implementation used upstream.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = UncertaintyWeightedLoss(config).to(device)
    optimizer = build_optimizer(model, loss_fn, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.mixed_precision and device.type == "cuda"))
    early_stopping = EarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta)
    exp_logger = ExperimentLogger(config)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, config.max_epochs + 1):
        epoch_train_losses: List[float] = []
        for features, labels in train_batches:
            breakdown = train_step(
                model, loss_fn, optimizer, scaler, features, labels, device,
                grad_clip_norm=config.grad_clip_norm,
            )
            epoch_train_losses.append(breakdown["total_loss"])
            exp_logger.log_scalars(breakdown, step=global_step, prefix="train")
            global_step += 1

        scheduler.step()

        epoch_val_losses = [
            eval_step(model, loss_fn, features, labels, device)["total_loss"]
            for features, labels in val_batches
        ]
        mean_val_loss = sum(epoch_val_losses) / max(len(epoch_val_losses), 1)
        mean_train_loss = sum(epoch_train_losses) / max(len(epoch_train_losses), 1)

        exp_logger.log_scalars(
            {"epoch_train_loss": mean_train_loss, "epoch_val_loss": mean_val_loss,
             "lr": scheduler.get_last_lr()[0]},
            step=epoch, prefix="epoch",
        )
        logger.info(
            "epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch, config.max_epochs, mean_train_loss, mean_val_loss,
        )

        is_best = mean_val_loss < best_val_loss
        if is_best:
            best_val_loss = mean_val_loss
        save_checkpoint(
            model, optimizer, loss_fn, epoch,
            metrics={"train_loss": mean_train_loss, "val_loss": mean_val_loss},
            checkpoint_dir=config.checkpoint_dir, is_best=is_best,
        )

        if early_stopping.step(mean_val_loss):
            logger.info("Early stopping triggered at epoch %d (best_val_loss=%.4f)",
                         epoch, early_stopping.best_loss)
            break

    exp_logger.close()
    return model


# --------------------------------------------------------------------------- #
# Deterministic inference
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict(
    model: AuthenticationNetwork,
    features: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Prediction:
    """
    Deterministic inference entry point (dropout disabled).

    Feature Vector -> Authentication Network -> Prediction Object,
    to be consumed downstream by the Risk / Policy / Decision engines.
    """
    device = device or next(model.parameters()).device
    model.eval()
    features = features.to(device)
    prediction = model(features)
    return prediction.to(device)


# --------------------------------------------------------------------------- #
# Calibrated inference: MC Dropout / Deep Ensembles
# --------------------------------------------------------------------------- #


def _enable_mc_dropout(model: AuthenticationNetwork) -> None:
    """Put the model in eval() but re-enable Dropout layers for MC sampling."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def predict_with_uncertainty(
    models: Sequence[AuthenticationNetwork],
    features: torch.Tensor,
    mc_samples: int = 20,
    device: Optional[torch.device] = None,
) -> UncertainPrediction:
    """
    Calibrated inference combining Monte Carlo Dropout and (optionally) a
    Deep Ensemble:

      - Pass a single model to get MC-Dropout uncertainty only
        (mc_samples stochastic forward passes with dropout active).
      - Pass multiple independently-trained models to additionally get
        Deep Ensemble uncertainty (each ensemble member also runs
        mc_samples MC-Dropout passes).

    Returns an UncertainPrediction with the mean Prediction plus the
    per-head standard deviation across all samples, which downstream
    Policy/Risk engines can use to fall back to a conservative decision
    (e.g. VOICE_AND_OTP) when the model is uncertain.
    """
    assert len(models) >= 1, "predict_with_uncertainty requires at least one model"
    device = device or next(models[0].parameters()).device
    features = features.to(device)

    trust_samples: List[torch.Tensor] = []
    risk_samples: List[torch.Tensor] = []
    decision_prob_samples: List[torch.Tensor] = []
    confidence_samples: List[torch.Tensor] = []
    last_prediction: Optional[Prediction] = None

    for m in models:
        m.to(device)
        _enable_mc_dropout(m)
        for _ in range(mc_samples):
            pred = m(features)
            trust_samples.append(pred.trust_score)
            risk_samples.append(pred.risk_score)
            decision_prob_samples.append(pred.decision_probability)
            confidence_samples.append(pred.confidence)
            last_prediction = pred

    trust_stack = torch.stack(trust_samples, dim=0)          # (N, batch)
    risk_stack = torch.stack(risk_samples, dim=0)             # (N, batch)
    decision_prob_stack = torch.stack(decision_prob_samples, dim=0)  # (N, batch, classes)
    confidence_stack = torch.stack(confidence_samples, dim=0)  # (N, batch)

    mean_pred = Prediction(
        trust_score=trust_stack.mean(dim=0),
        risk_score=risk_stack.mean(dim=0),
        decision_logits=torch.log(decision_prob_stack.mean(dim=0).clamp_min(1e-8)),
        confidence=confidence_stack.mean(dim=0),
        embedding=last_prediction.embedding if last_prediction is not None else torch.empty(0),
        feature_attention=(
            last_prediction.feature_attention if last_prediction is not None else None
        ),
    )

    return UncertainPrediction(
        mean=mean_pred,
        trust_std=trust_stack.std(dim=0),
        risk_std=risk_stack.std(dim=0),
        decision_probability_std=decision_prob_stack.std(dim=0),
        confidence_std=confidence_stack.std(dim=0),
        num_samples=trust_stack.shape[0],
    )


class DeepEnsemble:
    """
    Convenience wrapper holding several independently-trained
    AuthenticationNetwork instances (e.g. different seeds / data splits)
    for use with predict_with_uncertainty.
    """

    def __init__(self, models: Sequence[AuthenticationNetwork]):
        assert len(models) >= 1
        self.models = list(models)

    @classmethod
    def from_checkpoints(cls, paths: Sequence[str], device: Optional[torch.device] = None) -> "DeepEnsemble":
        models = [load_checkpoint(p, device=device)[0] for p in paths]
        return cls(models)

    def predict_with_uncertainty(
        self, features: torch.Tensor, mc_samples: int = 20,
        device: Optional[torch.device] = None,
    ) -> UncertainPrediction:
        return predict_with_uncertainty(self.models, features, mc_samples=mc_samples, device=device)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(0)

    config_path = (Path(__file__).resolve().parent.parent/ "config"/ "auth"/ "config.yaml")
    if config_path.exists():
        config = ModelConfig.from_yaml(str(config_path))
    else:
        config = ModelConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AuthenticationNetwork(config).to(device)

    batch_size = 8
    dummy_features = torch.randn(batch_size, config.num_features)
    dummy_labels = AuthenticationLabels(
        trust_score=torch.rand(batch_size),
        risk_score=torch.rand(batch_size),
        decision=torch.randint(0, config.num_decision_classes, (batch_size,)),
        confidence=torch.rand(batch_size),
    )

    # one training step
    loss_fn = UncertaintyWeightedLoss(config).to(device)
    optimizer = build_optimizer(model, loss_fn, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.mixed_precision and device.type == "cuda"))

    breakdown = train_step(
        model, loss_fn, optimizer, scaler, dummy_features, dummy_labels, device,
        grad_clip_norm=config.grad_clip_norm,
    )
    scheduler.step()
    print("train breakdown:", breakdown)

    # deterministic inference
    result = predict(model, dummy_features, device=device)
    print("trust_score:  ", result.trust_score)
    print("risk_score:    ", result.risk_score)
    print("decision:      ", [DECISION_LABELS[Decision(d.item())] for d in result.decision])
    print("confidence:    ", result.confidence)
    print("embedding shape:", result.embedding.shape)

    # calibrated (MC Dropout) inference
    uncertain = predict_with_uncertainty([model], dummy_features, mc_samples=10, device=device)
    print("trust_std (MC Dropout):", uncertain.trust_std)
    print("risk_std (MC Dropout): ", uncertain.risk_std)