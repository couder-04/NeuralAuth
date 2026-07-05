"""
training/evaluate.py

Evaluation script for the Authentication Network.

Evaluates a trained checkpoint on the validation/test dataset and reports:

- Total Loss
- Trust Loss
- Risk Loss
- Confidence Loss
- Decision Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The network architecture and loss implementation are imported directly
from authentication_network.py.
"""

from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from torch.utils.data import DataLoader

from engines.authentication_network import (
    AuthenticationLabels,
    UncertaintyWeightedLoss,
    AuthenticationNetwork,
    ModelConfig,
    load_checkpoint,
)

from training.dataset import AuthenticationDataset


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "auth" / "config.yaml"

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "auth_net_v1.0.0_best.pt"
)

DATASET = (
    PROJECT_ROOT
    / "data"
    / "authentication_dataset.csv"
)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

@torch.no_grad()
def evaluate():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --------------------------------------------------------

    config = ModelConfig.from_yaml(CONFIG_PATH)

    model, _ = load_checkpoint(
        CHECKPOINT,
        device=device,
    )

    model.eval()

    loss_fn = UncertaintyWeightedLoss(config).to(device)

    dataset = AuthenticationDataset(DATASET)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
    )

    total_loss = 0

    predictions = []
    labels = []

    trust_errors = []
    risk_errors = []

    # --------------------------------------------------------

    for features, target in loader:

        features = features.to(device)
        target = target.to(device)

        output = model(features)

        loss, breakdown = loss_fn(
            output,
            target,
        )

        total_loss += loss.item()

        pred = output.decision.cpu().numpy()
        gt = target.decision.cpu().numpy()

        predictions.extend(pred)
        labels.extend(gt)

        trust_errors.extend(
            torch.abs(
                output.trust_score -
                target.trust_score
            ).cpu().tolist()
        )

        risk_errors.extend(
            torch.abs(
                output.risk_score -
                target.risk_score
            ).cpu().tolist()
        )

    # --------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

    print()

    print("=" * 60)
    print("Authentication Network Evaluation")
    print("=" * 60)

    print(f"Samples              : {len(dataset)}")
    print(f"Loss                 : {total_loss/len(loader):.4f}")

    print()

    print(f"Decision Accuracy    : {accuracy:.4f}")
    print(f"Precision            : {precision:.4f}")
    print(f"Recall               : {recall:.4f}")
    print(f"F1 Score             : {f1:.4f}")

    print()

    print(
        f"Average Trust Error  : {sum(trust_errors)/len(trust_errors):.4f}"
    )

    print(
        f"Average Risk Error   : {sum(risk_errors)/len(risk_errors):.4f}"
    )

    print()

    print("Confusion Matrix")

    print(cm)

    print("=" * 60)


# ------------------------------------------------------------

if __name__ == "__main__":
    evaluate()