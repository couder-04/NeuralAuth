"""
tests/test_network.py

Unit tests for Authentication Network.
"""

import torch

from engines.authentication_network import (
    AuthenticationNetwork,
    ModelConfig,
    Prediction,
    AuthenticationLabels,
    UncertaintyWeightedLoss,
    build_optimizer,
    predict,
    predict_with_uncertainty,
)


def create_model():

    config = ModelConfig()

    model = AuthenticationNetwork(config)

    return model, config


def create_features(config, batch_size=4):

    return torch.randn(
        batch_size,
        config.num_features,
    )


def create_labels(config, batch_size=4):

    return AuthenticationLabels(

        trust_score=torch.rand(batch_size),

        risk_score=torch.rand(batch_size),

        decision=torch.randint(
            0,
            config.num_decision_classes,
            (batch_size,),
        ),

        confidence=torch.rand(batch_size),
    )


# ----------------------------------------------------------
# Model Creation
# ----------------------------------------------------------

def test_model_initialization():

    model, config = create_model()

    assert isinstance(model, AuthenticationNetwork)

    assert model.config == config


# ----------------------------------------------------------
# Forward Pass
# ----------------------------------------------------------

def test_forward_pass():

    model, config = create_model()

    x = create_features(config)

    output = model(x)

    assert isinstance(output, Prediction)

    assert output.trust_score.shape == (4,)
    assert output.risk_score.shape == (4,)
    assert output.confidence.shape == (4,)
    assert output.embedding.shape == (
        4,
        config.embedding_dim,
    )
    assert output.decision_logits.shape == (
        4,
        config.num_decision_classes,
    )


# ----------------------------------------------------------
# Decision probabilities
# ----------------------------------------------------------

def test_decision_probabilities():

    model, config = create_model()

    x = create_features(config)

    output = model(x)

    probs = output.decision_probability

    assert probs.shape == (
        4,
        config.num_decision_classes,
    )

    sums = probs.sum(dim=1)

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5,
    )


# ----------------------------------------------------------
# Loss
# ----------------------------------------------------------

def test_loss_function():

    model, config = create_model()

    loss_fn = UncertaintyWeightedLoss(config)

    x = create_features(config)

    labels = create_labels(config)

    prediction = model(x)

    loss, breakdown = loss_fn(
        prediction,
        labels,
    )

    assert loss.item() > 0

    assert "trust_loss" in breakdown
    assert "risk_loss" in breakdown
    assert "decision_loss" in breakdown
    assert "confidence_loss" in breakdown
    assert "total_loss" in breakdown


# ----------------------------------------------------------
# Optimizer
# ----------------------------------------------------------

def test_optimizer_creation():

    model, config = create_model()

    loss_fn = UncertaintyWeightedLoss(config)

    optimizer = build_optimizer(
        model,
        loss_fn,
        config,
    )

    assert optimizer is not None


# ----------------------------------------------------------
# Inference
# ----------------------------------------------------------

def test_predict():

    model, config = create_model()

    x = create_features(config)

    prediction = predict(
        model,
        x,
    )

    assert isinstance(
        prediction,
        Prediction,
    )

    assert prediction.embedding.shape == (
        4,
        config.embedding_dim,
    )


# ----------------------------------------------------------
# Monte Carlo Dropout
# ----------------------------------------------------------

def test_uncertainty_prediction():

    model, config = create_model()

    x = create_features(config)

    result = predict_with_uncertainty(

        [model],

        x,

        mc_samples=3,

    )

    assert result.num_samples == 3

    assert result.trust_std.shape == (4,)
    assert result.risk_std.shape == (4,)
    assert result.confidence_std.shape == (4,)


# ----------------------------------------------------------
# Batch Size 1
# ----------------------------------------------------------

def test_single_sample():

    model, config = create_model()

    x = create_features(
        config,
        batch_size=1,
    )

    prediction = model(x)

    assert prediction.trust_score.shape == (1,)
    assert prediction.risk_score.shape == (1,)


# ----------------------------------------------------------
# Gradient Flow
# ----------------------------------------------------------

def test_backward_pass():

    model, config = create_model()

    loss_fn = UncertaintyWeightedLoss(config)

    x = create_features(config)

    labels = create_labels(config)

    prediction = model(x)

    loss, _ = loss_fn(
        prediction,
        labels,
    )

    loss.backward()

    grads = [
        p.grad
        for p in model.parameters()
        if p.grad is not None
    ]

    assert len(grads) > 0