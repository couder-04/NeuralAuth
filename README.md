# 🚗 Adaptive Transaction Authentication Engine

> An AI-powered, multi-layer transaction authentication framework for next-generation connected vehicles.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-green.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

</p>

---

## Overview

The **Adaptive Transaction Authentication Engine** is an intelligent security framework for **voice-enabled financial transactions inside connected vehicles**.

Unlike traditional banking authentication, which relies on static OTPs or passwords, this system combines multiple AI models, behavioral biometrics, contextual signals, and deterministic security policies into a single pipeline. Every transaction is evaluated across:

- 👤 User identity
- 🎤 Voice biometrics
- 🗣 Behavioral biometrics
- 🚗 Vehicle context
- 🧠 User intent
- 📈 Historical behavior
- 💳 Transaction characteristics
- ⚠️ Risk level
- 📜 Security policy

Rather than a single-factor pass/fail check, the engine performs **multi-factor adaptive authentication** — reducing fraud while keeping the experience frictionless for legitimate users.

---

## Motivation

Traditional banking authentication has several recurring weaknesses:

- Static, one-size-fits-all rules
- Blanket OTP verification regardless of context
- Poor user experience and high false-rejection rates
- Little to no context awareness or behavioral intelligence
- Vulnerability to phishing-style workflows

Connected vehicles introduce a new interaction paradigm — users issue financial commands by voice, e.g. *"Transfer twenty thousand rupees to Rahul"* or *"Pay my electricity bill."* For each command, the system must determine:

- Is the speaker genuine, and is the voice live (not a replay or synthetic clone)?
- Is this consistent with the user's normal behavior?
- Is the driver actually present in the vehicle?
- Is the transaction itself suspicious, and does it warrant stronger authentication?

This project addresses those questions by combining deep learning with deterministic security rules, rather than relying on either alone.

---

## Key Features

**AI-based authentication** — a multi-task neural network jointly predicts a trust score, risk score, authentication decision, confidence, and a shared transaction embedding.

**Voice biometrics** — speaker similarity, liveness detection, audio quality, and enrollment matching.

**Behavioral biometrics** — speaking rate, pronunciation similarity, command familiarity, and stress/hesitation estimation.

**Vehicle context awareness** — vehicle speed, driver presence, seatbelt status, engine state, and familiarity of location/time.

**Intent understanding** — an LLM parses natural language into a structured transaction (intent, amount, beneficiary, purpose).

**Risk analysis** — a dedicated, deterministic engine combines authentication output, transaction features, and historical trust into a LOW/MEDIUM/HIGH/CRITICAL risk level.

**Rule-based policy engine** — YAML-defined, auditable policies decide the final security action (allow, challenge, OTP, manual review, or reject).

**Decision engine** — merges all upstream outputs into a single, client-facing response.

**REST API** — a production-ready FastAPI server exposing `POST /authenticate`.

**Explainability** — every prediction exposes feature attention weights, the shared embedding, and confidence, so nothing is a black box.

**Testing** — a pytest suite covering the network, feature extractor, intent engine, policy engine, and API.

**Production features** — mixed precision training, gradient clipping, early stopping, checkpointing, TensorBoard and Weights & Biases logging, Monte Carlo dropout, deep ensemble support, and fully YAML-driven hyperparameters.

---

## Architecture

```
                    Raw Transaction Request
                               │
                               ▼
                     Feature Extractor
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
 Authentication Network   Intent Engine        Transaction Data
   (trust / risk /          (Gemma LLM)
    decision / conf.)             │
          │                       ▼
          │                Structured Intent
          └──────────────┬────────┘
                          ▼
                    Risk Engine
                          │
                          ▼
                   Policy Engine
                          │
                          ▼
                  Decision Engine
                          │
                          ▼
                  Final API Response
```

**End-to-end example:**

```
"Transfer ₹20,000 to Rahul"
        │
        ▼
Authentication Network → Trust 0.93 · Risk 0.14 · Confidence 0.97 · Decision ALLOW
        │
        ▼
Intent Engine → MONEY_TRANSFER · amount 20000 · beneficiary "Rahul" · purpose PERSONAL_TRANSFER
        │
        ▼
Risk Engine → LOW
        │
        ▼
Policy Engine → ALLOW
        │
        ▼
Decision Engine → { "transaction_allowed": true, "voice_required": false, "otp_required": false }
```

---

## Repository Structure

```text
Transaction_Engine/
│
├── api/
│   └── server.py
│
├── assets/
│
├── config/
│   ├── auth/
│   │   └── config.yaml
│   └── intent/
│       ├── config.py
│       └── config.yaml
│
├── docs/
│   ├── ARCHITECTURE.png
│   ├── PIPELINE.png
│   └── ...
│
├── engines/
│   ├── authentication_network.py
│   ├── feature_extractor.py
│   ├── intent_engine.py
│   ├── risk_engine.py
│   ├── policy_engine.py
│   └── decision_engine.py
│
├── inference/
│   └── predictor.py
│
├── models/
│   ├── feature_vector.py
│   ├── request.py
│   ├── response.py
│   └── prediction.py
│
├── rules/
│   └── policy_rules.yaml
│
├── training/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── losses.py
│
├── tests/
│   ├── test_network.py
│   ├── test_policy.py
│   ├── test_intent.py
│   ├── test_api.py
│   └── test_feature_extractor.py
│
├── main.py
└── README.md
```

---

## Core Components

| Component | Responsibility |
|---|---|
| Feature Extractor | Converts raw request JSON into a structured, numerical `FeatureVector`. No normalization, scaling, or learning — pure encoding. |
| Authentication Network | Predicts trust, risk, a base decision, and confidence from the feature vector. |
| Intent Engine | Extracts structured transaction intent from natural language using an LLM. |
| Risk Engine | Deterministically combines model and contextual signals into a risk level. |
| Policy Engine | Applies auditable, YAML-defined banking policy to produce a security action. |
| Decision Engine | Merges all outputs into the final, client-facing response. No inference of its own. |
| FastAPI Server | Exposes the pipeline over REST. |

**Design philosophy:** *the AI predicts; the policies decide.* Machine learning estimates what is likely true; the deterministic policy layer decides what is permitted. This separation keeps regulatory-sensitive behavior (e.g. "always require OTP above ₹100,000") auditable and independent of model weights.

---

## Authentication Network

A multi-task neural network that estimates how trustworthy and risky a transaction is. Rather than training separate models for each task, it learns trust, risk, decision, and confidence jointly from a shared representation — improving generalization, reducing inference latency, and lowering memory footprint.

**Architecture:**

```
Feature Vector
      │
      ▼
Feature Attention
      │
      ▼
Projection Layer
      │
      ▼
Residual Encoder
      │
      ▼
Shared Embedding
      │
      ├──► Trust Head
      ├──► Risk Head
      ├──► Decision Head
      └──► Confidence Head
```

**Input feature domains:**

| Domain | Features |
|---|---|
| Identity | Account age, KYC status, phone/email verification, voice enrollment |
| Voice biometrics | Speaker similarity, liveness, audio quality, spoof probability |
| Behavioral biometrics | Speech-rate similarity, pronunciation similarity, command familiarity, stress score, hesitation score |
| Vehicle context | Speed, engine status, driver presence, seatbelt status, familiar location/time |
| Historical profile | Previous trust score, failed attempts, successful transactions, fraud history |
| Transaction | Amount, category, beneficiary type, beneficiary frequency |
| Intent | Intent type, LLM confidence |
| Risk | Transaction risk |

**Outputs:**

| Output | Description | Range |
|---|---|---|
| Trust Score | Probability the user is genuine | 0–1 |
| Risk Score | Estimated transaction risk | 0–1 |
| Decision | Base authentication recommendation | `ALLOW`, `VOICE_CHALLENGE`, `VOICE_AND_OTP`, `REJECT` |
| Confidence | Model confidence in its own prediction | 0–1 |
| Embedding | Shared latent representation, reused by downstream engines | — |

> **Note on decisions vs. actions:** the network's `Decision` output is a *base recommendation* with four classes. The Policy Engine (below) can override or escalate this into a wider set of six final actions — for example, adding a `MANUAL_REVIEW` or plain `OTP` step that the network itself never predicts directly. The network never has the authority to approve a transaction on its own; see [Decision Engine](#decision-engine).

---

## Intent Engine

Converts natural language into a structured transaction using an instruction-tuned Hugging Face LLM (Gemma family).

```
"Transfer twenty thousand rupees to Rahul"
        ↓
{
  "intent": "MONEY_TRANSFER",
  "amount": 20000,
  "currency": "INR",
  "beneficiary": "Rahul",
  "beneficiary_type": "SAVED",
  "purpose": "PERSONAL_TRANSFER"
}
```

**Responsibilities:** intent detection, amount/currency extraction, beneficiary extraction, purpose detection, confidence estimation.

**Supported intents:** `MONEY_TRANSFER`, `BILL_PAYMENT`, `BALANCE_INQUIRY`, `TRANSACTION_HISTORY`, `UNKNOWN`.

**The LLM is never trusted blindly.** Every response passes through JSON validation, schema validation, a retry mechanism, and beneficiary verification before use. Anything that fails falls back to `UNKNOWN` rather than being guessed at.

---

## Risk Engine

A deterministic, explainable engine — not a model — that combines:

- From the Authentication Network: trust score, confidence, base decision
- From the Intent Engine: intent, intent confidence
- From the Feature Vector: amount, failed attempts, previous trust, beneficiary type, context features

into one of four risk levels:

| Level | Meaning |
|---|---|
| LOW | Normal transaction |
| MEDIUM | Slight anomaly |
| HIGH | Suspicious transaction |
| CRITICAL | Fraud highly probable |

```python
RiskPrediction(score=0.82, level="HIGH")
```

---

## Policy Engine

Applies deterministic banking policy — **no machine learning** — converting AI predictions into a security action. Rules live in `rules/policy_rules.yaml` and are evaluated with no source-code changes required:

```yaml
- name: LargeTransaction
  priority: 50
  when:
    transaction_amount_gt: 100000
  action: OTP
  reason: Large transaction
```

**Inputs:** trust score, risk score, confidence, intent, intent confidence, risk level, transaction amount, beneficiary type, failed attempts, network decision.

**Supported actions:** `ALLOW`, `VOICE_CHALLENGE`, `OTP`, `VOICE_AND_OTP`, `MANUAL_REVIEW`, `REJECT`.

**Conflict resolution** when multiple rules match: highest priority wins first; ties are broken by highest severity.

**Every decision is explainable**, carrying the matched policy name, reason, timestamp, override status, and an audit log entry.

---

## Decision Engine

Merges the outputs of the Authentication Network, Intent Engine, Risk Engine, and Policy Engine into the final response. It performs **no inference** — only assembly.

```
Network:  ALLOW
Risk:     HIGH
Policy:   VOICE_AND_OTP
   ↓
Decision: VOICE_AND_OTP
```

---

## REST API

**Health checks**

```
GET /
GET /health
```

**Authentication**

```
POST /authenticate
```

Request:

```json
{
  "identity": { "...": "..." },
  "biometric": { "...": "..." },
  "behavior": { "...": "..." },
  "vehicle": { "...": "..." },
  "history": { "...": "..." },
  "transaction": { "...": "..." }
}
```

Response:

```json
{
  "status": "SUCCESS",
  "action": "ALLOW",
  "transaction_allowed": true,
  "authentication_required": false,
  "voice_required": false,
  "otp_required": false,
  "manual_review": false,
  "message": "Transaction approved.",
  "reason": "High trust score with low transaction risk.",
  "audit_log": {
    "matched_policy": "TrustedUser",
    "risk_level": "LOW",
    "timestamp": "2026-07-04T10:25:13Z"
  }
}
```

Interactive docs are available once the server is running:

```
http://127.0.0.1:8000/docs    (Swagger)
http://127.0.0.1:8000/redoc   (ReDoc)
```

---

## Installation

```bash
git clone https://github.com/your_username/Transaction_Engine.git
cd Transaction_Engine

python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the API

```bash
python api/server.py
# or
uvicorn api.server:app --reload
```

## Training

```bash
python training/train.py
python training/evaluate.py
```

Training features: AdamW optimizer, cosine annealing scheduler, early stopping, gradient clipping, mixed precision, TensorBoard, Weights & Biases, and checkpointing. The loss combines binary cross-entropy, cross-entropy, and confidence loss across tasks, with optional uncertainty-weighted multi-task balancing.

## Inference

```python
from inference.predictor import Predictor

predictor = Predictor(checkpoint="checkpoints/auth_best.pt")
prediction = predictor.predict(feature_vector)
print(prediction)
```

## Testing

```bash
pytest                          # full suite
pytest tests/test_network.py
pytest tests/test_policy.py
pytest tests/test_api.py
```

---

## Configuration

| File | Controls |
|---|---|
| `config/auth/config.yaml` | Authentication Network hyperparameters |
| `config/intent/config.yaml` | Intent Engine / LLM settings |
| `rules/policy_rules.yaml` | Policy Engine business rules |

Changing configuration never requires touching source code or retraining.

---

## Design Principles

- **Single Responsibility** — each module (feature extraction, network, intent, risk, policy, decision) does exactly one job.
- **Configuration over hardcoding** — hyperparameters and business rules live in YAML, not code.
- **Deterministic policy layer** — security-critical decisions are explicit, auditable rules, never opaque model weights.
- **Explainability** — every decision carries a reason, confidence, risk level, matched rule, and audit log.
- **Modularity** — engines are independently testable, maintainable, and replaceable without touching the rest of the pipeline.
- **Production readiness** — structured logging, checkpointing, mixed precision, early stopping, and a full test suite.

---

## Security Model

The system uses seven layers, with no single component solely responsible for approving a transaction:

1. Identity verification
2. Voice biometrics
3. Behavioral biometrics
4. Vehicle context
5. Intent verification
6. Risk estimation
7. Business policy enforcement

---

## Metrics to Monitor in Production

| Component | Metrics |
|---|---|
| Authentication Network | Trust/risk score distributions, prediction confidence, decision class distribution |
| Intent Engine | Intent accuracy, JSON validation failures, retry count, average latency |
| Risk Engine | Risk level frequency, fraud detection rate, false positive rate |
| Policy Engine | Rule hit frequency, override rate, manual review rate |
| API | Requests/sec, average latency, error rate, availability |

---

## FAQ

**Why both AI models and rule-based engines?** ML gives probabilistic estimates of what's *likely* true; the policy layer enforces what's *permitted*. Financial systems need the latter to be deterministic and auditable, independent of any model's weights.

**Why multi-task learning for the network?** Learning trust, risk, decision, and confidence from one shared representation generalizes better and is cheaper at inference time than four separate models.

**Can the Authentication Network approve a transaction directly?** No. Its output always flows through the Risk Engine → Policy Engine → Decision Engine before anything is returned to the client.

**Can policies change without touching code?** Yes — almost all policy changes are edits to `rules/policy_rules.yaml`, with no recompilation or retraining needed.

**Is the LLM trusted?** No — every intent response is schema-validated, retried on failure, and checked against beneficiary/business rules; failures fall back to `UNKNOWN`.

---

## Technology Stack

| Layer | Tools |
|---|---|
| Machine learning | PyTorch, Transformers (Hugging Face) |
| Backend | FastAPI, Uvicorn |
| Data validation | Pydantic |
| Configuration | PyYAML |
| Logging | TensorBoard, Weights & Biases |
| Testing | Pytest |

---

## Roadmap

**v1.0 (current)** — Authentication Network, Intent Engine, Risk Engine, Policy Engine, Decision Engine, FastAPI backend.

**v2.0** — Voice anti-spoofing, face authentication, online learning, deeper explainability (SHAP / Integrated Gradients), distributed inference.

**v3.0** — Multi-modal authentication, edge deployment, federated learning, continuous model updates, Kafka event streaming, Kubernetes deployment, Prometheus/Grafana monitoring.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a pull request

---

## Citation

```text
Adaptive Transaction Authentication Engine
An Explainable Multi-Modal AI Framework for
Secure Vehicle-Based Financial Transactions.
```

---

## License

Released under the **MIT License**.

---

## Project Status

| | |
|---|---|
| Version | 1.0.0 |
| Architecture | Modular AI + rule-based system |
| Backend | FastAPI |
| Deep learning framework | PyTorch |
| LLM framework | Hugging Face Transformers |
| Language | Python 3.11+ |
| License | MIT |

---

⭐ *If you find this project useful, consider starring the repository and contributing to its development.*