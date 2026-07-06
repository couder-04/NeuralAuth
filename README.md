# 🧠 NeuralAuth

<div align="center">

### AI-Powered Multi-Modal Transaction Authentication Engine

*Explainable, policy-aware, production-hardened transaction authentication.*

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)
![Tests](https://img.shields.io/badge/Tests-158%20passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-orange.svg)
![Coverage](https://img.shields.io/badge/Coverage-Core%20Paths-9cf.svg)
![Status](https://img.shields.io/badge/Status-Production--Hardened-success.svg)

</div>

---

## Table of Contents

- [🧠 NeuralAuth](#-neuralauth)
    - [AI-Powered Multi-Modal Transaction Authentication Engine](#ai-powered-multi-modal-transaction-authentication-engine)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Complete Architecture Graph](#complete-architecture-graph)
    - [High-level system view](#high-level-system-view)
  - [Request Lifecycle, Step by Step](#request-lifecycle-step-by-step)
  - [Startup \& Shutdown Lifecycle](#startup--shutdown-lifecycle)
  - [Core Components](#core-components)
    - [Authentication Network (`engines/authentication_network.py`)](#authentication-network-enginesauthentication_networkpy)
    - [Feature Extraction (`engines/feature_extractor.py`, `models/feature_vector.py`)](#feature-extraction-enginesfeature_extractorpy-modelsfeature_vectorpy)
    - [Inference (`inference/predictor.py`)](#inference-inferencepredictorpy)
    - [Intent Engine (`engines/intent_engine.py`)](#intent-engine-enginesintent_enginepy)
    - [Risk Engine (`engines/risk_engine.py`)](#risk-engine-enginesrisk_enginepy)
    - [Policy Context (`engines/policy_context.py`)](#policy-context-enginespolicy_contextpy)
    - [Policy Engine (`engines/policy_engine.py`, `rules/policy_rules.yaml`)](#policy-engine-enginespolicy_enginepy-rulespolicy_rulesyaml)
    - [Decision Engine (`engines/decision/`)](#decision-engine-enginesdecision)
    - [Dashboard (`dashboard.py`, launched by `app.py`)](#dashboard-dashboardpy-launched-by-apppy)
  - [Decision Engine Internals](#decision-engine-internals)
    - [Fusion Strategy Comparison](#fusion-strategy-comparison)
  - [Repository Structure](#repository-structure)
  - [Module Dependency Graph](#module-dependency-graph)
  - [Testing Pyramid](#testing-pyramid)
  - [Security Layers](#security-layers)
  - [Getting Started](#getting-started)
  - [Configuration](#configuration)
  - [Security](#security)
  - [Testing](#testing)
  - [Milestones](#milestones)
  - [Tech Stack](#tech-stack)
  - [Known Limitations \& Roadmap](#known-limitations--roadmap)
  - [Contributing](#contributing)
  - [License](#license)
  - [NeuralAuth](#neuralauth)
    - [Intelligent Authentication Through Deep Learning](#intelligent-authentication-through-deep-learning)

---

## Overview

**NeuralAuth** authenticates financial transactions in real time by combining a multi-task deep neural network, an LLM-backed intent parser, a deterministic risk engine, a YAML-driven policy engine, and a pluggable decision-fusion layer — all behind a single FastAPI endpoint.

Every stage produces a typed, explainable output that is threaded through to a full audit trail: the exact feature values that fed the model, the model's own attribution scores, which policy rules matched, and which fusion strategy produced the final action. Nothing is a black box.

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#6C5CE7','primaryTextColor':'#fff','primaryBorderColor':'#4834d4','lineColor':'#a29bfe','secondaryColor':'#00b894','tertiaryColor':'#fdcb6e'}}}%%
flowchart LR
    A(["🎙<br/>Voice + Context"]) --> B["🧠<br/>Neural Network"]
    A --> C["🤖<br/>Intent Engine"]
    B --> D["⚠️<br/>Risk Engine"]
    C --> D
    D --> E["📜<br/>Policy Engine"]
    E --> F["⚖️<br/>Decision Fusion"]
    F --> G(["✅<br/>Final Action"])

    style A fill:#00cec9,stroke:#00695c,color:#fff
    style B fill:#6c5ce7,stroke:#341f97,color:#fff
    style C fill:#fd79a8,stroke:#b53471,color:#fff
    style D fill:#fab1a0,stroke:#e17055,color:#2d3436
    style E fill:#ffeaa7,stroke:#fdcb6e,color:#2d3436
    style F fill:#74b9ff,stroke:#0984e3,color:#fff
    style G fill:#55efc4,stroke:#00b894,color:#2d3436
```

---

## Key Features

- 🧠 Multi-task deep neural Authentication Network (trust / risk / decision / confidence heads)
- 🎙 Voice biometrics + behavioral signal ingestion
- 🚗 Vehicle/location/time context analysis
- 🤖 LLM-backed Intent Engine (schema-validated, retrying, constrained decoding when available)
- ⚠️ Deterministic, YAML-driven Policy Engine (hot-reloadable rules, no code changes to add a rule)
- ⚖️ Pluggable Decision Fusion (majority/weighted/Bayesian/risk-first/policy-first strategies)
- 🔍 Full, explainable audit trail — every engineered feature, every model attribution, every matched rule, every fusion vote
- ⚡ FastAPI backend with **eager startup model loading** (no first-request latency) and **fail-fast startup** (won't serve traffic with a broken model)
- 🔒 Thread-safe singleton engines (double-checked locking — guaranteed single model load under concurrency)
- 🛡 Opt-in API-key authentication, input validation (GPS bounds, non-negative speed, ISO-8601 timestamps, non-blank identifiers), and no internal-exception leakage to clients
- 🧪 158 automated tests across unit / integration / concurrency / security / validation layers
- 📈 End-to-end offline training pipeline with synthetic dataset generation

<div align="center">

| 🧠 Neural | 🤖 LLM | ⚠️ Risk | 📜 Policy | ⚖️ Fusion | 🔍 Audit |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 4 heads | Schema-validated | 5-factor breakdown | 11 hot rules | 6 strategies | 31-field trace |

</div>

---

## Complete Architecture Graph

### High-level system view

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#0984e3','primaryTextColor':'#fff','primaryBorderColor':'#2d3436','lineColor':'#636e72'}}}%%
flowchart TD
    Client(["💻<br/>Client / Dashboard<br/>dashboard.py"]) -->|"POST /authenticate<br/>X-API-Key: optional"| API

    subgraph API["🚀 api/server.py — FastAPI"]
        Lifespan["🟢<br/>lifespan()<br/>eager-loads engines<br/>fails fast on error"]
        Auth["🔑<br/>require_api_key()<br/>opt-in"]
        Route["📥<br/>authenticate()<br/>request_id"]
    end

    Route --> FE["🧩<br/>FeatureExtractor<br/>31-field vector"]
    FE --> Pred["🧠<br/>Authentication<br/>Predictor"]
    Pred --> Net["🕸<br/>Authentication<br/>Network"]
    Net --> AuthResult["📊<br/>AuthResult"]

    Route --> IE["🤖<br/>IntentEngine<br/>LLM parse"]
    IE --> Txn["💬<br/>Transaction"]

    AuthResult --> RE["⚠️<br/>RiskEngine"]
    FE --> RE
    Txn --> RE
    RE --> RiskResult["📈<br/>RiskResult"]

    AuthResult --> PC["🗂<br/>policy_context"]
    FE --> PC
    Txn --> PC
    PC --> PI["📦<br/>PolicyInput"]
    RiskResult --> PI
    PI --> PE["📜<br/>PolicyEngine"]
    PE --> PolicyResult["✅<br/>PolicyResult"]

    AuthResult --> DE
    RiskResult --> DE
    PolicyResult --> DE
    Txn --> DE
    FE --> DE

    subgraph DE["⚖️ DecisionEngine.decide()"]
        Fusion["🔀<br/>Fusion"]
        Explain["💡<br/>Explanation"]
        AuditB["📁<br/>Audit"]
        Meta["🏷<br/>Metadata"]
        Hist["🕘<br/>History"]
        Metrics["📉<br/>Metrics"]
    end

    DE --> DecisionResult["🎯<br/>Decision<br/>Result"]
    DecisionResult --> Response["📤<br/>Response"]
    Response --> Client

    classDef client fill:#00b894,stroke:#00694c,color:#fff,stroke-width:2px
    classDef apilayer fill:#0984e3,stroke:#053e6e,color:#fff,stroke-width:2px
    classDef feature fill:#6c5ce7,stroke:#341f97,color:#fff,stroke-width:2px
    classDef intent fill:#fd79a8,stroke:#b53471,color:#fff,stroke-width:2px
    classDef risk fill:#e17055,stroke:#a84832,color:#fff,stroke-width:2px
    classDef policy fill:#fdcb6e,stroke:#e1a83c,color:#2d3436,stroke-width:2px
    classDef decision fill:#00cec9,stroke:#008b87,color:#fff,stroke-width:2px
    classDef out fill:#55efc4,stroke:#00b894,color:#2d3436,stroke-width:2px

    class Client client
    class API,Lifespan,Auth,Route apilayer
    class FE,Pred,Net,AuthResult feature
    class IE,Txn intent
    class RE,RiskResult risk
    class PC,PI,PE,PolicyResult policy
    class DE,Fusion,Explain,AuditB,Meta,Hist,Metrics decision
    class DecisionResult,Response out
```

---

## Request Lifecycle, Step by Step

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#6c5ce7','primaryTextColor':'#fff','actorBkg':'#0984e3','actorBorder':'#053e6e','actorTextColor':'#fff','signalColor':'#2d3436','signalTextColor':'#2d3436','activationBorderColor':'#00b894','activationBkgColor':'#dff9fb'}}}%%
sequenceDiagram
    participant C as Client
    participant API as API
    participant FE as Features
    participant NN as Neural Net
    participant IE as Intent
    participant RE as Risk
    participant PE as Policy
    participant DE as Decision

    C->>API: POST /authenticate
    Note over API: Pydantic validation (422 on failure)
    Note over API: X-API-Key check (401 on mismatch)
    API->>FE: extract(request)
    FE-->>API: FeatureVector (31 fields)
    API->>NN: predict(features)
    NN-->>API: trust / risk / confidence
    API->>IE: parse(transcript)
    IE-->>API: Transaction
    API->>RE: evaluate(auth, features, txn)
    RE-->>API: RiskResult
    API->>PE: evaluate(policy_input)
    PE-->>API: PolicyResult (rule_trace)
    API->>DE: decide(auth, risk, policy, txn)
    Note over DE: Policy CRITICAL priority always wins
    DE-->>API: DecisionResult + audit_log
    API-->>C: TransactionResponse (JSON)
```

1. `POST /authenticate` arrives with a `TransactionRequest` JSON body.
2. **Validation** — Pydantic rejects blank `user_id`/`transcript`, out-of-range GPS coordinates, negative speed, or malformed timestamps with a `422` before any engine runs.
3. **Auth gate** — if `TRANSACTION_ENGINE_API_KEY` is configured, the `X-API-Key` header must match, or the request is rejected with `401`.
4. **Feature Extraction** builds the 31-field `FeatureVector`.
5. **Authentication Network** (via the pre-warmed `AuthenticationPredictor`) produces trust/risk/confidence/decision-probability signals.
6. **Intent Engine** parses the voice transcript into a structured `Transaction`.
7. **Risk Engine** turns the network's risk score into a leveled, explainable assessment.
8. **Policy Engine** evaluates deterministic YAML rules against everything gathered so far — including real, request-derived `location_familiarity`, `time_familiarity`, `previous_trust_score`, and `failed_attempts` (not hardcoded placeholders).
9. **Decision Engine** fuses the AI and Policy recommendations (Policy's `CRITICAL` priority unconditionally wins), builds the full audit trail (including the complete `feature_vector`), and returns one final action.
10. **Response** is serialized back to the client; any unhandled exception at any stage is caught once, logged in full server-side, and returned to the client only as a generic message + `request_id`.

---

## Startup & Shutdown Lifecycle

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#00b894','primaryTextColor':'#fff','primaryBorderColor':'#00694c','lineColor':'#636e72'}}}%%
stateDiagram-v2
    [*] --> ProcessStart: process start
    ProcessStart --> LifespanEnter: FastAPI lifespan() enters

    state LifespanEnter {
        [*] --> LoadPredictor: get_predictor()
        LoadPredictor --> LoadIntent: get_intent_engine()
        LoadIntent --> LoadRisk: get_risk_engine()
        LoadRisk --> LoadPolicy: get_policy_engine()
        LoadPolicy --> LoadDecision: get_decision_engine()
        LoadDecision --> [*]
    }

    LifespanEnter --> StartupFailed: any loader raises
    StartupFailed --> [*]: process never serves traffic

    LifespanEnter --> Warm: "Startup complete: all engines are warm"
    Warm --> Serving: server accepts traffic (~4.7s startup, ~4ms first request)
    Serving --> ShutdownResume: shutdown triggered
    ShutdownResume --> Done: "Shutdown complete."
    Done --> [*]
```

---

## Core Components

### Authentication Network (`engines/authentication_network.py`)

The central model. `FeatureVector → FeatureAttention → ProjectionLayer → ResidualEncoder → shared embedding → {Trust, Risk, Decision, Confidence} heads`. Also hosts the training loop, checkpoint I/O, an `ExperimentLogger`, MC-dropout uncertainty utilities, and a `DeepEnsemble` — see [Known Limitations](#known-limitations--roadmap) for why this file is flagged as a future decomposition candidate.

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#6c5ce7','primaryTextColor':'#fff','primaryBorderColor':'#341f97','lineColor':'#a29bfe'}}}%%
flowchart LR
    FV["🧩<br/>Feature<br/>Vector"] --> FA["👁<br/>Feature<br/>Attention"]
    FA --> PL["📐<br/>Projection<br/>Layer"]
    PL --> RE2["🔁<br/>Residual<br/>Encoder"]
    RE2 --> EMB(("🌐<br/>Shared<br/>Embedding"))
    EMB --> T["🤝<br/>Trust Head"]
    EMB --> R["⚠️<br/>Risk Head"]
    EMB --> D["🎯<br/>Decision Head"]
    EMB --> CF["📊<br/>Confidence<br/>Head"]

    style FV fill:#00cec9,stroke:#008b87,color:#fff
    style FA fill:#74b9ff,stroke:#0984e3,color:#fff
    style PL fill:#a29bfe,stroke:#6c5ce7,color:#fff
    style RE2 fill:#fd79a8,stroke:#b53471,color:#fff
    style EMB fill:#fdcb6e,stroke:#e1a83c,color:#2d3436
    style T fill:#55efc4,stroke:#00b894,color:#2d3436
    style R fill:#ff7675,stroke:#c0392b,color:#fff
    style D fill:#74b9ff,stroke:#0984e3,color:#fff
    style CF fill:#ffeaa7,stroke:#fdcb6e,color:#2d3436
```

### Feature Extraction (`engines/feature_extractor.py`, `models/feature_vector.py`)

Deterministic, ML-free transformation of a raw request into the 31-field `FeatureVector`. No scoring, no normalization, no tensor conversion — that's the inference layer's job.

```mermaid
%%{init: {'theme':'base'}}%%
pie showData
    title FeatureVector — 31 Fields by Category
    "Identity (5)" : 5
    "Biometrics (4)" : 4
    "Behavior (5)" : 5
    "Vehicle (6)" : 6
    "History (4)" : 4
    "Transaction (4)" : 4
    "Intent (2)" : 2
    "Risk (1)" : 1
```

### Inference (`inference/predictor.py`)

`AuthenticationPredictor` loads and cross-validates six on-disk artifacts once, then exposes `preprocess()` / `predict()` / `predict_result()` as three explicitly separate steps. Thread-safe lazy singleton via `get_predictor()`.

### Intent Engine (`engines/intent_engine.py`)

Wraps an LLM (HF pipeline, LIGHT/HEAVY backend selectable via `config/intent/config.yaml`) with prompt construction, JSON-schema validation, retry-on-failure, and beneficiary SAVED/NEW/UNKNOWN classification.

### Risk Engine (`engines/risk_engine.py`)

Standardizes the network's risk score into `overall_risk`, `risk_level`, and an auditable `breakdown` — deliberately produces no recommended action of its own.

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'pie1':'#00b894','pie2':'#fdcb6e','pie3':'#e17055','pie4':'#6c5ce7','pie5':'#0984e3'}}}%%
pie showData
    title Risk Breakdown Weighting (illustrative)
    "Voice Risk" : 20
    "Behavior Risk" : 20
    "Location Risk" : 25
    "Device Risk" : 15
    "Transaction Risk" : 20
```

### Policy Context (`engines/policy_context.py`)

Bridges continuous `FeatureVector` scores (`location_familiarity`, `time_familiarity`) into the categorical labels the Policy Engine's rules key off of, with documented, configurable thresholds and an optional wall-clock "odd hour" signal.

### Policy Engine (`engines/policy_engine.py`, `rules/policy_rules.yaml`)

Deterministic, hot-reloadable, YAML-driven rule evaluation. Every rule is a declarative `when: {field_op: value}` block; conflicts resolve by priority then action severity.

### Decision Engine (`engines/decision/`)

See [Decision Engine Internals](#decision-engine-internals) below — this is a package, not a single file, by design.

### Dashboard (`dashboard.py`, launched by `app.py`)

A NiceGUI-based visualization client that talks to the API exclusively over HTTP, rendering the pipeline stages, risk/policy/fusion breakdowns, the full feature vector, and system telemetry.

---

## Decision Engine Internals

`engines/decision/` is a package, each module with exactly one responsibility:

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#0984e3','primaryTextColor':'#fff','primaryBorderColor':'#053e6e','lineColor':'#636e72'}}}%%
flowchart TD
    subgraph Pkg["⚖️ engines/decision/"]
        DEng["🧭<br/>decision_engine.py<br/>orchestration"]
        Cfg["⚙️<br/>config.py<br/>thresholds"]
        Fus["🔀<br/>fusion.py<br/>Majority · Weighted<br/>RiskWeighted<br/>Bayesian · RiskFirst<br/>PolicyFirst"]
        Expl["💡<br/>explanation.py<br/>top_reasons"]
        Aud["📁<br/>audit.py<br/>decision_trace<br/>feature_vector"]
        Meta2["🏷<br/>metadata.py<br/>request_id"]
        Hist2["🕘<br/>history.py<br/>HistoryStore"]
        Metr["📉<br/>metrics.py<br/>counters"]
        Hooks["🪝<br/>hooks.py<br/>before/after"]
        Ens["🧬<br/>ensemble.py<br/>combine"]
        Num["🔢<br/>numeric.py<br/>to_python()"]
        Ser["📤<br/>serializers.py<br/>to_json()"]
        Types["🏗<br/>types.py<br/>DecisionResult"]
    end

    DEng --> Cfg
    DEng --> Fus
    DEng --> Expl
    DEng --> Aud
    DEng --> Meta2
    DEng --> Hist2
    DEng --> Metr
    DEng --> Hooks
    Fus --> Ens
    Aud --> Num
    Aud --> Ser
    DEng --> Types

    classDef core fill:#6c5ce7,stroke:#341f97,color:#fff,stroke-width:2px
    classDef support fill:#00cec9,stroke:#008b87,color:#fff,stroke-width:2px
    classDef util fill:#fdcb6e,stroke:#e1a83c,color:#2d3436,stroke-width:2px
    class DEng core
    class Cfg,Fus,Expl,Aud support
    class Meta2,Hist2,Metr,Hooks,Ens,Num,Ser,Types util
```

`DecisionEngine.decide(authentication, risk, policy, intent=None, transaction=None, features=None, ...)` fuses the AI vote and the Policy vote (plus any `additional_recommendations`) via the configured strategy, builds `top_reasons` and `top_attributions` (the model's own top-5 attribution/attention scores — **not** the feature vector), and assembles the full audit trail — including, since the most recent fix, the complete 31-field `feature_vector` under `audit_log["feature_vector"]`, so dashboards can inspect every engineered signal, not just the top-5 model attributions.

### Fusion Strategy Comparison

```mermaid
%%{init: {'theme':'base'}}%%
quadrantChart
    title Fusion Strategies — Determinism vs. Risk Sensitivity
    x-axis Low Determinism --> High Determinism
    y-axis Low Risk Sensitivity --> High Risk Sensitivity
    quadrant-1 Cautious & Rigid
    quadrant-2 Cautious & Adaptive
    quadrant-3 Loose & Rigid
    quadrant-4 Loose & Adaptive
    "MajorityVoting": [0.7, 0.3]
    "WeightedVoting": [0.6, 0.5]
    "RiskWeightedFusion (default)": [0.55, 0.85]
    "BayesianFusion": [0.35, 0.7]
    "RiskFirst": [0.8, 0.95]
    "PolicyFirst": [0.9, 0.6]
```

---

## Repository Structure

```text
Transaction_engine/
│
├── api/
│   └── server.py                 FastAPI app: lifespan, auth dependency, /authenticate route
│
├── engines/
│   ├── authentication_network.py Model architecture + training loop + checkpoint I/O
│   ├── feature_extractor.py      Raw request → FeatureVector (31 fields)
│   ├── intent_engine.py          LLM-backed transcript → Transaction
│   ├── risk_engine.py            Risk standardization + breakdown
│   ├── policy_engine.py          YAML-driven deterministic rules
│   ├── policy_context.py         FeatureVector scores → policy categorical labels
│   └── decision/                 Fusion, audit, explanation, metadata, history, metrics, hooks
│
├── inference/
│   └── predictor.py               AuthenticationPredictor (artifact loading + inference)
│
├── models/
│   ├── request.py                 TransactionRequest (validated Pydantic model)
│   ├── response.py                TransactionResponse
│   ├── feature_vector.py          FeatureVector dataclass (31 fields)
│   └── prediction.py              Prediction / AuthenticationResult contracts
│
├── config/
│   ├── auth/config.yaml           Authentication Network hyperparameters
│   └── intent/config.py+.yaml     Intent Engine config (LIGHT/HEAVY backend selection)
│
├── rules/
│   └── policy_rules.yaml          Live, hot-reloadable Policy Engine rules
│
├── training/                       Independent offline pipeline: dataset generation,
│                                    LLM-based labeling/verification, train.py, evaluate.py
│
├── scripts/
│   └── smoke_test_qwen.py          Manual model smoke test (not a pytest test)
│
├── tests/                           158 tests: unit, integration, concurrency, security, validation
│
├── dashboard.py                    NiceGUI visualization client (HTTP-only, no business logic)
├── app.py                          Launches API (background thread) + dashboard together
├── .env.example                    Documented environment variables (no real secrets)
├── .gitignore                      Excludes secrets, caches, venvs, model artifacts
├── requirements.txt
└── README.md
```

---

## Module Dependency Graph

Verified to be a clean DAG — no circular imports anywhere in the runtime path:

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#00b894','primaryTextColor':'#fff','primaryBorderColor':'#00694c','lineColor':'#636e72'}}}%%
flowchart BT
    Models["📦<br/>models/"]
    Config["⚙️<br/>config/"]
    Engines["🧩<br/>engines/"]
    Inference["🧠<br/>inference/"]
    Api["🚀<br/>api/"]

    Models --> Engines
    Config --> Engines
    Engines --> Inference
    Inference --> Api

    style Models fill:#74b9ff,stroke:#0984e3,color:#fff
    style Config fill:#74b9ff,stroke:#0984e3,color:#fff
    style Engines fill:#6c5ce7,stroke:#341f97,color:#fff
    style Inference fill:#fd79a8,stroke:#b53471,color:#fff
    style Api fill:#00cec9,stroke:#008b87,color:#fff
```

- `api/server.py` → `models.*`, `engines.*`, `inference.predictor`
- `inference/predictor.py` → `engines.authentication_network`, `engines.feature_extractor`
- `engines/feature_extractor.py` → `models.feature_vector`
- `engines/intent_engine.py` → `config.intent.config`
- Nothing under `engines/`, `models/`, or `config/` imports from `api/` or `inference/` — the business/ML layer has zero knowledge of the web layer (correct dependency inversion).

---

## Testing Pyramid

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#fdcb6e','primaryTextColor':'#2d3436','primaryBorderColor':'#e1a83c','lineColor':'#636e72'}}}%%
flowchart TD
    Sec["🛡<br/>Security<br/>test_api_security.py"]
    Val["✅<br/>Validation<br/>test_request_validation.py"]
    Conc["🧵<br/>Concurrency<br/>test_concurrency.py"]
    Start["🟢<br/>Startup / Lifecycle<br/>test_startup.py"]
    Int["🔗<br/>Integration<br/>test_pipeline_integration.py"]
    Unit["🧩<br/>Unit<br/>feature_extractor, policy,<br/>network, intent"]

    Unit --> Int --> Start --> Conc --> Val --> Sec

    style Unit fill:#55efc4,stroke:#00b894,color:#2d3436
    style Int fill:#74b9ff,stroke:#0984e3,color:#fff
    style Start fill:#a29bfe,stroke:#6c5ce7,color:#fff
    style Conc fill:#fd79a8,stroke:#b53471,color:#fff
    style Val fill:#fdcb6e,stroke:#e1a83c,color:#2d3436
    style Sec fill:#ff7675,stroke:#c0392b,color:#fff
```

---

## Security Layers

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#e17055','primaryTextColor':'#fff','primaryBorderColor':'#a84832','lineColor':'#636e72'}}}%%
flowchart LR
    R["📥<br/>Incoming<br/>Request"] --> V["✅<br/>Pydantic<br/>Validation"]
    V --> K["🔑<br/>API-Key<br/>Gate"]
    K --> P["🧠<br/>Pipeline<br/>Execution"]
    P --> E["🙈<br/>Error<br/>Sanitization"]
    E --> Out["📤<br/>Safe<br/>Response"]

    style R fill:#dfe6e9,stroke:#636e72,color:#2d3436
    style V fill:#74b9ff,stroke:#0984e3,color:#fff
    style K fill:#fdcb6e,stroke:#e1a83c,color:#2d3436
    style P fill:#6c5ce7,stroke:#341f97,color:#fff
    style E fill:#ff7675,stroke:#c0392b,color:#fff
    style Out fill:#55efc4,stroke:#00b894,color:#2d3436
```

---

## Getting Started

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (optional — only needed for the LLM-based
#    training/labeling pipeline, and for enabling API-key auth)
cp .env.example .env
# edit .env with real values

# 4. Run the API standalone
python api/server.py
# → serves on http://127.0.0.1:8000 (GET /, GET /health, POST /authenticate)

# 5. OR run the API + dashboard together
python app.py
```

`api/server.py`'s FastAPI `lifespan` loads every model/engine before the server starts accepting traffic — expect a multi-second startup (dominated by the Authentication Network checkpoint and the Intent Engine's LLM) rather than a slow first request.

---

## Configuration

| File | Loaded by | Purpose |
|---|---|---|
| `config/auth/config.yaml` | `ModelConfig.from_yaml()` (`engines/authentication_network.py`) | Authentication Network architecture/training hyperparameters |
| `config/intent/config.yaml` | `load_config()` (`config/intent/config.py`) | Intent Engine backend selection (LIGHT/HEAVY model), retries, validation limits |
| `rules/policy_rules.yaml` | `PolicyEngine.load_rules()` | Live, hot-reloadable Policy Engine rules (`reload_yaml()` to pick up changes without a restart) |
| `.env` (from `.env.example`) | `training/config.py` (dotenv) | LLM provider API key for the offline training/labeling pipeline; `TRANSACTION_ENGINE_API_KEY` to enable API auth |

---

## Security

- **Authentication**: `POST /authenticate` is gated by an opt-in `X-API-Key` header check (`require_api_key`, `api/server.py`), enforced only when `TRANSACTION_ENGINE_API_KEY` is set. A startup warning is logged if it's left unset. `GET /` and `GET /health` are always open (health-check friendly).
- **No internal detail leakage**: unhandled exceptions are logged in full server-side (`logger.exception`) and returned to clients only as a generic message plus an opaque `request_id` for support correlation — never `str(exception)`.
- **Input validation**: GPS coordinates constrained to valid ranges, `vehicle_speed` non-negative, `user_id`/`transcript` rejected if blank, `timestamp` must be ISO-8601 — malformed input fails cleanly with `422` instead of crashing deep inside the pipeline.
- **Secrets hygiene**: `.gitignore` excludes `.env`, model artifacts, and caches; `.env.example` documents required variables without real values.
- **Safe deserialization practices**: all YAML loading uses `yaml.safe_load`; no `eval`/`exec`/`subprocess`/`shell=True` anywhere in the codebase.
- **Thread-safe model loading**: every lazy singleton (`get_predictor`, `get_intent_engine`, `get_risk_engine`, `get_policy_engine`, `get_decision_engine`) uses double-checked locking — verified under a 32-thread concurrency test that exactly one instance is ever constructed.

See [Known Limitations & Roadmap](#known-limitations--roadmap) for security items that are identified but intentionally deferred (PII redaction in `audit_log`, `torch.load`/`joblib.load` hardening, rate limiting).

---

## Testing

```bash
pytest                 # full suite — 158 tests, ~25-30s
pytest tests/test_concurrency.py -v      # thread-safety of every singleton
pytest tests/test_startup.py -v          # FastAPI lifespan behavior
pytest tests/test_api_security.py -v     # auth + error-leakage behavior
pytest tests/test_request_validation.py -v  # input validation
```

| Category | Example files |
|---|---|
| Unit | `test_feature_extractor.py`, `test_policy.py`, `test_policy_context.py`, `test_network.py`, `test_intent.py`, `test_intent_engine.py` |
| Integration | `test_pipeline_integration.py`, `test_api.py`, `test_dashboard_feature_vector.py` |
| Concurrency | `test_concurrency.py` |
| Startup/Lifecycle | `test_startup.py` |
| Security | `test_api_security.py` |
| Validation | `test_request_validation.py` |

All tests run against the real engines wherever practical (e.g. `test_pipeline_integration.py`, `test_dashboard_feature_vector.py`), with only the model-loading seams (`get_predictor`, `get_intent_engine`, etc.) monkeypatched where a real model would be too slow/unnecessary for the behavior under test.

---

## Milestones

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'cScale0':'#00b894','cScale1':'#0984e3','cScale2':'#fdcb6e'}}}%%
timeline
    title NeuralAuth Milestone Timeline
    section Foundation
        M0 Foundation : done
        M1 Feature Extraction : done
        M2 Authentication Network : done
    section Intelligence
        M3 Training Pipeline : done
        M4 Intent Engine : done
        M5 Risk Engine : done
        M6 Policy Engine : done
    section Decisioning
        M7 Decision Engine v1 : done
        M8 Enterprise Decision Package : done
        M9 Decision Fusion : done
        M10 End-to-End Pipeline : done
    section Hardening
        M11 Explainability : done
        M12 Startup & Thread Safety : done
        M13 Security Hardening : done
        M14 Testing & Validation : done
    section What's Next
        M15 Explainability Dashboard : in progress
        M16 Monitoring & Analytics : planned
        M17 PII Redaction : planned
        M18 Rate Limiting : planned
        M19 Deployment CI/CD : planned
```

| Milestone | Status |
|------------|--------|
| M0 – Foundation | ✅ |
| M1 – Feature Extraction | ✅ |
| M2 – Authentication Network | ✅ |
| M3 – Training Pipeline | ✅ |
| M4 – Intent Engine | ✅ |
| M5 – Risk Engine | ✅ |
| M6 – Policy Engine | ✅ |
| M7 – Decision Engine v1 | ✅ |
| M8 – Enterprise Decision Package | ✅ |
| M9 – Decision Fusion | ✅ |
| M10 – End-to-End Pipeline | ✅ |
| M11 – Explainability (full audit trail, feature vector, attributions) | ✅ |
| M12 – Startup Lifecycle & Thread Safety | ✅ |
| M13 – Security Hardening (auth, validation, error handling) | ✅ |
| M14 – Testing & Validation (158 automated tests) | ✅ |
| M15 – Explainability Dashboard | 🚧 |
| M16 – Monitoring & Analytics (drift detection, live metrics) | ⏳ |
| M17 – PII Redaction in Audit Trail | ⏳ |
| M18 – Rate Limiting | ⏳ |
| M19 – Deployment (Docker / Kubernetes / CI/CD) | ⏳ |

---

## Tech Stack

<div align="center">

| Layer | Technologies |
|---|---|
| 🧠 ML / Modeling | PyTorch, ONNX / ONNX Runtime, Scikit-Learn |
| 🤖 LLM | Transformers (HF pipeline) |
| 🚀 API | FastAPI, Uvicorn, Pydantic v2 |
| 📊 Data | NumPy, Pandas |
| ⚙️ Config | PyYAML |
| 🖥 Dashboard | NiceGUI |
| 🧪 Testing | pytest |
| 🐍 Runtime | Python 3.11+ |

</div>

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#fdcb6e','primaryTextColor':'#2d3436','primaryBorderColor':'#e1a83c','lineColor':'#636e72'}}}%%
flowchart TD
    Root(("🧠<br/>NeuralAuth<br/>Tech Stack"))

    Root --> ML["🔮<br/>ML / Modeling"]
    ML --> ML1["PyTorch"]
    ML --> ML2["ONNX Runtime"]
    ML --> ML3["Scikit-Learn"]

    Root --> LLM["🤖<br/>LLM"]
    LLM --> LLM1["Transformers"]
    LLM --> LLM2["HF Pipeline"]

    Root --> API2["🚀<br/>API"]
    API2 --> API1["FastAPI"]
    API2 --> API3["Uvicorn"]
    API2 --> API4["Pydantic v2"]

    Root --> Data2["📊<br/>Data"]
    Data2 --> Data1["NumPy"]
    Data2 --> Data3["Pandas"]

    Root --> Cfg2["⚙️<br/>Config"]
    Cfg2 --> Cfg1["PyYAML"]

    Root --> Dash2["🖥<br/>Dashboard"]
    Dash2 --> Dash1["NiceGUI"]

    Root --> Test2["🧪<br/>Testing"]
    Test2 --> Test1["pytest"]

    classDef root fill:#fdcb6e,stroke:#e1a83c,color:#2d3436,stroke-width:3px
    classDef branch fill:#6c5ce7,stroke:#341f97,color:#fff,stroke-width:2px
    classDef leaf fill:#dfe6e9,stroke:#636e72,color:#2d3436,stroke-width:1px

    class Root root
    class ML,LLM,API2,Data2,Cfg2,Dash2,Test2 branch
    class ML1,ML2,ML3,LLM1,LLM2,API1,API3,API4,Data1,Data3,Cfg1,Dash1,Test1 leaf
```

---

## Known Limitations & Roadmap

Identified during the most recent architecture/security review, tracked deliberately rather than fixed speculatively:

- **PII in audit trail** — `audit_log["transaction"]` currently includes the raw request (transcript, GPS, user_id) unredacted. Needs a coordinated redaction/allow-list design with the dashboard, not an isolated patch.
- **`torch.load`/`joblib.load` use `weights_only=False`** on local model artifacts — acceptable only as long as the artifact directory is protected from tampering; migrating to `weights_only=True` requires verifying the checkpoint's bundled `config` object against the actual production checkpoint format first.
- **No rate limiting** on `/authenticate` — a single endpoint runs the full ML pipeline; nothing currently prevents request-flooding.
- **Two large, multi-concern modules** — `engines/authentication_network.py` (architecture + training loop + checkpointing) and `engines/intent_engine.py` (LLM driving + validation/normalization) are flagged for a dedicated, test-first decomposition, not an incidental refactor.
- **Config loading is inconsistent** across `config/intent`, `engines/authentication_network.ModelConfig`, `engines/decision/config.py`, and `engines/policy_engine.py` — four different loading philosophies; worth unifying once a second real YAML-driven `DecisionConfig` consumer exists.
- **Dependency versions are unpinned** in `requirements.txt` — flagged for a proper `pip-compile`-style resolution pass.

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#ff7675','primaryTextColor':'#fff','primaryBorderColor':'#c0392b','lineColor':'#636e72'}}}%%
flowchart TD
    A["🔴<br/>PII in Audit Trail"]:::high
    B["🟠<br/>torch.load unsafe"]:::med
    C["🟠<br/>No Rate Limiting"]:::med
    D["🟡<br/>Large Modules"]:::low
    E["🟡<br/>Inconsistent Config"]:::low
    F["🟡<br/>Unpinned Deps"]:::low

    classDef high fill:#ff7675,stroke:#c0392b,color:#fff,stroke-width:2px
    classDef med fill:#fab1a0,stroke:#e17055,color:#2d3436,stroke-width:2px
    classDef low fill:#ffeaa7,stroke:#fdcb6e,color:#2d3436,stroke-width:2px
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add/update tests for your change (`pytest` must stay green)
4. Commit your changes
5. Open a Pull Request

---

## License

This project is released under the MIT License.

---

<div align="center">

## NeuralAuth

### Intelligent Authentication Through Deep Learning

**Secure • Explainable • Adaptive**

🧠 ⚡ 🔒 📊 ✅

</div>