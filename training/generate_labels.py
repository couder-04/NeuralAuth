#!/usr/bin/env python3
"""
generate_labels.py

Production-grade Synthetic Label Generation for AI-powered Voice Authentication.

Enhancements in this version:
- Feature-driven probabilistic personas.
- Magnitude-based feature transformations (log1p + min-max scaling).
- Decoupled trust and risk latent engines.
- Entropy and top-two margin confidence estimation.
- Iterative logit-bias distribution calibration (Temperature/Ratio scaling).
- Probabilistic rule engine (no hard thresholds).
- Internal debug metadata (dropped before output).
- Extensive validation (correlation, distributions, anomalies).
- Detailed execution summary.
"""

import logging
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Configuration & Constants ---
CONFIG = {
    'input_file': Path('training/data/transactions_modified.csv'),
    'output_file': Path('training/data/dataset.csv'),
    'decisions': {
        'ALLOW': 0,
        'VOICE_CHALLENGE': 1,
        'VOICE_AND_OTP': 2,
        'REJECT': 3
    },
    'target_distributions': {
        0: 0.62,  # ALLOW (~62%)
        1: 0.20,  # VOICE_CHALLENGE (~20%)
        2: 0.12,  # VOICE_AND_OTP (~12%)
        3: 0.06   # REJECT (~6%)
    },
    'random_seed': 42,
    'temperature': 1.2 # Softmax temperature for calibration smoothing
}

PERSONAS = [
    'Trusted Veteran', 'Average User', 'Newcomer', 
    'High-Velocity User', 'Stressed User', 'Fraudster', 'Bot/Replay'
]

class MathUtils:
    """Utility class for nonlinear mathematical operations."""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))
    
    @staticmethod
    def soft_clip(x: np.ndarray, min_val: float = 0.0, max_val: float = 1.0, 
                  sharpness: float = 5.0) -> np.ndarray:
        midpoint = (max_val + min_val) / 2
        range_val = max_val - min_val
        return min_val + range_val * MathUtils.sigmoid(sharpness * (x - midpoint))

    @staticmethod
    def softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
        scaled_logits = logits / temp
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
    @staticmethod
    def log1p_minmax(series: pd.Series) -> np.ndarray:
        """Apply log1p followed by min-max scaling for magnitude-based features."""
        val = np.log1p(series.clip(lower=0).to_numpy())
        val_min, val_max = np.min(val), np.max(val)
        if val_max == val_min:
            return np.zeros_like(val)
        return (val - val_min) / (val_max - val_min)
        
    @staticmethod
    def rank_normalize(series: pd.Series) -> np.ndarray:
        """Use rank normalization only when strict relative ordering is desired."""
        return series.rank(pct=True).to_numpy()
        
    @staticmethod
    def entropy(probs: np.ndarray) -> np.ndarray:
        """Calculate normalized Shannon entropy for confidence penalty."""
        p_safe = np.clip(probs, 1e-9, 1.0)
        ent = -np.sum(p_safe * np.log(p_safe), axis=1)
        max_ent = np.log(probs.shape[1])
        return ent / max_ent


class AuthenticationLabelGenerator:
    """Engine for generating realistic, decoupled, and probabilistically calibrated synthetic labels."""

    def __init__(self, df: pd.DataFrame):
        self.df_original = df
        self.df_internal = df.copy()
        self.n = len(df)
        self.rng = np.random.default_rng(CONFIG['random_seed'])
        
        self.latents: Dict[str, np.ndarray] = {}
        self.interactions: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Any] = {}
        
    def _normalize_inputs(self):
        """Prepare inputs with domain-specific transformations."""
        df = self.df_internal
        
        # Log1p + MinMax for magnitude-based features
        magnitude_feats = ['transaction_amount', 'successful_transactions', 
                           'failed_attempts', 'account_age_days']
        for f in magnitude_feats:
            if f in df.columns:
                df[f + '_norm'] = MathUtils.log1p_minmax(df[f])
                
        # Rank normalization for highly skewed frequency/relative features
        rank_feats = ['beneficiary_frequency', 'fraud_history']
        for f in rank_feats:
            if f in df.columns:
                df[f + '_norm'] = MathUtils.rank_normalize(df[f])
                
        # Bounded continuous features (clip to 0-1)
        bounded_feats = ['speaker_similarity', 'liveness_score', 'audio_quality', 
                         'spoof_probability', 'speech_rate_similarity', 
                         'pronunciation_similarity', 'stress_score', 'hesitation_score', 
                         'previous_trust_score', 'transaction_risk']
        for f in bounded_feats:
            if f in df.columns:
                df[f] = np.clip(df[f].astype(float), 0.0, 1.0)
                
    def _generate_personas(self):
        """Derive probabilistic personas from input features (No random assignment)."""
        df = self.df_internal
        
        # Calculate affinity scores for each persona based on feature combinations
        logits = np.zeros((self.n, len(PERSONAS)))
        
        # Trusted Veteran: High account age, high trust, many success
        logits[:, 0] = df['account_age_days_norm'] * 2 + df['previous_trust_score'] * 2 + df.get('successful_transactions_norm', 0)
        # Average User: Baseline
        logits[:, 1] = np.ones(self.n) * 1.5
        # Newcomer: Low account age
        logits[:, 2] = (1.0 - df['account_age_days_norm']) * 3
        # High-Velocity: Many successful and failed attempts, high frequency
        logits[:, 3] = df.get('successful_transactions_norm', 0) + df.get('failed_attempts_norm', 0) + df.get('transaction_amount_norm', 0)
        # Stressed User: High stress, high hesitation
        logits[:, 4] = df['stress_score'] * 3 + df['hesitation_score'] * 2
        # Fraudster: High fraud history, high failed attempts, high spoof
        logits[:, 5] = df.get('fraud_history_norm', 0) * 4 + df.get('failed_attempts_norm', 0) * 2 + df['spoof_probability'] * 3
        # Bot/Replay: High spoof, zero stress, impossible audio perfection
        logits[:, 6] = df['spoof_probability'] * 4 + (1.0 - df['stress_score']) + df['speaker_similarity'] * 2
        
        # Add slight Gumbel noise for variability, then select max
        # Normalize each persona logit so every persona competes fairly
        logits = (logits - logits.mean(axis=0)) / (logits.std(axis=0) + 1e-8)

        # Add stronger stochasticity
        gumbel_noise = self.rng.gumbel(0, 0.8, logits.shape)

        persona_indices = np.argmax(logits + gumbel_noise, axis=1)
        
        self.metadata['persona'] = [PERSONAS[i] for i in persona_indices]
        self.metadata['persona_idx'] = persona_indices

    def _calculate_latents(self):
        """Calculate decoupled Trust and Risk latents. Modulate via Persona."""
        df = self.df_internal
        p_idx = self.metadata['persona_idx']
        
        # --- Trust Latents ---
        self.latents['identity_strength'] = MathUtils.sigmoid(
            df['account_age_days_norm'] * 1.5 + df['voice_enrolled'] * 1.0 - 0.5
        )
        self.latents['voice_authenticity'] = np.clip(
            (df['speaker_similarity'] * 1.5 + df['liveness_score']) / 2.5 - df['spoof_probability'], 0, 1
        )
        self.latents['device_reputation'] = df['previous_trust_score'] * 0.8 + df.get('location_familiarity', 0.5) * 0.2
        self.latents['historical_trust'] = df['previous_trust_score'] * 0.7 + df.get('successful_transactions_norm', 0) * 0.3
        
        # --- Risk Latents ---
        self.latents['fraud_propensity'] = MathUtils.sigmoid(
            df.get('fraud_history_norm', 0) * 3 + df.get('failed_attempts_norm', 0) * 2 - 1.0
        )
        self.latents['transaction_velocity'] = df.get('successful_transactions_norm', 0) * 0.5 + df.get('failed_attempts_norm', 0) * 1.5
        self.latents['session_anomaly'] = df['spoof_probability'] * 2 + (1.0 - df['liveness_score'])
        self.latents['behavioral_anomaly'] = (df['stress_score'] + df['hesitation_score'] + (1.0 - df['speech_rate_similarity'])) / 3.0
        
        # --- Persona Infusion (Affects latents, not final scores directly) ---
        # Trusted Veteran (0): Boost history, lower velocity anomaly
        trusted = (p_idx == 0)
        self.latents['historical_trust'][trusted] = np.clip(self.latents['historical_trust'][trusted] * 1.3, 0, 1)
        self.latents['transaction_velocity'][trusted] *= 0.7
        
        # Fraudster (5): Boost fraud propensity and session anomaly
        fraudster = (p_idx == 5)
        self.latents['fraud_propensity'][fraudster] = np.clip(self.latents['fraud_propensity'][fraudster] * 1.5 + 0.2, 0, 1)
        self.latents['session_anomaly'][fraudster] = np.clip(self.latents['session_anomaly'][fraudster] * 1.5, 0, 1)
        
        # Bot/Replay (6): Max out behavioral anomaly and session anomaly
        bot = (p_idx == 6)
        self.latents['behavioral_anomaly'][bot] = np.clip(self.latents['behavioral_anomaly'][bot] * 2.0, 0, 1)
        self.latents['voice_authenticity'][bot] *= 0.3

    def _calculate_nonlinear_interactions(self):
        """Compute complex, decoupled interaction rules."""
        df = self.df_internal
        
        # 1. Conflicting Multimodal: Good voice, but highly stressed and poor liveness
        self.interactions['multimodal_conflict'] = df['speaker_similarity'] * df['stress_score'] * (1.0 - df['liveness_score'])
        
        # 2. Repeated Beneficiary Behavior: High frequency, low amount, trusted history (Safe)
        self.interactions['repeated_safe_behavior'] = df.get('beneficiary_frequency_norm', 0) * (1.0 - df.get('transaction_amount_norm', 0)) * df['previous_trust_score']
        
        # 3. Transaction Velocity Anomaly: Low account age + extremely high amount + high failed attempts
        self.interactions['velocity_anomaly'] = (1.0 - df['account_age_days_norm']) * df.get('transaction_amount_norm', 0) * df.get('failed_attempts_norm', 0)
        
        # 4. Biometric Impossible Scenario: 100% similarity, 0% liveness (Likely deepfake)
        self.interactions['deepfake_signature'] = df['speaker_similarity'] * (1.0 - df['liveness_score']) * df['spoof_probability']

    def _build_engines(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build independent Trust and Risk engines."""
        # TRUST ENGINE
        base_trust = (
            self.latents['identity_strength'] * 1.5 +
            self.latents['voice_authenticity'] * 2.0 +
            self.latents['device_reputation'] * 1.0 +
            self.latents['historical_trust'] * 1.5
        ) / 6.0
        
        trust_boost = self.interactions['repeated_safe_behavior'] * 0.5
        trust_penalty = self.interactions['multimodal_conflict'] * 0.8 + self.interactions['deepfake_signature'] * 2.0
        
        raw_trust = np.tanh(base_trust * 2 + trust_boost - trust_penalty)
        trust_score = MathUtils.soft_clip((raw_trust + 1) / 2, sharpness=4.0)

        # RISK ENGINE
        base_risk = (
            self.latents['fraud_propensity'] * 2.5 +
            self.latents['transaction_velocity'] * 1.5 +
            self.latents['session_anomaly'] * 2.0 +
            self.latents['behavioral_anomaly'] * 1.0 +
            self.df_internal['transaction_risk'] * 1.0
        ) / 8.0
        
        risk_boost = self.interactions['velocity_anomaly'] * 1.5 + self.interactions['deepfake_signature'] * 2.0
        risk_penalty = self.interactions['repeated_safe_behavior'] * 0.5
        
        raw_risk = base_risk + risk_boost - risk_penalty
        risk_score = MathUtils.soft_clip(raw_risk, sharpness=5.0)

        return trust_score, risk_score

    def _apply_probabilistic_rules(self, base_logits: np.ndarray) -> np.ndarray:
        """Apply rules probabilistically by shifting logits, tracking primary triggers."""
        logits = base_logits.copy()
        df = self.df_internal
        
        rule_triggers = np.zeros((self.n, 5)) # Matrix to track rule impacts for metadata
        
        # Rule 1: High Spoof -> Push heavily toward REJECT probabilistically
        spoof_impact = MathUtils.sigmoid((df['spoof_probability'] - 0.7) * 10) * 4.0
        logits[:, 3] += spoof_impact
        rule_triggers[:, 1] = spoof_impact
        
        # Rule 2: Fraud History + Failed Attempts -> Push toward VOICE_OTP and REJECT
        fraud_impact = (self.latents['fraud_propensity'] ** 2) * 3.0
        logits[:, 3] += fraud_impact * 0.7
        logits[:, 2] += fraud_impact * 0.5
        rule_triggers[:, 2] = fraud_impact
        
        # Rule 3: Conflicting Multimodal -> Push to CHALLENGE
        conflict_impact = self.interactions['multimodal_conflict'] * 3.5
        logits[:, 1] += conflict_impact
        logits[:, 0] -= conflict_impact * 0.5 # Steal from ALLOW
        rule_triggers[:, 3] = conflict_impact
        
        # Rule 4: Trusted + Safe behavior -> Push to ALLOW
        safe_impact = self.interactions['repeated_safe_behavior'] * 3.0
        logits[:, 0] += safe_impact
        rule_triggers[:, 4] = safe_impact
        
        # Determine Primary Triggered Rule for metadata
        rule_names = ['None', 'High_Spoof', 'Fraud_Propensity', 'Multimodal_Conflict', 'Safe_Behavior']
        primary_idx = np.argmax(rule_triggers, axis=1)
        # Only assign rule if impact is substantial (>1.0 logit shift)
        max_impact = np.max(rule_triggers, axis=1)
        primary_idx[max_impact < 2.0] = 0
        
        self.metadata['primary_triggered_rule'] = [rule_names[i] for i in primary_idx]
        return logits

    def _calibrate_and_sample(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Robust distribution calibration using iterative logit adjustment (Ratio Scaling)."""
        target_dist = np.array([CONFIG['target_distributions'][i] for i in range(4)])
        calibrated_logits = logits.copy()
        
        # Iterative Proportional Fitting on Logits to match target distributions exactly in expectation
        max_iter = 50
        for _ in range(max_iter):
            probs = MathUtils.softmax(calibrated_logits, temp=CONFIG['temperature'])
            current_dist = probs.mean(axis=0)
            
            # Ratio of Target to Current
            ratio = target_dist / (current_dist + 1e-9)
            
            # If we are within 0.5% of target for all classes, stop
            if np.all(np.abs(current_dist - target_dist) < 0.005):
                break
                
            # Update logits by log of the ratio
            calibrated_logits += np.log(ratio)
            
        final_probs = MathUtils.softmax(calibrated_logits, temp=CONFIG['temperature'])
        
        # Probabilistic Sampling via Gumbel-Max
        u = self.rng.uniform(size=final_probs.shape)
        gumbel = -np.log(-np.log(np.clip(u, 1e-9, 1.0)))
        decisions = np.argmax(np.log(np.clip(final_probs, 1e-9, 1.0)) + gumbel,axis=1)
        
        return final_probs, decisions

    def _calculate_confidence(self, probs: np.ndarray, trust_score: np.ndarray, risk_score: np.ndarray) -> np.ndarray:
        """Compute robust confidence using entropy, margin, and latent ambiguity."""
        # 1. Prediction Entropy (Lower entropy = Higher certainty)
        normalized_entropy = MathUtils.entropy(probs)
        
        # 2. Top-Two Margin
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        # 3. Signal Agreement / Ambiguity (Trust and Risk shouldn't both be high)
        signal_conflict = (trust_score * risk_score)
        
        # Base confidence calculation
        raw_confidence = 1.0 - (0.4 * normalized_entropy) + (0.3 * margin) - (0.3 * signal_conflict)
        
        # Inject realistic noise based on temperature
        noise = self.rng.uniform(-0.02, 0.02, size=self.n)
        
        return MathUtils.soft_clip(raw_confidence + noise, min_val=0.5, max_val=1.0, sharpness=8.0)

    def _validate(self, df_out: pd.DataFrame):
        """Extensive statistical validation of the generated labels."""
        logger.info("Running advanced dataset validations...")
        
        # Range Checks
        assert df_out['trust_score'].between(0, 1).all(), "trust_score bounds violated"
        assert df_out['risk_score'].between(0, 1).all(), "risk_score bounds violated"
        assert df_out['confidence'].between(0.5, 1).all(), "confidence bounds violated"
        
        # Class Distribution Check
        dist = df_out['decision'].value_counts(normalize=True).to_dict()
        for cls, target in CONFIG['target_distributions'].items():
            actual = dist.get(cls, 0.0)
            # Allow 3% absolute deviation due to sampling variance
            assert abs(actual - target) < 0.03, f"Class {cls} dist {actual:.2f} deviates too far from target {target:.2f}"
            
        # Decoupling & Correlation Validation
        trust = df_out['trust_score']
        risk = df_out['risk_score']
        
        # Trust and Risk should be negatively correlated, but NOT perfectly decoupled (r > -0.95)
        tr_corr, _ = pearsonr(trust, risk)
        assert tr_corr > -0.95, f"Trust and Risk are too heavily coupled (r={tr_corr:.3f})"
        
        # Risk should positively correlate with fraud history if it exists
        if 'fraud_history' in self.df_original.columns:
            rf_corr, _ = pearsonr(risk, self.df_original['fraud_history'])
            assert rf_corr > 0.05, f"Risk doesn't correlate positively with fraud history (r={rf_corr:.3f})"
            
        # Impossible state check (High trust, low risk, but REJECT)
        impossible = df_out[(trust > 0.9) & (risk < 0.1) & (df_out['decision'] == 3)]
        assert len(impossible) / self.n < 0.01, "Too many impossible decision mappings found"
        
        self.validation_stats = {
            'trust_risk_corr': tr_corr,
            'mean_confidence': df_out['confidence'].mean(),
            'actual_distribution': dist
        }

    def _generate_summary(self):
        """Generate a detailed execution and statistical summary."""
        print("\n" + "="*50)
        print("LABEL GENERATION SUMMARY")
        print("="*50)
        
        print("\n[1] Decision Distribution (Target vs Actual):")
        labels = ['ALLOW', 'VOICE_CHALLENGE', 'VOICE_AND_OTP', 'REJECT']
        for i, label in enumerate(labels):
            actual = self.validation_stats['actual_distribution'].get(i, 0.0)
            target = CONFIG['target_distributions'][i]
            print(f"  - {label}: {actual:.1%} (Target: {target:.1%})")
            
        print("\n[2] Engine Statistics:")
        print(f"  - Trust/Risk Correlation (Decoupling metric): {self.validation_stats['trust_risk_corr']:.3f} (Ideal: > -0.95)")
        print(f"  - Mean Confidence Score: {self.validation_stats['mean_confidence']:.3f}")
        
        print("\n[3] Persona Distribution (Feature-Driven):")
        persona_counts = pd.Series(self.metadata['persona']).value_counts(normalize=True)
        for p, pct in persona_counts.items():
            print(f"  - {p}: {pct:.1%}")
            
        print("\n[4] Rule Engine Triggers:")
        rule_counts = pd.Series(self.metadata['primary_triggered_rule']).value_counts(normalize=True)
        for r, pct in rule_counts.items():
            print(f"  - {r}: {pct:.1%}")
            
        print("="*50 + "\n")

    def run(self) -> pd.DataFrame:
        """Execute the pipeline, generate labels, and cleanup metadata."""
        logger.info(f"Starting label generation pipeline for {self.n} transactions...")
        
        self._normalize_inputs()
        self._generate_personas()
        self._calculate_latents()
        self._calculate_nonlinear_interactions()
        
        trust_score, risk_score = self._build_engines()
        trust_score = np.clip(trust_score, 1e-4, 1 - 1e-4)

        risk_score = np.clip(risk_score, 1e-4, 1 - 1e-4)
        
        # Base logits mapping Trust & Risk to decisions
        base_logits = np.zeros((self.n, 4))
        base_logits[:, 0] = trust_score * 3.5 - risk_score * 2.0
        base_logits[:, 1] = (1.0 - trust_score) * 1.5 + risk_score * 1.0
        base_logits[:, 2] = risk_score * 2.5 + self.latents['session_anomaly'] - trust_score * 1.0
        base_logits[:, 3] = risk_score * 4.0 - trust_score * 2.5 - 1.5
        
        rule_adjusted_logits = self._apply_probabilistic_rules(base_logits)
        final_probs, decisions = self._calibrate_and_sample(rule_adjusted_logits)
        
        confidence = self._calculate_confidence(final_probs, trust_score, risk_score)
        
        # Attach debugging metadata (temporarily)
        df_out = self.df_original.copy()
        df_out['trust_score'] = np.round(trust_score, 4)
        df_out['risk_score'] = np.round(risk_score, 4)
        df_out['decision'] = decisions
        df_out['confidence'] = np.round(confidence, 4)
        
        # Validation checks
        self._validate(df_out)
        self._generate_summary()
        
        # Final formatting (Ensure no debug columns remain and order is exact)
        final_columns = list(self.df_original.columns) + ['trust_score', 'risk_score', 'decision', 'confidence']
        return df_out[final_columns]


def main():
    """Main execution entrypoint."""
    input_path = CONFIG['input_file']
    output_path = CONFIG['output_file']
    
    # FAIL FAST if data does not exist
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {input_path}. "
            "Ensure upstream pipeline (modify_dataset.py) has run successfully."
        )
        
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
        
    generator = AuthenticationLabelGenerator(df)
    df_dataset = generator.run()
    
    # Ensure output directory exists and save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving fully calibrated dataset to {output_path}...")
    df_dataset.to_csv(output_path, index=False)
    logger.info("Label generation complete.")

if __name__ == "__main__":
    main()