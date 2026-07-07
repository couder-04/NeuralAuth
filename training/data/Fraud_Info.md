# Fraud Simulation Validation Report

- Rows in fraud_context.csv: 100,000
- Rows in dataset.csv: 100,000
- Rows joined on user_id: 100,000

## Fraud Scenario Counts

| Scenario | Tier | Count | % of Population |
|---|---|---|---|
| genuine | genuine | 94,590 | 94.59% |
| voice_replay_attack | opportunistic | 937 | 0.94% |
| gps_spoofing | opportunistic | 885 | 0.89% |
| behavioral_anomaly | opportunistic | 882 | 0.88% |
| deepfake_voice_attack | sophisticated | 678 | 0.68% |
| stolen_device | sophisticated | 656 | 0.66% |
| account_takeover | sophisticated | 630 | 0.63% |
| insider_fraud | coordinated | 373 | 0.37% |
| multi_stage_coordinated_attack | coordinated | 369 | 0.37% |

All non-genuine scenarios fired at least once. ✓

## Tier / Fraud Balance vs. Configuration

- Configured `fraud_rate`: 0.055
- Actual fraud rate (non-genuine / total): 5.41%

| Tier | Actual % |
|---|---|
| genuine | 94.59% |
| opportunistic | 2.70% |
| sophisticated | 1.96% |
| coordinated | 0.74% |

## Fraud vs. Genuine Feature Comparison

For each subsystem's evidence features: mean value for genuine rows vs. every fraud scenario. A scenario should only show large deviation on the subsystems it's designed to affect.

### Voice

| scenario | speaker_similarity | liveness_score | audio_quality | spoof_probability |
|---|---|---|---|---|
| genuine | 0.9992 | 0.9673 | 0.9139 | 0.0297 |
| account_takeover | 0.9107 | 0.7505 | 0.8444 | 0.1327 |
| behavioral_anomaly | 0.9956 | 0.9340 | 0.9041 | 0.0465 |
| deepfake_voice_attack | 0.9259 | 0.5714 | 0.8802 | 0.1471 |
| gps_spoofing | 0.9987 | 0.9560 | 0.9091 | 0.0353 |
| insider_fraud | 0.9973 | 0.9435 | 0.9042 | 0.0427 |
| multi_stage_coordinated_attack | 0.8460 | 0.6407 | 0.8217 | 0.1934 |
| stolen_device | 0.9559 | 0.8274 | 0.8668 | 0.0929 |
| voice_replay_attack | 0.8478 | 0.6089 | 0.8078 | 0.2099 |

### Behavior

| scenario | speech_rate_similarity | pronunciation_similarity | command_familiarity | stress_score | hesitation_score |
|---|---|---|---|---|---|
| genuine | 0.7381 | 0.8903 | 0.8758 | 0.2245 | 0.2783 |
| account_takeover | 0.5621 | 0.7500 | 0.6884 | 0.3975 | 0.4028 |
| behavioral_anomaly | 0.4282 | 0.6454 | 0.5470 | 0.5362 | 0.4953 |
| deepfake_voice_attack | 0.7008 | 0.8613 | 0.8339 | 0.2603 | 0.3031 |
| gps_spoofing | 0.6789 | 0.8433 | 0.8116 | 0.2797 | 0.3190 |
| insider_fraud | 0.7026 | 0.8647 | 0.8326 | 0.2476 | 0.2998 |
| multi_stage_coordinated_attack | 0.5084 | 0.7020 | 0.6335 | 0.4596 | 0.4469 |
| stolen_device | 0.5206 | 0.7192 | 0.6412 | 0.4290 | 0.4267 |
| voice_replay_attack | 0.7096 | 0.8664 | 0.8458 | 0.2502 | 0.2955 |

### Vehicle

| scenario | vehicle_speed | location_familiarity | time_familiarity |
|---|---|---|---|
| genuine | 27.8033 | 0.8703 | 0.8180 |
| account_takeover | 31.0486 | 0.7794 | 0.7588 |
| behavioral_anomaly | 29.4908 | 0.8171 | 0.7825 |
| deepfake_voice_attack | 26.9072 | 0.8621 | 0.8138 |
| gps_spoofing | 30.3986 | 0.6630 | 0.6459 |
| insider_fraud | 27.3954 | 0.8590 | 0.8093 |
| multi_stage_coordinated_attack | 35.7383 | 0.7124 | 0.6997 |
| stolen_device | 34.1900 | 0.7295 | 0.7218 |
| voice_replay_attack | 29.1886 | 0.8577 | 0.8117 |

### History

| scenario | previous_trust_score | failed_attempts | successful_transactions |
|---|---|---|---|
| genuine | 0.7307 | 0.5950 | 1808.3861 |
| account_takeover | 0.6938 | 0.8365 | 1686.4746 |
| behavioral_anomaly | 0.7105 | 0.7143 | 1728.4683 |
| deepfake_voice_attack | 0.7112 | 0.7389 | 1731.4218 |
| gps_spoofing | 0.7219 | 0.6475 | 1782.2441 |
| insider_fraud | 0.7059 | 0.8043 | 1757.2520 |
| multi_stage_coordinated_attack | 0.5965 | 1.5528 | 1341.8672 |
| stolen_device | 0.6677 | 0.9405 | 1567.9558 |
| voice_replay_attack | 0.7163 | 0.7663 | 1755.5304 |

### Transaction

| scenario | transaction_amount | transaction_risk | beneficiary_frequency |
|---|---|---|---|
| genuine | 15092.8759 | 0.1538 | 0.7072 |
| account_takeover | 34985.1517 | 0.2182 | 0.6230 |
| behavioral_anomaly | 17061.6875 | 0.1644 | 0.6980 |
| deepfake_voice_attack | 25743.7616 | 0.1866 | 0.6644 |
| gps_spoofing | 20954.4333 | 0.1719 | 0.6772 |
| insider_fraud | 40361.9469 | 0.2310 | 0.5905 |
| multi_stage_coordinated_attack | 44595.3810 | 0.2508 | 0.5983 |
| stolen_device | 27388.9538 | 0.2012 | 0.6559 |
| voice_replay_attack | 20330.6437 | 0.1736 | 0.6858 |

### Intent

| scenario | llm_confidence |
|---|---|
| genuine | 0.9600 |
| account_takeover | 0.9419 |
| behavioral_anomaly | 0.9511 |
| deepfake_voice_attack | 0.9503 |
| gps_spoofing | 0.9559 |
| insider_fraud | 0.9529 |
| multi_stage_coordinated_attack | 0.9308 |
| stolen_device | 0.9492 |
| voice_replay_attack | 0.9521 |

## Spoof Probability & Liveness Distribution (fraud vs. genuine)

**spoof_probability**

| Population | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| genuine | 0.0297 | 0.0288 | 0.0564 | 0.0745 | 0.1076 |
| fraud (any scenario) | 0.1110 | 0.0749 | 0.2739 | 0.4006 | 0.4006 |

**liveness_score**

| Population | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| genuine | 0.9673 | 0.9698 | 0.9997 | 1.0000 | 1.0000 |
| fraud (any scenario) | 0.7822 | 0.8282 | 0.9728 | 1.0000 | 1.0000 |

## Correlation: Fraud Scenario Presence vs. Trust/Risk/Decision

| Column | Pearson r vs. is_fraud |
|---|---|
| trust_score | -0.3137 |
| risk_score | 0.1875 |
| decision | 0.1016 |
| confidence | -0.1518 |

risk_score correlates positively with fraud presence (r=0.188). ✓

## Decision Class Balance (overall, and fraud-conditioned)

**Overall:**

| Decision | % |
|---|---|
| ALLOW | 61.41% |
| VOICE_CHALLENGE | 19.86% |
| VOICE_AND_OTP | 12.61% |
| REJECT | 6.11% |

**By fraud presence:**

| fraud | ALLOW | REJECT | VOICE_AND_OTP | VOICE_CHALLENGE |
|---|---|---|---|---|
| genuine | 0.6257 | 0.0589 | 0.1198 | 0.1956 |
| any_fraud | 0.4107 | 0.0998 | 0.2372 | 0.2523 |

## Decision Breakdown Per Scenario

| scenario | ALLOW | REJECT | VOICE_AND_OTP | VOICE_CHALLENGE |
|---|---|---|---|---|
| genuine | 0.6257 | 0.0589 | 0.1198 | 0.1956 |
| account_takeover | 0.3444 | 0.1127 | 0.2556 | 0.2873 |
| behavioral_anomaly | 0.5510 | 0.0703 | 0.1485 | 0.2302 |
| deepfake_voice_attack | 0.2773 | 0.1047 | 0.3053 | 0.3127 |
| gps_spoofing | 0.5989 | 0.0599 | 0.1458 | 0.1955 |
| insider_fraud | 0.5067 | 0.0804 | 0.1528 | 0.2601 |
| multi_stage_coordinated_attack | 0.1626 | 0.1897 | 0.3957 | 0.2520 |
| stolen_device | 0.4177 | 0.1174 | 0.2073 | 0.2576 |
| voice_replay_attack | 0.2967 | 0.1131 | 0.3372 | 0.2529 |

