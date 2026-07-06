# Dataset Information


## Dataset Overview

- Rows: 100,000
- Columns: 36
- Memory Usage: 27.47 MB
- Dataset Fingerprint (MD5): `224a57c63de4b1e1d9efdd76b6e7a9a1`


### Columns per Family

```
                         Columns
Identifier                     1
Identity                       5
Voice Biometrics               4
Behavior                       5
Vehicle Context                6
Historical Profile             4
Transaction                    5
Intent                         2
Unexpected / Non-Schema        4
```

## Schema Validation

- Expected columns: 33 (2 ID + 31 features)
- Actual columns: 36
- **Missing columns (1):** ['transaction_id']
- **Unexpected / extra columns (4):** ['trust_score', 'risk_score', 'decision', 'confidence']
- Column order matches the declared schema. ✓

## Label / Data Leakage Detection

- **Label columns found in this file: ['trust_score', 'risk_score', 'decision', 'confidence'].** If this is meant to be the pre-label feature dataset (transactions.csv), labels should not be present yet -- verify this file wasn't accidentally overwritten with dataset.csv, or that generate_labels.py hasn't been run upstream of this check.
- No features show suspiciously high correlation with label columns.

## Missing Values & Completeness

No missing values anywhere in the dataset. ✓

## Duplicate Analysis

- Duplicate rows (all columns identical): 0 (0.00%)
- Duplicate `user_id` values: 0

## Summary Statistics (Continuous Features)


### Identity

```
                      mean    median  mode       std     variance      cv   min     max   range  unique
account_age_days  667.4055  542.6841  30.0  500.8141  250814.7564  0.7504  30.0  5000.0  4970.0  100000
```

### Voice Biometrics

```
                      mean  median  mode     std  variance      cv     min     max   range  unique
speaker_similarity  0.9831  0.9852   1.0  0.0107    0.0001  0.0109  0.9310  1.0000  0.0690   99938
liveness_score      0.9676  0.9702   1.0  0.0259    0.0007  0.0268  0.8361  1.0000  0.1639   91923
audio_quality       0.9146  0.9192   1.0  0.0606    0.0037  0.0662  0.4727  1.0000  0.5273   95495
spoof_probability   0.0293  0.0284   0.0  0.0193    0.0004  0.6586  0.0000  0.0948  0.0948    2736
```

### Behavior

```
                            mean  median    mode     std  variance      cv     min     max   range  unique
speech_rate_similarity    0.7381  0.7391  0.4731  0.0584    0.0034  0.0792  0.4731  0.9613  0.4882  100000
pronunciation_similarity  0.8904  0.8905  1.0000  0.0348    0.0012  0.0391  0.7441  1.0000  0.2559   99962
command_familiarity       0.8756  0.8784  1.0000  0.0764    0.0058  0.0872  0.5364  1.0000  0.4636   96801
stress_score              0.2234  0.2224  0.0000  0.1018    0.0104  0.4558  0.0000  0.8403  0.8403   99346
hesitation_score          0.2780  0.2774  0.0000  0.0990    0.0098  0.3559  0.0000  0.7160  0.7160   99965
```

### Vehicle Context

```
                         mean   median  mode      std  variance      cv     min     max     range  unique
vehicle_speed         27.8406  28.3600   0.0  19.8907  395.6389  0.7144  0.0000  113.82  113.8200   48885
location_familiarity   0.8708   0.8751   1.0   0.0844    0.0071  0.0969  0.3426    1.00    0.6574   96278
time_familiarity       0.8187   0.8202   1.0   0.1007    0.0101  0.1231  0.4640    1.00    0.5360   98362
```

### Historical Profile

```
                              mean     median  mode       std     variance      cv      min       max      range  unique
previous_trust_score        0.7308     0.7334   1.0    0.1438       0.0207  0.1967   0.1493     1.000     0.8507   98823
successful_transactions  1807.8760  1798.0000  20.0  867.0637  751799.3780  0.4796  20.0000  4603.002  4583.0020   14279
```

### Transaction

```
                             mean     median    mode         std      variance      cv     min          max       range  unique
transaction_amount     15100.5349  6871.0150  0.0000  17225.3157  2.967115e+08  1.1407  0.0000  101062.0800  101062.080   69199
beneficiary_frequency      0.7045     0.7478  0.0000      0.2182  4.760000e-02  0.3098  0.0000       1.0000       1.000  100000
transaction_risk           0.0974     0.0878  0.0093      0.0517  2.700000e-03  0.5312  0.0093       0.4132       0.404  100000
```

### Intent

```
                mean  median  mode     std  variance      cv     min  max   range  unique
llm_confidence  0.96  0.9601   1.0  0.0217    0.0005  0.0226  0.8943  1.0  0.1057   98926
```

### Unexpected / Non-Schema

```
               mean  median    mode     std  variance      cv     min     max   range  unique
trust_score  0.8625  0.8656  0.8674  0.0108    0.0001  0.0125  0.7670  0.8775  0.1105     811
risk_score   0.4251  0.3071  0.2827  0.2562    0.0656  0.6025  0.0273  0.9987  0.9714    9560
confidence   0.7627  0.8197  0.5334  0.1675    0.0281  0.2196  0.5180  0.9923  0.4743    4731
```

## Distribution Analysis (Continuous Features)


### Identity

```
                  skewness  kurtosis
account_age_days    1.5525     3.578
```

### Voice Biometrics

```
                    skewness  kurtosis
speaker_similarity   -0.8379    0.3200
liveness_score       -0.5944   -0.2133
audio_quality        -0.9271    2.2025
spoof_probability     0.2715   -0.6897
```

### Behavior

```
                          skewness  kurtosis
speech_rate_similarity     -0.1091   -0.0903
pronunciation_similarity   -0.0225   -0.0504
command_familiarity        -0.3211   -0.3114
stress_score                0.2112    0.2037
hesitation_score            0.0622   -0.1469
```

### Vehicle Context

```
                      skewness  kurtosis
vehicle_speed           0.2136   -0.7177
location_familiarity   -0.3056   -0.5587
time_familiarity       -0.1302   -0.6768
```

### Historical Profile

```
                         skewness  kurtosis
previous_trust_score      -0.1285   -0.6704
successful_transactions    0.1020   -0.3571
```

### Transaction

```
                       skewness  kurtosis
transaction_amount       1.0503    0.1914
beneficiary_frequency   -0.7070   -0.2972
transaction_risk         0.9609    0.7586
```

### Intent

```
                skewness  kurtosis
llm_confidence   -0.0345   -0.8723
```

### Unexpected / Non-Schema

```
             skewness  kurtosis
trust_score   -2.0506    5.3079
risk_score     0.6445   -0.9577
confidence    -0.2904   -1.4895
```

## Percentiles (Continuous Features)


### Identity

```
                       p1        p5       p10       p25       p50       p75        p90        p95        p99
account_age_days  57.5661  112.1434  164.1077  299.2269  542.6841  902.7587  1335.2122  1645.9677  2354.2578
```

### Voice Biometrics

```
                        p1      p5     p10     p25     p50     p75     p90     p95     p99
speaker_similarity  0.9537  0.9625  0.9679  0.9765  0.9852  0.9916  0.9953  0.9967  0.9982
liveness_score      0.9004  0.9208  0.9316  0.9500  0.9702  0.9903  0.9999  1.0000  1.0000
audio_quality       0.7381  0.8146  0.8392  0.8776  0.9192  0.9598  0.9961  0.9999  1.0000
spoof_probability   0.0000  0.0000  0.0020  0.0139  0.0284  0.0433  0.0559  0.0625  0.0733
```

### Behavior

```
                              p1      p5     p10     p25     p50     p75     p90     p95     p99
speech_rate_similarity    0.5993  0.6403  0.6624  0.6986  0.7391  0.7787  0.8130  0.8321  0.8669
pronunciation_similarity  0.8096  0.8329  0.8457  0.8669  0.8905  0.9140  0.9351  0.9474  0.9711
command_familiarity       0.6880  0.7439  0.7737  0.8238  0.8784  0.9327  0.9805  0.9992  1.0000
stress_score              0.0010  0.0555  0.0921  0.1539  0.2224  0.2901  0.3529  0.3923  0.4721
hesitation_score          0.0542  0.1168  0.1498  0.2097  0.2774  0.3456  0.4060  0.4404  0.5094
```

### Vehicle Context

```
                          p1      p5     p10      p25      p50      p75      p90      p95      p99
vehicle_speed         0.0000  0.0771  0.2631  10.9100  28.3600  42.2936  54.0700  60.9299  73.6801
location_familiarity  0.6828  0.7240  0.7541   0.8100   0.8751   0.9387   0.9861   0.9995   1.0000
time_familiarity      0.6009  0.6483  0.6819   0.7456   0.8202   0.8947   0.9569   0.9847   1.0000
```

### Historical Profile

```
                              p1        p5       p10       p25        p50        p75        p90        p95     p99
previous_trust_score      0.4173    0.4890    0.5351     0.626     0.7334     0.8406     0.9244     0.9638     1.0
successful_transactions  20.0000  348.9916  665.0000  1201.000  1798.0000  2402.7715  2945.0000  3272.0712  3839.0
```

### Transaction

```
                           p1      p5     p10       p25        p50         p75         p90         p95         p99
transaction_amount     0.0000  0.0000  0.0000  100.0000  6871.0150  27097.6275  41534.2530  49066.9565  62776.4346
beneficiary_frequency  0.1423  0.2838  0.3807    0.5603     0.7478      0.8859      0.9567      0.9789      0.9958
transaction_risk       0.0213  0.0309  0.0396    0.0587     0.0878      0.1243      0.1721      0.2012      0.2480
```

### Intent

```
                   p1      p5     p10     p25     p50     p75   p90     p95  p99
llm_confidence  0.917  0.9245  0.9303  0.9432  0.9601  0.9768  0.99  0.9956  1.0
```

### Unexpected / Non-Schema

```
                 p1      p5     p10     p25     p50     p75     p90     p95     p99
trust_score  0.8229  0.8390  0.8475  0.8601  0.8656  0.8691  0.8715  0.8730  0.8754
risk_score   0.0609  0.1086  0.1584  0.2413  0.3071  0.6711  0.8263  0.8925  0.9631
confidence   0.5220  0.5266  0.5307  0.5558  0.8197  0.9032  0.9694  0.9816  0.9886
```

## Feature Range Validation


### Identity

```
                      Range Status  Invalid
Feature                                    
account_age_days  [1, 5000]   PASS        0
```

### Voice Biometrics

```
                         Range Status  Invalid
Feature                                       
speaker_similarity  [0.0, 1.0]   PASS        0
liveness_score      [0.0, 1.0]   PASS        0
audio_quality       [0.0, 1.0]   PASS        0
spoof_probability   [0.0, 1.0]   PASS        0
```

### Behavior

```
                               Range Status  Invalid
Feature                                             
speech_rate_similarity    [0.0, 1.0]   PASS        0
pronunciation_similarity  [0.0, 1.0]   PASS        0
command_familiarity       [0.0, 1.0]   PASS        0
stress_score              [0.0, 1.0]   PASS        0
hesitation_score          [0.0, 1.0]   PASS        0
```

### Vehicle Context

```
                           Range Status  Invalid
Feature                                         
vehicle_speed           [0, 250]   PASS        0
location_familiarity  [0.0, 1.0]   PASS        0
time_familiarity      [0.0, 1.0]   PASS        0
```

### Historical Profile

```
                              Range Status  Invalid
Feature                                            
previous_trust_score     [0.0, 1.0]   PASS        0
failed_attempts             [0, 50]   PASS        0
successful_transactions  [0, 10000]   PASS        0
```

### Transaction

```
                               Range Status  Invalid
Feature                                             
transaction_risk          [0.0, 1.0]   PASS        0
beneficiary_frequency     [0, 10000]   PASS        0
transaction_amount     [0, 10000000]   PASS        0
```

### Intent

```
                     Range Status  Invalid
Feature                                   
llm_confidence  [0.0, 1.0]   PASS        0
```

## Binary Feature Analysis


### Identity

```
                                          Distribution
Feature                                               
kyc_verified      1.0=98,010 (98.0%); 0.0=1,990 (2.0%)
phone_verified    1.0=98,277 (98.3%); 0.0=1,723 (1.7%)
email_verified    1.0=96,291 (96.3%); 0.0=3,709 (3.7%)
voice_enrolled  1.0=78,159 (78.2%); 0.0=21,841 (21.8%)
```

### Vehicle Context

```
                                             Distribution
Feature                                                  
engine_running     1.0=84,991 (85.0%); 0.0=15,009 (15.0%)
driver_present       1.0=97,424 (97.4%); 0.0=2,576 (2.6%)
seatbelt_fastened    1.0=93,271 (93.3%); 0.0=6,729 (6.7%)
```

### Historical Profile

```
                                       Distribution
Feature                                            
fraud_history  0.0=97,694 (97.7%); 1.0=2,306 (2.3%)
```

### Transaction

```
                                        Distribution
Feature                                             
beneficiary_type  0=88,561 (88.6%); 1=11,439 (11.4%)
```

## Categorical Feature Analysis


### Historical Profile

```
                 Categories  Top Category  Top Share %  Entropy (bits)  Normalized Entropy
Feature                                                                                   
failed_attempts           6           0.0        54.71           1.477               0.572
```

### Transaction

```
                      Categories  Top Category  Top Share %  Entropy (bits)  Normalized Entropy
Feature                                                                                        
transaction_category           4             1        55.28           1.677               0.838
```

### Intent

```
             Categories  Top Category  Top Share %  Entropy (bits)  Normalized Entropy
Feature                                                                               
intent_type           4             1        55.28           1.677               0.838
```

### Unexpected / Non-Schema

```
          Categories  Top Category  Top Share %  Entropy (bits)  Normalized Entropy
Feature                                                                            
decision           4             0        61.52           1.514               0.757
```

### failed_attempts -- Top Categories

- 0.0: 54,712 (54.71%)
- 1.0: 32,986 (32.99%)
- 2.0: 9,952 (9.95%)
- 3.0: 2,025 (2.02%)
- 4.0: 292 (0.29%)
- 5.0: 33 (0.03%)

### transaction_category -- Top Categories

- 1: 55,285 (55.28%)
- 2: 19,863 (19.86%)
- 3: 14,882 (14.88%)
- 0: 9,970 (9.97%)

### intent_type -- Top Categories

- 1: 55,285 (55.28%)
- 2: 19,863 (19.86%)
- 3: 14,882 (14.88%)
- 0: 9,970 (9.97%)

### decision -- Top Categories

- 0: 61,521 (61.52%)
- 1: 20,136 (20.14%)
- 2: 12,187 (12.19%)
- 3: 6,156 (6.16%)

## Cardinality


### Identifier

```
         Unique Values Type
user_id         100000   id
```

### Identity

```
                  Unique Values        Type
account_age_days         100000  continuous
kyc_verified                  2      binary
phone_verified                2      binary
email_verified                2      binary
voice_enrolled                2      binary
```

### Voice Biometrics

```
                    Unique Values        Type
speaker_similarity          99938  continuous
liveness_score              91923  continuous
audio_quality               95495  continuous
spoof_probability            2736  continuous
```

### Behavior

```
                          Unique Values        Type
speech_rate_similarity           100000  continuous
pronunciation_similarity          99962  continuous
command_familiarity               96801  continuous
stress_score                      99346  continuous
hesitation_score                  99965  continuous
```

### Vehicle Context

```
                      Unique Values        Type
vehicle_speed                 48885  continuous
engine_running                    2      binary
location_familiarity          96278  continuous
time_familiarity              98362  continuous
driver_present                    2      binary
seatbelt_fastened                 2      binary
```

### Historical Profile

```
                         Unique Values         Type
previous_trust_score             98823   continuous
failed_attempts                      6  categorical
successful_transactions          14279   continuous
fraud_history                        2       binary
```

### Transaction

```
                       Unique Values         Type
transaction_amount             69199   continuous
transaction_category               4  categorical
beneficiary_type                   2       binary
beneficiary_frequency         100000   continuous
transaction_risk              100000   continuous
```

### Intent

```
                Unique Values         Type
intent_type                 4  categorical
llm_confidence          98926   continuous
```

### Unexpected / Non-Schema

```
             Unique Values         Type
trust_score            811   continuous
risk_score            9560   continuous
decision                 4  categorical
confidence            4731   continuous
```

## Correlation Analysis


### Top Positive Correlations

```
              Feature A                Feature B        r
   previous_trust_score  successful_transactions 0.912588
           stress_score         hesitation_score 0.719170
 speech_rate_similarity      command_familiarity 0.693224
   location_familiarity         time_familiarity 0.491653
   previous_trust_score              trust_score 0.466307
successful_transactions              trust_score 0.452626
       transaction_risk               risk_score 0.435706
 speech_rate_similarity pronunciation_similarity 0.327042
     transaction_amount               risk_score 0.316768
     transaction_amount         transaction_risk 0.313098
```

### Top Negative Correlations

```
               Feature A         Feature B         r
              risk_score        confidence -0.962619
          liveness_score spoof_probability -0.570639
   beneficiary_frequency  transaction_risk -0.548069
pronunciation_similarity      stress_score -0.490103
        transaction_risk        confidence -0.445835
  speech_rate_similarity  hesitation_score -0.431946
      transaction_amount        confidence -0.365848
pronunciation_similarity  hesitation_score -0.360141
           audio_quality spoof_probability -0.359695
  speech_rate_similarity      stress_score -0.333438
```

### Highly Correlated Pairs (|r| >= 0.8)

```
           Feature A               Feature B         r
previous_trust_score successful_transactions  0.912588
          risk_score              confidence -0.962619
```

### Redundant Features (|r| >= 0.95)

```
 Feature A  Feature B         r
risk_score confidence -0.962619
```

## Multicollinearity (VIF)

```
                            VIF
confidence                16.86
risk_score                16.71
previous_trust_score       8.75
successful_transactions    8.16
speech_rate_similarity     3.30
command_familiarity        2.58
hesitation_score           2.52
stress_score               2.46
spoof_probability          1.92
transaction_risk           1.85
beneficiary_frequency      1.63
trust_score                1.61
liveness_score             1.61
pronunciation_similarity   1.47
transaction_amount         1.33
time_familiarity           1.32
location_familiarity       1.32
audio_quality              1.26
account_age_days           1.15
speaker_similarity         1.02
llm_confidence             1.02
vehicle_speed              1.00
```

Rule of thumb: VIF > 10 suggests problematic multicollinearity; VIF > 5 is worth investigating.

## Outlier Analysis (Continuous Features)


### Identity

```
                  IQR_Outliers  Z_Outliers  Outlier %
Feature                                              
account_age_days          3496        1518        3.5
```

### Voice Biometrics

```
                    IQR_Outliers  Z_Outliers  Outlier %
Feature                                                
speaker_similarity          1058         600       1.06
liveness_score               373         383       0.38
audio_quality               1259         938       1.26
spoof_probability             52          53       0.05
```

### Behavior

```
                          IQR_Outliers  Z_Outliers  Outlier %
Feature                                                      
speech_rate_similarity             548         225       0.55
pronunciation_similarity           671         271       0.67
command_familiarity                406         241       0.41
stress_score                       666         343       0.67
hesitation_score                   458         156       0.46
```

### Vehicle Context

```
                      IQR_Outliers  Z_Outliers  Outlier %
Feature                                                  
vehicle_speed                   68         101       0.10
location_familiarity            73          77       0.08
time_familiarity                40          29       0.04
```

### Historical Profile

```
                         IQR_Outliers  Z_Outliers  Outlier %
Feature                                                     
previous_trust_score               29          27       0.03
successful_transactions           159          39       0.16
```

### Transaction

```
                       IQR_Outliers  Z_Outliers  Outlier %
Feature                                                   
transaction_amount              532         586       0.59
beneficiary_frequency           226         110       0.23
transaction_risk               2597         834       2.60
```

### Intent

```
                IQR_Outliers  Z_Outliers  Outlier %
Feature                                            
llm_confidence             0           1        0.0
```

### Unexpected / Non-Schema

```
             IQR_Outliers  Z_Outliers  Outlier %
Feature                                         
trust_score          9312        2075       9.31
risk_score              0           0       0.00
confidence              0           0       0.00
```

## Feature Quality


### Identity

```
                  Quality
account_age_days    100.0
email_verified      100.0
voice_enrolled      100.0
kyc_verified         85.0
phone_verified       85.0
```

### Voice Biometrics

```
                    Quality
speaker_similarity    100.0
liveness_score        100.0
audio_quality         100.0
spoof_probability     100.0
```

### Behavior

```
                          Quality
speech_rate_similarity      100.0
pronunciation_similarity    100.0
command_familiarity         100.0
stress_score                100.0
hesitation_score            100.0
```

### Vehicle Context

```
                      Quality
vehicle_speed           100.0
engine_running          100.0
location_familiarity    100.0
time_familiarity        100.0
driver_present          100.0
seatbelt_fastened       100.0
```

### Historical Profile

```
                         Quality
previous_trust_score       100.0
failed_attempts            100.0
successful_transactions    100.0
fraud_history              100.0
```

### Transaction

```
                       Quality
transaction_amount       100.0
transaction_category     100.0
beneficiary_type         100.0
beneficiary_frequency    100.0
transaction_risk         100.0
```

### Intent

```
                Quality
intent_type       100.0
llm_confidence    100.0
```

### Unexpected / Non-Schema

```
             Quality
risk_score     100.0
decision       100.0
confidence     100.0
trust_score     75.0
```

## Recommendations

- Schema: add missing required columns: ['transaction_id'].
- Schema: ['trust_score', 'risk_score', 'decision', 'confidence'] are not part of the declared model schema -- confirm they're intentional (e.g. metadata) before training.
- Leakage: label columns ['trust_score', 'risk_score', 'decision', 'confidence'] are present in this file -- exclude them from model input features.
- Distribution: `trust_score` has skewness -2.05 -- consider a log/Box-Cox transform.
- Consistency: binary feature(s) ['kyc_verified', 'phone_verified'] are >98% one class -- consider class weighting or resampling.
- Distribution: ['trust_score'] have >5% flagged outliers -- verify these are real, not generator bugs.
- Correlation: near-duplicate feature pairs risk_score/confidence (r=-0.96) -- consider dropping one from each pair.
- Correlation: ['confidence', 'risk_score'] have VIF > 10 -- multicollinearity risk for linear models.

## Dataset Health Score

```
                 Weight %  Score /100  Weighted Contribution
Sub-score                                                   
Schema                 20        50.0                  10.00
Completeness           15       100.0                  15.00
Distribution           15        97.7                  14.66
Consistency            15        87.0                  13.05
Correlation            15        81.0                  12.15
Feature Quality        20        98.4                  19.69
```

- **Overall Weighted Health Score: 84.5/100 (Fair)**