# Dataset Information


## Dataset Overview

- Rows: 100,000
- Columns: 36
- Memory Usage: 27.47 MB
- Dataset Fingerprint (MD5): `aa3fa32a75533819cac80ce2958c750b`


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
                      mean  median  mode      std     variance      cv  min   max  range  unique
account_age_days  667.3341   543.0   354  500.757  250757.5437  0.7504   30  5000   4970    2931
```

### Voice Biometrics

```
                      mean  median  mode     std  variance      cv     min     max   range  unique
speaker_similarity  0.9959  1.0000   1.0  0.0241    0.0006  0.0242  0.6900  1.0000  0.3100   54660
liveness_score      0.9573  0.9682   1.0  0.0646    0.0042  0.0674  0.1318  1.0000  0.8682   92185
audio_quality       0.9114  0.9172   1.0  0.0645    0.0042  0.0708  0.3075  1.0000  0.6925   95472
spoof_probability   0.0341  0.0300   0.0  0.0348    0.0012  1.0207  0.0000  0.4006  0.4006    4539
```

### Behavior

```
                            mean  median    mode     std  variance      cv     min     max   range  unique
speech_rate_similarity    0.7308  0.7360  0.0567  0.0718    0.0052  0.0983  0.0567  0.9605  0.9038  100000
pronunciation_similarity  0.8845  0.8884  1.0000  0.0473    0.0022  0.0535  0.4551  1.0000  0.5449   99966
command_familiarity       0.8680  0.8748  1.0000  0.0897    0.0080  0.1033  0.0000  1.0000  1.0000   96870
stress_score              0.2316  0.2272  0.0000  0.1110    0.0123  0.4790  0.0000  0.9993  0.9993   99331
hesitation_score          0.2833  0.2812  0.0000  0.1037    0.0108  0.3660  0.0000  0.7799  0.7799   99944
```

### Vehicle Context

```
                         mean   median  mode      std  variance      cv     min       max     range  unique
vehicle_speed         27.9381  28.4661   0.0  19.9791  399.1630  0.7151  0.0000  112.3113  112.3113   48937
location_familiarity   0.8657   0.8710   1.0   0.0890    0.0079  0.1028  0.2958    1.0000    0.7042   96406
time_familiarity       0.8146   0.8173   1.0   0.1033    0.0107  0.1268  0.3705    1.0000    0.6295   98505
```

### Historical Profile

```
                              mean     median  mode       std     variance      cv      min     max      range  unique
previous_trust_score        0.7289     0.7323   1.0    0.1435       0.0206  0.1969   0.1699     1.0     0.8301   98867
successful_transactions  1802.1754  1793.0000  20.0  865.5752  749220.3534  0.4803  19.0000  4600.0  4581.0000    4180
```

### Transaction

```
                             mean     median    mode         std      variance      cv     min          max        range  unique
transaction_amount     15692.5079  6899.0850  0.0000  18519.0763  3.429562e+08  1.1801  0.0000  224730.0800  224730.0800   69442
beneficiary_frequency      0.7047     0.7493  0.0000      0.2187  4.780000e-02  0.3104  0.0000       1.0000       1.0000  100000
transaction_risk           0.1558     0.1376  0.0139      0.0830  6.900000e-03  0.5329  0.0139       0.6285       0.6146  100000
```

### Intent

```
                  mean  median  mode    std  variance      cv     min  max   range  unique
llm_confidence  0.9594  0.9597   1.0  0.022    0.0005  0.0229  0.8845  1.0  0.1155   99032
```

### Unexpected / Non-Schema

```
               mean  median    mode     std  variance      cv     min     max   range  unique
trust_score  0.8621  0.8659  0.8684  0.0133    0.0002  0.0154  0.6035  0.8773  0.2738    1149
risk_score   0.4187  0.3145  0.2653  0.2447    0.0599  0.5844  0.0281  0.9997  0.9716    9558
confidence   0.7609  0.8099  0.5316  0.1648    0.0272  0.2166  0.5185  0.9919  0.4734    4731
```

## Distribution Analysis (Continuous Features)


### Identity

```
                  skewness  kurtosis
account_age_days    1.5523    3.5766
```

### Voice Biometrics

```
                    skewness  kurtosis
speaker_similarity   -8.3163   75.8214
liveness_score       -5.4939   39.1155
audio_quality        -1.1189    2.9223
spoof_probability     4.9699   38.6346
```

### Behavior

```
                          skewness  kurtosis
speech_rate_similarity     -1.4334    6.2078
pronunciation_similarity   -2.0546    9.6799
command_familiarity        -1.2606    4.6836
stress_score                0.5764    1.3891
hesitation_score            0.2146    0.1852
```

### Vehicle Context

```
                      skewness  kurtosis
vehicle_speed           0.2139   -0.7035
location_familiarity   -0.4143   -0.2154
time_familiarity       -0.1696   -0.6025
```

### Historical Profile

```
                         skewness  kurtosis
previous_trust_score      -0.1332   -0.6536
successful_transactions    0.1028   -0.3514
```

### Transaction

```
                       skewness  kurtosis
transaction_amount       1.4045    2.8008
beneficiary_frequency   -0.7099   -0.3001
transaction_risk         1.6226    2.6207
```

### Intent

```
                skewness  kurtosis
llm_confidence   -0.0866   -0.7761
```

### Unexpected / Non-Schema

```
             skewness  kurtosis
trust_score   -4.1767   34.9141
risk_score     0.6525   -0.8388
confidence    -0.2383   -1.4979
```

## Percentiles (Continuous Features)


### Identity

```
                    p1     p5    p10    p25    p50    p75     p90      p95      p99
account_age_days  58.0  112.0  164.0  299.0  543.0  903.0  1335.0  1646.05  2352.01
```

### Voice Biometrics

```
                        p1      p5     p10     p25     p50     p75     p90     p95     p99
speaker_similarity  0.8497  0.9875  0.9991  0.9997  1.0000  1.0000  1.0000  1.0000  1.0000
liveness_score      0.6064  0.9056  0.9248  0.9466  0.9682  0.9890  0.9997  1.0000  1.0000
audio_quality       0.6970  0.8051  0.8335  0.8746  0.9172  0.9584  0.9959  0.9999  1.0000
spoof_probability   0.0000  0.0000  0.0028  0.0148  0.0300  0.0458  0.0604  0.0697  0.2055
```

### Behavior

```
                              p1      p5     p10     p25     p50     p75     p90     p95     p99
speech_rate_similarity    0.4747  0.6218  0.6517  0.6935  0.7360  0.7770  0.8111  0.8310  0.8672
pronunciation_similarity  0.6853  0.8200  0.8387  0.8635  0.8884  0.9125  0.9340  0.9465  0.9705
command_familiarity       0.5847  0.7234  0.7623  0.8171  0.8748  0.9310  0.9797  0.9989  1.0000
stress_score              0.0011  0.0577  0.0946  0.1570  0.2272  0.2984  0.3673  0.4153  0.5448
hesitation_score          0.0563  0.1181  0.1518  0.2113  0.2812  0.3514  0.4160  0.4558  0.5402
```

### Vehicle Context

```
                          p1      p5     p10      p25      p50      p75      p90      p95      p99
vehicle_speed         0.0000  0.0753  0.2585  10.8100  28.4661  42.3549  54.1400  61.1207  73.8114
location_familiarity  0.6561  0.7103  0.7439   0.8038   0.8710   0.9364   0.9853   0.9993   1.0000
time_familiarity      0.5844  0.6386  0.6746   0.7405   0.8173   0.8923   0.9549   0.9839   1.0000
```

### Historical Profile

```
                              p1        p5       p10        p25        p50        p75        p90        p95      p99
previous_trust_score      0.4147    0.4875    0.5333     0.6241     0.7323     0.8372     0.9218     0.9611     1.00
successful_transactions  20.0000  347.0000  660.0000  1196.0000  1793.0000  2392.0000  2937.0000  3266.0000  3823.01
```

### Transaction

```
                           p1      p5     p10       p25        p50         p75         p90         p95         p99
transaction_amount     0.0000  0.0000  0.0000  100.0000  6899.0850  27744.8525  42727.2590  50967.8615  68790.5451
beneficiary_frequency  0.1404  0.2828  0.3784    0.5600     0.7493      0.8870      0.9568      0.9786      0.9958
transaction_risk       0.0449  0.0636  0.0765    0.1025     0.1376      0.1776      0.2638      0.3648      0.4238
```

### Intent

```
                    p1      p5     p10     p25     p50     p75     p90     p95  p99
llm_confidence  0.9147  0.9234  0.9294  0.9427  0.9597  0.9766  0.9895  0.9952  1.0
```

### Unexpected / Non-Schema

```
                 p1      p5     p10     p25     p50     p75     p90     p95     p99
trust_score  0.8135  0.8376  0.8470  0.8603  0.8659  0.8694  0.8716  0.8731  0.8752
risk_score   0.0646  0.1131  0.1623  0.2393  0.3145  0.6365  0.7996  0.8746  0.9587
confidence   0.5230  0.5281  0.5331  0.5662  0.8099  0.9027  0.9672  0.9804  0.9881
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
kyc_verified      1=97,905 (97.9%); 0=2,095 (2.1%)
phone_verified    1=98,260 (98.3%); 0=1,740 (1.7%)
email_verified    1=96,283 (96.3%); 0=3,717 (3.7%)
voice_enrolled  1=78,000 (78.0%); 0=22,000 (22.0%)
```

### Vehicle Context

```
                                         Distribution
Feature                                              
engine_running     1=84,906 (84.9%); 0=15,094 (15.1%)
driver_present       1=97,300 (97.3%); 0=2,700 (2.7%)
seatbelt_fastened    1=92,822 (92.8%); 0=7,178 (7.2%)
```

### Historical Profile

```
                                   Distribution
Feature                                        
fraud_history  0=97,569 (97.6%); 1=2,431 (2.4%)
```

### Transaction

```
                                        Distribution
Feature                                             
beneficiary_type  0=88,266 (88.3%); 1=11,734 (11.7%)
```

## Categorical Feature Analysis


### Historical Profile

```
                 Categories  Top Category  Top Share %  Entropy (bits)  Normalized Entropy
Feature                                                                                   
failed_attempts           7             0        54.65           1.486               0.529
```

### Transaction

```
                      Categories  Top Category  Top Share %  Entropy (bits)  Normalized Entropy
Feature                                                                                        
transaction_category           4             1        55.25           1.677               0.838
```

### Intent

```
             Categories  Top Category  Top Share %  Entropy (bits)  Normalized Entropy
Feature                                                                               
intent_type           4             1        55.25           1.677               0.838
```

### Unexpected / Non-Schema

```
          Categories  Top Category  Top Share %  Entropy (bits)  Normalized Entropy
Feature                                                                            
decision           4             0        61.41           1.518               0.759
```

### failed_attempts -- Top Categories

- 0: 54,649 (54.65%)
- 1: 32,916 (32.92%)
- 2: 9,945 (9.95%)
- 3: 2,095 (2.10%)
- 4: 343 (0.34%)
- 5: 51 (0.05%)
- 6: 1 (0.00%)

### transaction_category -- Top Categories

- 1: 55,246 (55.25%)
- 2: 19,925 (19.93%)
- 3: 14,877 (14.88%)
- 0: 9,952 (9.95%)

### intent_type -- Top Categories

- 1: 55,246 (55.25%)
- 2: 19,925 (19.93%)
- 3: 14,877 (14.88%)
- 0: 9,952 (9.95%)

### decision -- Top Categories

- 0: 61,411 (61.41%)
- 1: 19,864 (19.86%)
- 2: 12,611 (12.61%)
- 3: 6,114 (6.11%)

## Cardinality


### Identifier

```
         Unique Values Type
user_id         100000   id
```

### Identity

```
                  Unique Values        Type
account_age_days           2931  continuous
kyc_verified                  2      binary
phone_verified                2      binary
email_verified                2      binary
voice_enrolled                2      binary
```

### Voice Biometrics

```
                    Unique Values        Type
speaker_similarity          54660  continuous
liveness_score              92185  continuous
audio_quality               95472  continuous
spoof_probability            4539  continuous
```

### Behavior

```
                          Unique Values        Type
speech_rate_similarity           100000  continuous
pronunciation_similarity          99966  continuous
command_familiarity               96870  continuous
stress_score                      99331  continuous
hesitation_score                  99944  continuous
```

### Vehicle Context

```
                      Unique Values        Type
vehicle_speed                 48937  continuous
engine_running                    2      binary
location_familiarity          96406  continuous
time_familiarity              98505  continuous
driver_present                    2      binary
seatbelt_fastened                 2      binary
```

### Historical Profile

```
                         Unique Values         Type
previous_trust_score             98867   continuous
failed_attempts                      7  categorical
successful_transactions           4180   continuous
fraud_history                        2       binary
```

### Transaction

```
                       Unique Values         Type
transaction_amount             69442   continuous
transaction_category               4  categorical
beneficiary_type                   2       binary
beneficiary_frequency         100000   continuous
transaction_risk              100000   continuous
```

### Intent

```
                Unique Values         Type
intent_type                 4  categorical
llm_confidence          99032   continuous
```

### Unexpected / Non-Schema

```
             Unique Values         Type
trust_score           1149   continuous
risk_score            9558   continuous
decision                 4  categorical
confidence            4731   continuous
```

## Correlation Analysis


### Top Positive Correlations

```
             Feature A                Feature B        r
  previous_trust_score  successful_transactions 0.910118
speech_rate_similarity      command_familiarity 0.781619
          stress_score         hesitation_score 0.744839
    speaker_similarity           liveness_score 0.712230
speech_rate_similarity pronunciation_similarity 0.580811
  location_familiarity         time_familiarity 0.527091
        liveness_score              trust_score 0.506238
    speaker_similarity              trust_score 0.500026
  previous_trust_score              trust_score 0.408204
      transaction_risk               risk_score 0.390415
```

### Top Negative Correlations

```
               Feature A         Feature B         r
              risk_score        confidence -0.956460
      speaker_similarity spoof_probability -0.802474
          liveness_score spoof_probability -0.771737
   beneficiary_frequency  transaction_risk -0.621375
pronunciation_similarity      stress_score -0.584006
       spoof_probability       trust_score -0.545916
  speech_rate_similarity  hesitation_score -0.488155
  speech_rate_similarity      stress_score -0.455926
pronunciation_similarity  hesitation_score -0.442708
           audio_quality spoof_probability -0.421008
```

### Highly Correlated Pairs (|r| >= 0.8)

```
           Feature A               Feature B         r
  speaker_similarity       spoof_probability -0.802474
previous_trust_score successful_transactions  0.910118
          risk_score              confidence -0.956460
```

### Redundant Features (|r| >= 0.95)

```
 Feature A  Feature B        r
risk_score confidence -0.95646
```

## Multicollinearity (VIF)

```
                            VIF
risk_score                14.89
confidence                14.69
previous_trust_score       8.57
successful_transactions    7.97
speech_rate_similarity     5.01
spoof_probability          4.52
command_familiarity        3.37
speaker_similarity         3.11
liveness_score             2.95
stress_score               2.81
hesitation_score           2.73
trust_score                2.12
pronunciation_similarity   2.03
transaction_risk           2.00
beneficiary_frequency      1.89
location_familiarity       1.41
time_familiarity           1.39
transaction_amount         1.34
audio_quality              1.32
account_age_days           1.13
llm_confidence             1.03
vehicle_speed              1.00
```

Rule of thumb: VIF > 10 suggests problematic multicollinearity; VIF > 5 is worth investigating.

## Outlier Analysis (Continuous Features)


### Identity

```
                  IQR_Outliers  Z_Outliers  Outlier %
Feature                                              
account_age_days          3482        1514       3.48
```

### Voice Biometrics

```
                    IQR_Outliers  Z_Outliers  Outlier %
Feature                                                
speaker_similarity         10250        1674      10.25
liveness_score              3313        2193       3.31
audio_quality               1842        1258       1.84
spoof_probability           1901        1372       1.90
```

### Behavior

```
                          IQR_Outliers  Z_Outliers  Outlier %
Feature                                                      
speech_rate_similarity            2251        1356       2.25
pronunciation_similarity          2911        1819       2.91
command_familiarity               1637        1099       1.64
stress_score                      1469         839       1.47
hesitation_score                   784         380       0.78
```

### Vehicle Context

```
                      IQR_Outliers  Z_Outliers  Outlier %
Feature                                                  
vehicle_speed                   89         118       0.12
location_familiarity           271         236       0.27
time_familiarity                84          59       0.08
```

### Historical Profile

```
                         IQR_Outliers  Z_Outliers  Outlier %
Feature                                                     
previous_trust_score               32          27       0.03
successful_transactions           187          50       0.19
```

### Transaction

```
                       IQR_Outliers  Z_Outliers  Outlier %
Feature                                                   
transaction_amount              968         838       0.97
beneficiary_frequency           230         113       0.23
transaction_risk               9011        1957       9.01
```

### Intent

```
                IQR_Outliers  Z_Outliers  Outlier %
Feature                                            
llm_confidence            55          63       0.06
```

### Unexpected / Non-Schema

```
             IQR_Outliers  Z_Outliers  Outlier %
Feature                                         
trust_score          9757        1642       9.76
risk_score              0           0       0.00
confidence              0           0       0.00
```

## Feature Quality


### Identity

```
                  Quality
account_age_days    100.0
kyc_verified        100.0
email_verified      100.0
voice_enrolled      100.0
phone_verified       85.0
```

### Voice Biometrics

```
                    Quality
audio_quality         100.0
liveness_score         80.0
spoof_probability      80.0
speaker_similarity     65.0
```

### Behavior

```
                          Quality
speech_rate_similarity      100.0
command_familiarity         100.0
stress_score                100.0
hesitation_score            100.0
pronunciation_similarity     80.0
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
transaction_risk          85.0
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
trust_score     65.0
```

## Recommendations

- Schema: add missing required columns: ['transaction_id'].
- Schema: ['trust_score', 'risk_score', 'decision', 'confidence'] are not part of the declared model schema -- confirm they're intentional (e.g. metadata) before training.
- Leakage: label columns ['trust_score', 'risk_score', 'decision', 'confidence'] are present in this file -- exclude them from model input features.
- Distribution: `speaker_similarity` has skewness -8.32 -- consider a log/Box-Cox transform.
- Distribution: `liveness_score` has skewness -5.49 -- consider a log/Box-Cox transform.
- Distribution: `spoof_probability` has skewness 4.97 -- consider a log/Box-Cox transform.
- Distribution: `pronunciation_similarity` has skewness -2.05 -- consider a log/Box-Cox transform.
- Distribution: `trust_score` has skewness -4.18 -- consider a log/Box-Cox transform.
- Distribution: `speaker_similarity` has kurtosis 75.82 -- heavy tails, check for extreme outliers.
- Distribution: `liveness_score` has kurtosis 39.12 -- heavy tails, check for extreme outliers.
- Distribution: `spoof_probability` has kurtosis 38.63 -- heavy tails, check for extreme outliers.
- Distribution: `pronunciation_similarity` has kurtosis 9.68 -- heavy tails, check for extreme outliers.
- Distribution: `trust_score` has kurtosis 34.91 -- heavy tails, check for extreme outliers.
- Consistency: binary feature(s) ['phone_verified'] are >98% one class -- consider class weighting or resampling.
- Distribution: ['speaker_similarity', 'transaction_risk', 'trust_score'] have >5% flagged outliers -- verify these are real, not generator bugs.
- Correlation: near-duplicate feature pairs risk_score/confidence (r=-0.96) -- consider dropping one from each pair.
- Correlation: ['risk_score', 'confidence'] have VIF > 10 -- multicollinearity risk for linear models.

## Dataset Health Score

```
                 Weight %  Score /100  Weighted Contribution
Sub-score                                                   
Schema                 20        50.0                  10.00
Completeness           15       100.0                  15.00
Distribution           15        77.3                  11.59
Consistency            15        86.0                  12.90
Correlation            15        79.0                  11.85
Feature Quality        20        95.4                  19.09
```

- **Overall Weighted Health Score: 80.4/100 (Fair)**