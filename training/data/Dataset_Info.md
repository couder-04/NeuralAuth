# Dataset Information


## Dataset Overview

- Rows: 100,000
- Columns: 36
- Memory Usage: 27.47 MB
- Dataset Fingerprint (MD5): `cac39634d3b685a9289ef6faf50ea1b6`


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
                      mean    median  mode      std     variance      cv   min     max   range  unique
account_age_days  667.3641  542.5923  30.0  500.777  250777.6028  0.7504  30.0  5000.0  4970.0  100000
```

### Voice Biometrics

```
                      mean  median  mode     std  variance      cv     min     max   range  unique
speaker_similarity  0.9831  0.9852   1.0  0.0106    0.0001  0.0108  0.9308  1.0000  0.0692   99940
liveness_score      0.9675  0.9700   1.0  0.0259    0.0007  0.0268  0.8397  1.0000  0.1603   91984
audio_quality       0.9147  0.9193   1.0  0.0610    0.0037  0.0667  0.4051  1.0000  0.5949   95348
spoof_probability   0.0293  0.0287   0.0  0.0193    0.0004  0.6566  0.0000  0.0945  0.0945    2718
```

### Behavior

```
                            mean  median    mode     std  variance      cv     min     max   range  unique
speech_rate_similarity    0.7379  0.7390  0.4994  0.0584    0.0034  0.0791  0.4994  0.9530  0.4537  100000
pronunciation_similarity  0.8902  0.8902  1.0000  0.0347    0.0012  0.0390  0.7420  1.0000  0.2580   99961
command_familiarity       0.8757  0.8784  1.0000  0.0766    0.0059  0.0875  0.5517  1.0000  0.4483   96728
stress_score              0.2244  0.2227  0.0000  0.1022    0.0104  0.4554  0.0000  0.8344  0.8344   99336
hesitation_score          0.2784  0.2777  0.0000  0.0988    0.0098  0.3549  0.0000  0.7243  0.7243   99952
```

### Vehicle Context

```
                         mean   median  mode      std  variance      cv     min      max     range  unique
vehicle_speed         27.7957  28.3800   0.0  19.9260  397.0453  0.7169  0.0000  109.336  109.3360   48961
location_familiarity   0.8707   0.8745   1.0   0.0845    0.0071  0.0970  0.3107    1.000    0.6893   96217
time_familiarity       0.8181   0.8203   1.0   0.1005    0.0101  0.1228  0.4584    1.000    0.5416   98470
```

### Historical Profile

```
                              mean     median  mode       std     variance      cv      min        max      range  unique
previous_trust_score        0.7308     0.7344   1.0    0.1442       0.0208  0.1973   0.1694     1.0000     0.8306   98802
successful_transactions  1807.9815  1802.0000  20.0  869.9056  756735.7668  0.4811  20.0000  4602.1295  4582.1295   14363
```

### Transaction

```
                             mean     median    mode         std      variance      cv     min          max        range  unique
transaction_amount     15008.3837  6744.4050  0.0000  17189.4842  2.954784e+08  1.1453  0.0000  104638.7700  104638.7700   68838
beneficiary_frequency      0.7045     0.7478  0.0000      0.2182  4.760000e-02  0.3098  0.0000       1.0000       1.0000  100000
transaction_risk           0.0973     0.0875  0.0098      0.0517  2.700000e-03  0.5317  0.0098       0.4155       0.4057  100000
```

### Intent

```
                  mean  median  mode     std  variance      cv     min  max   range  unique
llm_confidence  0.9599  0.9599   1.0  0.0215    0.0005  0.0224  0.8976  1.0  0.1024   98945
```

### Unexpected / Non-Schema

```
               mean  median    mode     std  variance      cv     min     max   range  unique
trust_score  0.8624  0.8656  0.8672  0.0108    0.0001  0.0125  0.7665  0.8773  0.1108     815
risk_score   0.4230  0.3063  0.2822  0.2554    0.0652  0.6037  0.0271  0.9984  0.9713    9557
confidence   0.7630  0.8183  0.5308  0.1672    0.0280  0.2192  0.5180  0.9924  0.4744    4734
```

## Distribution Analysis (Continuous Features)


### Identity

```
                  skewness  kurtosis
account_age_days    1.5522    3.5767
```

### Voice Biometrics

```
                    skewness  kurtosis
speaker_similarity   -0.8168    0.2241
liveness_score       -0.5855   -0.2393
audio_quality        -0.9628    2.4414
spoof_probability     0.2502   -0.7232
```

### Behavior

```
                          skewness  kurtosis
speech_rate_similarity     -0.1189   -0.1005
pronunciation_similarity   -0.0159   -0.0494
command_familiarity        -0.3110   -0.3464
stress_score                0.2238    0.1998
hesitation_score            0.0695   -0.1327
```

### Vehicle Context

```
                      skewness  kurtosis
vehicle_speed           0.2184   -0.7053
location_familiarity   -0.2953   -0.5908
time_familiarity       -0.1343   -0.6638
```

### Historical Profile

```
                         skewness  kurtosis
previous_trust_score      -0.1410   -0.6632
successful_transactions    0.0831   -0.3736
```

### Transaction

```
                       skewness  kurtosis
transaction_amount       1.0478    0.1620
beneficiary_frequency   -0.7069   -0.2975
transaction_risk         0.9732    0.8001
```

### Intent

```
                skewness  kurtosis
llm_confidence   -0.0292    -0.866
```

### Unexpected / Non-Schema

```
             skewness  kurtosis
trust_score   -2.0602    5.3672
risk_score     0.6542   -0.9389
confidence    -0.2907   -1.4820
```

## Percentiles (Continuous Features)


### Identity

```
                       p1        p5       p10       p25       p50       p75        p90        p95       p99
account_age_days  57.5853  112.1434  164.1023  299.1673  542.5923  902.7737  1335.2285  1645.6819  2353.857
```

### Voice Biometrics

```
                        p1      p5     p10     p25     p50     p75     p90     p95     p99
speaker_similarity  0.9541  0.9625  0.9679  0.9765  0.9852  0.9916  0.9953  0.9967  0.9982
liveness_score      0.9002  0.9207  0.9318  0.9498  0.9700  0.9902  0.9999  1.0000  1.0000
audio_quality       0.7352  0.8139  0.8388  0.8774  0.9193  0.9601  0.9969  0.9999  1.0000
spoof_probability   0.0000  0.0000  0.0020  0.0138  0.0287  0.0434  0.0560  0.0621  0.0729
```

### Behavior

```
                              p1      p5     p10     p25     p50     p75     p90     p95     p99
speech_rate_similarity    0.5988  0.6401  0.6619  0.6986  0.7390  0.7788  0.8126  0.8317  0.8660
pronunciation_similarity  0.8095  0.8330  0.8456  0.8666  0.8902  0.9138  0.9348  0.9473  0.9704
command_familiarity       0.6881  0.7440  0.7740  0.8231  0.8784  0.9334  0.9811  0.9992  1.0000
stress_score              0.0011  0.0567  0.0934  0.1546  0.2227  0.2917  0.3547  0.3939  0.4767
hesitation_score          0.0568  0.1167  0.1506  0.2097  0.2777  0.3455  0.4062  0.4423  0.5099
```

### Vehicle Context

```
                          p1      p5     p10      p25      p50      p75      p90      p95      p99
vehicle_speed         0.0000  0.0729  0.2500  10.6600  28.3800  42.2200  53.9945  60.9670  73.8234
location_familiarity  0.6803  0.7248  0.7541   0.8094   0.8745   0.9386   0.9865   0.9995   1.0000
time_familiarity      0.5990  0.6482  0.6814   0.7454   0.8203   0.8937   0.9554   0.9841   1.0000
```

### Historical Profile

```
                              p1        p5       p10        p25        p50        p75        p90        p95        p99
previous_trust_score      0.4162    0.4898    0.5338     0.6249     0.7344     0.8412     0.9237     0.9631     1.0000
successful_transactions  20.0000  324.0000  656.0000  1201.0000  1802.0000  2402.0000  2950.0000  3267.0000  3822.0051
```

### Transaction

```
                           p1      p5     p10     p25        p50         p75         p90         p95         p99
transaction_amount     0.0000  0.0000  0.0000  0.0000  6744.4050  27008.7375  41392.1020  48947.1620  62305.6072
beneficiary_frequency  0.1423  0.2837  0.3807  0.5603     0.7478      0.8859      0.9567      0.9789      0.9958
transaction_risk       0.0214  0.0310  0.0398  0.0587     0.0875      0.1241      0.1721      0.2012      0.2486
```

### Intent

```
                    p1      p5     p10     p25     p50     p75     p90     p95  p99
llm_confidence  0.9176  0.9246  0.9303  0.9433  0.9599  0.9765  0.9895  0.9954  1.0
```

### Unexpected / Non-Schema

```
                 p1      p5     p10     p25     p50     p75     p90     p95     p99
trust_score  0.8227  0.8389  0.8475  0.8601  0.8656  0.8691  0.8715  0.8730  0.8753
risk_score   0.0617  0.1080  0.1568  0.2399  0.3063  0.6669  0.8252  0.8902  0.9619
confidence   0.5220  0.5267  0.5308  0.5564  0.8183  0.9034  0.9695  0.9815  0.9885
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
kyc_verified      1.0=97,930 (97.9%); 0.0=2,070 (2.1%)
phone_verified    1.0=98,318 (98.3%); 0.0=1,682 (1.7%)
email_verified    1.0=96,127 (96.1%); 0.0=3,873 (3.9%)
voice_enrolled  1.0=78,093 (78.1%); 0.0=21,907 (21.9%)
```

### Vehicle Context

```
                                             Distribution
Feature                                                  
engine_running     1.0=84,897 (84.9%); 0.0=15,103 (15.1%)
driver_present       1.0=97,432 (97.4%); 0.0=2,568 (2.6%)
seatbelt_fastened    1.0=93,094 (93.1%); 0.0=6,906 (6.9%)
```

### Historical Profile

```
                                       Distribution
Feature                                            
fraud_history  0.0=97,760 (97.8%); 1.0=2,240 (2.2%)
```

### Transaction

```
                                        Distribution
Feature                                             
beneficiary_type  0=88,822 (88.8%); 1=11,178 (11.2%)
```

## Categorical Feature Analysis


### Historical Profile

```
                 Categories  Top Category  Top Share %  Entropy (bits)  Normalized Entropy
Feature                                                                                   
failed_attempts           6           0.0        54.85           1.474                0.57
```

### Transaction

```
                      Categories  Top Category  Top Share %  Entropy (bits)  Normalized Entropy
Feature                                                                                        
transaction_category           4             1        54.86           1.685               0.842
```

### Intent

```
             Categories  Top Category  Top Share %  Entropy (bits)  Normalized Entropy
Feature                                                                               
intent_type           4             1        54.86           1.685               0.842
```

### Unexpected / Non-Schema

```
          Categories  Top Category  Top Share %  Entropy (bits)  Normalized Entropy
Feature                                                                            
decision           4             0        61.52           1.515               0.757
```

### failed_attempts -- Top Categories

- 0.0: 54,851 (54.85%)
- 1.0: 32,980 (32.98%)
- 2.0: 9,838 (9.84%)
- 3.0: 2,016 (2.02%)
- 4.0: 276 (0.28%)
- 5.0: 39 (0.04%)

### transaction_category -- Top Categories

- 1: 54,863 (54.86%)
- 2: 19,985 (19.98%)
- 3: 15,017 (15.02%)
- 0: 10,135 (10.13%)

### intent_type -- Top Categories

- 1: 54,863 (54.86%)
- 2: 19,985 (19.98%)
- 3: 15,017 (15.02%)
- 0: 10,135 (10.13%)

### decision -- Top Categories

- 0: 61,517 (61.52%)
- 1: 20,141 (20.14%)
- 2: 12,180 (12.18%)
- 3: 6,162 (6.16%)

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
speaker_similarity          99940  continuous
liveness_score              91984  continuous
audio_quality               95348  continuous
spoof_probability            2718  continuous
```

### Behavior

```
                          Unique Values        Type
speech_rate_similarity           100000  continuous
pronunciation_similarity          99961  continuous
command_familiarity               96728  continuous
stress_score                      99336  continuous
hesitation_score                  99952  continuous
```

### Vehicle Context

```
                      Unique Values        Type
vehicle_speed                 48961  continuous
engine_running                    2      binary
location_familiarity          96217  continuous
time_familiarity              98470  continuous
driver_present                    2      binary
seatbelt_fastened                 2      binary
```

### Historical Profile

```
                         Unique Values         Type
previous_trust_score             98802   continuous
failed_attempts                      6  categorical
successful_transactions          14363   continuous
fraud_history                        2       binary
```

### Transaction

```
                       Unique Values         Type
transaction_amount             68838   continuous
transaction_category               4  categorical
beneficiary_type                   2       binary
beneficiary_frequency         100000   continuous
transaction_risk              100000   continuous
```

### Intent

```
                Unique Values         Type
intent_type                 4  categorical
llm_confidence          98945   continuous
```

### Unexpected / Non-Schema

```
             Unique Values         Type
trust_score            815   continuous
risk_score            9557   continuous
decision                 4  categorical
confidence            4734   continuous
```

## Correlation Analysis


### Top Positive Correlations

```
              Feature A                Feature B        r
   previous_trust_score  successful_transactions 0.912899
           stress_score         hesitation_score 0.719146
 speech_rate_similarity      command_familiarity 0.696582
   location_familiarity         time_familiarity 0.495201
   previous_trust_score              trust_score 0.468302
successful_transactions              trust_score 0.455929
       transaction_risk               risk_score 0.433898
 speech_rate_similarity pronunciation_similarity 0.324152
     transaction_amount               risk_score 0.321485
     transaction_amount         transaction_risk 0.319147
```

### Top Negative Correlations

```
               Feature A         Feature B         r
              risk_score        confidence -0.961811
          liveness_score spoof_probability -0.567669
   beneficiary_frequency  transaction_risk -0.544806
pronunciation_similarity      stress_score -0.488945
        transaction_risk        confidence -0.445071
  speech_rate_similarity  hesitation_score -0.427939
      transaction_amount        confidence -0.371884
pronunciation_similarity  hesitation_score -0.364284
           audio_quality spoof_probability -0.357683
  speech_rate_similarity      stress_score -0.328813
```

### Highly Correlated Pairs (|r| >= 0.8)

```
           Feature A               Feature B         r
previous_trust_score successful_transactions  0.912899
          risk_score              confidence -0.961811
```

### Redundant Features (|r| >= 0.95)

```
 Feature A  Feature B         r
risk_score confidence -0.961811
```

## Multicollinearity (VIF)

```
                            VIF
confidence                16.72
risk_score                16.52
previous_trust_score       8.79
successful_transactions    8.24
speech_rate_similarity     3.28
command_familiarity        2.58
hesitation_score           2.51
stress_score               2.45
spoof_probability          1.89
transaction_risk           1.84
beneficiary_frequency      1.63
trust_score                1.61
liveness_score             1.59
pronunciation_similarity   1.46
transaction_amount         1.34
time_familiarity           1.33
location_familiarity       1.33
audio_quality              1.25
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
account_age_days          3494        1516       3.49
```

### Voice Biometrics

```
                    IQR_Outliers  Z_Outliers  Outlier %
Feature                                                
speaker_similarity           949         510       0.95
liveness_score               362         385       0.39
audio_quality               1246         959       1.25
spoof_probability             39          43       0.04
```

### Behavior

```
                          IQR_Outliers  Z_Outliers  Outlier %
Feature                                                      
speech_rate_similarity             532         232       0.53
pronunciation_similarity           617         257       0.62
command_familiarity                367         230       0.37
stress_score                       675         356       0.68
hesitation_score                   469         157       0.47
```

### Vehicle Context

```
                      IQR_Outliers  Z_Outliers  Outlier %
Feature                                                  
vehicle_speed                   81         112       0.11
location_familiarity            71          71       0.07
time_familiarity                41          29       0.04
```

### Historical Profile

```
                         IQR_Outliers  Z_Outliers  Outlier %
Feature                                                     
previous_trust_score               38          37       0.04
successful_transactions           141          36       0.14
```

### Transaction

```
                       IQR_Outliers  Z_Outliers  Outlier %
Feature                                                   
transaction_amount              458         519       0.52
beneficiary_frequency           226         109       0.23
transaction_risk               2698         832       2.70
```

### Intent

```
                IQR_Outliers  Z_Outliers  Outlier %
Feature                                            
llm_confidence             0           0        0.0
```

### Unexpected / Non-Schema

```
             IQR_Outliers  Z_Outliers  Outlier %
Feature                                         
trust_score          9373        2104       9.37
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
- Distribution: `trust_score` has skewness -2.06 -- consider a log/Box-Cox transform.
- Consistency: binary feature(s) ['phone_verified'] are >98% one class -- consider class weighting or resampling.
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
Consistency            15        92.0                  13.80
Correlation            15        81.0                  12.15
Feature Quality        20        98.9                  19.77
```

- **Overall Weighted Health Score: 85.4/100 (Good)**