# IDS Study — Comparative ML Benchmark and HSR-IDS for Rare-Attack Detection on NSL-KDD

A reproducible study of network intrusion detection on the NSL-KDD benchmark, in two parts:

1. **A comparative benchmark** of seven classical machine learning algorithms across binary and five-class formulations, evaluated on eight metrics with cross-validation and paired significance testing.
2. **HSR-IDS** (*Hierarchical Specialist Router for Intrusion Detection*) — a two-stage architecture targeting the rare-attack failure mode the benchmark exposes.

> **Correction.** This README previously headlined Macro-F1 54.15% → 61.59% and R2L F1 1.51% → 23.40% at 11.39% routed. Those thresholds were selected on the test set. With thresholds selected on training data only, under a protocol [committed before it ran](experiments/hsr_validation/PROTOCOL.md), HSR-IDS reaches a **median Macro-F1 of 57.51%** over ten seeds (range 54.81–61.45), with **median R2L F1 of 4.13%** and a median 4.33% routed. It still beats its XGBoost gate (+3.36 points, 95% CI [1.50, 5.36]), mainly on U2R. However, **plain XGBoost with SMOTE on all classes reaches a median 61.61%** with no routing and no thresholds. See [Corrected results](#corrected-results).

All numbers in this README come from the committed files under `results/` and `experiments/hsr_validation/`. Nothing here is estimated or rounded from memory.

---

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Part 1 — Comparative Benchmark](#part-1--comparative-benchmark)
  - [Binary classification](#binary-classification-attack-vs-normal)
  - [Statistical validation](#statistical-validation)
  - [Five-class classification](#five-class-classification)
  - [Feature analysis](#feature-analysis)
- [Part 2 — HSR-IDS](#part-2--hsr-ids)
  - [Architecture](#architecture)
  - [The routing rule](#the-routing-rule-and-why-it-is-not-the-obvious-one)
  - [Corrected results](#corrected-results)
  - [Superseded test-tuned results](#superseded-test-tuned-results)
  - [Ablation](#ablation-what-actually-does-the-work)
  - [Threshold sensitivity](#threshold-sensitivity)
  - [SHAP interpretability](#shap-interpretability)
- [Repository Structure](#repository-structure)
- [Reproducing the Results](#reproducing-the-results)
- [Limitations and Honest Caveats](#limitations-and-honest-caveats)
- [Future Work](#future-work)
- [Citation](#citation)
- [License and Acknowledgements](#license-and-acknowledgements)

---

## Motivation

Intrusion detection systems are evaluated overwhelmingly on aggregate accuracy or binary F1. Those metrics are dominated by the majority classes. On NSL-KDD, DoS and normal traffic together account for roughly 90% of the training set, so a model can look excellent while being effectively blind to the attack categories that matter most operationally.

This study asks two questions:

1. **How bad is the rare-class problem, exactly, and is it universal?** We benchmark seven algorithms and report per-class F1, not just aggregates.
2. **Can it be fixed without degrading the common classes or the throughput?** Most rare-class remedies (global resampling, cost-sensitive loss, full ensembles) trade majority-class performance or inference cost for minority-class recall. HSR-IDS is designed to avoid both trades.

The central empirical finding motivating the method: **no single model is best everywhere.** XGBoost is the best binary model but only fifth of seven by Macro-F1, and it has the second-worst R2L F1 (1.51%; Logistic Regression is 0.48%). Naive Bayes has the worst overall Macro-F1 (34.95%) but the *best* R2L F1 (36.71%). KNN is best at U2R (42.00%). The models disagree most precisely on the rare classes — which is an opening, not just a nuisance.

---

## Dataset

**NSL-KDD**, the revised version of KDD Cup '99 produced at the University of New Brunswick, which removes the duplicate records that made the original trivially easy.

| File | Rows | Role |
|---|---|---|
| `KDDTrain+.txt` | 125,973 | Training |
| `KDDTest+.txt` | 22,544 | Testing |

Each row is one network connection described by 41 features (3 categorical: `protocol_type`, `service`, `flag`; 38 numeric) plus a label.

### Class distribution

| Class | Description | Train | Test |
|---|---|---:|---:|
| normal | Legitimate traffic | 67,343 | 9,711 |
| DoS | Denial of service | 45,927 | 7,458 |
| Probe | Reconnaissance / port scanning | 11,656 | 2,421 |
| R2L | Remote-to-local (e.g. password guessing) | **995** | 2,887 |
| U2R | User-to-root privilege escalation | **52** | 67 |

Two properties of this table drive everything that follows:

- **Severe imbalance.** U2R has 52 training instances against 67,343 normal — a ratio of roughly 1:1,295.
- **Train/test distribution shift.** The official test split deliberately contains attack variants absent from training, and the R2L proportion *inverts* (995 train → 2,887 test). This is why every model in the benchmark drops from >99% cross-validated F1 to 74–80% on the held-out test set. That gap is a feature of the benchmark, not a bug in the pipeline — it simulates a deployed detector meeting tomorrow's attacks.

> **Note on data files.** `KDDTrain+.txt` and `KDDTest+.txt` are included in `data/`, so the notebook runs without additional setup. NSL-KDD is publicly distributed for research by the [Canadian Institute for Cybersecurity, University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html).

---

## Preprocessing

Three steps, applied identically across every experiment:

1. **Label construction.** For the binary task, all four attack families collapse to `1`, normal to `0`. For the multiclass task, the 39 individual attack names in the raw labels are mapped to their five parent categories.
2. **Categorical encoding.** `protocol_type`, `service`, and `flag` are label-encoded. The encoder is fit on the *union* of train and test category vocabularies so that services appearing only in the test split do not raise unseen-label errors. This is deliberate and does not leak label information — only the set of category names is shared, never any target value or distributional statistic.
3. **Standardization.** All features are scaled with `StandardScaler`. **The scaler is fit on training data only** and then applied to test data. `src_bytes` ranges into the millions while `logged_in` is binary; without scaling, distance- and gradient-based models would weight features by magnitude rather than relevance.

**On leakage:** scaler statistics, SMOTE synthesis, and all model fitting occur strictly on the training partition. SMOTE in particular is applied *only* to the specialist's training set, never to any data used for evaluation — oversampling before splitting is the most common way published IDS results become inflated, and it is avoided here.

---

## Part 1 — Comparative Benchmark

Seven algorithms: Decision Tree, Random Forest, Naive Bayes, K-Nearest Neighbors (k=5), Logistic Regression, Gradient Boosting, and XGBoost.

### Binary classification (attack vs. normal)

Source: `results/main_results.csv`, `results/roc_auc.csv`

| Model | Accuracy % | Precision % | Recall % | F1 % | FPR % | AUC | Train (s) | Predict (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **80.67** | 96.79 | **68.32** | **80.10** | 3.00 | **0.9706** | 0.277 | 0.0033 |
| Gradient Boosting | 80.64 | 96.81 | 68.24 | 80.05 | 2.98 | 0.9625 | 10.324 | 0.0133 |
| Decision Tree | 78.77 | 96.49 | 65.07 | 77.72 | 3.13 | 0.8104 | 0.381 | 0.0012 |
| Naive Bayes | 77.14 | 91.14 | 66.29 | 76.75 | 8.52 | 0.8299 | 0.018 | 0.0021 |
| Random Forest | 76.99 | 96.61 | 61.75 | 75.34 | 2.86 | 0.9621 | 0.596 | 0.0127 |
| K-Nearest Neighbors | 76.76 | **97.12** | 60.98 | 74.92 | **2.39** | 0.8228 | 0.004 | 1.0606 |
| Logistic Regression | 75.37 | 92.51 | 61.73 | 74.05 | 6.60 | 0.8714 | 0.304 | 0.0003 |

**Confusion matrix for XGBoost** (the best model): TP 8,767 · TN 9,420 · FP 291 · FN 4,066.

Three observations worth stating plainly:

- **Precision is uniformly high (91–97%), recall is uniformly poor (61–68%).** Every model is conservative: when it raises an alarm it is usually right, but it misses roughly a third of attacks. Those 4,066 false negatives are novel attack variants in the test split.
- **XGBoost matches Gradient Boosting on accuracy while training ~37× faster** (0.277s vs 10.324s). This efficiency, plus the best AUC, is why XGBoost becomes the backbone of HSR-IDS.
- **KNN is operationally disqualified.** Its 1.06s prediction time — ~300× slower than XGBoost — reflects the lazy-learner cost of comparing every query against 125,973 stored instances. Unusable at line rate.

### Statistical validation

5-fold cross-validation on the training partition (`results/cross_validation.csv`) followed by paired t-tests (`results/statistical_tests.csv`).

| Model | Mean F1 % | Std | Min | Max |
|---|---:|---:|---:|---:|
| XGBoost | 99.90 | 0.03 | 99.85 | 99.93 |
| Random Forest | 99.88 | 0.03 | 99.84 | 99.91 |
| Decision Tree | 99.78 | 0.02 | 99.74 | 99.81 |
| Gradient Boosting | 99.56 | 0.02 | 99.53 | 99.59 |
| K-Nearest Neighbors | 99.55 | 0.02 | 99.53 | 99.59 |
| Logistic Regression | 94.99 | 0.25 | 94.75 | 95.32 |
| Naive Bayes | 88.62 | 0.35 | 88.18 | 89.04 |

The ~20-point gap between these cross-validated scores and the held-out test scores above is the **generalization gap** induced by the deliberate distribution shift in `KDDTest+`. Reporting only the CV column — as some published work does — would overstate real-world performance by roughly 20 points.

| Comparison | t-statistic | p-value | Significant (α=0.05) |
|---|---:|---:|:---:|
| XGBoost vs Decision Tree | 16.4704 | 0.0001 | ✅ |
| XGBoost vs Random Forest | 2.2517 | 0.0875 | ❌ |
| XGBoost vs Naive Bayes | 66.7337 | <0.0001 | ✅ |
| XGBoost vs K-Nearest Neighbors | 18.4777 | 0.0001 | ✅ |
| XGBoost vs Logistic Regression | 41.0278 | <0.0001 | ✅ |
| XGBoost vs Gradient Boosting | 58.3618 | <0.0001 | ✅ |

XGBoost's advantage is statistically significant over five of six competitors. **Against Random Forest it is not** (p = 0.0875) — reported here rather than omitted.

**What these tests cover.** They compare binary F1 by cross-validation inside `KDDTrain+`, where all strong models score near 99%. They say nothing about test performance, where XGBoost and Gradient Boosting are 80.10 and 80.05, or about HSR-IDS. The tests for HSR-IDS are in [Corrected results](#corrected-results).

### Five-class classification

Source: `results/multiclass_results.csv`. This is where the rare-class problem becomes unavoidable.

| Model | Accuracy % | **Macro-F1 %** | Weighted-F1 % | DoS | Probe | **R2L** | **U2R** | normal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 76.74 | **58.03** | 72.88 | 88.52 | 72.60 | 12.58 | 37.36 | 79.10 |
| Decision Tree | 76.90 | 57.40 | 73.62 | 87.38 | 66.49 | 21.17 | 31.25 | **80.71** |
| K-Nearest Neighbors | 74.98 | 56.93 | 70.58 | 84.87 | 74.51 | 4.91 | **42.00** | 78.35 |
| Logistic Regression | 76.14 | 54.53 | 71.35 | 86.88 | **79.30** | 0.48 | 27.16 | 78.82 |
| XGBoost | 76.12 | 54.15 | 71.41 | 87.33 | 78.36 | 1.51 | 25.00 | 78.55 |
| Random Forest | 75.08 | 50.40 | 70.85 | 85.36 | 76.31 | 7.01 | 5.56 | 77.78 |
| Naive Bayes | 43.31 | 34.95 | 34.75 | 55.73 | 66.84 | **36.71** | 5.19 | 10.26 |

**Finding 1 — DoS is solved.** Every model reaches 85–88% (Naive Bayes excepted). DoS floods produce extreme `src_bytes` and connection-count signatures that are trivially separable.

**Finding 2 — R2L is a universal failure.** The range is 0.48% to 36.71%. R2L traffic is behaviourally near-identical to a legitimate user mistyping a password; the feature set simply does not carry enough signal, and 995 training examples are not enough to find what signal exists.

**Finding 3 — The ranking inverts by class.** XGBoost, the strongest binary model, is the *second worst* at R2L. Naive Bayes, the weakest overall, is the *best* at R2L — its independence assumption, wrong for network data in general, happens not to be swamped by the majority classes the way discriminative models are. KNN leads on U2R.

Finding 3 is the direct motivation for HSR-IDS: if models have complementary and class-specific blind spots, a single monolithic classifier is the wrong shape for the problem.

**Note the gap between Macro-F1 and Weighted-F1** — for XGBoost, 54.15% vs 71.41%. Weighted-F1 lets the rare classes disappear into the majority. Macro-F1 refuses to. **Macro-F1 is the primary metric of this study.**

### Feature analysis

`results/feature_importance.csv` ranks all 41 features by ANOVA F-score and Random Forest impurity importance. The two rankings disagree instructively:

| Feature | ANOVA rank | RF rank |
|---|---:|---:|
| `same_srv_rate` | 1 | 5 |
| `dst_host_srv_count` | 2 | 6 |
| `dst_host_same_srv_rate` | 3 | 8 |
| `logged_in` | 4 | 4 |
| `src_bytes` | 35 | **1** |
| `dst_bytes` | 36 | **2** |
| `flag` | 9 | 3 |

`src_bytes` and `dst_bytes` rank near the bottom by ANOVA but first and second by RF importance. ANOVA tests for a *linear* mean-difference between classes; byte-volume features are informative through highly non-linear thresholds instead, which tree ensembles exploit and ANOVA cannot see. Any feature-selection pipeline relying on univariate filtering alone would discard the two most useful features in the dataset.

`results/feature_ablation.csv` reports F1 against top-*k* feature subsets:

| k | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 41 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 % | 79.58 | 75.61 | 78.71 | 79.31 | 78.50 | 78.21 | 79.57 | **80.06** |

The full 41-feature set wins, but only by 0.48 points over the top 5 — and the curve is non-monotonic (k=10 is the worst point). For deployment on constrained hardware, a 5-feature model recovering 99.4% of full-model F1 is a defensible trade. **All headline experiments use all 41 features.**

---

## Part 2 — HSR-IDS

### Architecture

HSR-IDS is a two-stage cascade. The analogy is triage: a general practitioner sees every patient, handles the routine cases, and refers only the ambiguous ones to a specialist trained deeply on exactly those hard cases.

```
                    incoming connection
                            │
                            ▼
             ┌──────────────────────────────┐
             │  STAGE 1 — THE GATE          │
             │  XGBoost, 5-class            │
             │  trained on full train set   │
             └──────────────┬───────────────┘
                            │  prediction ŷ + confidence p
                            ▼
                    ┌───────────────┐
                    │ ROUTING RULE  │
                    └───┬───────┬───┘
             route      │       │      accept gate verdict
     (see results)      │       │
                        ▼       ▼
        ┌───────────────────────┐   final prediction
        │ STAGE 2 — SPECIALIST  │
        │ XGBoost, 3-class      │
        │ {normal, R2L, U2R}    │
        │ SMOTE-balanced        │
        └───────────┬───────────┘
                    ▼
            final prediction
```

**Stage 1 — The Gate.** A standard 5-class XGBoost trained on the full training set. It produces a predicted class and a confidence (max softmax probability). It handles DoS, Probe, and unambiguous normal traffic well.

**Stage 2 — The Specialist.** A second XGBoost trained on a *restricted* three-class problem: `{normal, R2L, U2R}`. Narrowing the label space means the decision boundary no longer has to accommodate DoS and Probe, and the rare classes are no longer competing against 45,927 DoS instances for model capacity.

Its training set is SMOTE-balanced: R2L is synthesized from 995 up to 67,343, U2R from 52 up to 67,343, matching the normal class. SMOTE interpolates between real minority instances and their nearest minority neighbours, producing plausible synthetic points rather than duplicates.

### The routing rule, and why it is not the obvious one

A sample is escalated to the specialist if **either**:

- **(a)** Gate confidence < *low* — the Gate is genuinely uncertain; **or**
- **(b)** Gate predicts `normal` with confidence < *certain* — the Gate is confident, but not near-certain.

The values are selected from a 5×5 grid. Training-data selection gives (0.5, 0.9) most often; the published (0.5, 0.999) came from the test set.

Condition (b) is the non-obvious component and it is what makes the method work.

The intuitive design is "escalate when the model is unsure." **That design fails on R2L**, because the Gate is *not* unsure about R2L — it is **confidently wrong**, assigning ~99% probability to `normal`. A pure low-confidence trigger never fires on the exact samples the system exists to catch.

Condition (b) therefore re-examines *suspiciously confident normal verdicts*, exploiting an asymmetry in the cost structure: a `normal` verdict that is wrong is a missed intrusion, whereas an attack verdict that is wrong is a false alarm. The threshold is asymmetric because the errors are.

This is also why HSR-IDS is not an ordinary ensemble. Standard ensembles evaluate every model on every sample and average. HSR-IDS (i) invokes the second model on a minority of samples, (ii) scopes it to the specific known weakness, and (iii) triggers on a *directional* error mode rather than on generic uncertainty.

### Corrected results

Source: `experiments/hsr_validation/`. The protocol was fixed before tuning ran. `tune.py` cannot open `KDDTest+`, and thresholds were committed before `final_eval.py` ran once. `reproduce_baseline.py` reproduces the earlier run exactly, so these changes come from the protocol alone.

The primary arm holds out whole attack types for validation, a third per category, to preserve the novelty shift HSR-IDS targets. Ten seeds; each result refits on all of `KDDTrain+`.

| Model | Macro-F1 % | R2L F1 % | U2R F1 % | Routed % |
|---|---:|---:|---:|---:|
| XGBoost gate alone | 54.15 | 1.51 | 25.00 | 0 |
| Gradient Boosting | 58.03 | 12.58 | 37.36 | — |
| XGBoost, balanced class weights | 58.77 | 16.39 | 30.23 | — |
| **HSR-IDS, thresholds from training data** (median of 10) | **57.51** (54.81–61.45) | **4.13** (4.06–25.74) | **38.42** (34.48–42.70) | 4.33 (3.15–11.39) |
| HSR-IDS, stratified control arm (median of 10) | 57.53 (55.59–59.88) | 4.46 (4.06–11.76) | 38.86 (34.48–45.16) | 4.52 (3.15–6.14) |
| **XGBoost, SMOTE on all classes** (median of 10) | **61.61** (61.04–62.79) | **21.99** (18.44–24.55) | **40.84** (36.17–45.36) | — |
| *Superseded: HSR-IDS tuned on test* | *61.59* | *23.40* | *39.13* | *11.39* |

Per-class comparator scores come from `comparators_detail.py`, an exploratory addition: the protocol fixed only their Macro-F1.

**Reported model** (modal thresholds 0.5 and 0.9, seed 42):

| Class | Gate precision / recall / F1 | HSR-IDS precision / recall / F1 |
|---|---:|---:|
| DoS | 96.07 / 80.05 / 87.33 | 96.07 / 80.05 / 87.33 |
| Probe | 86.48 / 71.62 / 78.36 | 86.48 / 71.58 / 78.33 |
| R2L | 81.48 / 0.76 / 1.51 | 89.71 / 2.11 / 4.13 |
| U2R | 76.92 / 14.93 / 25.00 | 85.00 / 25.37 / 39.08 |
| normal | 65.97 / 97.04 / 78.55 | 66.18 / 97.03 / 78.69 |

Macro-F1 difference against the gate: **+3.36, 95% paired-bootstrap CI [1.50, 5.36]**. McNemar: 46 connections correct only under HSR-IDS, 2 only under the gate, exact p ≈ 8×10⁻¹². Test rows are resampled individually.

**What survives.** The gate's failure is confident error, as diagnosed. A specialist improves on the gate, chiefly on U2R, at low routing cost, and common classes hold.

**What does not.**

- **The R2L claim.** R2L F1 of 23.40% depends on `certain` = 0.999, and training data selected that value on 1 seed of 10. Without it, R2L stays near 4%.
- **The Macro-F1 claim.** Test-tuned selection on the same models gives a median 61.63%, so selection on the test set contributed a median 4.29 points. An earlier version of this README called the optimism small. It is not.
- **The claim to beat simple resampling.** Global SMOTE on the same XGBoost beats validated HSR-IDS by about 4 points and matches the published figure, without thresholds. Against the gate its DoS F1 falls by about 1.5 points and Probe rises by about 2.
- **The stability claim.** Selected thresholds vary across seeds, and a flat surface around 0.999 on the test set says nothing about whether 0.999 can be found without it.

**Sensitivity.** Using partition-trained models without refitting gives a median 51.39% (33.73–53.82) in the primary arm, *below the gate*, since those models never saw the held-out attack types. The stratified arm gives 57.89%.

**Limits.** The authors had already seen test results when designing the rule, the grid and the 15% cap, so this reduces the bias and cannot remove it. Attack-type hold-out approximates the real shift. A validated threshold may also be poorly chosen when validation attacks differ from test attacks; that is the problem a deployed system faces too.

### Superseded test-tuned results

The rest of Part 2 records the original run, with thresholds selected on `KDDTest+`. It is kept for transparency and should not be cited as an estimate of performance.

Source: `results/hsr_ids_results.csv`

| Class | Gate alone (XGBoost) F1 % | HSR-IDS F1 % | Δ |
|---|---:|---:|---:|
| DoS | 87.33 | 87.33 | 0.00 |
| Probe | 78.36 | 78.33 | −0.03 |
| **R2L** | **1.51** | **23.40** | **+21.89 (≈16×)** |
| **U2R** | **25.00** | **39.13** | **+14.13** |
| normal | 78.55 | 79.77 | +1.22 |
| **Macro-F1** | **54.15** | **61.59** | **+7.44** |

The shape of this table is the result. Large gains concentrated on the two target classes; the common classes are untouched (Probe moves by −0.03, which is noise); normal *improves* slightly, because the specialist corrects some false-normal verdicts. A rare-class intervention that degrades the majority classes has simply moved the problem around — this one does not.

**Cost.** Only **11.39%** of test traffic reaches the specialist; 88.61% is resolved by the Gate alone. Expected inference cost is roughly 1.11× a single XGBoost pass, versus 2× for an always-consult cascade and *n*× for an *n*-model ensemble. The gate's prediction time is 0.0033s per test batch, so the cascade remains viable on modest hardware.

### Ablation: what actually does the work

HSR-IDS combines two mechanisms — SMOTE balancing and selective routing. A 2×2 factorial ablation isolates each. Source: `results/ablation_2x2_results.csv`

| Config | SMOTE | Selective routing | Macro-F1 % | Routed % | Accuracy % | R2L | U2R |
|---|:---:|:---:|---:|---:|---:|---:|---:|
| Gate alone (baseline) | — | — | 54.15 | 0.00 | 76.12 | 1.51 | 25.00 |
| A | ✗ | ✗ (always consult) | 60.83 | 63.54 | 77.78 | 23.77 | 34.88 |
| B | ✗ | ✓ | 60.27 | 11.39 | 77.28 | 17.64 | 38.64 |
| C | ✓ | ✗ (always consult) | 61.52 | 63.54 | 77.75 | 23.41 | 38.71 |
| **D = HSR-IDS** | ✓ | ✓ | **61.59** | **11.39** | 77.74 | 23.40 | 39.13 |

Reading it honestly:

- **Baseline → A (+6.68 points) is the largest single jump.** Most of the benefit comes simply from *having a narrowly-scoped specialist* — before SMOTE, before selective routing. The mechanism is label-space restriction, not resampling.
- **B → D (+1.32 points): SMOTE contributes a real but secondary gain**, concentrated in R2L (17.64 → 23.40).
- **C → D (+0.07 points): selective routing is essentially free accuracy-wise but cuts routed traffic from 63.54% to 11.39%** — a 5.6× reduction in specialist invocations for 0.07 points of Macro-F1. It is an efficiency mechanism, and it is reported as one.

Stating this breakdown rather than attributing everything to SMOTE is the point of running the ablation. **All of it uses the test-tuned thresholds**, so it describes behaviour at that operating point only.

### Threshold sensitivity

This grid was evaluated on `KDDTest+`, which is the defect corrected above (`results/hsr_threshold_search.csv`).

| low-conf ↓ / normal-certainty → | 0.9 | 0.95 | 0.99 | 0.999 | 0.9999 |
|---|---:|---:|---:|---:|---:|
| **0.5** | 57.51 | 57.64 | 59.02 | **61.59** | 61.61 |
| 0.6 | 57.13 | 57.26 | 58.64 | 61.21 | 61.23 |
| 0.7 | 57.17 | 57.30 | 58.68 | 61.23 | 61.25 |
| 0.8 | 56.79 | 56.92 | 58.30 | 60.86 | 60.88 |
| 0.9 | 55.64 | 55.78 | 57.16 | 59.73 | 59.75 |

Two things this shows:

- **The normal-certainty threshold dominates.** Moving it from 0.9 to 0.999 gains ~4 Macro-F1 points at every low-confidence setting. Moving the low-confidence threshold across its whole range costs ~2 points. Condition (b) is doing the heavy lifting, confirming the design rationale above.
- **(0.5, 0.9999) scores marginally higher (61.61 vs 61.59) but routes 19.92% of traffic instead of 11.39%.** We selected 0.999: +0.02 Macro-F1 does not justify a 75% increase in specialist invocations. The selection criterion is stated so the choice can be audited.

The row is flat around 0.999, but the step from 0.99 to 0.999 is about 2.6 points, and that step is the one training data does not support.

`results/hybrid_threshold_sweep.csv` records an earlier single-threshold sweep, which peaked at 58.69% Macro-F1 (threshold 0.95, 7.11% routed). The two-condition rule outperforms every point on that curve, which is the empirical justification for adding condition (b).

### SHAP interpretability

TreeSHAP applied to the specialist's R2L predictions (`results/hsr_shap_r2l_top_features.csv`), mean absolute SHAP value:

| Rank | Feature | Mean \|SHAP\| | Security interpretation |
|---:|---|---:|---|
| 1 | `src_bytes` | 1.573 | Payload volume — credential attempts have characteristic small, regular sizes |
| 2 | `dst_host_srv_diff_host_rate` | 1.336 | Same service contacted across many hosts — lateral spread |
| 3 | `service` | 1.180 | Target service (telnet, ftp, imap are classic R2L vectors) |
| 4 | `dst_host_same_src_port_rate` | 0.917 | Source port reuse — scripted rather than human sessions |
| 5 | `logged_in` | 0.889 | Authentication outcome |
| 6 | `dst_host_serror_rate` | 0.756 | Connection error rate — failed attempts |
| 7 | `duration` | 0.725 | Session length |
| 8 | `dst_host_count` | 0.401 | Host fan-out |
| 9 | `is_guest_login` | 0.343 | Guest account usage |
| 10 | `count` | 0.254 | Connection frequency |

These are coherent R2L signatures, not artifacts: the top features describe a source contacting a specific service repeatedly across hosts with reused ports and authentication failures — which is what password guessing looks like on the wire. This supports the claim that the specialist learned a real attack signature rather than exploiting a dataset quirk.

---

## Repository Structure

```
ids-study/
├── data/
│   ├── KDDTrain+.txt     # 125,973 connections
│   └── KDDTest+.txt      # 22,544 connections
├── notebooks/
│   └── experiment.ipynb  # full pipeline: benchmark, multiclass, HSR-IDS, ablation, SHAP
├── results/              # All CSV outputs — every number in this README is sourced here
│   ├── main_results.csv                  # binary benchmark, 7 models, incl. confusion counts
│   ├── roc_auc.csv                       # binary AUC per model
│   ├── cross_validation.csv              # 5-fold CV F1 (mean/std/min/max)
│   ├── statistical_tests.csv             # paired t-tests vs XGBoost
│   ├── multiclass_results.csv            # 5-class per-model, per-class F1
│   ├── feature_importance.csv            # ANOVA F-score vs RF importance, all 41 features
│   ├── feature_ablation.csv              # F1 vs top-k feature subsets
│   ├── hsr_ids_results.csv               # SUPERSEDED: test-tuned Gate vs HSR-IDS
│   ├── ablation_2x2_results.csv          # SMOTE × routing factorial ablation
│   ├── hsr_threshold_search.csv          # 5×5 routing threshold grid search
│   ├── hsr_shap_r2l_top_features.csv     # TreeSHAP, specialist R2L predictions
│   ├── hybrid_threshold_sweep.csv        # superseded single-threshold sweep (see below)
│   └── proposed_hybrid_results.csv       # superseded intermediate hybrid (see below)
├── experiments/hsr_validation/  # Corrected protocol, thresholds and results
├── figures/              # Generated plots
├── requirements.txt
└── README.md
```

### Superseded files — read this before comparing numbers

Two files in `results/` are from **earlier design iterations and are not the reported method.** They are retained for transparency about the development path:

- **`proposed_hybrid_results.csv`** — an intermediate hybrid reaching R2L 3.45% and U2R 40.74%. It used a different specialist scope and a single-condition routing rule. **It is not HSR-IDS.** The final method is in `hsr_ids_results.csv` (R2L **23.40%**).
- **`hybrid_threshold_sweep.csv`** — the single-threshold sweep that preceded the two-condition rule, peaking at 58.69% Macro-F1.

`hsr_ids_results.csv`, `ablation_2x2_results.csv`, `hsr_threshold_search.csv` and `hsr_shap_r2l_top_features.csv` are also superseded as estimates: they use test-tuned thresholds. If you are citing this work, **`experiments/hsr_validation/validated_results.json` is the authoritative file.**

---

## Reproducing the Results

**Requirements:** Python 3.9+

```bash
git clone https://github.com/anishc23/ids-study.git
cd ids-study
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dataset is bundled in `data/`, so no further setup is needed. Open `notebooks/experiment.ipynb` and run all cells — it executes the full pipeline end to end and regenerates everything in `results/` and `figures/`.

To reproduce the correction, from `experiments/hsr_validation/`: `python reproduce_baseline.py`, `python tune.py`, `python final_eval.py`, `python comparators_detail.py`.

Core dependencies: `scikit-learn`, `xgboost`, `imbalanced-learn` (SMOTE), `shap`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`.

**Determinism.** Random seeds are fixed throughout (models, SMOTE, CV splits). Metrics should reproduce exactly on the same library versions. Timing columns in `main_results.csv` are hardware-dependent and will differ — the *ratios* (XGBoost ~37× faster than Gradient Boosting; KNN ~300× slower at inference) are the meaningful quantity.

---

## Limitations and Honest Caveats

Stated explicitly, because a reader should not have to discover them:

1. **Validated HSR-IDS does not beat simple resampling.** Global SMOTE on XGBoost reaches a median 61.61% against 57.51%, and R2L remains poorly detected by every approach.
2. **NSL-KDD is a dated benchmark.** It derives from 1998 DARPA traffic and does not reflect modern encrypted traffic, application-layer attacks, or contemporary protocol mixes. It remains valuable as a *comparative* benchmark because the field's literature is calibrated on it, but results here should not be read as deployment-ready performance figures.
3. **XGBoost's advantage over Random Forest is not statistically significant** (p = 0.0875). The choice of XGBoost as Gate rests on its AUC and training speed as much as on its F1.
4. **The earlier thresholds were tuned on the test set, and the effect was not small.** Selection on training data lowers median Macro-F1 by 4.29 points and median R2L F1 to 4.13%. The rule itself was designed after seeing test results, so even the corrected estimate is not fully independent.
5. **The specialist inherits SMOTE's limitations.** Synthesizing 67,343 instances from 52 real U2R samples means the specialist's U2R decision region is largely interpolated. The U2R gain (25.00 → about 38 to 39) should be treated as more fragile than the R2L gain.
6. **Single-dataset evaluation.** No cross-dataset generalization is demonstrated.

---

## Future Work

- **Validate on modern traffic** — CICIDS2017 or UNSW-NB15, with a proper temporal split rather than a random one, to test whether the routing mechanism transfers.
- **Test the diagnosis against resampling** — establish whether routing adds anything to a globally SMOTE-trained model, using training-data thresholds.
- **Per-class routing thresholds** instead of the single two-condition rule, since R2L and U2R fail through different mechanisms.
- **Alternative gates** — a deep gate (1D-CNN or transformer over connection features) may produce better-calibrated confidences, which would directly improve routing decisions.
- **Confidence calibration** — condition (b) depends on the meaningfulness of the 0.999 threshold; temperature scaling or isotonic regression on the Gate's outputs may sharpen it.
- **Cascade depth** — a third stage scoped to `{R2L, U2R}` alone.

---

## Citation

If this work is useful to you:

```bibtex
@misc{choudhury_idsstudy_2026,
  author = {Choudhury, Anish},
  title  = {IDS Study: Comparative ML Benchmark and HSR-IDS for
            Rare-Attack Detection on NSL-KDD},
  year   = {2026},
  url    = {https://github.com/anishc23/ids-study}
}
```

---

## License and Acknowledgements

**Dataset.** NSL-KDD is provided by the Canadian Institute for Cybersecurity, University of New Brunswick. Please cite Tavallaee et al., *A Detailed Analysis of the KDD CUP 99 Data Set* (CISDA 2009) when using it.

**Contributors.** [@anishc23](https://github.com/anishc23) · [@AbhiShrek07](https://github.com/AbhiShrek07)

Developed at Ramrao Adik Institute of Technology, D.Y. Patil Deemed to be University, Navi Mumbai.

**License.** Code released under the MIT License — see `LICENSE`. The NSL-KDD files in `data/` are not covered by the MIT License; they are redistributed under UNB's terms, which require citing the dataset and the paper above.
