# KATS: Kinetic Adaptive Triage Stacker

Code and experimental records for:

**Cloud Task Priority Triage Under Class Imbalance: An Explainable Stacked Ensemble Approach**

KATS combines LightGBM, Random Forest, and isotonic-calibrated Gaussian
Naive Bayes through a weighted logistic-regression meta-learner.

The revised study evaluates when this configuration helps or harms
three-class priority classification. It does not establish a universally
preferred classifier, an imbalance-ratio-based deployment rule, or
observed improvements in scheduling, SLA compliance, or financial cost.

## Current evaluation: Repair R1

The current manuscript uses four datasets, eight models including KATS,
and five seeds: `42, 7, 13, 99, 2026`.

CloudTask is excluded from the current quantitative benchmark.
Historical five-dataset results must not be substituted for Repair-R1
results.

| Dataset | Benchmark rows | Predictors | IR, approximately | Evaluation scope |
|---|---:|---:|---:|---|
| GoogleCluster | 60,000 | 18 | 1.95 | Retrospective, study-defined policy-tier classification |
| ITIncident | 24,918 incidents | 11 | 34.61 | Recorded final-state incident priority |
| MultiCloud | 1,000 | 8 | 1.00 | Constructed, corpus-relative QoS tiers |
| CICIDS2017 | 60,000 | 52 | 20.99 | Auxiliary intrusion-tier classification |

IR is the largest class count divided by the smallest class count.
ITIncident contains 678 High-priority incidents, approximately 2.72%
of the retained corpus.

These datasets do not represent one common, validated operational
urgency target.

## Main findings

The revised results include unfavorable outcomes:

- ITIncident KATS achieves mean High recall of 0.9561, but precision is
  0.0387 and it produces 3,189 mean false High alarms per test split.
- CICIDS2017 KATS has mean Macro-F1 of 0.5467, compared with 0.9967 for
  standalone LightGBM. Removing stacking yields Macro-F1 of 0.9960.
- Collection-grouped GoogleCluster evaluation qualifies the near-ceiling
  results obtained under the main splitting protocol.
- No KATS-versus-baseline correctness comparison is significant on
  MultiCloud after the study-wide Holm correction.
- Calibration of the Naive Bayes component does not establish calibration
  of the complete stack.
- Illustrative FN/FP error costs are not observed SLA or financial outcomes.

Consult the versioned result files for uncertainty, per-seed values,
and the exact experimental settings.

## Fitting and evaluation

### Fold-first stacking

KATS constructs base-model probabilities using three OOF splits.

Within each permitted training subset:

1. B1 and B2 fit their own preprocessing and any enabled resampler.
2. B3 uses nested, three-split isotonic calibration, with training-only
   preprocessing for each GaussianNB child.
3. Held-out probabilities are aligned to `[High, Low, Medium]`.
4. Nine base probabilities are concatenated with retained predictors.
5. Eligible original OOF observations train the meta-learner; synthetic
   observations do not train the meta-learner.

Grouped or stratified splits are used where applicable. Chronological
analyses use forward-time splits; initial observations without complete
OOF predictions are excluded from meta fitting.

### Sampling and weights

Main sampling is dataset-selected, not a universal runtime IR gate:

| Input setting | Synthetic sampler |
|---|---|
| Main mixed-feature ITIncident | SMOTENC |
| Numeric CICIDS2017 | SMOTE |
| GoogleCluster and MultiCloud | None |
| Separate categorical-only ITIncident sensitivities | SMOTEN |

Enabled main samplers operate within B1/B2 training subsets.
B3 receives neither synthetic sampling nor explicit class weights.

B1 and the meta-learner use inverse-frequency weights multiplied by a
separate asymmetric penalty:

\[
w_c = \frac{n}{3n_c}\alpha_c,
\qquad
\alpha_c =
\begin{cases}
5, & c=\mathrm{High}\ \text{and training IR}>3,\\
1, & \text{otherwise}.
\end{cases}
\]

B2 uses its own library-balanced weighting. The “No class weights”
ablation removes both inverse-frequency weights and the additional High
multiplier from B1/meta; it is not an ablation of the multiplier alone.

### Decision rules and uncertainty

The main configuration uses fixed estimator settings, not a
hyperparameter search.

KATS selects a High threshold using a training-only validation holdout,
then refits on the outer-training partition. Final labels use multiclass
argmax with a High override. Standalone baselines use multiclass argmax.

Main summaries report five-seed means and sample standard deviations.
Overlapping splits are not independent population replicates.

Exact McNemar tests assess paired overall correctness on seed-2026
observations, with Holm correction across 28 comparisons. They do not
directly test High recall, Macro-F1, or kappa. Nonsignificance is not
equivalence.

Conditional paired bootstrap intervals hold the fitted models fixed.
They do not measure training-seed or population-generalization uncertainty.

## Published materials

The current result directory is
[`KATS_REVIEWER_REPAIR_R1_RESULTS`](KATS_REVIEWER_REPAIR_R1_RESULTS/).

Key records include:

| Material | File |
|---|---|
| Per-seed metrics | `all_completed_metrics_per_seed.csv` |
| Experiment mean/SD summaries | `all_experiment_mean_sd.csv` |
| Configuration and software versions | `run_configuration.json` |
| Dataset audits | `*_data_audit.json` |
| Retained predictor schemas | `*_feature_schema.csv` |
| Main split identifiers | `main_split_*.csv` |
| McNemar results | `McNemar_seed2026_exact_Holm.csv` |
| Conditional paired intervals | `paired_conditional_test_CIs.csv` |
| High reliability intervals | `High_reliability_group_bootstrap_CIs.csv` |
| Fitted-stack explanation agreement | `canonical_fitted_stack_fidelity_summary.csv` |
| Local explanation agreement | `canonical_fitted_stack_local_fidelity.csv` |
| Repeated inference timing | `latency_raw_repetitions.csv`, `latency_summary.csv` |
| Timing scope and hardware metadata | `timing_scope.json` |
| Completion and failure records | `FINAL_STATUS.json`, `FAILED_CASES.json` |

The directory also contains a `Cases/` subdirectory and numerical-audit
records. Check configuration, case identifiers and status records before
combining outputs.

## Repository navigation

| Location | Purpose |
|---|---|
| [`Final Notebook.ipynb`](Final%20Notebook.ipynb) | Published final notebook |
| [`src/`](src/) | Scripts, including multiple historical versions |
| [`src/Updated Code/`](src/Updated%20Code/) | Additional versioned scripts and notebook |
| [`KATS_REVIEWER_REPAIR_R1_RESULTS/`](KATS_REVIEWER_REPAIR_R1_RESULTS/) | Published R1 result and audit records |
| [`Experiments Added/`](Experiments%20Added/) | Additional experimental materials |
| [`results/`](results/) | Other result files; establish provenance before use |
| [`Secondary RES/`](Secondary%20RES/) and [`Mix of RES/`](Mix%20of%20RES/) | Additional result collections requiring version checks |
| [`datasets/`](datasets/) | Dataset-related materials |
| [`experiment_registry.md`](experiment_registry.md) | Experiment documentation |
| [`key_findings_summary.md`](key_findings_summary.md) | Findings documentation; cross-check against the current R1 source |

Legacy scripts and summaries are retained for provenance. Their presence
does not make their outputs current manuscript results.

## Reproduction and version control

Clone the repository:

```bash
git clone [https://github.com/hamidborkot/KATS-Cloud-Priority-Triage.git](https://github.com/hamidborkot/KATS-Cloud-Priority-Triage.git)
cd KATS-Cloud-Priority-Triage
```

Before running experiments:

1. Review the published notebook and the intended pipeline version.
2. Match its settings to the applicable `run_configuration.json`.
3. Obtain the corresponding source datasets and configure input paths.
4. Keep main R1, categorical-only retry, and earlier-protocol outputs
   separately identified.
5. Inspect completion/failure records and case identifiers before
   generating manuscript summaries.

No single legacy script is advertised here as a verified one-command
reproduction of every revised result. Earlier-protocol sensitivity and
fusion outputs must not replace the current main benchmark.

## Interpretation limits

- Retained incident and trace-state predictors are not necessarily
  observable at task submission or incident creation.
- Predictor grouping does not automatically establish entity- or
  session-independent generalization.
- GoogleCluster retains a scheduler policy-proxy limitation.
- The cleaned CICIDS derivative lacks capture/session identifiers.
- Earlier-protocol sensitivity results do not establish a universal IR rule.
- The fitted-stack explanation audit uses one seed and a class-enriched
  ITIncident sample, not a prevalence-representative operational sample.
- Timing describes the recorded execution environment and operation,
  not end-to-end concurrent deployment latency.
- Error-cost analysis is illustrative; no scheduling simulation or
  observed SLA improvement is claimed.

## Manuscript and license

The associated manuscript is under review at *Future Generation Computer
Systems*. No acceptance or publication claim is made here.

See [`LICENSE`](LICENSE) for the repository license text.
