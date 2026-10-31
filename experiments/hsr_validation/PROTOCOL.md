# HSR-IDS threshold selection: corrected protocol

**Status: specification, committed and pushed before any corrected tuning or
evaluation ran.** The commit adding this file is the record that the rule was
fixed in advance. Nothing below consults a corrected result, because none exists
at the time of writing.

## The defect

Cell 17 of `notebooks/experiment.ipynb` grid-searches the routing thresholds
against `KDDTest+` and reports the best row. The published 61.59% macro-F1 is
therefore selected on the data used to report it. Two related problems:

- `results/statistical_tests.csv` tests cross-validated binary F1 inside
  `KDDTrain+`, so no test addresses HSR-IDS itself.
- The reported figure rests on one seed.

`reproduce_baseline.py` reproduces the published run exactly (gate 54.15, HSR-IDS
61.59, 11.39% routed), so any change below is due to the protocol.

## What is not clean, and cannot be made clean

The authors have already seen test results. The grid, the 15% routing cap, the
two-condition rule and the idea of re-examining confident Normal predictions were
all formed after looking at `KDDTest+`. This protocol removes threshold selection
from the test set. It cannot remove that history, so the result is less optimistic,
not untouched. Only a new dataset can give a fully independent estimate.

## Fixed in advance

**Unchanged:** the gate and specialist (default XGBoost, SMOTE with k=5 on Normal,
R2L, U2R), the routing rule, the grid (`low` in 0.5...0.9, `certain` in
0.9...0.9999), and the selection rule: highest validation macro-F1 with at most
15% routed; ties go to the first row in grid order, as in the notebook.

**Changed:** encoders and scaler are fitted on training data only.

## Two validation arms

Both use only `KDDTrain+`. `tune.py` raises an error if it tries to open
`KDDTest+`. Seeds 0 to 9 set the partition, SMOTE and XGBoost.

**Arm S, shift-preserving (primary).** `KDDTest+` contains attack types absent from
training, and that shift is what HSR-IDS targets. So a third of the attack types in
each category (rounded, minimum one) are held out whole for validation, with 20% of
Normal. A random stratified split would not contain this shift, because on it the
gate is nearly perfect.

**Arm V, stratified (control).** A plain 80/20 split stratified by category.

## Final evaluation

`final_eval.py` runs once, after the thresholds are committed.

1. **Primary result:** per seed, refit gate and specialist on all of `KDDTrain+`
   with that seed, apply that seed's Arm S thresholds, and score `KDDTest+`.
   Report the median and range of macro-F1 over the ten seeds.
2. **Reported model:** the modal Arm S thresholds (ties to the lower routed share
   on validation), refit at seed 42. Per-class precision, recall and F1, then:
   a paired bootstrap on test rows (5,000 resamples) for the macro-F1 difference
   against the gate, and an exact McNemar test on correctness.
3. **Same for Arm V**, as the control.
4. **Sensitivity:** the frozen partition-trained models, without refitting.
5. **Comparators, on all of `KDDTrain+`:** XGBoost with balanced sample weights,
   and XGBoost on all classes after SMOTE, each over the ten seeds.
6. **Optimism:** the test-tuned maximum on the same refit models, minus the
   validated result. This is reported only as a measure of the bias.

## Decision rule

Whatever item 1 gives replaces 61.59% as the headline, including if it is lower
than the gate's 54.15%. The earlier figure is kept in `results/` as superseded and
labelled test-tuned. The expected direction is lower than 61.59, and Arm S may
choose a different `certain` threshold from 0.999.

## Known limits

Test rows are resampled individually, and NSL-KDD has no session structure to
group on. Attack-type hold-out approximates, and does not equal, the real shift.
This is one dataset from 1998 traffic.
