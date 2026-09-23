# Session 17 — Final project brief

**Status:** This directory contains an assignment brief only. There are no
data files, model checkpoints, training scripts, notebooks, evaluation scripts,
or results here. The [theory note](../../notes/md/note_17_final_project.md)
provides broader context; the implemented toy RLHF exercise is in
[Session 15](../15_rlhf_basics/README.md).

## Goal

Build a small, reproducible preference-learning pipeline. A possible sequence
is supervised fine-tuning, preference data collection, reward-model training,
then PPO or DPO. Start with a small permitted dataset and a fixed baseline;
do not interpret a toy exercise as a safety or deployment qualification.

## Deliverables

- Document data provenance, consent/licensing, preprocessing, and
  train/validation/test separation.
- Version the model, reference model, tokenizer, dataset, code, dependencies,
  configuration, and random seeds.
- Provide one runnable smoke command and tests for data and loss contracts.
- Generate evaluation tables from saved predictions. Include uncertainty and
  failure cases; compare methods only under a matched protocol.
- Report hardware, wall time, and memory from an actual run if those numbers
  are relevant. Keep human evaluation separate from automated scores.
- Explain limitations, including reward-model error, distribution shift,
  and the gap between offline scores and real-world safety.

No training, win-rate, safety, speed, or memory result has been measured for
this final project in this repository. Completion boxes should be checked
only after the corresponding artifact and validation exist.
