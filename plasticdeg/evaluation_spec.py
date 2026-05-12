"""
Phase 0 — frozen evaluation contract for train / report scripts.

All metrics are computed on the held-out homology split unless a script
explicitly documents a secondary probe (e.g. hard negatives).
"""

from __future__ import annotations

# Primary ranking metric under ~1:8 imbalance (report first).
PRIMARY_METRIC = "pr_auc"

# Single-number confusion-matrix summary at a *stated* threshold.
MCC_AT_THRESHOLD = "mcc_at_threshold"

# Practical screening readout (threshold chosen on train, e.g. recall ≥ 0.8).
PRECISION_AT_RECALL = "precision_at_recall_0.8"

RECALL_TARGET_FOR_THRESHOLD = 0.8
