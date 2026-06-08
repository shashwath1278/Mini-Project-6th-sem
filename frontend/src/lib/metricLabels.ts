/** Short plain-language strings for UI subtitles and native tooltips (`title`). */

export const METRIC_TIPS = {
  esm: "ESM-2 is a protein language model. It converts the amino-acid sequence into a fixed-length numeric vector (embedding) that the classifiers use.",
  rf: "Random forest (RF): an ensemble of decision trees trained on ESM embeddings; outputs a score between 0 and 1.",
  lr: "Logistic regression (LR): a linear classifier on scaled embeddings; outputs a probability between 0 and 1.",
  tauRf:
    "τ_RF: cutoff probability for the random forest. If RF score ≥ τ_RF, the sequence is called “positive” for that head.",
  tauLr:
    "τ_LR: cutoff probability for logistic regression. If LR score ≥ τ_LR, the sequence is called “positive” for that head.",
  rfProb: "RF score: random forest output in [0, 1]. Higher = more similar to training positives in RF’s view.",
  lrProb: "LR score: logistic regression probability in [0, 1]. Higher = more similar to training positives in LR’s view.",
  rfPlus: "RF call: “positive” if RF score ≥ τ_RF, else “negative” (training-defined class, not a lab assay).",
  lrPlus: "LR call: “positive” if LR score ≥ τ_LR, else “negative”.",
  trainThrRf:
    "Cutoff probability chosen on the training set to meet a minimum recall on training positives, then frozen (not tuned on test).",
  trainThrLr:
    "Same idea for logistic regression: training-derived cutoff (τ_LR) for minimum positive recall, then frozen for test.",
  prAuc:
    "PR-AUC: area under the precision–recall curve (0–1). Summarizes ranking quality for the positive class on held-out test data.",
  rocAuc:
    "ROC-AUC: area under the receiver operating characteristic curve (0–1). Summarizes how well scores separate positives from negatives on test.",
  mcc:
    "MCC: Matthews correlation coefficient. Single number summarizing confusion matrix quality; +1 is perfect, 0 is random-like.",
  testCombined:
    "test_combined: metrics computed on the full held-out homology test split (positives + easy negatives together).",
  tn: "True negatives: negatives correctly predicted as negative.",
  fp: "False positives: negatives incorrectly predicted as positive.",
  fn: "False negatives: positives incorrectly predicted as negative.",
  tp: "True positives: positives correctly predicted as positive.",
  fprHard:
    "FPR on hard negatives: fraction of curated lipase/esterase sequences (not in the positive list) that RF still scores above the frozen cutoff.",
  predPosHard: "How many hard-negative sequences were called “positive” at the frozen RF cutoff.",
  silverDown:
    "Silver tier BLAST-expanded positives count half toward training loss vs gold PAZy rows (see metrics JSON).",
  hardN: "Number of reviewed lipase / esterase proteins used as hard negatives (similar chemistry, not labeled as plastic positives).",
  easyNegN:
    "Number of easy (broad Swiss-Prot) negatives on the held-out test split, used only for context in this probe summary.",
  baseline: "Which model variant produced this probe file (e.g. v2 random forest).",
  tierProbe: "Optional breakdown of false-positive rate by evidence tier (gold vs silver positives, etc.).",
} as const;
