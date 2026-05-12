# Evaluation snippets (copy into report)

Generated automatically. Edit tone to match your institution’s style guide.

---

## Numbers to cite (current bundle)

- After BLAST expansion across external databases, we retained **189** sequence-unique positive proteins (**160** Tier-1 / PAZy-aligned Gold, **29** Tier-2 Silver), merging all raw accessions that map to the same amino-acid sequence (see `accession_sequence_alias_map.csv`).
- Homology-aware train/test split manifest: **1155** training IDs and **330** test IDs (positives + negatives combined in split files).

---

## Limitations (Discussion / Methods)

- **Negative design:** Random Swiss-Prot negatives are useful for a first baseline but do not exhaust “hard” confounders (e.g. lipases/esterases). We therefore report a separate **Phase-4 stress test** on reviewed hydrolase negatives.
- **Homology split:** Train and test were separated at the whole-cluster level (CD-HIT/MMseqs2-style clustering) to reduce near-duplicate leakage; residual remote homology can still exist at lower identity.
- **Silver tier:** Tier-2 positives are homology-propagated under strict BLAST identity/coverage rules; they are down-weighted during training but are not direct wet-lab confirmations.
- **Frozen embeddings:** ESM-2 representations are not fine-tuned here; performance reflects separability in a fixed representation space plus a shallow classifier, not a full end-to-end sequence model.
- **Metric saturation:** Very high test PR-AUC/ROC-AUC on the primary split can indicate strong embedding separability and/or an “easy” negative distribution; interpret alongside stress tests.
- **Scope:** The model is a polyester-relevant enzyme discovery aid within the PAZy-centered label set; generalization to unseen families or environmental metagenomes is not claimed from this benchmark alone.

---

## Interpreting high test AUC vs. the hard-negative probe

On the official homology test split, the frozen-embedding Random Forest baseline reported PR-AUC≈**1.0000** and ROC-AUC≈**1.0000** (MCC at the train-chosen recall threshold ≈**0.6961**; n_test=330). A high score here primarily means that **ESM-2 embeddings linearly/nonlinearly separate your positives from the sampled negatives in this benchmark**, which is expected when negatives are drawn broadly from unrelated Swiss-Prot proteins.

We therefore **do not** treat AUC≈1.0 as proof of biochemical specificity in the wild. Instead, we complement the headline metrics with a **hard-negative probe**: reviewed EC 3.1.1/3.1.2 hydrolases that are evolutionarily closer to true lipases than random negatives, scored with the **same frozen threshold** chosen on training positives (no threshold tuning on the probe).

In the current run, **80** hard negatives were scored; **false-positive rate at the frozen threshold** was **0.0** (see `hard_negative_probe_v2.json`). Mean probe score ≈**0.0289** vs. easy test negatives ≈**0.0263** when available—use this to discuss whether the model assigns systematically higher mass to hydrolase-like sequences.

---

## Suggested one-sentence “integrity” line

> We report BLAST expansion breadth (raw accessions), then train only on **189** sequence-unique positives, with an explicit alias table and a hydrolase stress test at a frozen threshold.
