"""
One-shot bundle for a tight deadline: (1) manuscript artifacts + (2) evaluation text.

1. Refreshes tables/accession_sequence_alias_map.csv from expansion + merged positives
   without re-running the full integrate merge.
2. Copies key JSON/CSV into reports/manuscript_bundle/ for Word/LaTeX appendices.
3. Writes reports/manuscript_bundle/EVALUATION_SNIPPETS.md (limitations + AUC/stress-test
   wording you can paste into Discussion / Limitations).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from plasticdeg import paths
from plasticdeg.expand.integrate_expansion import write_alias_map_from_files


def _count_lines(p: Path) -> int | None:
    if not p.is_file():
        return None
    return sum(1 for _ in p.read_text(encoding="utf-8", errors="replace").splitlines() if _.strip())


def _safe_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_evaluation_snippets(
    out_md: Path,
    *,
    manifest: dict | None,
    metrics_v2: dict | None,
    hard_probe: dict | None,
    n_train: int | None,
    n_test: int | None,
) -> None:
    n_canonical = (manifest or {}).get("n_rows")
    t1 = (manifest or {}).get("tier1")
    t2 = (manifest or {}).get("tier2")
    rf = (metrics_v2 or {}).get("models", {}).get("random_forest", {})
    tc = rf.get("test_combined") or {}
    pr = tc.get("pr_auc")
    roc = tc.get("roc_auc")
    mcc = tc.get("mcc_at_threshold")
    n_test_combined = rf.get("n_test_combined")
    hn = hard_probe or {}
    h_n = hn.get("hard_negative_count")
    h_fp = hn.get("hard_negative_false_positive_rate")
    h_mean = hn.get("hard_negative_score_mean")
    ez_mean = hn.get("easy_test_negative_score_mean")

    lines: list[str] = [
        "# Evaluation snippets (copy into report)",
        "",
        "Generated automatically. Edit tone to match your institution’s style guide.",
        "",
        "---",
        "",
        "## Numbers to cite (current bundle)",
        "",
    ]
    if n_canonical is not None:
        lines.append(
            f"- After BLAST expansion across external databases, we retained **{n_canonical}** "
            f"sequence-unique positive proteins (**{t1}** Tier-1 / PAZy-aligned Gold, **{t2}** Tier-2 Silver), "
            "merging all raw accessions that map to the same amino-acid sequence (see `accession_sequence_alias_map.csv`)."
        )
    if n_train is not None and n_test is not None:
        lines.append(
            f"- Homology-aware train/test split manifest: **{n_train}** training IDs and **{n_test}** test IDs "
            "(positives + negatives combined in split files)."
        )
    lines.extend(
        [
            "",
            "---",
            "",
            "## Limitations (Discussion / Methods)",
            "",
            "- **Negative design:** Random Swiss-Prot negatives are useful for a first baseline but do not exhaust "
            "“hard” confounders (e.g. lipases/esterases). We therefore report a separate **Phase-4 stress test** on "
            "reviewed hydrolase negatives.",
            "- **Homology split:** Train and test were separated at the whole-cluster level (CD-HIT/MMseqs2-style "
            "clustering) to reduce near-duplicate leakage; residual remote homology can still exist at lower identity.",
            "- **Silver tier:** Tier-2 positives are homology-propagated under strict BLAST identity/coverage rules; "
            "they are down-weighted during training but are not direct wet-lab confirmations.",
            "- **Frozen embeddings:** ESM-2 representations are not fine-tuned here; performance reflects separability "
            "in a fixed representation space plus a shallow classifier, not a full end-to-end sequence model.",
            "- **Metric saturation:** Very high test PR-AUC/ROC-AUC on the primary split can indicate strong "
            "embedding separability and/or an “easy” negative distribution; interpret alongside stress tests.",
            "- **Scope:** The model is a polyester-relevant enzyme discovery aid within the PAZy-centered label set; "
            "generalization to unseen families or environmental metagenomes is not claimed from this benchmark alone.",
            "",
            "---",
            "",
            "## Interpreting high test AUC vs. the hard-negative probe",
            "",
        ]
    )
    def _f4(x: object) -> str:
        return f"{float(x):.4f}" if isinstance(x, (int, float)) else "n/a"

    if isinstance(n_test_combined, int) and pr is not None:
        lines.append(
            f"On the official homology test split, the frozen-embedding Random Forest baseline reported "
            f"PR-AUC≈**{_f4(pr)}** and ROC-AUC≈**{_f4(roc)}** (MCC at the train-chosen recall threshold ≈**{_f4(mcc)}**; "
            f"n_test={n_test_combined}). "
            "A high score here primarily means that **ESM-2 embeddings linearly/nonlinearly separate your positives "
            "from the sampled negatives in this benchmark**, which is expected when negatives are drawn broadly from "
            "unrelated Swiss-Prot proteins."
        )
    else:
        lines.append(
            "On the official homology test split, the frozen-embedding Random Forest achieved very high ranking metrics; "
            "see `metrics_esm_baseline_v2.json` for exact values."
        )
    lines.extend(
        [
            "",
            "We therefore **do not** treat AUC≈1.0 as proof of biochemical specificity in the wild. Instead, we "
            "complement the headline metrics with a **hard-negative probe**: reviewed EC 3.1.1/3.1.2 hydrolases that "
            "are evolutionarily closer to true lipases than random negatives, scored with the **same frozen threshold** "
            "chosen on training positives (no threshold tuning on the probe).",
            "",
        ]
    )
    if h_n is not None:
        if isinstance(h_mean, (int, float)) and isinstance(ez_mean, (int, float)):
            lines.append(
                f"In the current run, **{int(h_n)}** hard negatives were scored; **false-positive rate at the frozen "
                f"threshold** was **{h_fp}** (see `hard_negative_probe_v2.json`). "
                f"Mean probe score ≈**{float(h_mean):.4f}** vs. easy test negatives ≈**{float(ez_mean):.4f}** when available—use this "
                "to discuss whether the model assigns systematically higher mass to hydrolase-like sequences."
            )
        else:
            lines.append(
                f"In the current run, **{int(h_n)}** hard negatives were scored; see `hard_negative_probe_v2.json`."
            )
    lines.extend(["", "---", "", "## Suggested one-sentence “integrity” line", ""])
    if n_canonical is not None:
        lines.append(
            f"> We report BLAST expansion breadth (raw accessions), then train only on **{n_canonical}** "
            "sequence-unique positives, with an explicit alias table and a hydrolase stress test at a frozen threshold."
        )
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manuscript bundle: alias map + snapshots + evaluation snippets")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=paths.manuscript_bundle_dir(),
    )
    parser.add_argument(
        "--expanded-csv",
        type=Path,
        default=paths.expanded_positives_csv(),
    )
    parser.add_argument(
        "--deduped-positives-csv",
        type=Path,
        default=paths.positives_gt_expanded_csv(),
    )
    args = parser.parse_args(argv)

    if not args.expanded_csv.is_file():
        print(f"ERROR: missing expansion CSV: {args.expanded_csv}", file=sys.stderr)
        return 1
    if not args.deduped_positives_csv.is_file():
        print(f"ERROR: missing merged positives CSV: {args.deduped_positives_csv}", file=sys.stderr)
        return 1

    args.bundle_dir.mkdir(parents=True, exist_ok=True)

    alias_path = paths.accession_sequence_alias_map_csv()
    write_alias_map_from_files(args.expanded_csv, args.deduped_positives_csv, alias_path)
    print(f"  Wrote / refreshed {alias_path}", flush=True)

    copies: list[tuple[Path, str]] = [
        (alias_path, "accession_sequence_alias_map.csv"),
        (paths.metrics_esm_baseline_v2_json(), "metrics_esm_baseline_v2.json"),
        (paths.hard_negative_probe_v2_json(), "hard_negative_probe_v2.json"),
        (paths.hard_negative_probe_json(), "hard_negative_probe_v1.json"),
        (paths.tier_probe_summary_json(), "tier_probe_summary.json"),
        (paths.tables_dir() / "integrate_expansion_manifest.json", "integrate_expansion_manifest.json"),
        (paths.blast_expansion_log_json(), "blast_expansion_log.json"),
    ]
    for src, name in copies:
        if not src.is_file():
            print(f"  skip (missing): {src}", flush=True)
            continue
        dst = args.bundle_dir / name
        shutil.copyfile(src, dst)
        print(f"  copied -> {dst}", flush=True)

    manifest = _safe_json(paths.tables_dir() / "integrate_expansion_manifest.json")
    metrics_v2 = _safe_json(paths.metrics_esm_baseline_v2_json())
    hard = _safe_json(paths.hard_negative_probe_v2_json()) or _safe_json(paths.hard_negative_probe_json())
    n_train = _count_lines(paths.split_train_txt())
    n_test = _count_lines(paths.split_test_txt())

    eval_path = args.bundle_dir / "EVALUATION_SNIPPETS.md"
    _write_evaluation_snippets(
        eval_path,
        manifest=manifest,
        metrics_v2=metrics_v2,
        hard_probe=hard,
        n_train=n_train,
        n_test=n_test,
    )
    print(f"  Wrote {eval_path}", flush=True)
    print("\nDone. Open reports/manuscript_bundle/ for appendix files + EVALUATION_SNIPPETS.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
