# PAZy track — ground truth and rules (v2)

This folder holds the **frozen label source** for the parallel “PAZy-centered”
implementation. Nothing in `data/processed_v2/` should be regenerated
without updating `ground_truth.json` and bumping the **revision note** here.

## Pilot decision (locked)

**Scope:** **Polyester-class** positives only (one binary “in-class vs not” model for v1).

**Meaning:** You include PAZy enzymes whose listed plastics are **hydrolytic polyesters / PHA** (ester-type backbones you defend as one biochemical family). You **do not** pretend one model covers **polyolefins** (PE, PP) or **PVC** in v1.

**Nylon / PUR:** **Out of the v1 polyester-class positive list** unless you later open a second task. They are **polyamide / mixed urethane** chemistry—honest to treat separately from “polyester-class.”

When curating `ground_truth.json`, each row must satisfy:

1. **Source:** PAZy (tier 1).  
2. **Verified activity** on site = yes (or equivalent), when shown—skip ambiguous rows if possible.  
3. **`substrate`** is one plastic string taken from PAZy that falls in the **in-scope list below**.

## Polyester-class scope (v1 default)

**In scope for `substrate` / positive curation** (PAZy “Plastics” column must match
**at least one** of these, or a clear synonym PAZy uses):

| Include | Examples |
|---------|----------|
| Aromatic polyesters | **PET**, **PEF**, **PBT** (and PAZy synonyms you document) |
| Aliphatic polyesters | **PLA**, **PCL**, **PBS** |
| Biological polyester | **PHA**, **PHB** |
| Cutin / suberin | Only when you treat the row as a **polyester-hydrolase proxy** (note in `source_note`) |

**Out of scope for v1 positives (unless you open a second task):**

- Laccase, Mn-peroxidase, lignin peroxidase, alkane hydroxylase–style **oxidative**
  mechanisms unless PAZy explicitly lists a **hydrolytic** plastic substrate you accept  
- “PE / PP / PS / PVC degraders” without hydrolytic backbone evidence (see `DATASET_GUIDE.md`)

## Evidence tiers

| `evidence_tier` | Meaning |
|-----------------|--------|
| 1 | PAZy (or PlasticDB) curated row with literature pointer you trust |
| 2 | Optional: secondary database or UniProt-reviewed function only — **use sparingly** |

Until PAZy accessions are pasted, `ground_truth.json` may contain **seed**
accessions for pipeline smoke tests only — mark `source_note` accordingly.

## PAZy export without UniProt column

If **Export CSV (with sequences)** only has `protein_id`, `protein_name`,
`amino_acid_sequence`, that export is still valid — PAZy stores UniProt on
each **protein detail page**, not in that file.

Resolve automatically:

```bash
python -m plasticdeg.ingest.pazy_csv_to_ground_truth --csv data/exported/pazy_proteins_sequences.csv
```

This reads each `protein_id`, fetches `https://pazy.eu/proteins/{id}`, parses
UniProt + plastics, applies length filter, dedupes accessions, and overwrites
`data/pazy/ground_truth.json`. Rows with **no UniProt link on PAZy** are
skipped and listed in `ground_truth.build_errors.txt`.

### When PAZy has no UniProt link (gap fill)

**Option A — automatic second resolver (implemented)**  
Many PAZy pages still carry **RefSeq protein** URLs (`…/protein/WP_…`, `NP_…`) or
**ENA** links (`ebi.ac.uk/ena/browser/view/…`) and NCBI BLAST deep links with the
same accession. The resolver can call **UniProt’s async ID mapping API** for those
IDs (`RefSeq_Protein` or `EMBL-GenBank-DDBJ` / `EMBL-GenBank-DDBJ_CDS` → `UniProtKB`).

Important: submit the job as **HTML form fields** (`from`, `to`, `ids`), not a raw
JSON-only body — otherwise the API returns **400**.

```bash
python -m plasticdeg.ingest.pazy_csv_to_ground_truth --csv data/exported/pazy_proteins_sequences.csv --idmap-fallback
```

Use `--sleep-uniprot` (default `0.5`) to stay polite to `rest.uniprot.org`. Some
INSDC accessions still fail to map (no UniProt cross-reference); those rows stay
in `ground_truth.build_errors.txt`.

**Option B — UniProt web UI (batch)**  
[UniProt ID mapping](https://www.uniprot.org/id-mapping) — paste RefSeq / EMBL
accessions, choose source and target databases, download mapped UniProt IDs, then
merge into curation or `ground_truth.json` by hand.

**Option C — sequence BLAST**  
Use **NCBI blastp** or **UniProt BLAST** with the CSV sequence, require high
identity/coverage, and pick a **reviewed (Swiss-Prot)** or **same-strain**
UniProt entry when the paper supports it. Treat as higher manual scrutiny
(evidence tier / `source_note`).

**Option D — literature / PAZy UI**  
Open the PAZy detail page and linked references; copy the UniProt accession the
authors state, then append a row to `ground_truth.json` (or a sidecar TSV you
merge once).

Dry-run first five rows:

```bash
python -m plasticdeg.ingest.pazy_csv_to_ground_truth --dry-run --limit 5
```

## Files

| File | Role |
|------|------|
| `ground_truth.template.json` | Schema + empty example |
| `ground_truth.json` | **Authoritative** list (from curation or `pazy_csv_to_ground_truth`) |
| `../exported/` | Your PAZy CSV exports |
| `../processed_v2/` | Pipeline outputs (`sequences/`, `tables/`, …); see `plasticdeg/paths.py` |

## Revision log

- **v0:** Scaffold + UniProt fetch; replace seeds with PAZy export when ready.
- **v1:** Pilot scope locked to **polyester-class**; nylon/PUR excluded from v1 positives unless a new task is opened.
