# Dataset Guide — Plastics, Microbes, and Sourcing

A reference for how the PlasticDeg dataset is conceptually structured: which plastics exist, which microbes/enzymes degrade each, and where the data actually comes from.

---

## Part 1 — Plastic Variants and Their Microbes

### Plastics organized by chemistry

The chemistry determines which enzymes work — *that's why* you group by bond type, not by commercial product name. An enzyme doesn't care that something is a "water bottle" vs a "yogurt cup"; it cares about the chemical bond it's attacking.

| Group | Plastics | Backbone bond | Hydrolyzable? |
|---|---|---|---|
| **Polyesters** | PET, PLA, PBS, PHA/PHB, PCL | C–O–C(=O) (ester) | Yes — easy |
| **Polyamides** | Nylon-6, Nylon-6,6 | C–N–C(=O) (amide) | Yes — moderate |
| **Polyurethanes** | PUR (foams) | Mixed: ester + urethane (carbamate) | Partial |
| **Polyolefins** | PE (LDPE, HDPE), PP | C–C only | **No** — needs oxidation first |
| **Styrenics** | PS (Styrofoam), ABS | C–C + aromatic ring | **No** — needs oxidation first |
| **Vinyls** | PVC | C–C + Cl | **No** — extremely recalcitrant |

**The key dividing line:** polymers with oxygen/nitrogen in the backbone (esters, amides, urethanes) are *hydrolyzable* — water + an enzyme breaks them. Pure carbon backbones (PE, PP, PS, PVC) are *not* — they need oxidative attack (like burning) before any enzyme can act on the fragments.

### Microbes and enzymes for each plastic

| Plastic | Microbes (most validated) | Enzymes | Evidence quality |
|---|---|---|---|
| **PET** | *Ideonella sakaiensis*, *Thermobifida fusca*, leaf-branch compost metagenome | IsPETase, MHETase, LCC, TfCut2, cutinases | **Tier 1** — strong, in vitro validated |
| **PLA** | *Amycolatopsis* sp., *Cryptococcus* | Proteases, lipases, cutinases | **Tier 1–2** |
| **PHA/PHB** | Many bacteria & fungi (it's biological plastic — degraders are everywhere) | PHA depolymerases | **Tier 1** |
| **PUR** | *Pseudomonas* sp., *Comamonas acidovorans* | Esterases, ureases, amidases | **Tier 2** |
| **Nylon** | *Flavobacterium* sp. K172 (one famous strain) | NylB (6-aminohexanoate hydrolase) — textbook example of evolution of new function | **Tier 1** but extremely narrow |
| **PE** | *Pseudomonas*, *Bacillus*, mealworm gut microbiome | Laccases, alkane hydroxylases (AlkB), MnP — but causal link often unproven | **Tier 2–3** — most claims are weak |
| **PP** | Similar to PE, fewer studies | Similar | **Tier 3** |
| **PS** | Mealworm gut microbes, *Pseudomonas* sp. | MnP, laccases, styrene monooxygenases | **Tier 2–3** |
| **PVC** | A handful of disputed reports | Unknown | **Tier 4** — basically no enzyme-level evidence |

### The honest evidence problem nobody talks about

Most "microbe X degrades PE" papers don't actually prove an enzyme is doing it. The microbe might be:

- Eating **additives** (plasticizers, antioxidants), not the polymer itself
- Working on **pre-oxidized** PE (UV-aged, corona-treated) — much easier than virgin PE
- Forming a **biofilm** that *physically* fragments the plastic without chemical degradation

So when you read "Bacillus sp. degrades polyethylene," the question "which protein in Bacillus is responsible?" often has no clean answer. That's why PE/PS prediction is data-starved at the enzyme level even though there are many "PE-eating microbes" in the literature.

### Realistic data counts after deduplication and clustering

A sequence-based ML project only cares about enzyme-level evidence, which collapses the landscape sharply:

| Group | Realistic positive count | Trainable? |
|---|---|---|
| PET / cutinases | ~80–150 sequences | **Yes** — strong |
| PHA / PLA hydrolases | ~50–100 | Yes — moderate |
| Other polyesters / PUR esterases | ~30–60 | Borderline |
| Nylon hydrolases (NylB family) | ~5–15 | No — too narrow, also too easy (very specific family) |
| PE / PP oxidative enzymes (laccase / MnP / AlkB with PE evidence) | ~10–25 | **No** — too few, evidence too weak |
| PS-active enzymes | ~5–10 | No |

**The real ML opportunity is the polyester hydrolases.** Everything else is either a lookup table (Nylon) or wishful thinking (PE/PS) at current data scales.

### Three honest framings for the project scope

1. **Single-substrate (PET-only)** — narrowest, strongest evidence
   - Positives: only PETases + cutinases (~80–120 after clustering)
   - Question: "Is this enzyme PET-active?"
   - Strongest scientific claim, smallest dataset

2. **Polyester-class (recommended for mini-project)**
   - Positives: PETases + MHETases + cutinases + PLA/PHA hydrolases + polyester esterases (~150–250)
   - Hard negatives: other esterases (lipases on triglycerides, carboxylesterases on natural substrates)
   - Question: "Is this enzyme a polyester-active hydrolase?"
   - Best balance of data size and biological coherence

3. **Hydrolyzable-plastic-class (broader)** — adds polyamide + polyurethane
   - Positives: above + nylonases + polyurethanases (~200–300)
   - Question: "Is this enzyme active on any hydrolyzable plastic?"
   - Risks lumping mechanistically distinct enzymes; harder to defend

---

## Part 2 — Where the Dataset Actually Comes From

### UniProt is the right primary source, but it's not enough alone

UniProt is a general protein database — it has every reviewed protein sequence, but **there is no "this enzyme degrades PET" field anywhere in the schema**.

| What you need | UniProt provides? |
|---|---|
| Protein sequence | Yes — gold standard |
| Source organism (microbe) | Yes — every entry has taxonomy |
| Generic enzyme function (e.g., "esterase") | Yes — function annotations + EC numbers |
| **Specific plastic substrate (PET vs PLA vs PE)** | **No** — not a field. Would have to read each entry's literature manually. |
| **Evidence tier (in vitro proven on plastic vs homology guess)** | **No** — Swiss-Prot is curated for general function, not plastic activity. |
| Direct query "all PET-degrading enzymes" | **No** — only via proxy (keywords, EC numbers, names) — leaky and incomplete |

The current `fetch_uniprot.py` works around this by searching for *enzyme family names* ("PETase", "cutinase") as a proxy for plastic activity. That captures most correct entries but:

- Pulls in Tier 3 homologs that were *named* PETase based on similarity, not experimental evidence
- Misses validated plastic-active enzymes named generically (e.g., "esterase from *Thermobifida fusca*" that turns out to be a cutinase)
- Cannot distinguish a PETase that was tested on PET from one only tested on PHA

### What you actually need: a curated mapping database

Curated databases like PAZy and PlasticDB exist precisely to be the human-curated bridge that UniProt doesn't have.

| Resource | What it provides | Quality |
|---|---|---|
| **PAZy** (Plastics-Active enZYmes) | ~150–300 entries: enzyme name, **UniProt accession**, **specific plastic substrate**, source organism, **literature citation** | Tier 1 — manually curated by experts at BAM (German Federal Inst. for Materials Research) |
| **PlasticDB** | Similar, more environmental microbe focus, includes whole-organism degradation data | Tier 1–2 |
| **MEROPS** | Peptidase database — useful for nylonases | Tier 1 |
| **CAZy** | Carbohydrate-active enzymes — overlaps with cutinases | Tier 1 |
| **BRENDA** | Enzyme function/reaction DB — sometimes includes synthetic substrates like PET | Tier 1 |

### The standard bioinformatics pattern

Every serious plastic-enzyme paper builds its dataset this way:

```
PAZy   →  list of UniProt accessions with verified plastic activity
  ↓
UniProt   →  fetch sequences + organism + EC + cross-refs for those accessions
  ↓
Your dataset
```

PAZy tells you *what* to fetch and *what plastic* it degrades. UniProt provides the actual sequence data. Neither alone is sufficient.

### Practical access to PAZy

PAZy doesn't have a clean REST API. Three options:

1. **Manual copy (recommended for mini-project)**: visit pazy.eu, copy the accession column from each enzyme family table into a JSON file. ~150 IDs, takes 15 minutes one-time.
2. **HTML scraping**: brittle, breaks when their site updates, but automatable.
3. **Email the curators**: they're academics and usually share data dumps if asked. Slower.

Suggested file format at `data/raw/pazy_accessions.json`:

```json
{
  "PET": ["A0A0K8P6T7", "G9BY57", "G4FEV4", "..."],
  "PLA": ["...", "..."],
  "PHA": ["...", "..."],
  "PUR": ["...", "..."],
  "PE":  ["...", "..."],
  "PS":  ["...", "..."]
}
```

This file is the **ground-truth mapping**. Everything downstream — sequence fetch, evidence tiering, substrate annotation — derives from it.

### What adding PAZy changes about the project

A current dataset built only from UniProt keyword search is on Tier 3 homology-based annotations. Not wrong, but not maximally rigorous. Adding PAZy gives:

1. **Tier 1 confidence flag**: an `evidence_tier` column distinguishing PAZy-validated vs. UniProt-keyword-only entries.
2. **Substrate granularity**: PETase entries from PAZy can be flagged "validated on PET" specifically, vs. cutinases that PAZy validated on PLA or cutin only.
3. **Stronger negatives**: anything in EC 3.1.1.- that is *not* in PAZy is much more confidently a non-plastic-active hydrolase.
4. **A defensible result line**: "PR-AUC on the PAZy-validated test subset (Tier 1 only): X" — the number that goes in the project report and that a reviewer cannot poke holes in.

### Honest summary

- **For sequences**: UniProt is necessary and sufficient.
- **For the plastic-to-enzyme mapping**: UniProt alone is insufficient. PAZy (or PlasticDB) is required as the curated bridge. There's no shortcut — this is work human curators have done, and it can't be obtained from any single API.
- **Without PAZy**: project is still valid, but framed as "predicting plastic-degrading enzyme *families*" (using UniProt keyword as a noisy proxy) rather than "predicting plastic-degrading enzymes" (using PAZy as ground truth).
- **With PAZy**: the project becomes substantially more defensible and matches the methodology of published work.

### One concrete enhancement

A `substrate` column added to `dataset.csv` would make the polyester scope visible in the data, not just in the writeup:

| label | substrate | example enzymes |
|---|---|---|
| 1 | PET | PETase, MHETase, LCC, TfCut |
| 1 | PLA | PLA depolymerase, *Amycolatopsis* protease |
| 1 | PHA | PHA depolymerase |
| 1 | cutin | Cutinase (natural substrate, also acts on polyesters) |
| 0 | triglyceride | Lipase (natural substrate, NOT polyester) |
| 0 | other | DNA pol, RNA pol, etc. |

This is just metadata — the model still trains on binary `label`. But the evaluation can then report **per-substrate recall**: "the model recovers 92% of PETases, 85% of cutinases, 70% of PHA depolymerases" — which is much more informative than a single F1 score.
