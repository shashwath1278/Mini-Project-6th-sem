# `plasticdeg` — sole pipeline code

Legacy `src/` and `run_pipeline.py` were removed. All new implementation lives in **subpackage folders**; the package root only keeps `paths.py`, `evaluation_spec.py`, and `__init__.py`.

## Layout (where the scripts live)

| Subpackage | Run with `python -m plasticdeg.<path>` |
|------------|----------------------------------------|
| `plasticdeg.ingest.*` | Sequences, negatives, PlasticDB/PAZy import, ground-truth cleanup |
| `plasticdeg.expand.*` | BLAST expansion, integrate merge |
| `plasticdeg.split.*` | Homology / stratified splits (`homology_split`) |
| `plasticdeg.embed.*` | ESM-2 embedding pipelines |
| `plasticdeg.train.*` | RF baselines on embeddings |
| `plasticdeg.eval.*` | Reports, probes, label audit |
| `plasticdeg.bundle.*` | Manuscript artifact bundle |
| `plasticdeg.support.*` | UniProt ID map helpers, dead-accession list (library use; no CLI) |

## Run (from repository root)

```bash
pip install -r requirements.txt
python -m plasticdeg.ingest.fetch_sequences
# After homology_split (train/test id lists exist):
python -m plasticdeg.embed.embed_sequences   # writes embeddings_esm2_t33_mean.npz (large; gitignored)
```

**Windows + PyTorch:** If `import torch` fails with `shm.dll` / `WinError 127`, you are almost certainly using **Miniconda base Python 3.13** (wrong interpreter). Create env `plasticdeg` with **Python 3.12**, then verify with **`where.exe python`** (in PowerShell, `where` alone is `Where-Object`, not PATH search). First line must be `...\envs\plasticdeg\python.exe`.

If `conda activate plasticdeg` does not stick, use (with **live** output — avoids looking “stuck”):

`conda run --no-capture-output -n plasticdeg python -u -m plasticdeg.embed.embed_sequences --batch-size 4`

(`-u` unbuffered; `--no-capture-output` streams child stdout/stderr.)

**CPU runtime:** `esm2_t33_650M_UR50D` over ~1.5k sequences can take **1–4+ hours** on a laptop CPU (normal). For a quicker baseline embed, use `--model esm2_t12_35M_UR50D` and e.g. `--out data/processed_v2/embeddings/embeddings_esm2_t12_mean.npz` (smaller, faster; different embedding dimension).

- If `conda create` fails with **CondaToSNonInteractiveError**, run the three `conda tos accept ...` lines Conda prints, **or** create from conda-forge only:  
  `conda create -n plasticdeg python=3.12 -c conda-forge --override-channels -y`

- **OMP duplicate lib** when importing sklearn+torch: set `KMP_DUPLICATE_LIB_OK=TRUE` for that session (workaround only).

- Install **VC++ Redistributable (x64)** if DLL errors persist. **WSL Ubuntu** + `pip install torch fair-esm` is a reliable fallback to run `python -m plasticdeg.embed.embed_sequences`.

Inputs: `data/pazy/ground_truth.json`  
Outputs: `data/processed_v2/sequences/positives_from_gt.fasta`, `tables/positives_gt.csv` (defaults are centralized in `plasticdeg/paths.py`).

See `data/pazy/README.md` for curation rules.
