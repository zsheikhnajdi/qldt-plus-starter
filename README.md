# QLDT+ — Interpretable Quantum‑Logic Decision Trees

Code for building and evaluating QLDT+ trees (quantum‑logic inspired decision trees) and CART baselines on public datasets (Pima, Transfusion, Heart).

## Quickstart

### 0) Prerequisites
- Python 3.10–3.11
- Graphviz runtime for PNG export

### 1) Create environment
```bash
# conda
conda env create -f environment.yml
conda activate qldt

# or pip
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Get data
Either put original CSVs into `data/raw/` using **these exact names**:
- `pima-indians-diabetes.data.csv`
- `transfusion.csv`
- `heart.csv`

**or** run the fetcher:
```bash
python scripts/fetch_data.py --all
```

### 3) Run a demo
```bash
python src/run_qldt.py --dataset pima --grid --draw
python src/run_qldt.py --dataset transfusion --draw
python src/run_qldt.py --dataset heart --draw
```

Outputs appear in `artifacts/`:
- `qldt_*.png`, `cart_*.png`
- `qldt_param_grid_search_results.csv`
- `qldt_interpretability_table.html`
- `qldt_feature_importance.png`

## Project layout
```
QLDT-Plus/
├─ src/
│  ├─ qldtplus.py
│  ├─ run_qldt.py
│  ├─ utils_io.py
│  └─__init__.py
├─ data/
│  ├─ sample/   # tiny csvs (optional)
│  └─ raw/      # full datasets (gitignored)
├─ artifacts/   # figures/csv/html outputs
├─ scripts/
│  └─ fetch_data.py
├─ tests/
│  └─ test_smoke.py
├─ environment.yml
├─ requirements.txt
├─ .gitignore
├─ .gitattributes        # (Git LFS)
└─ README.md
```

## Reproducibility
- Fixed random seeds where applicable.
- Balanced splits.
- Regenerates all figures with the commands above.

## License
This project is released under the **MIT License (Simplified Attribution Version)**.  
© 2025 **Zahra Sheikh Najdi**

You are free to use, modify, and distribute this software,  
provided that proper credit is given to the author:

> "Developed by Zahra Sheikh Najdi (2025)"

See the [LICENSE](./LICENSE) file for full details.

