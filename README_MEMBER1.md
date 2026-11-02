# Member 1 — Motasem's Data Collection and Cleaning

**Use this as an overlay in the existing project** `Roland-143/MLDM-Linear-And-Logistic-Modeling`. It adds files; it does not replace the team's `src/preprocessing.py`, `src/train.py`, or `README.md`.

## Run (Mac / Windows / Linux)

At the existing repo root, after copying the package files into their matching paths:

```bash
python -m pip install -r requirements.txt
python -m src.download_kc
python -m src.member1_prepare
python -m pytest -q
```

- The first module downloads and verifies the real 21,613-row King County housing dataset (CC0 data, originally from Kaggle).
- The second audits the data, exports CSVs for Tableau, creates repeat-property-safe train/test splits, and fits encoding/scaling **on training data only**.
- The final command checks tests.
- Build the **Tableau** dashboard following `docs/TABLEAU_DASHBOARD.md`.
- Give Nick `docs/MEMBER1_HANDOFF.md` for the report; Daniel and Monish can use the split/preprocessing outputs in `data/processed/`.

**Important integration:** The team's pre-existing `src/train.py` uses a row-level split. It should use `split_sales()` from `src.member1_prepare` to match the Member 1 property-group split. Do not combine model results from different splits. Change the trainer only after agreeing with the modeling teammates.

## What is supplied

- `src/download_kc.py` — dataset download and structural validation.
- `src/member1_prepare.py` — data-quality audit, Tableau CSV exports, split and encoding.
- `notebooks/01_eda.ipynb` — reproducible EDA notebook (statistics; final visualizations in Tableau).
- `tests/test_member1.py` — three preprocessing/EDA smoke tests.
- `docs/MEMBER1_HANDOFF.md` — results and documentation for Nick and model developers.
- `docs/TABLEAU_DASHBOARD.md` — seven concrete Tableau worksheet specifications.
- `tableau/data_quality_snapshot.csv` and `tableau/correlation_to_price_snapshot.csv` — small verified findings from the original dataset. More Tableau exports are generated when the scripts run.

## Share via GitHub

In Terminal, while inside the team repository:

```bash
git checkout -b member1-motasem-data-eda
# Copy the contents of this ZIP into the repository, preserving src/, notebooks/, tests/, docs/, and tableau/.
python -m src.download_kc
python -m src.member1_prepare
git add src/download_kc.py src/member1_prepare.py notebooks/01_eda.ipynb tests/test_member1.py docs/ README_MEMBER1.md tableau/
git commit -m "Add Member 1 data collection, EDA, and Tableau exports"
git push -u origin member1-motasem-data-eda
```

Then create a Pull Request on GitHub into `main`. Because this is the team's repo, a PR is safer than overwriting `main`.

Note: `.gitignore` keeps `data/raw/` and `data/processed/` out of GitHub by default; this is intentional. Team members can generate those local data files via the two Python commands. The tableau files are trackable under `tableau/`.
