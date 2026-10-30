# Collaborative AI Mini-Research Starter Repository

This repository provides a clean starter structure for a small collaborative AI mini-research project.

## Project Structure

```text
.
├── data/                  # Dataset inputs (kept in repo with .gitkeep)
├── paper/
│   ├── main.tex           # Starter LaTeX paper scaffold
│   └── references.bib     # BibLaTeX reference file (APA 7)
├── requirements.txt       # Python dependencies
├── results/               # Generated plots/tables (kept in repo with .gitkeep)
└── src/
    ├── main.py            # Program entry point
    ├── model.py           # Starter numerical model
    └── visualization.py   # Plotting and table export helpers
```

## Prerequisites

- Python 3.10+
- `pip`
- (Optional for paper compilation) A LaTeX distribution with `pdflatex` + `biber`

## Setup and Run Instructions

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the starter simulation**

   ```bash
   python src/main.py
   ```

4. **Check outputs**

   Running the script creates:
   - `results/population_trajectory.png`
   - `results/population_results.csv`

## Reproducibility Notes

- Keep all raw/processed datasets in `data/`.
- Keep generated figures and tabulated outputs in `results/`.
- Record parameter choices and assumptions in both code and paper sections as the project evolves.

## Required Dependencies

Dependencies are pinned minimally in `requirements.txt`:

- `numpy`
- `pandas`
- `matplotlib`

Install them with:

```bash
pip install -r requirements.txt
```

## LaTeX Paper Scaffold

The `paper/main.tex` file includes required sections and packages (`amsmath`, `listings`, `hyperref`, `biblatex`, `geometry`) and is configured for APA 7 references.

To compile (example):

```bash
cd paper
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```
