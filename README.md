# PCA-Formative-Assignment

https://colab.research.google.com/drive/14xRWzOe02905W2GgV7DtjkgHIE3SpqsG?usp=sharing

Principal Component Analysis (PCA) workflow for the Advanced Linear Algebra formative assignment. The repository contains:

- `data/african_development_indicators.csv` &mdash; an African-focused dataset with missing values and the non-numeric `Country` column.
- `notebooks/pca_formative_assignment.ipynb` &mdash; the completed assessment notebook with every template "TO DO" implemented and outputs rendered.
- `requirements.txt` &mdash; dependencies needed to reproduce the results locally or in Google Colab.

## 🗂️ Dataset highlights

- Covers 10 African countries (2019 &amp; 2020) with socio-economic metrics (electricity access, internet adoption, education index, etc.).
- Includes intentional `NaN` entries and categorical data to demonstrate imputation and encoding as required by the rubric.

## 🚀 Getting started

1. **Clone** the repo and create a virtual environment (recommended).
2. **Install dependencies**:

	```powershell
	pip install -r requirements.txt
	```

3. **Open the notebook** in VS Code or Jupyter/Colab and run cells from top to bottom to regenerate outputs.

## 🧭 Notebook roadmap

| Section | Description |
| --- | --- |
| Assignment context | Summarises rubric requirements and dataset compliance. |
| Dataset inspection | Loads the CSV, surfaces missing values, and classifies column types. |
| Data preparation | Imputes numeric/categorical fields and one-hot encodes `Country`. |
| Step 1 | Pure NumPy standardisation matching the provided formula. |
| Step 3 | Covariance matrix computation. |
| Step 4 | Eigendecomposition using `numpy.linalg.eigh`. |
| Step 5 | Sorting eigenpairs, explained variance ratios, and visualisation. |
| Step 6–7 | Dynamic component selection (95% threshold) and projection. |
| Step 8 | Side-by-side visual comparison (original vs. PCA space). |
| Task 3 | Runtime benchmark on a large synthetic dataset (optimisation requirement). |
| Interpretation | Summarises findings and ties them back to the rubric. |

## ✅ Submission checklist

- [ ] All notebook cells executed with visible outputs (plots, tables, variance metrics).
- [ ] Reduced-dimensional dataset shape reported and makes sense.
- [ ] Explained variance plot demonstrates the 95% threshold selection.
- [ ] Benchmark cell runs without errors (NumPy-only PCA implementation).
- [ ] README, dataset, and notebook pushed to GitHub before submitting the repository link.

Feel free to adapt the dataset or threshold if your instructor requests variations; just keep the rubric requirements in mind.
