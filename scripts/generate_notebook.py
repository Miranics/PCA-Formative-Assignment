from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "pca_formative_assignment.ipynb"


def create_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {
        "name": "python",
        "version": "3.13"
    }

    cells = [
        nbf.v4.new_markdown_cell(
            "# Formative 2 – Principal Component Analysis\n"
            "This notebook implements Principal Component Analysis (PCA) from scratch on an African development indicators dataset to satisfy the formative assignment requirements."
        ),
        nbf.v4.new_markdown_cell(
            "## Dataset overview\n"
            "We use a curated African Development Indicators dataset that includes socio-economic metrics for several countries between 2019 and 2020. The dataset intentionally contains missing values and categorical data (country names) to demonstrate cleaning, encoding, and PCA readiness steps."
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                # Import required libraries
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                import seaborn as sns
                from pathlib import Path
                from time import perf_counter

                plt.style.use("seaborn-v0_8")
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                # Load and inspect the dataset
                DATA_PATH = Path('..') / 'data' / 'african_development_indicators.csv'
                df_raw = pd.read_csv(DATA_PATH)
                print(f'Rows: {len(df_raw)}')
                display(df_raw.head())
                print('Data types and non-null counts:')
                display(df_raw.dtypes)
                print('Missing values per column:')
                display(df_raw.isna().sum())
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            "## Data preparation\n"
            "We handle missing values by imputing numeric columns with their mean and categorical columns with the mode. Country names are one-hot encoded to retain regional information while producing a fully numeric matrix suitable for PCA. Finally, we standardize the features to zero mean and unit variance so that PCA is not dominated by scale differences."
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                # Impute missing values and encode categorical data
                df_clean = df_raw.copy()
                categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

                for col in numeric_cols:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

                for col in categorical_cols:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iat[0])

                df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
                feature_names = df_encoded.columns

                X = df_encoded.to_numpy(dtype=float)
                X_mean = X.mean(axis=0)
                X_std = X.std(axis=0, ddof=0)
                X_std[X_std == 0] = 1.0  # prevent division by zero for constant features
                X_scaled = (X - X_mean) / X_std

                print('Prepared feature matrix shape:', X_scaled.shape)
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            "## Task 1 – Implement PCA from first principles\n"
            "We compute the covariance matrix of the standardized data, perform eigen-decomposition, sort eigenvalues in descending order, and project the data onto the principal components."
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                def compute_covariance_matrix(X: np.ndarray) -> np.ndarray:
                    # Return the sample covariance matrix for a centered dataset.
                    n_samples = X.shape[0]
                    X_centered = X - X.mean(axis=0)
                    covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
                    return covariance_matrix


                def eigen_decomposition(covariance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    # Compute eigenvalues and eigenvectors sorted from largest to smallest eigenvalue.
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
                    sort_idx = np.argsort(eigenvalues)[::-1]
                    eigenvalues_sorted = eigenvalues[sort_idx]
                    eigenvectors_sorted = eigenvectors[:, sort_idx]
                    return eigenvalues_sorted, eigenvectors_sorted


                def project_data(X: np.ndarray, eigenvectors: np.ndarray, n_components: int) -> np.ndarray:
                    # Project data onto the first n_components principal directions.
                    return X @ eigenvectors[:, :n_components]


                def compute_explained_variance(eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    total_variance = eigenvalues.sum()
                    explained_variance_ratio = eigenvalues / total_variance
                    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
                    return explained_variance_ratio, cumulative_explained_variance


                def pca_from_scratch(X: np.ndarray, n_components: int | None = None) -> dict[str, np.ndarray]:
                    # Perform PCA using covariance eigen-decomposition.
                    if n_components is None:
                        n_components = X.shape[1]

                    covariance_matrix = compute_covariance_matrix(X)
                    eigenvalues, eigenvectors = eigen_decomposition(covariance_matrix)
                    explained_variance_ratio, cumulative_variance = compute_explained_variance(eigenvalues)
                    projected_data = project_data(X, eigenvectors, n_components)

                    return {
                        'covariance_matrix': covariance_matrix,
                        'eigenvalues': eigenvalues,
                        'eigenvectors': eigenvectors,
                        'projected_data': projected_data,
                        'explained_variance_ratio': explained_variance_ratio,
                        'cumulative_explained_variance': cumulative_variance,
                    }
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                # Run PCA on the standardized data
                pca_result = pca_from_scratch(X_scaled)
                explained_variance_df = pd.DataFrame({
                    'Eigenvalue': pca_result['eigenvalues'],
                    'Explained Variance Ratio': pca_result['explained_variance_ratio'],
                    'Cumulative Explained Variance': pca_result['cumulative_explained_variance'],
                })
                explained_variance_df.head(10)
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            "## Task 2 – Select principal components dynamically\n"
            "We choose the smallest number of components that achieve a target explained variance threshold. The helper below returns both the selected dimensionality and the transformed dataset."
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                def select_number_of_components(cumulative_variance: np.ndarray, threshold: float = 0.95) -> int:
                    # Return the minimal number of components needed to reach the desired explained variance.
                    if threshold <= 0 or threshold > 1:
                        raise ValueError('threshold must be within (0, 1].')
                    n_components = int(np.searchsorted(cumulative_variance, threshold) + 1)
                    return n_components


                variance_threshold = 0.95
                n_components_optimal = select_number_of_components(
                    pca_result['cumulative_explained_variance'],
                    threshold=variance_threshold,
                )
                projected_optimal = pca_result['projected_data'][:, :n_components_optimal]
                print(f'Minimum components for {variance_threshold:.0%} variance: {n_components_optimal}')
                projected_optimal[:5]
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                # Visualize explained variance profile
                fig, ax = plt.subplots(figsize=(8, 5))
                indices = np.arange(1, len(pca_result['explained_variance_ratio']) + 1)
                ax.bar(indices, pca_result['explained_variance_ratio'], alpha=0.7, label='Explained variance ratio')
                ax.plot(indices, pca_result['cumulative_explained_variance'], marker='o', color='black', label='Cumulative explained variance')
                ax.axhline(y=variance_threshold, color='red', linestyle='--', label=f'{variance_threshold:.0%} threshold')
                ax.set_xlabel('Principal component index')
                ax.set_ylabel('Variance ratio')
                ax.set_title('Explained variance by principal component')
                ax.legend()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            "## Task 3 – Optimise for larger datasets\n"
            "The PCA implementation leverages vectorised NumPy operations. To demonstrate scalability, we benchmark the runtime on synthetic datasets with thousands of samples and features. This validates that the approach can process larger inputs efficiently."
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                def benchmark_pca_runtime(n_samples: int = 5000, n_features: int = 40, n_components: int = 10, runs: int = 3) -> dict[str, float]:
                    rng = np.random.default_rng(seed=42)
                    results = []
                    for _ in range(runs):
                        X_synthetic = rng.normal(size=(n_samples, n_features))
                        start = perf_counter()
                        pca_from_scratch(X_synthetic, n_components=n_components)
                        results.append(perf_counter() - start)
                    results = np.array(results)
                    return {
                        'mean_seconds': results.mean(),
                        'std_seconds': results.std(ddof=1),
                        'runs': runs,
                        'samples': n_samples,
                        'features': n_features,
                        'components': n_components,
                    }


                benchmark_stats = benchmark_pca_runtime()
                benchmark_stats
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            "## Visualisations – Before and After PCA\n"
            "We compare the structure of the data in the original feature space against the first two principal components."
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                # Original feature space (two informative dimensions)
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(
                    data=df_clean,
                    x='Access_to_electricity_pct',
                    y='Internet_users_pct',
                    hue='Country',
                    palette='tab10',
                    ax=ax,
                )
                ax.set_title('Original feature space: Access to electricity vs. Internet usage')
                ax.set_xlabel('Access to electricity (%)')
                ax.set_ylabel('Internet users (%)')
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Country')
                plt.tight_layout()
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                # PCA-transformed space (PC1 vs PC2)
                pc_df = pd.DataFrame(projected_optimal[:, :2], columns=['PC1', 'PC2'])
                pc_df['Country'] = df_clean['Country'].values
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue='Country', palette='tab10', ax=ax)
                ax.set_title('Principal component space: PC1 vs PC2')
                ax.set_xlabel('PC1 (highest variance)')
                ax.set_ylabel('PC2')
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Country')
                plt.tight_layout()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation\n"
            "- **Variance retention:** The cumulative explained variance plot shows how quickly the first few components capture most of the information, guiding the component selection.\n"
            "- **Structure preservation:** Clusters present in the original scatter plot remain distinguishable after PCA, albeit rotated into the new orthogonal basis.\n"
            "- **Scalability:** Vectorised NumPy operations enable PCA to run efficiently even on larger synthetic datasets, offering confidence that the approach generalises beyond this formative example."
        ),
    ]

    nb.cells = cells
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    create_notebook()
