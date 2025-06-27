#!/usr/bin/env python3


import os
import pickle
import sqlite3
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ─── Configuration ────────────────────────────────────────────────────────────
DB_PATH: str = "experimental/autolabel/250626_SK450_Gen_1p0-completed.db"
LABEL_FILTER: Optional[str] = None  # e.g. "1" or "N/A"; None means no filter
OUTPUT_FILE: str = "lda_result.png"
# ──────────────────────────────────────────────────────────────────────────────


# ─── Data Loading & Preparation ───────────────────────────────────────────────
def fetch_contours(
    db_path: str, label: Optional[str] = None
) -> Tuple[List[np.ndarray], List[str]]:
    """Fetch contours (as numpy arrays) and their labels from the DB."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = "SELECT contour, manual_label FROM cells"
    params: Tuple = ()
    if label is not None:
        query += " WHERE manual_label = ?"
        params = (label,)

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    contours, labels = [], []
    for contour_blob, manual_label in rows:
        contour = np.asarray(pickle.loads(contour_blob)).reshape(-1, 2)
        contours.append(contour)
        labels.append(str(manual_label))

    return contours, labels


def contour_to_vector(contour: np.ndarray) -> np.ndarray:
    """Convert a contour (N×2) to a 1-D vector of radial distances."""
    center = contour.mean(axis=0)
    distances = np.linalg.norm(contour - center, axis=1)
    return np.sort(distances)  # rotation/translation invariant


def prepare_vectors(contours: List[np.ndarray]) -> np.ndarray:
    """Pad all feature vectors to equal length and stack into a 2-D array."""
    vectors = [contour_to_vector(c) for c in contours]
    max_len = max(len(v) for v in vectors)
    padded = [np.pad(v, (0, max_len - len(v)), constant_values=0.0) for v in vectors]
    return np.vstack(padded)


# ─── Dimensionality Reduction ────────────────────────────────────────────────
def project_vectors(vectors: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, str]:
    """
    Reduce dimensionality to 2-D.
    * LDA is used when ≥ 2 classes exist.
    * Falls back to PCA when only one class is present.
    """
    unique = np.unique(labels)

    if len(unique) >= 2:
        # LDA: components ≤ min(n_features, n_classes − 1, 2)
        n_comp = min(2, vectors.shape[1], len(unique) - 1)
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        transformed = lda.fit_transform(vectors, labels)
        if n_comp == 1:  # add dummy axis so we can still plot 2-D
            transformed = np.column_stack([transformed, np.zeros_like(transformed)])
        method = f"LDA ({n_comp}-D)"
    else:
        # Only one class → Linear Discriminant is undefined
        n_comp = 2 if vectors.shape[1] >= 2 else 1
        pca = PCA(n_components=n_comp)
        transformed = pca.fit_transform(vectors)
        if n_comp == 1:
            transformed = np.column_stack([transformed, np.zeros_like(transformed)])
        method = f"PCA ({n_comp}-D, fallback)"

    return transformed, method


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_vectors_2d(
    embeddings: np.ndarray, labels: List[str], title: str, output: str
) -> None:
    """Scatter-plot the 2-D embeddings, one color per label."""
    plt.figure(figsize=(6, 6))
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=lab, s=14)
    if len(set(labels)) > 1:
        plt.legend(markerscale=1.5, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved figure to {output}")


# ─── Main Routine ────────────────────────────────────────────────────────────
def main() -> None:
    contours, labels = fetch_contours(DB_PATH, LABEL_FILTER)
    if not contours:
        print("No data found.")
        return

    vectors = prepare_vectors(contours)
    print(f"Prepared {len(vectors)} vectors with shape {vectors.shape}")

    embeddings, method = project_vectors(vectors, labels)
    plot_vectors_2d(embeddings, labels, method, OUTPUT_FILE)


if __name__ == "__main__":
    main()
