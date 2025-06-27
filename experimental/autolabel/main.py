#!/usr/bin/env python3
"""
Visualize cell contours.

Generates two figures:
1. A 2-D embedding scatter plot (LDA or PCA fallback)              → lda_result.png
2. An overlay of all radial-distance feature vectors (per contour) → vectors_overlay.png
"""

import os
import pickle
import sqlite3
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ─── Configuration ────────────────────────────────────────────────────────────
DB_PATH: str = "experimental/autolabel/250626_SK450_Gen_1p0-completed.db"
LABEL_FILTER: Optional[str] = None  # e.g. "1" or "N/A"; None means no filter
EMBEDDING_PLOT: str = "lda_result.png"
OVERLAY_PLOT: str = "vectors_overlay.png"
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
    """Convert a contour (N×2) to a 1-D vector of radial distances (sorted)."""
    center = contour.mean(axis=0)
    distances = np.linalg.norm(contour - center, axis=1)
    return np.sort(distances)


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
    * LDA is used when two or more classes exist.
    * Falls back to PCA when only one class is present.
    """
    unique = np.unique(labels)

    if len(unique) >= 2:
        # LDA: components ≤ min(n_features, n_classes − 1, 2)
        n_comp = min(2, vectors.shape[1], len(unique) - 1)
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        transformed = lda.fit_transform(vectors, labels)
        if n_comp == 1:  # pad to 2-D for plotting
            transformed = np.column_stack([transformed, np.zeros_like(transformed)])
        method = f"LDA ({n_comp}-D)"
    else:
        # Only one class → use PCA
        n_comp = 2 if vectors.shape[1] >= 2 else 1
        pca = PCA(n_components=n_comp)
        transformed = pca.fit_transform(vectors)
        if n_comp == 1:
            transformed = np.column_stack([transformed, np.zeros_like(transformed)])
        method = f"PCA ({n_comp}-D fallback)"

    return transformed, method


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_embedding(
    embeddings: np.ndarray, labels: List[str], title: str, output: str
) -> None:
    """Scatter plot of 2-D embeddings."""
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
    print(f"Saved embedding plot → {output}")


def plot_overlay(
    vectors: np.ndarray,
    labels: List[str],
    output: str,
    color_cycle: str = "tab10",
    alpha: float = 0.25,
) -> None:
    """
    Overlay all radial-distance vectors.
    Each vector is a polyline; colours cycle by label.
    """
    plt.figure(figsize=(8, 5))
    cmap = get_cmap(color_cycle)
    label_to_color: Dict[str, str] = {
        lab: cmap(i % cmap.N) for i, lab in enumerate(sorted(set(labels)))
    }

    x = np.arange(vectors.shape[1])
    for vec, lab in zip(vectors, labels):
        plt.plot(x, vec, color=label_to_color[lab], alpha=alpha, linewidth=0.8)

    # Create custom legend entries (one per class)
    for lab, col in label_to_color.items():
        plt.plot([], [], color=col, label=lab, linewidth=2)
    if len(label_to_color) > 1:
        plt.legend(fontsize=8)

    plt.xlabel("Index (padded)")
    plt.ylabel("Radial distance (pixels)")
    plt.title("Overlay of all radial-distance feature vectors")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved overlay plot   → {output}")


# ─── Main Routine ────────────────────────────────────────────────────────────
def main() -> None:
    contours, labels = fetch_contours(DB_PATH, LABEL_FILTER)
    if not contours:
        print("No data found.")
        return

    vectors = prepare_vectors(contours)
    print(f"Prepared {len(vectors)} vectors with shape {vectors.shape}")

    # 1) 2-D embedding (LDA / PCA)
    embeddings, method = project_vectors(vectors, labels)
    plot_embedding(embeddings, labels, method, EMBEDDING_PLOT)

    # 2) Overlay of raw vectors
    plot_overlay(vectors, labels, OVERLAY_PLOT)


if __name__ == "__main__":
    main()
