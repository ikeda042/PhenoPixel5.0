#!/usr/bin/env python3
"""
Visualize cell contours.

Produces two figures:
1. lda_result.png     – 2-D scatter (LDA, or PCA fallback when one class).
2. vectors_overlay.png – Overlay of *true* radial-distance vectors
                         (trailing zero padding is now hidden).

Changes vs previous version
---------------------------
* `prepare_vectors` now returns both the zero-padded matrix **and** the
  true length of each original vector.
* `plot_overlay` now draws only the first *length* elements for every
  contour, so artificial zeros from padding never appear.
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
LABEL_FILTER: Optional[str] = None  # e.g. "1" or "N/A"; None = no filter
EMBEDDING_PLOT = "lda_result.png"
OVERLAY_PLOT = "vectors_overlay.png"
# ──────────────────────────────────────────────────────────────────────────────


# ─── Data Loading & Preparation ───────────────────────────────────────────────
def fetch_contours(
    db_path: str, label: Optional[str] = None
) -> Tuple[List[np.ndarray], List[str]]:
    """Load contours (as numpy arrays) and their labels from the SQLite DB."""
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
    """Convert a contour (N×2) → sorted radial-distance vector."""
    center = contour.mean(axis=0)
    distances = np.linalg.norm(contour - center, axis=1)
    return np.sort(distances)


def prepare_vectors(
    contours: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      padded_vectors : (M × L) float array (zero-padded)
      lengths        : (M,) int array of true vector lengths
    """
    raw_vecs = [contour_to_vector(c) for c in contours]
    lengths = np.array([len(v) for v in raw_vecs])
    max_len = lengths.max()
    padded = [
        np.pad(v, (0, max_len - len(v)), constant_values=0.0) for v in raw_vecs
    ]
    return np.vstack(padded), lengths


# ─── Dimensionality Reduction ────────────────────────────────────────────────
def project_vectors(
    vectors: np.ndarray, labels: List[str]
) -> Tuple[np.ndarray, str]:
    """Return 2-D embedding + method string (LDA or PCA fallback)."""
    classes = np.unique(labels)

    if len(classes) >= 2:
        n_comp = min(2, vectors.shape[1], len(classes) - 1)
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        emb = lda.fit_transform(vectors, labels)
        if n_comp == 1:  # pad to 2-D
            emb = np.column_stack([emb, np.zeros_like(emb)])
        method = f"LDA ({n_comp}-D)"
    else:
        n_comp = 2 if vectors.shape[1] >= 2 else 1
        pca = PCA(n_components=n_comp)
        emb = pca.fit_transform(vectors)
        if n_comp == 1:
            emb = np.column_stack([emb, np.zeros_like(emb)])
        method = f"PCA ({n_comp}-D fallback)"

    return emb, method


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_embedding(
    emb: np.ndarray, labels: List[str], title: str, out_path: str
) -> None:
    """Scatter plot of 2-D embeddings."""
    plt.figure(figsize=(6, 6))
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(emb[idx, 0], emb[idx, 1], label=lab, s=14)
    if len(set(labels)) > 1:
        plt.legend(markerscale=1.5, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved embedding plot → {out_path}")


def plot_overlay(
    vectors: np.ndarray,
    lengths: np.ndarray,
    labels: List[str],
    out_path: str,
    cmap_name: str = "tab10",
    alpha: float = 0.25,
) -> None:
    """
    Overlay all radial-distance vectors without showing zero padding.
    Each actual vector length is respected.
    """
    plt.figure(figsize=(9, 5))
    cmap = get_cmap(cmap_name)
    label_colors: Dict[str, str] = {
        lab: cmap(i % cmap.N) for i, lab in enumerate(sorted(set(labels)))
    }

    for vec, n, lab in zip(vectors, lengths, labels):
        plt.plot(range(n), vec[:n], color=label_colors[lab], alpha=alpha, lw=0.8)

    # legend (one entry per class)
    for lab, col in label_colors.items():
        plt.plot([], [], color=col, label=lab, lw=2)
    if len(label_colors) > 1:
        plt.legend(fontsize=8)

    plt.xlabel("Index (native length)")
    plt.ylabel("Radial distance (pixels)")
    plt.title("Overlay of radial-distance vectors (padding removed)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved overlay plot   → {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    contours, labels = fetch_contours(DB_PATH, LABEL_FILTER)
    if not contours:
        print("No data found in DB.")
        return

    vectors, lengths = prepare_vectors(contours)
    print(f"Prepared {len(vectors)} vectors | padded shape = {vectors.shape}")

    # 1) 2-D embedding
    emb, method = project_vectors(vectors, labels)
    plot_embedding(emb, labels, method, EMBEDDING_PLOT)

    # 2) Overlay (true vectors)
    plot_overlay(vectors, lengths, labels, OVERLAY_PLOT)


if __name__ == "__main__":
    main()
