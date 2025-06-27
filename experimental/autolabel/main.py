#!/usr/bin/env python3
"""
Visualize cell contours.

Produces three figures:
1. lda_result.png      – 1-D LDA scatter（y はランダムジッタで重なり可視化）
2. pca_result.png      – 2-D PCA scatter
3. vectors_overlay.png – Overlay of true radial-distance vectors

Compatible with NumPy ≥ 2.0 (uses np.ptp).
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
LABEL_FILTER: Optional[str] = None
TARGET_LEN: int = 256
EMBEDDING_PLOT = "lda_result.png"
PCA_PLOT       = "pca_result.png"
OVERLAY_PLOT   = "vectors_overlay.png"
JITTER_SCALE   = 0.05
RNG_SEED       = 42
# ──────────────────────────────────────────────────────────────────────────────


# ─── Data Loading & Preparation ───────────────────────────────────────────────
def fetch_contours(
    db_path: str, label: Optional[str] = None
) -> Tuple[List[np.ndarray], List[str]]:
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
    center = contour.mean(axis=0)
    distances = np.linalg.norm(contour - center, axis=1)
    return np.sort(distances)


def resample_vector(vec: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    if len(vec) == target_len:
        return vec
    old_x = np.linspace(0.0, 1.0, len(vec))
    new_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(new_x, old_x, vec)


def prepare_resampled_vectors(
    contours: List[np.ndarray], target_len: int = TARGET_LEN
) -> np.ndarray:
    raw_vecs = [contour_to_vector(c) for c in contours]
    return np.vstack([resample_vector(v, target_len) for v in raw_vecs])


def prepare_vectors_for_overlay(
    contours: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    raw_vecs = [contour_to_vector(c) for c in contours]
    lengths = np.array([len(v) for v in raw_vecs])
    max_len = lengths.max()
    padded = [np.pad(v, (0, max_len - len(v)), constant_values=0.0) for v in raw_vecs]
    return np.vstack(padded), lengths


# ─── Dimensionality Reduction ────────────────────────────────────────────────
def project_vectors(
    vectors: np.ndarray, labels: List[str]
) -> Tuple[np.ndarray, str]:
    classes = np.unique(labels)
    if len(classes) >= 2:
        # ★ CHANGED – shrinkage='auto' で数値安定性向上
        lda = LinearDiscriminantAnalysis(n_components=1, solver="lsqr",
                                         shrinkage="auto")
        emb = lda.fit_transform(vectors, labels)
        method = "LDA (1-D, shrinkage)"
    else:
        pca = PCA(n_components=1)
        emb = pca.fit_transform(vectors)
        method = "PCA (1-D fallback)"
    return emb, method


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_embedding(
    emb: np.ndarray, labels: List[str], title: str, out_path: str
) -> None:
    emb = np.asarray(emb)
    if emb.ndim == 1:
        emb = emb[:, None]

    if emb.shape[1] == 1:                      # 1-D → jitter scatter
        x = emb[:, 0]
        rng = np.random.default_rng(RNG_SEED)
        # ★ CHANGED – x.ptp() → np.ptp(x) で NumPy 2.0 対応
        y = rng.normal(0.0,
                       JITTER_SCALE * (np.ptp(x) if np.ptp(x) > 0 else 1.0),
                       size=len(x))
        coords = np.column_stack([x, y])
    else:
        coords = emb[:, :2]

    plt.figure(figsize=(6, 6))
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(coords[idx, 0], coords[idx, 1], label=lab, s=14)
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
    plt.figure(figsize=(9, 5))
    cmap = get_cmap(cmap_name)
    label_colors: Dict[str, str] = {
        lab: cmap(i % cmap.N) for i, lab in enumerate(sorted(set(labels)))
    }
    for vec, n, lab in zip(vectors, lengths, labels):
        plt.plot(range(n), vec[:n], color=label_colors[lab], alpha=alpha, lw=0.8)

    for lab, col in label_colors.items():      # legend
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

    resampled_vectors = prepare_resampled_vectors(contours, TARGET_LEN)
    print(f"Prepared {len(resampled_vectors)} resampled vectors "
          f"| shape = {resampled_vectors.shape}")

    emb_1d, method = project_vectors(resampled_vectors, labels)
    plot_embedding(emb_1d, labels, method, EMBEDDING_PLOT)

    pca_emb = PCA(n_components=2).fit_transform(resampled_vectors)
    plot_embedding(pca_emb, labels, "PCA (2-D)", PCA_PLOT)

    padded_vectors, lengths = prepare_vectors_for_overlay(contours)
    plot_overlay(padded_vectors, lengths, labels, OVERLAY_PLOT)


if __name__ == "__main__":
    main()
