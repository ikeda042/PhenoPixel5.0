#!/usr/bin/env python3
"""
Visualize cell contours *and* persist the 1-D LDA mapping.

Produces four artefacts:
1. lda_result.png      – 1-D scatter (x = LDA, y = jitter)
2. pca_result.png      – 2-D PCA scatter
3. vectors_overlay.png – Overlay of true radial-distance vectors
4. lda_model.joblib    – Trained `LinearDiscriminantAnalysis` object
                         (use `map_contours()` to project new contours)

NumPy ≥ 2.0 / scikit-learn ≥ 1.5 compatible
"""

import os
import pickle
import sqlite3
from typing import Dict, List, Optional, Tuple

import joblib                        # ← NEW
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
LDA_MODEL_FILE = "lda_model.joblib"       # ← NEW
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
) -> Tuple[np.ndarray, LinearDiscriminantAnalysis, str]:
    """
    Fit 1-D LDA (or PCA fallback) and return:
      embedding (N,1), fitted model, description string
    """
    classes = np.unique(labels)
    if len(classes) >= 2:
        lda = LinearDiscriminantAnalysis(
            n_components=1, solver="eigen", shrinkage="auto"
        )
        emb = lda.fit_transform(vectors, labels)
        method = "LDA (1-D, shrinkage)"
        return emb, lda, method
    else:
        pca = PCA(n_components=1)
        emb = pca.fit_transform(vectors)
        return emb, None, "PCA (1-D fallback)"


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_embedding(
    emb: np.ndarray, labels: List[str], title: str, out_path: str
) -> None:
    emb = emb.ravel() if emb.ndim == 2 and emb.shape[1] == 1 else emb
    if emb.ndim == 1:  # 1-D → jitter
        x = emb
        rng = np.random.default_rng(RNG_SEED)
        y = rng.normal(0.0,
                       JITTER_SCALE * (np.ptp(x) if np.ptp(x) > 0 else 1.0),
                       size=len(x))
        coords = np.column_stack([x, y])
    else:              # 2-D
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


# ─── Persisted-model helper ──────────────────────────────────────────────────
def map_contours(
    contours: List[np.ndarray],
    model_path: str = LDA_MODEL_FILE,
    target_len: int = TARGET_LEN,
) -> np.ndarray:
    """
    Project *new* contours into the saved 1-D LDA space.

    Returns an array of shape (N, 1).
    """
    lda: LinearDiscriminantAnalysis = joblib.load(model_path)
    vecs = prepare_resampled_vectors(contours, target_len)
    return lda.transform(vecs)


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    contours, labels = fetch_contours(DB_PATH, LABEL_FILTER)
    if not contours:
        print("No data found in DB.")
        return

    resampled_vectors = prepare_resampled_vectors(contours, TARGET_LEN)
    print(f"Prepared {len(resampled_vectors)} resampled vectors "
          f"| shape = {resampled_vectors.shape}")

    # 1) 1-D LDA (or PCA fallback)
    emb_1d, lda_model, method = project_vectors(resampled_vectors, labels)
    plot_embedding(emb_1d, labels, method, EMBEDDING_PLOT)

    # Save LDA model if it exists
    if lda_model is not None:
        joblib.dump(lda_model, LDA_MODEL_FILE)
        print(f"Saved LDA model     → {LDA_MODEL_FILE}")

    # 2) 2-D PCA
    pca_emb = PCA(n_components=2).fit_transform(resampled_vectors)
    plot_embedding(pca_emb, labels, "PCA (2-D)", PCA_PLOT)

    # 3) Overlay plot
    padded_vectors, lengths = prepare_vectors_for_overlay(contours)
    plot_overlay(padded_vectors, lengths, labels, OVERLAY_PLOT)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
#  Example: mapping an unseen contour later on
# ---------------------------------------------------------------------------
# from autolabel.main import map_contours
# new_emb = map_contours([new_contour_ndarray])   # ← returns (1,1) array
# print("Projected coordinate:", new_emb[0, 0])
