import os
import pickle
import sqlite3
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ─── 設定 ──────────────────────────────────────────────────────────────
DB_PATH: str = "experimental/autolabel/250626_SK450_Gen_1p0-completed.db" 
LABEL_FILTER: Optional[str] = None   # 例: "1" や "N/A"。フィルタしないなら None
OUTPUT_FILE: str = "lda_result.png"
# ───────────────────────────────────────────────────────────────────────


def fetch_contours(db_path: str, label: Optional[str] = None) -> Tuple[List[np.ndarray], List[str]]:
    """Fetch contours and labels from the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if label is None:
        cur.execute("SELECT contour, manual_label FROM cells")
    else:
        cur.execute(
            "SELECT contour, manual_label FROM cells WHERE manual_label = ?",
            (label,),
        )
    rows = cur.fetchall()
    conn.close()

    contours: List[np.ndarray] = []
    labels: List[str] = []
    for contour_blob, manual_label in rows:
        contour = pickle.loads(contour_blob)
        contour = np.asarray(contour).reshape(-1, 2)
        contours.append(contour)
        labels.append(str(manual_label))
    return contours, labels


def contour_to_vector(contour: np.ndarray) -> np.ndarray:
    center = contour.mean(axis=0)
    distances = np.linalg.norm(contour - center, axis=1)
    return np.sort(distances)


def prepare_vectors(contours: List[np.ndarray]) -> np.ndarray:
    vectors = [contour_to_vector(c) for c in contours]
    max_len = max(len(v) for v in vectors)
    padded = [np.pad(v, (0, max_len - len(v))) for v in vectors]
    return np.vstack(padded)


def plot_lda(vectors: np.ndarray, labels: List[str], output: str) -> None:
    lda = LinearDiscriminantAnalysis(n_components=2)
    transformed = lda.fit_transform(vectors, labels)

    plt.figure(figsize=(6, 6))
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(transformed[idx, 0], transformed[idx, 1], label=lab)
    plt.legend()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=300)


def main() -> None:
    # 設定に従って処理を実行
    contours, labels = fetch_contours(DB_PATH, LABEL_FILTER)
    if not contours:
        print("No data found.")
        return

    vectors = prepare_vectors(contours)
    print(f"Prepared {len(vectors)} vectors with shape {vectors.shape}")
    for i in range(len(vectors)):
        print(f"Vector {i}: {vectors[i]} (Label: {labels[i]})")
    plot_lda(vectors, labels, OUTPUT_FILE)
    print(f"Saved figure to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
