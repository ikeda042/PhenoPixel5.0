#!/usr/bin/env python3
"""Train a classifier to separate cell contours labelled as 'N/A' and '1'.
This script uses a Support Vector Machine (SVM) instead of LDA and reports
cross‑validated accuracy. The database format is the same as used in
``experimental/autolabel/main.py``.
"""

import pickle
import sqlite3
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Configuration ---------------------------------------------------------------
DB_PATH: str = "experimental/autolabel/250626_SK450_Gen_1p0-completed.db"
TARGET_LEN: int = 256
RNG_SEED: int = 42

# Data loading ----------------------------------------------------------------
def fetch_contours(db_path: str, label: Optional[str] = None) -> Tuple[List[np.ndarray], List[str]]:
    """Fetch contours and labels from SQLite DB."""
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

# Feature preparation ---------------------------------------------------------
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

def prepare_vectors(contours: List[np.ndarray], target_len: int = TARGET_LEN) -> np.ndarray:
    raw_vecs = [contour_to_vector(c) for c in contours]
    return np.vstack([resample_vector(v, target_len) for v in raw_vecs])

# Training & evaluation -------------------------------------------------------
def train_svm_classifier(db_path: str = DB_PATH) -> SVC:
    """Train an SVM classifier and print cross‑validated accuracy."""
    contours, labels = fetch_contours(db_path)
    if not contours:
        print("No data found in DB.")
        return None

    X = prepare_vectors(contours)
    y = np.array(labels)

    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", gamma="scale", random_state=RNG_SEED)
    )

    scores = cross_val_score(svm, X, y, cv=5)
    print(f"Cross-validated accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    svm.fit(X, y)
    return svm

# Example ---------------------------------------------------------------------
if __name__ == "__main__":
    train_svm_classifier()
