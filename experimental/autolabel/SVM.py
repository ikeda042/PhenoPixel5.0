#!/usr/bin/env python3
"""
Train and persist an SVM-based classifier to separate cell contours
labelled as “N/A” and “1”, then reuse it for unseen samples.

➡  Usage examples
-----------------
# (1) Train and save the model
$ python cell_svm.py --train

# (2) Predict an unknown contour from inside another script
from cell_svm import load_svm_classifier, predict_contour, TARGET_LEN
model = load_svm_classifier()          # loads default 'svm_cell_classifier.pkl'
label = predict_contour(unknown_cnt, model, target_len=TARGET_LEN)
print(label)
"""

# ─────────────────────────── Imports ────────────────────────────
import argparse
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy import BLOB, FLOAT, Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ────────────────────────── Configuration ───────────────────────
DB_PATH: str = "experimental/autolabel/250626_SK450_Gen_1p0-completed.db"
MODEL_PATH: str = "svm_cell_classifier.pkl"          # ← saved model file
TARGET_LEN: int = 256
RNG_SEED: int = 42

# ──────────────────────── SQLAlchemy setup ──────────────────────
Base = declarative_base()


class Cell(Base):
    """ORM mapping for the `cells` table."""
    __tablename__ = "cells"

    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    area = Column(FLOAT)
    perimeter = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB, nullable=True)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)
    user_id = Column(String, nullable=True)


# ──────────────────────────── Helpers ───────────────────────────
def fetch_contours(
    db_path: str, label: Optional[str] = None
) -> Tuple[List[np.ndarray], List[str]]:
    """Fetch contours and labels from the DB."""
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    with Session() as session:
        query = session.query(Cell.contour, Cell.manual_label)
        if label is not None:
            query = query.filter(Cell.manual_label == label)
        rows = query.all()

    contours, labels = [], []
    for contour_blob, manual_label in rows:
        contour = np.asarray(pickle.loads(contour_blob)).reshape(-1, 2)
        contours.append(contour)
        labels.append(str(manual_label))
    return contours, labels


# -------- contour → fixed-length vector -------------------------
def contour_to_vector(contour: np.ndarray) -> np.ndarray:
    """Convert a 2-D contour (N×2) to ordered radial-distance vector."""
    center = contour.mean(axis=0)
    distances = np.linalg.norm(contour - center, axis=1)
    return np.sort(distances)


def resample_vector(vec: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """Linear-resample a 1-D vector to `target_len` samples."""
    if len(vec) == target_len:
        return vec
    old_x = np.linspace(0.0, 1.0, len(vec))
    new_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(new_x, old_x, vec)


def prepare_vectors(contours: List[np.ndarray], target_len: int = TARGET_LEN) -> np.ndarray:
    """Create a design matrix (num_samples × target_len)."""
    raw_vecs = [contour_to_vector(c) for c in contours]
    return np.vstack([resample_vector(v, target_len) for v in raw_vecs])


# ──────────────────────── Train & persist ───────────────────────
def train_svm_classifier(
    db_path: str = DB_PATH,
    model_path: str = MODEL_PATH,
    target_len: int = TARGET_LEN,
) -> Optional[SVC]:
    """Train an SVM classifier, report CV accuracy, and save the model."""
    contours, labels = fetch_contours(db_path)
    if not contours:
        print("No data found in DB.")
        return None

    X = prepare_vectors(contours, target_len)
    y = np.array(labels)

    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", gamma="scale", random_state=RNG_SEED),
    )

    scores = cross_val_score(svm, X, y, cv=5)
    print(f"Cross-validated accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    svm.fit(X, y)

    # ── Save the fitted pipeline ──
    with open(model_path, "wb") as f:
        pickle.dump(svm, f)
    print(f"Model saved to: {os.path.abspath(model_path)}")

    return svm


# ────────────────────────── Re-use API ──────────────────────────
def load_svm_classifier(model_path: str = MODEL_PATH) -> SVC:
    """Load a previously saved SVM pipeline."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Train it first with --train."
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_contour(
    contour: np.ndarray,
    model: SVC,
    target_len: int = TARGET_LEN,
    return_proba: bool = False,
) -> str:
    """
    Predict the label ('N/A' or '1') for a single contour.

    Parameters
    ----------
    contour : np.ndarray
        Raw contour (N×2) from your image-processing pipeline.
    model   : SVC
        The fitted pipeline from `load_svm_classifier`.
    return_proba : bool
        If True, also return the decision function value.

    Returns
    -------
    str  (or Tuple[str, float] if return_proba is True)
    """
    vec = prepare_vectors([contour], target_len)[0].reshape(1, -1)
    label = model.predict(vec)[0]
    if return_proba:
        score = model.decision_function(vec)[0]
        return label, float(score)
    return label


# ─────────────────────────── CLI entry ──────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or query the SVM cell classifier.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train",
        action="store_true",
        help="(Re)train the model and save it to disk.",
    )
    group.add_argument(
        "--predict",
        metavar="NPY_FILE",
        help="Path to .npy file containing an unknown contour (N×2) to classify.",
    )
    parser.add_argument("--db", default=DB_PATH, help="SQLite DB path.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to save/load the model.")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()

    if args.train:
        train_svm_classifier(db_path=args.db, model_path=args.model)
    else:  # predict mode
        contour = np.load(args.predict)
        model = load_svm_classifier(args.model)
        label, score = predict_contour(contour, model, return_proba=True)
        print(f"Predicted label: {label}   (decision score: {score:.4f})")


if __name__ == "__main__":
    _main()
