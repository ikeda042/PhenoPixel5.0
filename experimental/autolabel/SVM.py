#!/usr/bin/env python3
"""
Cell-contour classifier (SVM)

This module:

1. Trains an SVM pipeline on contours stored in an SQLite database.
2. Saves the fitted model to disk so it can be reused later.
3. Exposes helpers to load the model and predict the label of an
   unknown contour.

Typical usage
-------------
# ——————————————————————————————————————————
# (A) Train once (creates 'svm_cell_classifier.pkl')
# ——————————————————————————————————————————
import cell_svm
cell_svm.train_svm_classifier()      # prints CV accuracy & save location

# ——————————————————————————————————————————
# (B) Anywhere else in your codebase
# ——————————————————————————————————————————
from cell_svm import load_svm_classifier, predict_contour
model = load_svm_classifier()        # loads the saved pipeline
label, score = predict_contour(contour_ndarray, model, return_proba=True)
print(label, score)
"""

# ─────────────────────────── Imports ────────────────────────────
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
DB_PATH: str = os.path.join(
    os.path.dirname(__file__), "..", "..", "backend", "app", "databases", "test_database.db"
)
MODEL_PATH: str = "svm_cell_classifier.pkl"          # file created by train
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
    db_path: str = DB_PATH, label: Optional[str] = None
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Retrieve contours (as numpy arrays) and labels from the SQLite DB.
    If *label* is given, only that class is fetched.
    """
    abs_path = os.path.abspath(db_path)
    engine = create_engine(f"sqlite:///{abs_path}")
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


def contour_to_vector(contour: np.ndarray) -> np.ndarray:
    """Convert an N×2 contour to an ordered vector of radial distances."""
    center = contour.mean(axis=0)
    distances = np.linalg.norm(contour - center, axis=1)
    return np.sort(distances)  # sort for rotation invariance


def resample_vector(vec: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """Linearly resample *vec* to *target_len* points."""
    if len(vec) == target_len:
        return vec
    old_x = np.linspace(0.0, 1.0, len(vec))
    new_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(new_x, old_x, vec)


def prepare_vectors(
    contours: List[np.ndarray], target_len: int = TARGET_LEN
) -> np.ndarray:
    """Stack resampled contour vectors into a 2-D design matrix."""
    raw_vecs = [contour_to_vector(c) for c in contours]
    return np.vstack([resample_vector(v, target_len) for v in raw_vecs])


# ─────────────────────── Train & persist model ──────────────────
def train_svm_classifier(
    db_path: str = DB_PATH,
    model_path: str = MODEL_PATH,
    target_len: int = TARGET_LEN,
    rng_seed: int = RNG_SEED,
) -> Optional[SVC]:
    """
    Train an SVM classifier, report CV accuracy, and pickle to *model_path*.

    Returns
    -------
    sklearn.svm.SVC
        The fitted Pipeline (StandardScaler → SVC) or None if no data.
    """
    contours, labels = fetch_contours(db_path)
    if not contours:
        print("No data found in DB.")
        return None

    X = prepare_vectors(contours, target_len)
    y = np.array(labels)

    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", gamma="scale", random_state=rng_seed),
    )

    scores = cross_val_score(svm, X, y, cv=5)
    print(f"Cross-validated accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    svm.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(svm, f)
    print(f"Model saved → {os.path.abspath(model_path)}")

    return svm


# ─────────────────────────── Re-use helpers ─────────────────────
def load_svm_classifier(model_path: str = MODEL_PATH) -> SVC:
    """
    Load and return a previously saved classifier.

    Raises
    ------
    FileNotFoundError
        If the pickle file does not exist.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Run train_svm_classifier() first."
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_contour(
    contour: np.ndarray,
    model: SVC,
    target_len: int = TARGET_LEN,
    return_proba: bool = False,
) -> Tuple[str, float] | str:
    """
    Classify a single contour.

    Parameters
    ----------
    contour : np.ndarray (N×2)
        Raw contour points.
    model   : SVC
        Trained pipeline from load_svm_classifier().
    return_proba : bool, default=False
        If True, also return the decision function score.

    Returns
    -------
    str
        Predicted class label.
    float  (optional)
        Decision function value (signed distance to the separating hyper-plane).
    """
    vec = prepare_vectors([contour], target_len)[0].reshape(1, -1)
    label = model.predict(vec)[0]
    if return_proba:
        score = model.decision_function(vec)[0]
        return label, float(score)
    return label


# ──────────────────────── Demo workflow (optional) ──────────────
if __name__ == "__main__":
    # 1. Train the model and save it
    model = train_svm_classifier()

    # 2. Example of loading the model later
    model = load_svm_classifier()

    # 3. Dummy prediction example (replace with real contour)
    # unknown_contour = np.random.rand(150, 2) * 10  # Fake contour for demo
    # label, score = predict_contour(unknown_contour, model, return_proba=True)
    # print(f"Predicted label: {label}   (decision score: {score:.4f})")

    print("\nDemo complete. Uncomment the lines above to test prediction.")
