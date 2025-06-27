#!/usr/bin/env python3
"""
Cell-contour classifier (SVM) – “1”  と “N/A” を 2 クラスとして扱う版
------------------------------------------------------------------------
* manual_label = "1"（整数 1 が入ってくる場合もある）と "N/A" を
  そのままクラスラベルとして学習します。
* manual_label が None / 空文字 は欠損とみなし除外します。
* 特徴量は
    - 距離プロファイル 256 次元
    - Hu モーメント        7 次元
  の計 263 次元。
* SVM は class_weight="balanced"。
"""

# ───────── Imports ─────────
import os
import pickle
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy import BLOB, FLOAT, Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ───────── Config ─────────
DB_PATH   = os.path.join(
    os.path.dirname(__file__), "..", "..", "backend", "app", "databases",
    "test_database.db"
)
MODEL_PATH = "svm_cell_classifier.pkl"
TARGET_LEN = 256
RNG_SEED   = 42

# ───────── SQLAlchemy ─────────
Base = declarative_base()

class Cell(Base):
    __tablename__ = "cells"
    id           = Column(Integer, primary_key=True)
    manual_label = Column(String)   # ← 文字列として扱う
    contour      = Column(BLOB)

# ───────── Helper functions ─────────
def fetch_contours(
    db_path: str = DB_PATH,
    label: Optional[str] = None
) -> Tuple[List[np.ndarray], List[str]]:
    """
    SQLite から輪郭とラベルを取得する。

    Parameters
    ----------
    label : str | None
        "1" または "N/A" を指定するとそのクラスだけ取得。
        None のときは両方取得。
    """
    abs_path = os.path.abspath(db_path)
    engine   = create_engine(f"sqlite:///{abs_path}")
    Session  = sessionmaker(bind=engine)

    with Session() as session:
        q = session.query(Cell.contour, Cell.manual_label)
        if label is not None:
            q = q.filter(Cell.manual_label == label)
        rows = q.all()

    contours, labels = [], []
    for contour_blob, lab in rows:
        # ----- 欠損判定 -----
        if lab is None or str(lab).strip() == "":
            continue  # 欠損はスキップ
        # save as float32 array (N,2)
        cnt = np.asarray(pickle.loads(contour_blob)).reshape(-1, 2).astype(np.float32)
        contours.append(cnt)
        labels.append(str(lab).strip())   # "1" も "N/A" も文字列
    return contours, labels


def _resample(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) == n:
        return arr
    old = np.linspace(0.0, 1.0, len(arr))
    new = np.linspace(0.0, 1.0, n)
    return np.interp(new, old, arr)


def contour_to_features(cnt: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """
    1 輪郭 → 263 次元特徴
    """
    center   = cnt.mean(axis=0)
    dist_vec = np.linalg.norm(cnt - center, axis=1)
    profile  = _resample(np.sort(dist_vec), target_len)         # 256
    hu       = cv2.HuMoments(cv2.moments(cnt)).flatten()        # 7
    return np.hstack([profile, hu])                             # 263


def prepare_features(contours: List[np.ndarray], target_len: int) -> np.ndarray:
    return np.vstack([contour_to_features(c, target_len) for c in contours])


# ───────── Training ─────────
def train_svm_classifier(
    db_path: str = DB_PATH,
    model_path: str = MODEL_PATH,
    target_len: int = TARGET_LEN,
    rng_seed: int = RNG_SEED,
) -> SVC | None:
    """
    DB を読み込み，SVM を学習・保存して返す。
    """
    contours, labels = fetch_contours(db_path)
    if not contours:
        print("◆ エラー: 学習可能なデータがありません。")
        return None

    X = prepare_features(contours, target_len)
    y = np.array(labels)

    # クラス確認
    classes, counts = np.unique(y, return_counts=True)
    print("▼ Label distribution  :", dict(zip(classes, counts)))
    if len(classes) < 2:
        raise RuntimeError("“1” と “N/A” の両クラスを含むデータを用意してください。")

    min_cnt = counts.min()
    cv_splits = min(5, max(2, min_cnt))   # 最低 2 fold

    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", gamma="scale",
            class_weight="balanced", random_state=rng_seed)
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=rng_seed)
    scores = cross_val_score(svm, X, y, cv=cv)
    print(f"▼ CV accuracy ({cv_splits}-fold): {scores.mean():.3f} ± {scores.std():.3f}")

    svm.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(svm, f)
    print(f"● Model saved → {os.path.abspath(model_path)}")
    return svm


# ───────── Load / Predict ─────────
def load_svm_classifier(model_path: str = MODEL_PATH) -> SVC:
    if not os.path.isfile(model_path):
        raise FileNotFoundError("モデルがありません。train_svm_classifier() を先に実行してください。")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_contour(
    contour: np.ndarray,
    model: SVC,
    target_len: int = TARGET_LEN,
    return_proba: bool = False,
):
    """1 輪郭を分類する"""
    vec = contour_to_features(contour, target_len).reshape(1, -1)
    label = model.predict(vec)[0]
    if return_proba:
        score = model.decision_function(vec)[0]
        return label, float(score)
    return label


# ───────── CLI demo ─────────
if __name__ == "__main__":
    # 学習してモデルを保存
    model = train_svm_classifier()
    # 例: 予測テスト
    # model = load_svm_classifier()
    # dummy = np.random.rand(180, 2).astype(np.float32) * 20
    # print(predict_contour(dummy, model, return_proba=True))
