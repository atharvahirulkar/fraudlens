"""
Feature engineering pipeline for IEEE-CIS Fraud Detection.

Run as a script:
    python -m src.train.features

Outputs:
    data/processed/train.parquet   — feature matrix + isFraud target
    data/processed/pipeline.pkl    — fitted pipeline for inference
"""

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# TransactionDT is seconds elapsed since this date (confirmed by public IEEE-CIS EDA)
_DT_ORIGIN = pd.Timestamp("2017-11-30")

# - Feature column groups

_TARGET_ENCODE_COLS = [
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
]

_LABEL_ENCODE_COLS = [
    "ProductCD", "card4", "card6",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType",
]

_C_COLS = [f"C{i}" for i in range(1, 15)]
_D_COLS = [f"D{i}" for i in range(1, 16)]

# Curated V-feature subset — all 339 Vesta features exist, but only the ranges
# below carry meaningful signal after collinearity pruning in public benchmarks
_V_RANGES = [(1, 12), (12, 35), (35, 53), (53, 75), (75, 95), (95, 138), (279, 322)]
_V_COLS = [f"V{i}" for lo, hi in _V_RANGES for i in range(lo, hi)]

_ID_NUMERIC_COLS = [f"id_{i:02d}" for i in range(1, 12)]
_ID_CATEGORICAL_COLS = [f"id_{i:02d}" for i in range(12, 39)] + ["DeviceType"]


# - Pipeline state (survives pickling, used at inference time)

@dataclass
class PipelineState:
    global_amt_mean: float = 0.0
    global_amt_std: float = 1.0
    card1_agg: pd.DataFrame = field(default_factory=pd.DataFrame)
    uid_agg: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_encode_maps: dict = field(default_factory=dict)   # col → pd.Series
    label_encode_maps: dict = field(default_factory=dict)    # col → dict[str, int]
    numeric_medians: pd.Series = field(default_factory=pd.Series)
    feature_cols: list = field(default_factory=list)


# - Pure-function transformations

def _load_raw(split: str) -> pd.DataFrame:
    txn = pd.read_csv(RAW_DIR / f"{split}_transaction.csv")
    identity_path = RAW_DIR / f"{split}_identity.csv"
    if identity_path.exists():
        idn = pd.read_csv(identity_path)
        return txn.merge(idn, on="TransactionID", how="left")
    return txn


def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    dt = _DT_ORIGIN + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["day_of_week"] = dt.dt.dayofweek.astype(np.int8)
    df["day_of_month"] = dt.dt.day.astype(np.int8)
    df["month"] = dt.dt.month.astype(np.int8)
    return df


def _add_email_features(df: pd.DataFrame) -> pd.DataFrame:
    df["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(np.int8)

    _anon = {"anonymous.com", "protonmail.com", "guerrillamail.com", "mailnull.com"}
    df["r_email_anon"] = df["R_emaildomain"].isin(_anon).astype(np.int8)
    df["p_email_anon"] = df["P_emaildomain"].isin(_anon).astype(np.int8)

    # Root domain collapses provider variants (gmail.com / googlemail.com → gmail)
    df["p_email_root"] = df["P_emaildomain"].str.split(".").str[0].fillna("missing")
    df["r_email_root"] = df["R_emaildomain"].str.split(".").str[0].fillna("missing")

    return df


def _add_card_aggregates_fit(df: pd.DataFrame, state: PipelineState) -> pd.DataFrame:
    """Compute and store card-level aggregates from training data."""
    df["uid"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)

    state.global_amt_mean = df["TransactionAmt"].mean()
    state.global_amt_std = df["TransactionAmt"].std()

    state.card1_agg = (
        df.groupby("card1")["TransactionAmt"]
        .agg(card1_amt_mean="mean", card1_amt_std="std", card1_txn_count="count")
        .reset_index()
    )
    state.uid_agg = (
        df.groupby("uid")["TransactionAmt"]
        .agg(uid_amt_mean="mean", uid_amt_std="std", uid_txn_count="count")
        .reset_index()
    )

    return _apply_card_aggregates(df, state)


def _apply_card_aggregates(df: pd.DataFrame, state: PipelineState) -> pd.DataFrame:
    """Join pre-computed card aggregates; fall back to global stats for unseen cards."""
    if "uid" not in df.columns:
        df["uid"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)

    df = df.merge(state.card1_agg, on="card1", how="left")
    df = df.merge(state.uid_agg, on="uid", how="left")

    df["card1_amt_mean"].fillna(state.global_amt_mean, inplace=True)
    df["card1_amt_std"].fillna(state.global_amt_std, inplace=True)
    df["card1_txn_count"].fillna(1, inplace=True)
    df["uid_amt_mean"].fillna(state.global_amt_mean, inplace=True)
    df["uid_amt_std"].fillna(state.global_amt_std, inplace=True)
    df["uid_txn_count"].fillna(1, inplace=True)

    # How much does this transaction deviate from the card's normal spend?
    df["amt_zscore_card1"] = (
        (df["TransactionAmt"] - df["card1_amt_mean"]) / df["card1_amt_std"].clip(lower=1e-6)
    )
    df["amt_zscore_uid"] = (
        (df["TransactionAmt"] - df["uid_amt_mean"]) / df["uid_amt_std"].clip(lower=1e-6)
    )

    return df


# - Target encoding (smoothed, cross-validated to avoid leakage)

def _smooth_encode(
    fold_train: pd.DataFrame,
    col: str,
    target: str,
    global_mean: float,
    smoothing: int = 10,
) -> pd.Series:
    """Return a smoothed mean-target map fitted on fold_train."""
    agg = fold_train.groupby(col)[target].agg(["mean", "count"])
    smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (
        agg["count"] + smoothing
    )
    return smooth  # index = category value, values = smoothed fraud rate


def _target_encode_fit(
    df: pd.DataFrame,
    cols: list,
    target: str,
    state: PipelineState,
    n_splits: int = 5,
    smoothing: int = 10,
) -> pd.DataFrame:
    """
    Out-of-fold target encoding for training data.
    Stores final maps (fitted on full training data) in state for inference.
    """
    global_mean = df[target].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in cols:
        if col not in df.columns:
            continue
        encoded = np.full(len(df), global_mean)
        for train_idx, val_idx in kf.split(df):
            smooth_map = _smooth_encode(
                df.iloc[train_idx], col, target, global_mean, smoothing
            )
            encoded[val_idx] = df[col].iloc[val_idx].map(smooth_map).fillna(global_mean)

        df[f"{col}_te"] = encoded

        # Full-data map saved for inference
        state.target_encode_maps[col] = _smooth_encode(
            df, col, target, global_mean, smoothing
        )

    return df


def _target_encode_transform(df: pd.DataFrame, state: PipelineState) -> pd.DataFrame:
    for col, smooth_map in state.target_encode_maps.items():
        if col not in df.columns:
            continue
        global_mean = smooth_map.mean()
        df[f"{col}_te"] = df[col].map(smooth_map).fillna(global_mean)
    return df


# - Label encoding

def _label_encode_fit(
    df: pd.DataFrame, cols: list, state: PipelineState
) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("missing")
        mapping = {v: i for i, v in enumerate(sorted(df[col].unique()))}
        state.label_encode_maps[col] = mapping
        df[col] = df[col].map(mapping).fillna(-1).astype(np.int16)
    return df


def _label_encode_transform(df: pd.DataFrame, state: PipelineState) -> pd.DataFrame:
    for col, mapping in state.label_encode_maps.items():
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("missing").map(mapping).fillna(-1).astype(np.int16)
    return df


# - M-feature encoding

def _encode_m_features(df: pd.DataFrame) -> pd.DataFrame:
    """M1–M9 are boolean match flags ('T'/'F'/NaN) → 1/0/-1."""
    m_cols = [c for c in [f"M{i}" for i in range(1, 10)] if c in df.columns]
    for col in m_cols:
        df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype(np.int8)
    return df


# - Imputation

def _impute_fit(df: pd.DataFrame, state: PipelineState) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    state.numeric_medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(state.numeric_medians)

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df[object_cols] = df[object_cols].fillna("missing")

    return df


def _impute_transform(df: pd.DataFrame, state: PipelineState) -> pd.DataFrame:
    numeric_cols = [c for c in state.numeric_medians.index if c in df.columns]
    df[numeric_cols] = df[numeric_cols].fillna(state.numeric_medians[numeric_cols])

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df[object_cols] = df[object_cols].fillna("missing")

    return df


# - Column selection

def _select_features(df: pd.DataFrame, include_target: bool = True) -> list:
    base = [
        "TransactionAmt",
        "hour", "day_of_week", "day_of_month", "month",
        "email_match", "r_email_anon", "p_email_anon",
        "card1_amt_mean", "card1_amt_std", "card1_txn_count",
        "uid_amt_mean", "uid_amt_std", "uid_txn_count",
        "amt_zscore_card1", "amt_zscore_uid",
        "dist1", "dist2",
    ]
    te_cols = [f"{c}_te" for c in _TARGET_ENCODE_COLS]
    label_cols = _LABEL_ENCODE_COLS
    v_cols = [c for c in _V_COLS if c in df.columns]
    c_cols = [c for c in _C_COLS if c in df.columns]
    d_cols = [c for c in _D_COLS if c in df.columns]
    id_numeric = [c for c in _ID_NUMERIC_COLS if c in df.columns]

    feature_cols = (
        [c for c in base if c in df.columns]
        + [c for c in te_cols if c in df.columns]
        + [c for c in label_cols if c in df.columns]
        + v_cols + c_cols + d_cols + id_numeric
    )

    if include_target and "isFraud" in df.columns:
        feature_cols = ["isFraud"] + feature_cols

    return feature_cols


# - Public API

class FeaturePipeline:
    """Fit on training data; transform training and inference data consistently."""

    def __init__(self, n_te_splits: int = 5, te_smoothing: int = 10):
        self.n_te_splits = n_te_splits
        self.te_smoothing = te_smoothing
        self.state = PipelineState()

    def fit_transform(self, df: pd.DataFrame, target: str = "isFraud") -> pd.DataFrame:
        df = df.copy()
        df = _add_temporal(df)
        df = _add_email_features(df)
        df = _encode_m_features(df)
        df = _add_card_aggregates_fit(df, self.state)
        df = _target_encode_fit(
            df, _TARGET_ENCODE_COLS, target, self.state,
            self.n_te_splits, self.te_smoothing
        )
        df = _label_encode_fit(df, _LABEL_ENCODE_COLS + ["p_email_root", "r_email_root"], self.state)
        df = _impute_fit(df, self.state)

        feature_cols = _select_features(df, include_target=(target in df.columns))
        self.state.feature_cols = [c for c in feature_cols if c != target]
        return df[feature_cols]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = _add_temporal(df)
        df = _add_email_features(df)
        df = _encode_m_features(df)
        df = _apply_card_aggregates(df, self.state)
        df = _target_encode_transform(df, self.state)
        df = _label_encode_transform(df, self.state)
        df = _impute_transform(df, self.state)

        available = [c for c in self.state.feature_cols if c in df.columns]
        return df[available]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "FeaturePipeline":
        with open(path, "rb") as f:
            return pickle.load(f)


# - Script entry point

def build(split: str = "train") -> pd.DataFrame:
    """Load raw data, apply pipeline, write parquet. Returns processed DataFrame."""
    print(f"Loading {split} data from {RAW_DIR}...")
    df = _load_raw(split)
    print(f"  Raw shape: {df.shape}")

    pipeline = FeaturePipeline()

    if split == "train":
        processed = pipeline.fit_transform(df, target="isFraud")
        out_path = PROCESSED_DIR / "train.parquet"
        pipeline_path = PROCESSED_DIR / "pipeline.pkl"

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        processed.to_parquet(out_path, index=False)
        pipeline.save(pipeline_path)

        print(f"  Processed shape: {processed.shape}")
        print(f"  Saved → {out_path}")
        print(f"  Pipeline → {pipeline_path}")
        return processed
    else:
        # Load existing pipeline so test data is encoded identically to train
        pipeline = FeaturePipeline.load(PROCESSED_DIR / "pipeline.pkl")
        processed = pipeline.transform(df)
        out_path = PROCESSED_DIR / f"{split}.parquet"
        processed.to_parquet(out_path, index=False)
        print(f"  Processed shape: {processed.shape}")
        print(f"  Saved → {out_path}")
        return processed


if __name__ == "__main__":
    build("train")
