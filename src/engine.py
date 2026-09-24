"""
ML Agent engine.

Builds real scikit-learn pipelines from a JSON config, trains every model with
train / validation / test splits and cross-validation, tunes hyperparameters,
and produces the analysis the dashboard shows. It runs unchanged in the browser
(Pyodide, inside a Web Worker) and in normal CPython, which is how the exported
project (train.py / serve.py) reuses it.

Every public function takes and returns JSON strings so the JavaScript side
never has to deal with Python objects.
"""
import ast
import base64
import contextlib
import csv
import functools
import importlib
import io
import json
import math
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn import metrics as M  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.decomposition import PCA, FastICA, KernelPCA, TruncatedSVD  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, IsolationForest,
    StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor,
)
from sklearn.experimental import enable_halving_search_cv  # noqa: E402,F401
from sklearn.experimental import enable_iterative_imputer  # noqa: E402,F401
from sklearn.feature_selection import (  # noqa: E402
    RFE, RFECV, SelectFromModel, SelectKBest, SelectPercentile, SequentialFeatureSelector,
    VarianceThreshold, chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import Lasso, LogisticRegression, Ridge, RidgeCV  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    cross_val_score, GridSearchCV, GroupKFold, HalvingGridSearchCV, HalvingRandomSearchCV, KFold, RandomizedSearchCV,
    RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit,
    TimeSeriesSplit, cross_validate, learning_curve, train_test_split,
)
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    KBinsDiscretizer, MaxAbsScaler, MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, PowerTransformer, QuantileTransformer, RobustScaler, SplineTransformer,
    StandardScaler,
)
from sklearn.svm import LinearSVC  # noqa: E402
from sklearn.utils.metaestimators import available_if  # noqa: E402

try:
    from sklearn.preprocessing import TargetEncoder
except ImportError:  # scikit-learn < 1.3
    TargetEncoder = None

IN_BROWSER = sys.platform == "emscripten"
N_JOBS = 1
S = {}            # engine state: data, splits, fitted models, results
PROGRESS = None   # callable(json_str) set by the host


# ---------------------------------------------------------------- utilities
def clean(o):
    """Make any result JSON-safe (numpy types, NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if o is None or isinstance(o, (str, int)):
        return o
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(o)
    return str(o)


def dumps(o):
    return json.dumps(clean(o))


def set_progress(fn):
    global PROGRESS
    PROGRESS = fn


def progress(stage, message, i=None, n=None, **extra):
    msg = {"stage": stage, "message": message, "i": i, "n": n, **extra}
    if PROGRESS is not None:
        try:
            PROGRESS(dumps(msg))
        except Exception:
            pass


def log(message, level="info"):
    S.setdefault("log", []).append({"t": time.time(), "level": level, "message": message})
    progress("log", message, level=level)


def safe(fn):
    """Wrap a public function: always return JSON, never raise into JS."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            return dumps({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]})
    return wrapper


def sample_idx(n, k, seed=0):
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.RandomState(seed).choice(n, k, replace=False))


def downsample_curve(x, y, extra=None, k=150):
    x = np.asarray(x)
    idx = np.unique(np.linspace(0, len(x) - 1, min(k, len(x))).astype(int))
    out = {"x": x[idx], "y": np.asarray(y)[idx]}
    if extra is not None:
        out["t"] = np.asarray(extra)[np.minimum(idx, len(extra) - 1)]
    return out


# ------------------------------------------------------ custom transformers
class OutlierClipper(TransformerMixin, BaseEstimator):
    """Clip numeric columns using IQR, z-score or percentile limits learned on training data."""

    def __init__(self, method="iqr", factor=1.5):
        self.method = method
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.method == "zscore":
            mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
            self.lo_, self.hi_ = mu - self.factor * sd, mu + self.factor * sd
        elif self.method == "percentile":
            p = min(max(float(self.factor), 0.0), 0.25) * 100
            self.lo_, self.hi_ = np.nanpercentile(X, p, 0), np.nanpercentile(X, 100 - p, 0)
        else:
            q1, q3 = np.nanpercentile(X, 25, 0), np.nanpercentile(X, 75, 0)
            iqr = q3 - q1
            self.lo_, self.hi_ = q1 - self.factor * iqr, q3 + self.factor * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class SkewTransformer(TransformerMixin, BaseEstimator):
    """log1p / Yeo-Johnson / Box-Cox / quantile transform, optionally only on skewed columns."""

    def __init__(self, method="yeo-johnson", skewed_only=True, threshold=1.0):
        self.method = method
        self.skewed_only = skewed_only
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        sd = X.std(0)
        sk = np.where(sd > 0, ((X - X.mean(0)) ** 3).mean(0) / np.where(sd > 0, sd, 1) ** 3, 0)
        self.cols_ = np.where(np.abs(sk) > self.threshold)[0] if self.skewed_only else np.arange(X.shape[1])
        self.min_ = X.min(0)
        self.tr_ = None
        if len(self.cols_) and self.method != "log1p":
            Z = self._shift(X[:, self.cols_])
            if self.method in ("yeo-johnson", "box-cox"):
                self.tr_ = PowerTransformer(method=self.method, standardize=False).fit(Z)
            else:
                dist = "normal" if self.method == "quantile_normal" else "uniform"
                self.tr_ = QuantileTransformer(n_quantiles=min(1000, len(Z)), output_distribution=dist, random_state=0).fit(Z)
        return self

    def _shift(self, Z):
        if self.method in ("box-cox", "log1p"):
            m = self.min_[self.cols_]
            return Z - np.minimum(m, 0) + (1e-6 if self.method == "box-cox" else 0)
        return Z

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if len(self.cols_):
            Z = self._shift(X[:, self.cols_])
            X[:, self.cols_] = np.log1p(np.maximum(Z, 0)) if self.method == "log1p" else self.tr_.transform(np.maximum(Z, 1e-12) if self.method == "box-cox" else Z)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace each category with its frequency in the training data."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = [X.iloc[:, j].value_counts(normalize=True).to_dict() for j in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
        self.names_ = list(X.columns) if not isinstance(X.columns, pd.RangeIndex) else None
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([X.iloc[:, j].map(self.maps_[j]).fillna(0.0).astype(float).values for j in range(X.shape[1])])

    def get_feature_names_out(self, input_features=None):
        base = input_features if input_features is not None else [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray([f"{c}_freq" for c in base], dtype=object)


class DropCorrelated(TransformerMixin, BaseEstimator):
    """Drop features whose absolute correlation with an earlier kept feature exceeds `threshold`."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        rows = X[sample_idx(len(X), 3000)]
        with np.errstate(all="ignore"):
            c = np.nan_to_num(np.abs(np.corrcoef(rows, rowvar=False)))
        if c.ndim == 0:
            self.keep_ = np.arange(X.shape[1])
            return self
        keep = []
        for j in range(X.shape[1]):
            if all(c[j, i] <= self.threshold for i in keep):
                keep.append(j)
        self.keep_ = np.array(keep, dtype=int)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        base = np.asarray(input_features if input_features is not None else [f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return base[self.keep_]


def chi2_nonneg(X, y):
    """chi2 needs non-negative input, so shift each column to start at zero."""
    X = np.asarray(X, dtype=float)
    return chi2(X - X.min(0), y)


def _clamped(base, attr, limit):
    """Subclass `base` so an integer `attr` larger than the data allows is clamped at fit time."""
    def _clamp(self, X, y):
        v = getattr(self, attr)
        lim = limit(self, np.asarray(X), y)
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v > lim:
            setattr(self, attr, max(1, int(lim)))

    def fit(self, X, y=None, **kw):
        _clamp(self, X, y)
        return base.fit(self, X, y, **kw)

    def fit_transform(self, X, y=None, **kw):
        _clamp(self, X, y)
        return base.fit_transform(self, X, y, **kw)

    name = "Safe" + base.__name__
    cls = type(name, (base,), {"fit": fit, "fit_transform": fit_transform, "__module__": __name__})
    cls.__qualname__ = name
    return cls


def _nfeat(self, X, y):
    return X.shape[1]


SafeSelectKBest = _clamped(SelectKBest, "k", _nfeat)
SafeRFE = _clamped(RFE, "n_features_to_select", _nfeat)
SafeSelectFromModel = _clamped(SelectFromModel, "max_features", _nfeat)
SafeSFS = _clamped(SequentialFeatureSelector, "n_features_to_select", lambda s, X, y: max(1, X.shape[1] - 1))
SafePCA = _clamped(PCA, "n_components", lambda s, X, y: min(X.shape))
SafeSVD = _clamped(TruncatedSVD, "n_components", lambda s, X, y: max(1, X.shape[1] - 1))
SafeICA = _clamped(FastICA, "n_components", lambda s, X, y: X.shape[1])
SafeKernelPCA = _clamped(KernelPCA, "n_components", lambda s, X, y: min(X.shape))
SafeLDA = _clamped(LinearDiscriminantAnalysis, "n_components", lambda s, X, y: max(1, min(X.shape[1], len(np.unique(y)) - 1)))


def resample_xy(X, y, method, random_state=0, k=5):
    """Random over/under-sampling and a small SMOTE implementation (training data only)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if method == "random_under":
        m = counts.min()
        idx = np.concatenate([rng.choice(np.where(y == c)[0], m, replace=False) for c in classes])
        return X[idx], y[idx]
    target = counts.max()
    Xs, ys = [X], [y]
    for c, n in zip(classes, counts):
        need = target - n
        if need <= 0:
            continue
        pos = np.where(y == c)[0]
        if method == "smote" and n > 1:
            nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(X[pos])
            base = rng.choice(len(pos), need)
            neigh = nn.kneighbors(X[pos][base], return_distance=False)[:, 1:]
            pick = neigh[np.arange(need), rng.randint(0, neigh.shape[1], need)]
            gap = rng.rand(need, 1)
            Xs.append(X[pos][base] + gap * (X[pos][pick] - X[pos][base]))
        else:
            Xs.append(X[rng.choice(pos, need)])
        ys.append(np.full(need, c))
    return np.vstack(Xs), np.concatenate(ys)


class ResampledClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a classifier so the training data is resampled inside every fit (so it stays inside CV folds)."""

    def __init__(self, estimator=None, method="smote", random_state=0):
        self.estimator = estimator
        self.method = method
        self.random_state = random_state

    def fit(self, X, y, **kw):
        Xr, yr = resample_xy(X, y, self.method, self.random_state)
        self.estimator_ = clone(self.estimator).fit(Xr, yr)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X):
        return self.estimator_.decision_function(X)


# ----------------------------------------------------------- data loading
def _read_csv_text(text):
    head = "\n".join(text.splitlines()[:20])
    try:
        sep = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    return pd.read_csv(io.StringIO(text), sep=sep, na_values=["", "NA", "N/A", "na", "n/a", "NaN", "nan", "null", "NULL", "None", "?", "-"], keep_default_na=True)


@safe
def load_csv(text, name="data.csv"):
    df = _read_csv_text(text)
    return set_frame(df, name)


def set_frame(df, name):
    df = df.copy()
    df.columns = [str(c).strip() or f"column_{i + 1}" for i, c in enumerate(df.columns)]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df = df.dropna(how="all")
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).str.strip()
            df[c] = df[c].where(df[c].isna(), s)
    S.clear()
    S["df"] = df.reset_index(drop=True)
    S["name"] = name
    S["log"] = []
    return dumps({"profile": profile(S["df"]), "name": name})


@safe
def load_sample(key):
    from sklearn import datasets
    if key == "student_performance":
        df = _student_sample()
    elif key == "breast_cancer":
        d = datasets.load_breast_cancer(as_frame=True)
        df = d.frame.rename(columns={"target": "diagnosis"})
        df["diagnosis"] = df["diagnosis"].map({0: "malignant", 1: "benign"})
    elif key == "wine":
        d = datasets.load_wine(as_frame=True)
        df = d.frame.rename(columns={"target": "cultivar"})
        df["cultivar"] = "class_" + df["cultivar"].astype(str)
    elif key == "iris":
        d = datasets.load_iris(as_frame=True)
        df = d.frame
        df["species"] = [d.target_names[i] for i in d.target]
        df = df.drop(columns=["target"])
    elif key == "diabetes":
        df = datasets.load_diabetes(as_frame=True).frame.rename(columns={"target": "progression"})
    elif key == "digits":
        d = datasets.load_digits(as_frame=True)
        df = d.frame.rename(columns={"target": "digit"})
        df["digit"] = df["digit"].astype(str)
    else:
        raise ValueError(f"Unknown sample dataset '{key}'")
    return set_frame(df, f"{key}.csv")


def _student_sample(n=10000, seed=7):
    r = np.random.RandomState(seed)
    edu = np.array(["High School", "Bachelor", "Master", "PhD"])
    study = np.clip(4.5 + 2.2 * r.randn(n), 0, 10).round(1)
    att = np.clip(78 + 12 * r.randn(n), 35, 100).round()
    prev = np.clip(66 + 13 * r.randn(n), 25, 100).round()
    pe = edu[np.minimum(3, (r.rand(n) ** 1.3 * 4).astype(int))]
    extra = np.where(r.rand(n) < 0.42, "Yes", "No")
    sleep = np.clip(7 + 1.2 * r.randn(n), 3.5, 10).round(1)
    screen = np.clip(4 + 2 * r.randn(n), 0, 12).round(1)
    z = (0.95 * (study - 4.5) / 2.2 + 0.6 * (att - 78) / 12 + 0.45 * (prev - 66) / 13 - 0.3 * (screen - 4) / 2
         - 0.25 * np.abs(sleep - 7.5) + 0.12 * np.array([list(edu).index(v) for v in pe]) + np.where(extra == "Yes", 0.12, 0) + 0.7 * r.randn(n))
    df = pd.DataFrame({
        "id": np.arange(1, n + 1), "age": 17 + r.randint(0, 8, n).astype(float), "gender": np.where(r.rand(n) < 0.5, "Male", "Female"),
        "study_hours": study, "attendance": att, "previous_score": prev, "parental_education": pe,
        "extracurricular": extra, "sleep_hours": sleep, "screen_time": screen,
        "test_score": np.clip(64 + 11 * z + 8 * r.randn(n), 20, 100).round(), "pass": (z > -0.15).astype(int),
    })
    df.loc[r.rand(n) < 0.03, "age"] = np.nan
    df.loc[r.rand(n) < 0.03, "test_score"] = np.nan
    return df


DATE_RE = r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})"


def col_kind(s):
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    non = s.dropna().astype(str)
    if not len(non):
        return "categorical"
    smp = non.sample(min(300, len(non)), random_state=0)
    if smp.str.match(DATE_RE).mean() > 0.9:
        return "datetime"
    if non.nunique() > 0.5 * len(non) and smp.str.len().mean() > 25:
        return "text"
    return "categorical"


def profile(df):
    cols = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        kind = col_kind(s)
        nun = int(s.nunique(dropna=True))
        d = {"name": c, "kind": kind, "dtype": str(s.dtype), "missing": int(s.isna().sum()), "unique": nun}
        d["missing_pct"] = d["missing"] / max(1, n)
        if kind == "numeric":
            v = pd.to_numeric(s, errors="coerce").dropna()
            d.update(mean=v.mean(), std=v.std(), min=v.min(), q1=v.quantile(.25), median=v.median(), q3=v.quantile(.75), max=v.max(), skew=v.skew() if len(v) > 2 else 0, integer=bool(len(v) and np.all(np.mod(v, 1) == 0)))
        d["top"] = [[str(k), int(v)] for k, v in s.astype(str).where(s.notna(), None).value_counts().head(8).items()]
        d["id_like"] = bool((nun >= 0.95 * n and n > 50 and (kind != "numeric" or d.get("integer"))) and (c.lower() in ("id", "index", "uuid", "row", "key") or c.lower().endswith("_id") or c.lower().endswith("id")))
        d["constant"] = nun <= 1
        cols.append(d)
    target_guess = None
    pref = ["target", "label", "class", "y", "outcome", "pass", "churn", "survived", "default", "price", "diagnosis", "species", "cultivar", "digit", "progression"]
    for p in pref:
        for c in cols:
            if c["name"].lower() == p:
                target_guess = c["name"]
                break
        if target_guess:
            break
    if not target_guess:
        target_guess = next((c["name"] for c in reversed(cols) if not c["id_like"]), cols[-1]["name"] if cols else None)
    return {"rows": n, "cols": cols, "target_guess": target_guess, "memory": int(df.memory_usage(deep=True).sum())}


@safe
def eda(target=None, max_cols=60):
    df = S["df"]
    out = {"columns": {}, "missing": {}, "corr": None, "target": None}
    for c in list(df.columns)[:max_cols]:
        s = df[c]
        kind = col_kind(s)
        out["missing"][c] = int(s.isna().sum())
        if kind == "numeric":
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v):
                cnt, edges = np.histogram(v, bins=min(30, max(5, int(v.nunique()))))
                out["columns"][c] = {"kind": "numeric", "counts": cnt, "edges": edges, "box": [v.min(), v.quantile(.25), v.median(), v.quantile(.75), v.max()]}
        else:
            vc = s.astype(str).where(s.notna(), "(missing)").value_counts().head(20)
            out["columns"][c] = {"kind": kind, "labels": list(vc.index), "counts": vc.values}
    num = df.select_dtypes("number").iloc[:, :30]
    if num.shape[1] >= 2:
        out["corr"] = {"cols": list(num.columns), "values": num.corr().round(3).values}
    if target and target in df.columns:
        t = df[target]
        rel = {}
        tk = col_kind(t)
        for c in list(num.columns)[:30]:
            if c == target:
                continue
            if tk == "numeric" and t.nunique() > 20:
                ix = sample_idx(len(df), 600)
                rel[c] = {"type": "scatter", "x": df[c].iloc[ix].values, "y": t.iloc[ix].values}
            else:
                g = df.groupby(t.astype(str))[c]
                rel[c] = {"type": "box", "groups": {k: [v.min(), v.quantile(.25), v.median(), v.quantile(.75), v.max(), v.mean()] for k, v in g}}
        out["target"] = {"name": target, "relations": rel}
    return dumps(out)


# --------------------------------------------------------- frame preparation
def _date_parts(df, cols):
    for c in cols:
        d = pd.to_datetime(df[c], errors="coerce")
        df[f"{c}_year"], df[f"{c}_month"], df[f"{c}_day"] = d.dt.year, d.dt.month, d.dt.day
        df[f"{c}_weekday"], df[f"{c}_is_weekend"] = d.dt.dayofweek, (d.dt.dayofweek >= 5).astype(float).where(d.notna())
        if (d.dt.hour.fillna(0) != 0).any():
            df[f"{c}_hour"] = d.dt.hour
    return df.drop(columns=cols)


def apply_frame(raw, spec):
    """Turn raw rows into the model input frame (custom features, date parts, column selection)."""
    df = raw.copy()
    for f in spec["custom"]:
        try:
            df[f["name"]] = pd.to_numeric(df.eval(f["expr"], engine="python"), errors="coerce")
        except Exception:
            df[f["name"]] = np.nan
    if spec["date_cols"]:
        df = _date_parts(df, [c for c in spec["date_cols"] if c in df.columns])
    for c in spec["features"]:
        if c not in df.columns:
            df[c] = np.nan
    X = df[spec["features"]].copy()
    for c in spec["num"] + spec.get("bool_num", []):
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in spec["cat"] + spec["highcard"]:
        X[c] = X[c].where(X[c].isna(), X[c].astype(str)).astype(object)
    return X


def prepare(cfg):
    df = S["df"].copy()
    D = cfg["dataset"]
    P = cfg.get("preprocess") or {}
    FE = cfg.get("fe") or {}
    target = D["target"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not in the data.")
    before = len(df)
    df = df[df[target].notna()]
    if len(df) < before:
        log(f"Dropped {before - len(df)} rows with a missing target.")
    custom = []
    for f in FE.get("custom_features") or []:
        name, expr = str(f.get("name", "")).strip(), str(f.get("expr", "")).strip()
        if not name or not expr:
            continue
        try:
            vals = pd.to_numeric(df.eval(expr, engine="python"), errors="coerce")
            df[name] = vals
            custom.append({"name": name, "expr": expr})
            log(f"Created feature '{name}' = {expr} ({int(vals.isna().sum())} missing values).")
        except Exception as e:  # noqa: BLE001
            log(f"Could not create feature '{name}' from '{expr}': {e}", "warn")
    group_col, time_col = D.get("group_col") or None, D.get("time_col") or None
    groups = df[group_col].astype(str).values if group_col and group_col in df.columns else None
    if time_col and time_col in df.columns:
        df = df.assign(_t=pd.to_datetime(df[time_col], errors="coerce")).sort_values("_t", kind="stable").drop(columns="_t")
        if groups is not None:
            groups = df[group_col].astype(str).values
    exclude = set(D.get("exclude") or []) | {target}
    if group_col:
        exclude.add(group_col)
    date_cols, num, cat, highcard, dropped = [], [], [], [], []
    hc_thr = int(P.get("high_card_threshold", 50))
    miss_thr = float(P.get("drop_missing_threshold", 0.6))
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        kind = col_kind(s)
        if s.isna().mean() > miss_thr and P.get("drop_high_missing", True):
            dropped.append((c, f"more than {int(miss_thr * 100)}% missing"))
            continue
        if P.get("drop_constant", True) and s.nunique(dropna=True) <= 1:
            dropped.append((c, "constant"))
            continue
        if kind == "datetime":
            if FE.get("date_parts", True):
                date_cols.append(c)
            else:
                dropped.append((c, "date column (turn on date extraction to use it)"))
        elif kind == "text":
            dropped.append((c, "free text"))
        elif kind == "numeric":
            num.append(c)
        else:
            (highcard if s.nunique() > hc_thr else cat).append(c)
    if dropped:
        log("Left out columns: " + "; ".join(f"{c} ({why})" for c, why in dropped), "warn")
    feats = num + cat + highcard
    if date_cols:
        tmp = _date_parts(df[date_cols].copy(), date_cols)
        derived = [c for c in tmp.columns if tmp[c].notna().any()]
        num += derived
        feats = num + cat + highcard
        log(f"Extracted date parts from {', '.join(date_cols)}: {len(derived)} new numeric columns.")
    if not feats:
        raise ValueError("No usable feature columns are left. Check the excluded columns in the Dataset block.")
    spec = {"custom": custom, "date_cols": date_cols, "features": feats, "num": num, "cat": cat, "highcard": highcard, "dropped": dropped}
    X = apply_frame(df, spec)
    y_raw = df[target]
    if P.get("drop_duplicates", True):
        dup = pd.concat([X, y_raw], axis=1).duplicated()
        if dup.any():
            X, y_raw, df = X[~dup], y_raw[~dup], df[~dup]
            if groups is not None:
                groups = groups[~dup.values]
            log(f"Removed {int(dup.sum())} duplicate rows.")
    if P.get("num_impute") == "drop":
        keep = X.notna().all(1)
        if (~keep).any():
            X, y_raw, df = X[keep], y_raw[keep], df[keep]
            if groups is not None:
                groups = groups[keep.values]
            log(f"Dropped {int((~keep).sum())} rows with missing values.")
    task = D.get("task", "auto")
    if task == "auto":
        if col_kind(y_raw) != "numeric":
            task = "classification"
        else:
            v = pd.to_numeric(y_raw, errors="coerce")
            task = "classification" if (v.nunique() <= 20 and np.all(np.mod(v.dropna(), 1) == 0)) else "regression"
    classes = None
    if task == "classification":
        vals = y_raw.astype(str)
        uniq = sorted(vals.unique(), key=lambda v: (float(v) if _isnum(v) else math.inf, v))
        if len(uniq) < 2:
            raise ValueError("The target has only one class.")
        if len(uniq) > 50:
            raise ValueError(f"The target has {len(uniq)} classes. Set the task to regression or choose another target.")
        classes = uniq
        y = vals.map({c: i for i, c in enumerate(uniq)}).values.astype(int)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").values.astype(float)
        ok = ~np.isnan(y)
        X, df, y = X[ok], df[ok], y[ok]
        if groups is not None:
            groups = groups[ok]
    S["frame_spec"] = spec
    return task, X.reset_index(drop=True), y, groups, classes, df.reset_index(drop=True)


def _isnum(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


# --------------------------------------------------------- pipeline builder
def _num_imputer(P, seed):
    m = P.get("num_impute", "mean")
    if m == "knn":
        return KNNImputer(n_neighbors=int(P.get("knn_neighbors", 5)))
    if m == "iterative":
        return IterativeImputer(max_iter=10, random_state=seed)
    if m == "constant":
        return SimpleImputer(strategy="constant", fill_value=float(P.get("fill_value", 0)))
    if m in ("mean", "median", "most_frequent"):
        return SimpleImputer(strategy=m)
    return SimpleImputer(strategy="median")


def _scaler(name):
    return {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler(), "maxabs": MaxAbsScaler(),
            "normalizer": Normalizer(), "quantile_normal": QuantileTransformer(output_distribution="normal", random_state=0),
            "quantile_uniform": QuantileTransformer(random_state=0), "power": PowerTransformer()}.get(name, "passthrough")


def _cat_encoder(name, P, task, seed):
    mc = int(P.get("max_categories", 20)) or None
    if name == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    if name == "frequency":
        return FrequencyEncoder()
    if name == "target" and TargetEncoder is not None:
        return TargetEncoder(target_type="continuous" if task == "regression" else "auto", random_state=seed)
    if name == "onehot_drop_first":
        return OneHotEncoder(drop="first", handle_unknown="ignore", max_categories=mc, sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", max_categories=mc, sparse_output=False)


def build_preprocessor(cfg, task):
    P = cfg.get("preprocess") or {}
    FE = cfg.get("fe") or {}
    spec = S["frame_spec"]
    seed = int(cfg["split"].get("seed", 42))
    num, cat, hc = list(spec["num"]), list(spec["cat"]), list(spec["highcard"])
    tr = []
    poly_deg = int(FE.get("polynomial", 0) or 0)
    poly_cols = [c for c in (FE.get("poly_columns") or []) if c in num]
    spline_cols = [c for c in (FE.get("spline_columns") or []) if c in num] if FE.get("splines") else []
    bin_cols = [c for c in (FE.get("bin_columns") or []) if c in num]

    def num_steps(with_poly):
        st = [("impute", _num_imputer(P, seed))]
        if P.get("outliers") in ("iqr", "zscore", "percentile"):
            st.append(("clip", OutlierClipper(P["outliers"], float(P.get("outlier_factor", 1.5 if P["outliers"] == "iqr" else 3 if P["outliers"] == "zscore" else 0.01)))))
        if P.get("transform", "none") not in ("none", None):
            st.append(("transform", SkewTransformer(P["transform"], bool(P.get("transform_skewed_only", True)), float(P.get("skew_threshold", 1.0)))))
        st.append(("scale", _scaler(P.get("scaling", "standard"))))
        if with_poly and poly_deg >= 2:
            st.append(("poly", PolynomialFeatures(poly_deg, interaction_only=bool(FE.get("interaction_only", False)), include_bias=False)))
        return st

    rest = [c for c in num if c not in poly_cols] if poly_cols else num
    if rest:
        tr.append(("num", Pipeline(num_steps(poly_deg >= 2 and not poly_cols)), rest))
    if poly_cols and poly_deg >= 2:
        tr.append(("poly", Pipeline(num_steps(True)), poly_cols))
    if spline_cols:
        tr.append(("spline", Pipeline([("impute", SimpleImputer(strategy="median")), ("spline", SplineTransformer(n_knots=int(FE.get("spline_knots", 5)), degree=3))]), spline_cols))
    if bin_cols:
        tr.append(("bins", Pipeline([("impute", SimpleImputer(strategy="median")), ("bin", KBinsDiscretizer(n_bins=int(FE.get("n_bins", 5)), encode="onehot-dense", strategy=FE.get("bin_strategy", "quantile")))]), bin_cols))
    cat_imp = SimpleImputer(strategy="constant", fill_value="missing") if P.get("cat_impute") == "constant" else SimpleImputer(strategy="most_frequent")
    if cat:
        tr.append(("cat", Pipeline([("impute", cat_imp), ("encode", _cat_encoder(P.get("encoding", "onehot"), P, task, seed))]), cat))
    if hc and P.get("high_card_encoding", "frequency") != "drop":
        tr.append(("highcard", Pipeline([("impute", clone(cat_imp)), ("encode", _cat_encoder(P.get("high_card_encoding", "frequency"), P, task, seed))]), hc))
    return ColumnTransformer(tr, remainder="drop", sparse_threshold=0, verbose_feature_names_out=True)


def _fs_estimator(name, task, seed):
    cls = task == "classification"
    if name == "rf":
        return (ExtraTreesClassifier if cls else ExtraTreesRegressor)(n_estimators=100, random_state=seed, n_jobs=N_JOBS)
    return LogisticRegression(max_iter=2000) if cls else Ridge()


def build_fs_steps(cfg, task, n_classes):
    F = cfg.get("fs") or {}
    seed = int(cfg["split"].get("seed", 42))
    cls = task == "classification"
    st = []
    if F.get("variance_filter"):
        st.append(("variance", VarianceThreshold(float(F.get("variance_threshold", 0.0)))))
    if F.get("drop_correlated"):
        st.append(("decorrelate", DropCorrelated(float(F.get("corr_threshold", 0.95)))))
    m, k = F.get("method", "none"), int(F.get("k", 20))
    fscore = f_classif if cls else f_regression
    if m == "kbest_f":
        st.append(("select", SafeSelectKBest(fscore, k=k)))
    elif m == "kbest_mi":
        st.append(("select", SafeSelectKBest(functools.partial(mutual_info_classif if cls else mutual_info_regression, random_state=seed), k=k)))
    elif m == "kbest_chi2":
        st.append(("select", SafeSelectKBest(chi2_nonneg if cls else f_regression, k=k)))
    elif m == "percentile_f":
        st.append(("select", SelectPercentile(fscore, percentile=int(F.get("percentile", 50)))))
    elif m == "rfe":
        st.append(("select", SafeRFE(_fs_estimator(F.get("estimator", "linear"), task, seed), n_features_to_select=k, step=0.1)))
    elif m == "rfecv":
        st.append(("select", RFECV(_fs_estimator(F.get("estimator", "linear"), task, seed), step=1, cv=3, min_features_to_select=1)))
    elif m == "l1":
        est = LinearSVC(penalty="l1", dual=False, C=float(F.get("l1_C", 0.5)), max_iter=5000) if cls else Lasso(alpha=float(F.get("l1_alpha", 0.01)), max_iter=5000)
        st.append(("select", SafeSelectFromModel(est, max_features=k, threshold=1e-5)))
    elif m == "tree_importance":
        st.append(("select", SafeSelectFromModel(_fs_estimator("rf", task, seed), max_features=k, threshold=-np.inf)))
    elif m in ("sfs_forward", "sfs_backward"):
        est = LogisticRegression(max_iter=500) if cls else Ridge()
        st.append(("select", SafeSFS(est, n_features_to_select=k, direction="forward" if m == "sfs_forward" else "backward", cv=3)))
    r = F.get("reduction", "none")
    nc = F.get("n_components", 0.95)
    nc = float(nc) if float(nc) < 1 else int(nc)
    if r == "pca":
        st.append(("reduce", SafePCA(n_components=nc, random_state=seed)))
    elif r == "svd":
        st.append(("reduce", SafeSVD(n_components=int(nc) if nc >= 1 else 10, random_state=seed)))
    elif r == "ica":
        st.append(("reduce", SafeICA(n_components=int(nc) if nc >= 1 else 10, random_state=seed, max_iter=500)))
    elif r == "kernel_pca":
        st.append(("reduce", SafeKernelPCA(n_components=int(nc) if nc >= 1 else 10, kernel=F.get("kernel", "rbf"), random_state=seed)))
    elif r == "lda" and cls:
        st.append(("reduce", SafeLDA(n_components=int(nc) if nc >= 1 else max(1, n_classes - 1))))
    return st


def _convert_param(v):
    if isinstance(v, str):
        s = v.strip()
        if s.lower() == "none":
            return None
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        if "," in s and all(p.strip().lstrip("-").isdigit() for p in s.split(",") if p.strip()):
            return tuple(int(p) for p in s.split(",") if p.strip())
    if isinstance(v, list):
        return tuple(v)
    return v


_TAGS_CHECKED = set()


def _compat_tags(self):
    from sklearn.base import RegressorMixin
    from sklearn.utils import ClassifierTags, RegressorTags
    tags = BaseEstimator.__sklearn_tags__(self)
    if isinstance(self, ClassifierMixin):
        tags.estimator_type, tags.classifier_tags = "classifier", ClassifierTags()
        tags.target_tags.required = True
    elif isinstance(self, RegressorMixin):
        tags.estimator_type, tags.regressor_tags = "regressor", RegressorTags()
        tags.target_tags.required = True
    return tags


def fix_sklearn_tags(Est):
    """XGBoost < 2.1.4 (bundled with Pyodide 0.27) is incompatible with scikit-learn 1.6's estimator tags:
    it keeps the old `_more_tags` and lists sklearn's mixins after BaseEstimator, so tag lookup fails with
    "'super' object has no attribute '__sklearn_tags__'". Giving every class in its MRO that defines
    `_more_tags` a direct `__sklearn_tags__` makes scikit-learn use the modern, working code path."""
    if Est in _TAGS_CHECKED:
        return Est
    _TAGS_CHECKED.add(Est)
    try:
        from sklearn.utils import get_tags
    except ImportError:  # scikit-learn < 1.6 has no new tags, nothing to fix
        return Est
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            get_tags(Est())
        return Est
    except AttributeError:
        pass
    except Exception:
        return Est
    for klass in Est.__mro__:
        if "_more_tags" in vars(klass) and "__sklearn_tags__" not in vars(klass):
            klass.__sklearn_tags__ = _compat_tags
    if "__sklearn_tags__" not in vars(Est):
        Est.__sklearn_tags__ = _compat_tags
    return Est


def make_estimator(spec, task, seed, overrides=None):
    path = spec["cls"][task] if isinstance(spec["cls"], dict) else spec["cls"]
    mod, name = path.split(":")
    try:
        Est = fix_sklearn_tags(getattr(importlib.import_module(mod), name))
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"{spec.get('name', name)} needs the '{mod.split('.')[0]}' package, which is not available: {e}")
    valid = Est().get_params(deep=False)
    params = {}
    for k, v in {**(spec.get("params") or {}), **(overrides or {})}.items():
        if k in valid:
            params[k] = _convert_param(v)
    if "random_state" in valid and "random_state" not in params:
        params["random_state"] = seed
    if "n_jobs" in valid:
        params["n_jobs"] = N_JOBS
    if mod.startswith("lightgbm"):
        params.setdefault("verbose", -1)
    if mod.startswith("xgboost"):
        params.setdefault("verbosity", 0)
    return Est(**params)


def build_pipeline(cfg, spec, task, n_classes, overrides=None):
    seed = int(cfg["split"].get("seed", 42))
    P = cfg.get("preprocess") or {}
    est = make_estimator(spec, task, seed, overrides)
    imb = P.get("imbalance", "none") if task == "classification" else "none"
    if imb == "class_weight" and "class_weight" in est.get_params(deep=False):
        est.set_params(class_weight="balanced")
    steps = [("prep", build_preprocessor(cfg, task))] + build_fs_steps(cfg, task, n_classes)
    if (spec.get("flags") or {}).get("nonneg"):
        steps.append(("nonneg", MinMaxScaler()))
    if imb in ("random_over", "random_under", "smote"):
        est = ResampledClassifier(est, imb, seed)
    steps.append(("model", est))
    return Pipeline(steps)


def make_cv(cfg, task, y, groups, folds=None):
    C = cfg.get("cv") or {}
    s = C.get("strategy", "stratified_kfold")
    if s == "none":
        return None
    k = int(folds or C.get("folds", 5))
    seed = int(cfg["split"].get("seed", 42))
    cls = task == "classification"
    if cls and s in ("stratified_kfold", "repeated_stratified_kfold"):
        m = int(np.bincount(y).min())
        if m < k:
            log(f"Smallest class has {m} training rows, so CV uses {max(2, m)} folds instead of {k}.", "warn")
            k = max(2, m)
    if s == "kfold":
        return KFold(k, shuffle=True, random_state=seed)
    if s == "stratified_kfold":
        return StratifiedKFold(k, shuffle=True, random_state=seed) if cls else KFold(k, shuffle=True, random_state=seed)
    if s == "repeated_kfold":
        return RepeatedKFold(n_splits=k, n_repeats=int(C.get("repeats", 2)), random_state=seed)
    if s == "repeated_stratified_kfold":
        return (RepeatedStratifiedKFold if cls else RepeatedKFold)(n_splits=k, n_repeats=int(C.get("repeats", 2)), random_state=seed)
    if s == "shuffle_split":
        return ShuffleSplit(n_splits=k, test_size=float(C.get("test_size", 0.2)), random_state=seed)
    if s == "stratified_shuffle_split":
        return (StratifiedShuffleSplit if cls else ShuffleSplit)(n_splits=k, test_size=float(C.get("test_size", 0.2)), random_state=seed)
    if s == "time_series":
        return TimeSeriesSplit(n_splits=k)
    if s == "group_kfold":
        if groups is None:
            log("Group K-fold needs a group column in the Dataset block; using K-fold instead.", "warn")
            return KFold(k, shuffle=True, random_state=seed)
        return GroupKFold(n_splits=min(k, len(np.unique(groups))))
    return KFold(k, shuffle=True, random_state=seed)


# ------------------------------------------------------------------ metrics
LOWER_BETTER = {"rmse", "mae", "mape", "medae", "log_loss", "max_error"}
CLS_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f1_weighted", "roc_auc", "average_precision", "mcc", "kappa", "log_loss"]
REG_METRICS = ["r2", "rmse", "mae", "mape", "medae", "explained_variance", "max_error"]


def scorers(task, binary):
    if task == "classification":
        s = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
             "precision": "precision" if binary else "precision_macro", "recall": "recall" if binary else "recall_macro",
             "f1": "f1" if binary else "f1_macro", "f1_weighted": "f1_weighted",
             "roc_auc": "roc_auc" if binary else "roc_auc_ovr", "mcc": M.make_scorer(M.matthews_corrcoef),
             "log_loss": "neg_log_loss"}
        if binary:
            s["average_precision"] = "average_precision"
        return s
    return {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error",
            "mape": "neg_mean_absolute_percentage_error", "medae": "neg_median_absolute_error", "explained_variance": "explained_variance"}


def get_scores(model, X, K):
    """Class probabilities (or a softmax of decision scores) for ROC/AUC."""
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X), dtype=float), True
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            d = np.asarray(model.decision_function(X), dtype=float)
            if d.ndim == 1:
                p = 1 / (1 + np.exp(-d))
                return np.column_stack([1 - p, p]), False
            e = np.exp(d - d.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True), False
        except Exception:
            pass
    return None, False


def compute_metrics(task, y, pred, scores=None, is_proba=False, K=2):
    if task == "classification":
        binary = K == 2
        avg = "binary" if binary else "macro"
        m = {"accuracy": M.accuracy_score(y, pred), "balanced_accuracy": M.balanced_accuracy_score(y, pred),
             "precision": M.precision_score(y, pred, average=avg, zero_division=0), "recall": M.recall_score(y, pred, average=avg, zero_division=0),
             "f1": M.f1_score(y, pred, average=avg, zero_division=0), "f1_weighted": M.f1_score(y, pred, average="weighted", zero_division=0),
             "mcc": M.matthews_corrcoef(y, pred), "kappa": M.cohen_kappa_score(y, pred)}
        if scores is not None:
            try:
                if binary:
                    m["roc_auc"] = M.roc_auc_score(y, scores[:, 1])
                    m["average_precision"] = M.average_precision_score(y, scores[:, 1])
                elif len(np.unique(y)) == K:
                    m["roc_auc"] = M.roc_auc_score(y, scores, multi_class="ovr", labels=list(range(K)))
            except Exception:
                pass
            if is_proba:
                try:
                    m["log_loss"] = M.log_loss(y, np.clip(scores, 1e-15, 1), labels=list(range(K)))
                except Exception:
                    pass
        return m
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    m = {"r2": M.r2_score(y, pred), "rmse": math.sqrt(M.mean_squared_error(y, pred)), "mae": M.mean_absolute_error(y, pred),
         "medae": M.median_absolute_error(y, pred), "explained_variance": M.explained_variance_score(y, pred), "max_error": M.max_error(y, pred)}
    if np.all(np.abs(y) > 1e-9):
        m["mape"] = M.mean_absolute_percentage_error(y, pred)
    return m


def resolve_primary(cfg, task, y, K):
    m = (cfg.get("eval") or {}).get("primary_metric", "auto")
    valid = CLS_METRICS if task == "classification" else REG_METRICS
    if m in valid and m not in ("max_error", "kappa"):
        if m == "average_precision" and K != 2:
            return "f1"
        return m
    if task == "classification":
        c = np.bincount(y)
        return "f1" if (c.min() / c.sum() < 0.3) else "accuracy"
    return "r2"


def better(a, b, metric):
    if a is None:
        return False
    if b is None:
        return True
    return a < b if metric in LOWER_BETTER else a > b


# ----------------------------------------------------------- model details
def model_details(pipe, X, y, task, K):
    out = {}
    pred = pipe.predict(X)
    if task == "classification":
        scores, is_proba = get_scores(pipe, X, K)
        out["confusion"] = M.confusion_matrix(y, pred, labels=list(range(K)))
        p, r, f, s = M.precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
        out["report"] = [{"precision": p[i], "recall": r[i], "f1": f[i], "support": s[i]} for i in range(K)]
        if scores is not None:
            if K == 2:
                fpr, tpr, thr = M.roc_curve(y, scores[:, 1])
                out["roc"] = [{"cls": 1, **downsample_curve(fpr, tpr), "auc": M.auc(fpr, tpr)}]
                pr, rc, _ = M.precision_recall_curve(y, scores[:, 1])
                out["pr"] = [{"cls": 1, **downsample_curve(rc[::-1], pr[::-1]), "ap": M.average_precision_score(y, scores[:, 1])}]
                if is_proba:
                    pt, pp = _calibration(y, scores[:, 1])
                    out["calibration"] = {"prob_true": pt, "prob_pred": pp}
                ths = np.round(np.linspace(0.05, 0.95, 19), 2)
                rows = []
                for t in ths:
                    pr_t = (scores[:, 1] >= t).astype(int)
                    rows.append({"threshold": t, "precision": M.precision_score(y, pr_t, zero_division=0), "recall": M.recall_score(y, pr_t, zero_division=0),
                                 "f1": M.f1_score(y, pr_t, zero_division=0), "accuracy": M.accuracy_score(y, pr_t)})
                out["threshold"] = rows
                out["score_hist"] = {"neg": np.histogram(scores[y == 0, 1], bins=20, range=(0, 1))[0], "pos": np.histogram(scores[y == 1, 1], bins=20, range=(0, 1))[0]}
            else:
                out["roc"], out["pr"] = [], []
                for c in range(min(K, 10)):
                    yc = (y == c).astype(int)
                    if yc.sum() == 0 or yc.sum() == len(yc):
                        continue
                    fpr, tpr, _ = M.roc_curve(yc, scores[:, c])
                    out["roc"].append({"cls": c, **downsample_curve(fpr, tpr, k=80), "auc": M.auc(fpr, tpr)})
                    pr, rc, _ = M.precision_recall_curve(yc, scores[:, c])
                    out["pr"].append({"cls": c, **downsample_curve(rc[::-1], pr[::-1], k=80), "ap": M.average_precision_score(yc, scores[:, c])})
    else:
        y = np.asarray(y, float)
        pred = np.asarray(pred, float)
        res = y - pred
        ix = sample_idx(len(y), 600)
        out["scatter"] = {"y": y[ix], "pred": pred[ix], "res": res[ix]}
        cnt, edges = np.histogram(res, bins=30)
        out["res_hist"] = {"counts": cnt, "edges": edges}
        q = np.linspace(0.01, 0.99, 50)
        z = np.sort((res - res.mean()) / (res.std() or 1))
        from scipy import stats
        out["qq"] = {"theory": stats.norm.ppf(q), "sample": np.quantile(z, q)}
    return out


def _calibration(y, p, bins=10):
    try:
        from sklearn.calibration import calibration_curve
        return calibration_curve(y, p, n_bins=bins, strategy="quantile")
    except Exception:
        return [], []


def native_importance(pipe, top=40):
    try:
        model = pipe.steps[-1][1]
        if isinstance(model, ResampledClassifier):
            model = model.estimator_
        if hasattr(model, "feature_importances_"):
            vals, kind = np.asarray(model.feature_importances_, float), "feature_importances_"
        elif hasattr(model, "coef_"):
            c = np.asarray(model.coef_, float)
            vals, kind = (np.abs(c).mean(0) if c.ndim > 1 else np.abs(c)), "|coef_|"
        else:
            return None
        names = processed_names(pipe, len(vals))
        order = np.argsort(-vals)[:top]
        return {"kind": kind, "items": [{"feature": names[i], "value": vals[i]} for i in order]}
    except Exception:
        return None


def processed_names(pipe, n=None):
    try:
        names = [str(x) for x in pipe[:-1].get_feature_names_out()]
        names = [x.split("__", 1)[1] if "__" in x else x for x in names]
        if n is None or len(names) == n:
            return names
    except Exception:
        pass
    return [f"f{i}" for i in range(n or 0)]


# ------------------------------------------------------------------- training
def _fit_rows(spec, X, y, cfg, groups=None):
    lim = int((cfg.get("advanced") or {}).get("slow_model_rows", 3000))
    if (spec.get("flags") or {}).get("slow") and len(X) > lim:
        ix = sample_idx(len(X), lim, 1)
        log(f"{spec['name']} is slow on large data, so it trains on {lim} of {len(X)} training rows.", "warn")
        return X.iloc[ix], y[ix], (groups[ix] if groups is not None else None)
    return X, y, groups


def train_one(mid, name, key, make, cfg, D, cv_override=None, family="", params=None, spec=None):
    task, K, primary = D["task"], D["K"], D["primary"]
    r = {"id": mid, "name": name, "key": key, "family": family, "status": "ok", "params": params or {}}
    Xtr, ytr, gtr = D["X_train"], D["y_train"], D["g_train"]
    if spec is not None:
        Xtr, ytr, gtr = _fit_rows(spec, Xtr, ytr, cfg, gtr)
    t0 = time.time()
    if cv_override is not None:
        r["cv"] = cv_override
    elif D["cv"] is not None:
        probe = make()
        sc = {k: v for k, v in D["scorers"].items() if not (k == "log_loss" and not hasattr(probe, "predict_proba"))}
        res = cross_validate(probe, Xtr, ytr, cv=D["cv"], groups=gtr, scoring=sc, error_score=np.nan, n_jobs=N_JOBS)
        r["cv"] = {}
        for mname in sc:
            v = np.asarray(res[f"test_{mname}"], float)
            if mname in LOWER_BETTER:
                v = -v
            if np.all(np.isnan(v)):
                continue
            r["cv"][mname] = {"mean": np.nanmean(v), "std": np.nanstd(v), "folds": v}
        r["cv_fit_time"] = float(np.mean(res["fit_time"]))
    t1 = time.time()
    pipe = make()
    if cfg["split"].get("refit_train_val") and D["X_val"] is not None and spec is not None and not (spec.get("flags") or {}).get("slow"):
        pipe.fit(pd.concat([Xtr, D["X_val"]]), np.concatenate([ytr, D["y_val"]]))
    else:
        pipe.fit(Xtr, ytr)
    r["fit_time"] = time.time() - t1
    r["cv_time"] = t1 - t0
    t2 = time.time()
    for part in ("train", "val", "test"):
        X, y = D[f"X_{part}"], D[f"y_{part}"]
        if X is None or not len(X):
            continue
        if part == "train" and len(X) > 5000:
            ix = sample_idx(len(X), 5000, 2)
            X, y = X.iloc[ix], y[ix]
        pred = pipe.predict(X)
        sc, isp = get_scores(pipe, X, K) if task == "classification" else (None, False)
        r[part] = compute_metrics(task, y, pred, sc, isp, K)
    r["predict_time"] = time.time() - t2
    r["details"] = model_details(pipe, D["X_test"], D["y_test"], task, K)
    r["native_importance"] = native_importance(pipe)
    r["has_proba"] = hasattr(pipe, "predict_proba")
    r["score"] = r["cv"][primary]["mean"] if r.get("cv") and primary in r["cv"] else (r.get("val") or r["test"]).get(primary)
    r["score_source"] = "cv" if r.get("cv") and primary in r["cv"] else ("val" if r.get("val") else "test")
    tr, te = (r.get("train") or {}).get(primary), r["test"].get(primary)
    r["overfit_gap"] = (tr - te) if (tr is not None and te is not None) else None
    if primary in LOWER_BETTER and r["overfit_gap"] is not None:
        r["overfit_gap"] = -r["overfit_gap"]
    return r, pipe


def _summary_row(r, primary):
    return {k: r.get(k) for k in ("id", "name", "key", "family", "status", "score", "score_source", "fit_time", "overfit_gap", "baseline", "tuned_from", "ensemble")}


@safe
def run(config_json):
    cfg = json.loads(config_json)
    t_start = time.time()
    S["log"] = []
    S["cfg"] = cfg
    seed = int(cfg["split"].get("seed", 42))
    progress("prepare", "Preparing data")
    task, X, y, groups, classes, frame = prepare(cfg)
    K = len(classes) if classes else 1
    binary = K == 2
    log(f"Task: {task}{f' with {K} classes' if classes else ''}. {X.shape[1]} input columns, {len(X)} rows.")
    sp = cfg["split"]
    test_size, val_size = float(sp.get("test_size", 0.2)), float(sp.get("val_size", 0.1))
    idx = np.arange(len(X))
    cvs = (cfg.get("cv") or {}).get("strategy", "stratified_kfold")
    ordered = cvs == "time_series" or not sp.get("shuffle", True)
    strat = y if (task == "classification" and sp.get("stratify", True) and not ordered) else None

    def split(ix, size, st):
        if size <= 0:
            return ix, np.array([], dtype=int)
        if ordered:
            cut = int(round(len(ix) * (1 - size)))
            return ix[:cut], ix[cut:]
        try:
            return train_test_split(ix, test_size=size, random_state=seed, stratify=st)
        except ValueError:
            log("Stratified split was not possible (a class is too small), so a random split was used.", "warn")
            return train_test_split(ix, test_size=size, random_state=seed)

    tmp, te = split(idx, test_size, strat)
    vs = val_size / max(1e-9, 1 - test_size)
    tr, va = split(np.sort(tmp) if ordered else tmp, vs if val_size > 0 else 0, strat[tmp] if strat is not None else None)
    P = cfg.get("preprocess") or {}
    if P.get("outliers") == "isolation_forest" and len(tr) > 20:
        num = S["frame_spec"]["num"]
        if num:
            Z = SimpleImputer(strategy="median").fit_transform(X.iloc[tr][num])
            keep = IsolationForest(contamination=float(P.get("outlier_factor", 0.02)) if float(P.get("outlier_factor", 0.02)) < 0.5 else "auto", random_state=seed).fit_predict(Z) == 1
            log(f"Isolation Forest removed {int((~keep).sum())} outlier rows from the training set.")
            tr = np.asarray(tr)[keep]
    mx = int((cfg.get("advanced") or {}).get("max_train_rows", 0) or 0)
    if mx and len(tr) > mx:
        tr = np.asarray(tr)[sample_idx(len(tr), mx, 3)]
        log(f"Training set capped at {mx} rows (Settings → max training rows).", "warn")
    D = {"task": task, "K": K, "classes": classes}
    for part, ix in (("train", tr), ("val", va), ("test", te)):
        ix = np.asarray(ix, dtype=int)
        D[f"X_{part}"] = X.iloc[ix].reset_index(drop=True) if len(ix) else None
        D[f"y_{part}"] = y[ix] if len(ix) else None
        D[f"idx_{part}"] = ix
    D["g_train"] = groups[np.asarray(tr, dtype=int)] if groups is not None else None
    if D["X_test"] is None:
        raise ValueError("The test set is empty. Increase the test size.")
    log(f"Split: {len(tr)} train, {len(va)} validation, {len(te)} test rows{' (stratified)' if strat is not None else ''}{' (time-ordered)' if ordered else ''}.")
    D["cv"] = make_cv(cfg, task, D["y_train"], D["g_train"])
    if D["cv"] is not None:
        log(f"Cross-validation: {type(D['cv']).__name__} with {D['cv'].get_n_splits(D['X_train'], D['y_train'], D['g_train'])} splits on the training set.")
    D["scorers"] = scorers(task, binary)
    D["primary"] = primary = resolve_primary(cfg, task, y, K)
    log(f"Primary metric: {primary} ({'lower' if primary in LOWER_BETTER else 'higher'} is better). Models are ranked by {'cross-validation' if D['cv'] is not None else 'validation' if len(va) else 'test'} score.")
    S.update({"D": D, "task": task, "classes": classes, "frame": frame, "models": {}, "results": {}})
    results = S["results"]

    specs = list(cfg.get("models") or [])
    if (cfg.get("eval") or {}).get("baseline", True):
        specs.insert(0, {"id": "baseline", "key": "dummy", "name": "Baseline (dummy)", "family": "Baseline",
                         "cls": {"classification": "sklearn.dummy:DummyClassifier", "regression": "sklearn.dummy:DummyRegressor"},
                         "params": {"strategy": "prior" if task == "classification" else "mean"}})
    n_total = len(specs) + len(cfg.get("ensembles") or [])
    for i, spec in enumerate(specs):
        mid = spec["id"]
        if task not in (spec.get("tasks") or ["classification", "regression"]):
            log(f"{spec['name']} does not support {task}, so it was skipped.", "warn")
            continue
        progress("model", f"Training {spec['name']}", i + 1, n_total, model=mid)
        try:
            r, pipe = train_one(mid, spec["name"], spec["key"], lambda s=spec: build_pipeline(cfg, s, task, K), cfg, D, family=spec.get("family", ""), params=spec.get("params"), spec=spec)
            r["baseline"] = mid == "baseline"
            results[mid] = r
            S["models"][mid] = pipe
            log(f"{spec['name']}: {primary} {_fmt(r['score'])} ({r['score_source']}), test {_fmt(r['test'].get(primary))}, fit {r['fit_time']:.2f}s.")
        except Exception as e:  # noqa: BLE001
            results[mid] = {"id": mid, "name": spec["name"], "key": spec["key"], "family": spec.get("family", ""), "status": "error", "error": f"{type(e).__name__}: {e}"}
            log(f"{spec['name']} failed: {type(e).__name__}: {e}", "error")
        progress("model_done", spec["name"], i + 1, n_total, model=mid, status=results[mid]["status"])

    ranked = _ranked(results, primary)
    for j, ens in enumerate(cfg.get("ensembles") or []):
        progress("model", f"Training {ens['name']}", len(specs) + j + 1, n_total, model=ens["id"])
        try:
            k = int(ens.get("top_k", 3))
            base_ids = [m for m in (ens.get("base") or []) if m in S["models"]] or [r["id"] for r in ranked[:k]]
            if len(base_ids) < 2:
                raise ValueError("needs at least two trained models")
            spec_by = {s["id"]: s for s in specs}
            members = [(bid, spec_by[bid]) for bid in base_ids if bid in spec_by]
            make = _ensemble_factory(ens, members, cfg, task, K)
            r, pipe = train_one(ens["id"], ens["name"], ens["key"], make, cfg, D, family="Ensemble", params={"members": [results[b]["name"] for b in base_ids], **(ens.get("params") or {})})
            r["ensemble"] = True
            results[ens["id"]] = r
            S["models"][ens["id"]] = pipe
            log(f"{ens['name']} of {', '.join(results[b]['name'] for b in base_ids)}: {primary} {_fmt(r['score'])}.")
        except Exception as e:  # noqa: BLE001
            results[ens["id"]] = {"id": ens["id"], "name": ens["name"], "key": ens["key"], "family": "Ensemble", "status": "error", "error": f"{type(e).__name__}: {e}"}
            log(f"{ens['name']} failed: {e}", "error")

    T = cfg.get("tuning") or {}
    if T.get("enabled"):
        tune_models(cfg, T, specs, D)

    E = cfg.get("eval") or {}
    ranked = _ranked(results, primary)
    if not ranked:
        raise ValueError("No model finished training. See the log for the errors.")
    best = ranked[0]["id"]
    S["best"] = best
    which = E.get("permutation", "top3")
    perm_ids = [] if which == "none" else [best] if which == "best" else [r["id"] for r in ranked[:3]] if which == "top3" else [r["id"] for r in ranked]
    for i, mid in enumerate(perm_ids):
        progress("explain", f"Permutation importance: {results[mid]['name']}", i + 1, len(perm_ids))
        try:
            results[mid]["permutation"] = _permutation(mid, int(E.get("perm_repeats", 5)))
        except Exception as e:  # noqa: BLE001
            log(f"Permutation importance failed for {results[mid]['name']}: {e}", "warn")
    shap_which = E.get("shap", "best")
    shap_ids = [] if shap_which == "none" else [best] if shap_which == "best" else [r["id"] for r in ranked[:3]]
    for mid in shap_ids:
        progress("explain", f"SHAP values: {results[mid]['name']}")
        try:
            results[mid]["shap"] = _shap(mid, int(E.get("shap_rows", 40)), None)
        except Exception as e:  # noqa: BLE001
            log(f"SHAP failed for {results[mid]['name']}: {e}", "warn")
    if E.get("learning_curve", "none") != "none":
        for mid in ([best] if E.get("learning_curve") == "best" else [r["id"] for r in ranked[:3]]):
            progress("explain", f"Learning curve: {results[mid]['name']}")
            try:
                results[mid]["learning_curve"] = _learning_curve(mid)
            except Exception as e:  # noqa: BLE001
                log(f"Learning curve failed for {results[mid]['name']}: {e}", "warn")
    S["duration"] = time.time() - t_start
    log(f"Finished in {S['duration']:.1f}s. Best model: {results[best]['name']} ({primary} {_fmt(results[best]['score'])}).")
    return dumps(summary())


def _ensemble_factory(ens, members, cfg, task, K):
    cls = task == "classification"
    ests = [(f"m{i}_{s['key']}", build_pipeline(cfg, s, task, K)) for i, (_, s) in enumerate(members)]
    if ens["key"] == "voting":
        if cls:
            soft = all(hasattr(e, "predict_proba") for _, e in ests) and ens.get("params", {}).get("voting", "soft") == "soft"
            return lambda: VotingClassifier([(n, clone(e)) for n, e in ests], voting="soft" if soft else "hard", n_jobs=N_JOBS)
        return lambda: VotingRegressor([(n, clone(e)) for n, e in ests], n_jobs=N_JOBS)
    final = LogisticRegression(max_iter=2000) if cls else RidgeCV()
    return lambda: (StackingClassifier if cls else StackingRegressor)([(n, clone(e)) for n, e in ests], final_estimator=clone(final), cv=int(ens.get("params", {}).get("cv", 3)), n_jobs=N_JOBS)


def _dims(raw, meta):
    """Turn a search space (lists or {low, high, log, type} ranges) into dimensions for Bayesian search."""
    dims = []
    for k, vals in raw.items():
        mm = (meta or {}).get(k) or {}
        if isinstance(vals, dict):
            lo, hi = float(vals.get("low", vals.get("min", 0))), float(vals.get("high", vals.get("max", 1)))
            kind = "int" if vals.get("type") == "int" else "float"
            dims.append({"k": k, "kind": kind, "lo": lo, "hi": hi, "log": bool(vals.get("log")) and lo > 0})
            continue
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not mm.get("cat") and len(nums) == len(vals) and len(set(nums)) >= 2:
            lo, hi = min(nums), max(nums)
            dims.append({"k": k, "kind": "int" if all(isinstance(v, int) for v in nums) else "float", "lo": lo, "hi": hi,
                         "log": bool(lo > 0 and (mm.get("log") or hi / lo >= 50))})
        else:
            dims.append({"k": k, "kind": "cat", "choices": list(vals)})
    return dims


def _label(v):
    return "None" if v is None else ",".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)


def _search_optuna(dims, evaluate, n, timeout, seed, sampler_name, report):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "qmc":
        sampler = optuna.samplers.QMCSampler(seed=seed, warn_independent_sampling=False)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(5, max(2, n // 4)), multivariate=True, warn_independent_sampling=False)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    labels = {d["k"]: {} for d in dims if d["kind"] == "cat"}
    for d in dims:
        if d["kind"] == "cat":
            for j, c in enumerate(d["choices"]):
                lab = _label(c)
                labels[d["k"]][lab if lab not in labels[d["k"]] else f"{lab}#{j}"] = c
    out = []

    def objective(trial):
        params = {}
        for d in dims:
            if d["kind"] == "cat":
                params[d["k"]] = labels[d["k"]][trial.suggest_categorical(d["k"], list(labels[d["k"]]))]
            elif d["kind"] == "int":
                params[d["k"]] = trial.suggest_int(d["k"], int(d["lo"]), int(d["hi"]), log=d["log"])
            else:
                params[d["k"]] = trial.suggest_float(d["k"], float(d["lo"]), float(d["hi"]), log=d["log"])
        rec = evaluate(params)
        out.append(rec)
        report(len(out))
        return rec["score_raw"] if rec["score_raw"] is not None else -1e12

    study.optimize(objective, n_trials=n, timeout=timeout or None, catch=(Exception,))
    return out


def _search_gp(dims, evaluate, n, timeout, seed, report):
    """Built-in Bayesian optimisation: Gaussian-process surrogate + expected improvement."""
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    rng = np.random.RandomState(seed)
    nd = len(dims)

    def decode(u):
        params = {}
        for j, d in enumerate(dims):
            x = float(u[j])
            if d["kind"] == "cat":
                params[d["k"]] = d["choices"][min(len(d["choices"]) - 1, int(x * len(d["choices"])))]
            else:
                v = math.exp(math.log(d["lo"]) + x * (math.log(d["hi"]) - math.log(d["lo"]))) if d["log"] else d["lo"] + x * (d["hi"] - d["lo"])
                params[d["k"]] = int(round(v)) if d["kind"] == "int" else float(v)
        return params

    U, Y, out = [], [], []
    n_init = min(n, max(3, min(6, n // 3)))
    t0 = time.time()
    for it in range(n):
        if timeout and time.time() - t0 > timeout:
            break
        if it < n_init or len(Y) < 3:
            u = rng.rand(nd)
        else:
            gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * Matern(length_scale=np.full(nd, 0.3), length_scale_bounds=(1e-2, 10), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1)),
                                          normalize_y=True, random_state=seed, n_restarts_optimizer=2)
            gp.fit(np.array(U), np.array(Y))
            cand = rng.rand(2000, nd)
            mu, sd = gp.predict(cand, return_std=True)
            imp = mu - max(Y) - 0.01 * (np.std(Y) or 1e-3)
            z = imp / np.maximum(sd, 1e-9)
            u = cand[int(np.argmax(imp * norm.cdf(z) + sd * norm.pdf(z)))]
        rec = evaluate(decode(u))
        out.append(rec)
        U.append(u)
        Y.append(rec["score_raw"] if rec["score_raw"] is not None else (min(Y) if Y else 0) - 1)
        report(len(out))
    return out


def _param_importance(trials, keys):
    ok = [t for t in trials if t["score_raw"] is not None]
    if len(ok) < 6 or not keys:
        return None
    cols = []
    for k in keys:
        vals = [t["params"].get(k) for t in ok]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            cols.append(np.array(vals, float))
        else:
            cols.append(pd.Categorical([_label(v) for v in vals]).codes.astype(float))
    y = np.array([t["score_raw"] for t in ok])
    if np.std(y) == 0:
        return None
    rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=N_JOBS).fit(np.column_stack(cols), y)
    return sorted(({"param": k, "importance": float(v)} for k, v in zip(keys, rf.feature_importances_)), key=lambda x: -x["importance"])


def tune_models(cfg, T, specs, D):
    results, primary, task, K = S["results"], D["primary"], D["task"], D["K"]
    ranked = [r for r in _ranked(results, primary) if not r.get("ensemble") and not r.get("tuned_from")]
    which = T.get("models", "best")
    if isinstance(which, list):
        ids = [m for m in which if m in results and results[m]["status"] == "ok"]
    else:
        n = {"best": 1, "top3": 3, "top5": 5, "all": len(ranked)}.get(which, 1)
        ids = [r["id"] for r in ranked[:n]]
    spec_by = {s["id"]: s for s in specs}
    method = T.get("method", "optuna")
    n_iter = int(T.get("n_iter", 20))
    folds = int(T.get("cv_folds", 3))
    seed = int(cfg["split"].get("seed", 42))
    cv = make_cv(cfg, task, D["y_train"], D["g_train"], folds=folds) or KFold(folds, shuffle=True, random_state=0)
    sc = D["scorers"][primary]
    sign = -1 if primary in LOWER_BETTER else 1
    grids = T.get("grids") or {}
    for i, mid in enumerate(ids):
        spec = spec_by.get(mid)
        grid = (grids.get(mid) or grids.get(spec["key"]) if spec else None) or (spec or {}).get("grid")
        if not spec or not grid:
            log(f"No search space for {results[mid]['name']}; skipped tuning.", "warn")
            continue
        progress("tune", f"Tuning {spec['name']}", i + 1, len(ids), model=mid)
        base = build_pipeline(cfg, spec, task, K)
        prefix = "model__estimator__" if isinstance(base.steps[-1][1], ResampledClassifier) else "model__"
        valid = base.steps[-1][1].get_params(deep=True) if prefix == "model__" else base.steps[-1][1].estimator.get_params()
        raw = {k: v for k, v in grid.items() if k in valid and ((isinstance(v, list) and v) or (isinstance(v, dict) and ("low" in v or "min" in v)))}
        if not raw:
            log(f"The search space for {spec['name']} has no valid parameters; skipped.", "warn")
            continue
        Xtr, ytr, gtr = _fit_rows(spec, D["X_train"], D["y_train"], cfg, D["g_train"])
        m = method
        note = None
        t0 = time.time()

        def evaluate(params):
            pipe = clone(base).set_params(**{prefix + k: _convert_param(v) for k, v in params.items()})
            ts = time.time()
            s = cross_val_score(pipe, Xtr, ytr, cv=cv, groups=gtr, scoring=sc, error_score=np.nan, n_jobs=N_JOBS)
            ok = not np.all(np.isnan(s))
            mean = float(np.nanmean(s)) if ok else None
            return {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}, "raw": params, "score_raw": mean,
                    "mean": None if mean is None else sign * mean, "std": float(np.nanstd(s)) if ok else None, "folds": list(sign * s),
                    "fit_time": (time.time() - ts) / max(1, len(s))}

        def report(k, total=n_iter, name=spec["name"], idx=i):
            progress("tune", f"Tuning {name}: trial {k}/{total}", idx + 1, len(ids), model=mid)

        try:
            if m == "optuna":
                try:
                    import optuna  # noqa: F401
                except ImportError:
                    m, note = "bayesian", "Optuna is not installed here, so the built-in Gaussian-process Bayesian optimisation was used."
                    log(note, "warn")
            if m == "optuna":
                trials = _search_optuna(_dims(raw, spec.get("space")), evaluate, n_iter, float(T.get("timeout", 0) or 0), seed, T.get("sampler", "tpe"), report)
            elif m == "bayesian":
                trials = _search_gp(_dims(raw, spec.get("space")), evaluate, n_iter, float(T.get("timeout", 0) or 0), seed, report)
            else:
                g = {prefix + k: [_convert_param(v) for v in vals] for k, vals in raw.items() if isinstance(vals, list)}
                if len(g) < len(raw):
                    log(f"Ranges like {{low, high}} need Optuna or Bayesian search; {spec['name']} uses only the list parameters.", "warn")
                combos = int(np.prod([len(v) for v in g.values()]))
                if m == "grid" and combos > int(T.get("max_grid", 60)):
                    log(f"The grid for {spec['name']} has {combos} combinations; using random search with {n_iter} instead.", "warn")
                    m = "random"
                if m == "grid":
                    search = GridSearchCV(base, g, scoring=sc, cv=cv, n_jobs=N_JOBS, error_score=np.nan)
                elif m == "halving_grid":
                    search = HalvingGridSearchCV(base, g, scoring=sc, cv=cv, factor=3, random_state=0, n_jobs=N_JOBS, error_score=np.nan)
                elif m == "halving_random":
                    search = HalvingRandomSearchCV(base, g, scoring=sc, cv=cv, factor=3, random_state=0, n_jobs=N_JOBS, error_score=np.nan)
                else:
                    search = RandomizedSearchCV(base, g, n_iter=min(n_iter, combos), scoring=sc, cv=cv, random_state=0, n_jobs=N_JOBS, error_score=np.nan)
                search.fit(Xtr, ytr, **({"groups": gtr} if gtr is not None else {}))
                cvr = search.cv_results_
                nsplit = sum(1 for k in cvr if k.startswith("split") and k.endswith("_test_score"))
                trials = []
                for j in range(len(cvr["params"])):
                    params = {k.replace(prefix, ""): v for k, v in cvr["params"][j].items()}
                    mean = cvr["mean_test_score"][j]
                    ok = np.isfinite(mean)
                    trials.append({"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}, "raw": params,
                                   "score_raw": float(mean) if ok else None, "mean": sign * float(mean) if ok else None, "std": float(cvr["std_test_score"][j]) if ok else None,
                                   "folds": [sign * cvr[f"split{s}_test_score"][j] for s in range(nsplit)], "fit_time": float(cvr["mean_fit_time"][j])})
        except Exception as e:  # noqa: BLE001
            log(f"Tuning {spec['name']} failed: {type(e).__name__}: {e}", "error")
            continue
        good = [t for t in trials if t["score_raw"] is not None]
        if not good:
            log(f"Every tuning trial for {spec['name']} failed.", "error")
            continue
        for j, t in enumerate(trials):
            t["number"] = j
        order = sorted(trials, key=lambda t: -t["score_raw"] if t["score_raw"] is not None else math.inf)
        for rank, t in enumerate(order, 1):
            t["rank"] = rank
        best_t = order[0]
        history, best_raw = [], -math.inf
        for t in trials:
            if t["score_raw"] is not None:
                best_raw = max(best_raw, t["score_raw"])
            history.append({"number": t["number"], "value": t["mean"], "best": sign * best_raw if best_raw > -math.inf else None})
        keys = list(raw)
        importance = _param_importance(trials, keys)
        folds_scores = np.array(best_t["folds"], float)
        best_params = best_t["raw"]
        tid = f"{mid}__tuned"
        merged = {**(spec.get("params") or {}), **{k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()}}
        tspec = {**spec, "id": tid, "params": merged}
        try:
            r, pipe = train_one(tid, f"{spec['name']} (tuned)", spec["key"], lambda ts=tspec: build_pipeline(cfg, ts, task, K), cfg, D,
                                cv_override={primary: {"mean": float(np.nanmean(folds_scores)), "std": float(np.nanstd(folds_scores)), "folds": folds_scores}},
                                family=spec.get("family", ""), params=merged, spec=tspec)
        except Exception as e:  # noqa: BLE001
            log(f"Refitting tuned {spec['name']} failed: {e}", "error")
            continue
        r["tuned_from"] = mid
        grid_keys = {k: (v if isinstance(v, dict) else [(list(x) if isinstance(x, tuple) else x) for x in v]) for k, v in raw.items()}
        r["tuning"] = {"method": m, "sampler": T.get("sampler", "tpe") if m == "optuna" else None, "note": note,
                       "trials": [{k: t[k] for k in ("number", "params", "mean", "std", "rank", "fit_time")} for t in order[:150]],
                       "n_trials": len(trials), "n_failed": len(trials) - len(good), "best_params": best_t["params"], "time": time.time() - t0,
                       "before": results[mid].get("score"), "after": r["score"], "folds": len(folds_scores), "grid": grid_keys,
                       "history": history, "importance": importance}
        results[tid] = r
        S["models"][tid] = pipe
        spec_by[tid] = tspec
        label = {"optuna": "Optuna " + (T.get("sampler", "tpe")).upper(), "bayesian": "Bayesian (GP)"}.get(m, m)
        log(f"Tuned {spec['name']} with {label} search ({len(trials)} trials): CV {primary} {_fmt(results[mid].get('score'))} → {_fmt(r['score'])}. Best: {best_t['params']}.")


def _ranked(results, primary):
    ok = [r for r in results.values() if r.get("status") == "ok" and not r.get("baseline") and r.get("score") is not None]
    return sorted(ok, key=lambda r: r["score"], reverse=primary not in LOWER_BETTER)


def _fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def summary():
    D = S["D"]
    res = S["results"]
    primary = D["primary"]
    ranked = _ranked(res, primary)
    best = S.get("best") or (ranked[0]["id"] if ranked else None)
    names_in = list(S["frame_spec"]["features"])
    try:
        processed = processed_names(S["models"][best]) if best in S["models"] else []
    except Exception:
        processed = []
    return {"task": D["task"], "classes": D["classes"], "primary": primary, "lower_is_better": primary in LOWER_BETTER,
            "best": best, "ranking": [r["id"] for r in ranked], "models": res,
            "sizes": {"train": len(D["idx_train"]), "val": len(D["idx_val"]), "test": len(D["idx_test"])},
            "cv": None if D["cv"] is None else {"name": type(D["cv"]).__name__, "splits": D["cv"].get_n_splits(D["X_train"], D["y_train"], D["g_train"])},
            "features": {"input": names_in, "processed": processed, "dropped": S["frame_spec"]["dropped"], "custom": S["frame_spec"]["custom"]},
            "pipeline_text": _pipeline_text(best), "duration": S.get("duration"), "log": S.get("log", [])}


def _pipeline_text(mid):
    try:
        from sklearn.utils._estimator_html_repr import estimator_html_repr  # noqa: F401
    except Exception:
        pass
    try:
        return str(S["models"][mid])
    except Exception:
        return ""


# ------------------------------------------------------------- explainability
def _test_raw(n=None, seed=0):
    D = S["D"]
    X, y = D["X_test"], D["y_test"]
    if n and len(X) > n:
        ix = sample_idx(len(X), n, seed)
        return X.iloc[ix].reset_index(drop=True), y[ix], ix
    return X, y, np.arange(len(X))


def _permutation(mid, repeats=5):
    D = S["D"]
    X, y, _ = _test_raw(1500)
    res = permutation_importance(S["models"][mid], X, y, scoring=D["scorers"][D["primary"]], n_repeats=repeats, random_state=0, n_jobs=N_JOBS)
    order = np.argsort(-res.importances_mean)
    return [{"feature": X.columns[i], "mean": res.importances_mean[i], "std": res.importances_std[i]} for i in order]


def _out_fn(mid, cls):
    pipe = S["models"][mid]
    if S["D"]["task"] == "classification":
        def f(X):
            sc, _ = get_scores(pipe, X, S["D"]["K"])
            if sc is None:
                return (pipe.predict(X) == cls).astype(float)
            return sc[:, cls]
        return f
    return lambda X: np.asarray(pipe.predict(X), float)


def _shap(mid, n_rows=40, cls=None, rows=None, perms=10, bg_n=25):
    """Model-agnostic SHAP values by permutation sampling on the original input columns."""
    D = S["D"]
    task, K = D["task"], D["K"]
    if task == "classification" and cls is None:
        cls = 1 if K == 2 else int(np.bincount(D["y_train"]).argmax())
    f = _out_fn(mid, cls)
    Xtr = D["X_train"]
    bg = Xtr.iloc[sample_idx(len(Xtr), bg_n, 11)].reset_index(drop=True)
    Xte = D["X_test"]
    ix = np.asarray(rows if rows is not None else sample_idx(len(Xte), n_rows, 5), dtype=int)
    X = Xte.iloc[ix].reset_index(drop=True)
    cols = list(X.columns)
    F = len(cols)
    rng = np.random.RandomState(0)
    phi = np.zeros((len(X), F))
    fx = np.zeros(len(X))
    for r in range(len(X)):
        batch, orders = [], []
        for p in range(perms):
            z = bg.iloc[rng.randint(len(bg))].copy()
            batch.append(z.copy())
            order = rng.permutation(F)
            orders.append(order)
            for j in order:
                z.iloc[j] = X.iloc[r, j]
                batch.append(z.copy())
        out = f(pd.DataFrame(batch, columns=cols).astype(X.dtypes.to_dict(), errors="ignore"))
        k = 0
        for p in range(perms):
            prev = out[k]
            k += 1
            for j in orders[p]:
                cur = out[k]
                phi[r, j] += cur - prev
                prev = cur
                k += 1
        phi[r] /= perms
        fx[r] = out[-1]
    base = float(np.mean(f(bg)))
    gl = np.abs(phi).mean(0)
    order = np.argsort(-gl)
    values = []
    for j in range(F):
        col = X.iloc[:, j]
        if pd.api.types.is_numeric_dtype(col):
            v = col.astype(float).values
            lo, hi = np.nanmin(v), np.nanmax(v)
            values.append(((v - lo) / (hi - lo)) if hi > lo else np.zeros(len(v)))
        else:
            codes = pd.Categorical(col.astype(str)).codes.astype(float)
            values.append(codes / max(1, codes.max()))
    return {"class": cls, "base": base, "features": cols, "order": order, "global": gl, "phi": phi, "fx": fx, "rows": ix,
            "norm_values": np.array(values).T, "raw_values": X.astype(str).values}


def _learning_curve(mid):
    D = S["D"]
    pipe = S["models"][mid]
    X, y = D["X_train"], D["y_train"]
    if len(X) > 4000:
        ix = sample_idx(len(X), 4000, 4)
        X, y = X.iloc[ix], y[ix]
    cv = make_cv(S["cfg"], D["task"], y, None, folds=3) or KFold(3, shuffle=True, random_state=0)
    sizes, tr, te = learning_curve(clone(pipe), X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 6), scoring=D["scorers"][D["primary"]], n_jobs=N_JOBS, error_score=np.nan)
    sign = -1 if D["primary"] in LOWER_BETTER else 1
    return {"sizes": sizes, "train_mean": sign * tr.mean(1), "train_std": tr.std(1), "val_mean": sign * te.mean(1), "val_std": te.std(1)}


def _pdp(mid, feature, cls=None, grid=20):
    D = S["D"]
    task, K = D["task"], D["K"]
    if task == "classification" and cls is None:
        cls = 1 if K == 2 else 0
    f = _out_fn(mid, cls)
    X, _, _ = _test_raw(400)
    col = X[feature]
    if pd.api.types.is_numeric_dtype(col):
        vals = np.unique(np.nanquantile(col.astype(float), np.linspace(0.02, 0.98, grid)))
        kind = "numeric"
    else:
        vals = col.astype(str).value_counts().head(15).index.tolist()
        kind = "categorical"
    means, ice = [], []
    ice_ix = sample_idx(len(X), 30, 9)
    for v in vals:
        Z = X.copy()
        Z[feature] = v
        out = f(Z)
        means.append(float(np.mean(out)))
        ice.append(out[ice_ix])
    return {"feature": feature, "kind": kind, "values": vals, "mean": means, "ice": np.array(ice).T, "class": cls}


@safe
def analyze(kind, model_id, params_json="{}"):
    p = json.loads(params_json or "{}")
    if "D" not in S:
        raise ValueError("Run the pipeline first.")
    if model_id not in S["models"]:
        raise ValueError(f"Unknown model '{model_id}'. Available: {', '.join(S['models'])}")
    if kind == "shap":
        out = _shap(model_id, int(p.get("rows", 40)), p.get("cls"))
    elif kind == "local_shap":
        out = _shap(model_id, 1, p.get("cls"), rows=[int(p.get("row", 0))], perms=int(p.get("perms", 40)))
    elif kind == "learning_curve":
        out = _learning_curve(model_id)
    elif kind == "pdp":
        out = _pdp(model_id, p["feature"], p.get("cls"))
    elif kind == "permutation":
        out = _permutation(model_id, int(p.get("repeats", 5)))
    else:
        raise ValueError(f"Unknown analysis '{kind}'")
    if kind in ("shap", "learning_curve", "permutation"):
        S["results"][model_id][{"shap": "shap", "learning_curve": "learning_curve", "permutation": "permutation"}[kind]] = out
    return dumps(out)


# ------------------------------------------------------------ predictions
def _decode(pred):
    cl = S["D"]["classes"]
    return [cl[int(v)] for v in pred] if cl else [float(v) for v in pred]


def _predict_frame(mid, raw):
    X = apply_frame(raw, S["frame_spec"])
    pipe = S["models"][mid]
    pred = pipe.predict(X)
    out = {"prediction": _decode(pred)}
    if S["D"]["task"] == "classification":
        sc, _ = get_scores(pipe, X, S["D"]["K"])
        if sc is not None:
            out["probabilities"] = [{c: float(sc[i, j]) for j, c in enumerate(S["D"]["classes"])} for i in range(len(X))]
    return out


@safe
def predict(model_id, rows_json):
    rows = json.loads(rows_json)
    if isinstance(rows, dict):
        rows = [rows]
    raw = pd.DataFrame(rows)
    for c in S["df"].columns:
        if c not in raw.columns:
            raw[c] = np.nan
    return dumps(_predict_frame(model_id or S["best"], raw))


@safe
def predict_csv(model_id, text):
    raw = _read_csv_text(text)
    out = _predict_frame(model_id or S["best"], raw)
    raw["prediction"] = out["prediction"]
    for k in (out.get("probabilities") or [{}])[0].keys():
        raw[f"prob_{k}"] = [p[k] for p in out["probabilities"]]
    return dumps({"csv": raw.to_csv(index=False), "rows": len(raw)})


@safe
def test_rows(start=0, n=20):
    D = S["D"]
    raw = S["frame"].iloc[D["idx_test"]].reset_index(drop=True)
    part = raw.iloc[start:start + n]
    return dumps({"columns": list(raw.columns), "rows": part.astype(object).where(part.notna(), None).values.tolist(), "total": len(raw)})


@safe
def what_if(model_id, row, changes_json):
    D = S["D"]
    raw = S["frame"].iloc[D["idx_test"]].reset_index(drop=True).iloc[[int(row)]].copy()
    before = _predict_frame(model_id, raw)
    ch = json.loads(changes_json)
    for k, v in ch.items():
        if k not in raw.columns:
            raise ValueError(f"Unknown column '{k}'")
        raw[k] = v
    after = _predict_frame(model_id, raw)
    return dumps({"before": before, "after": after, "changes": ch})


@safe
def test_predictions(model_id):
    D = S["D"]
    raw = S["frame"].iloc[D["idx_test"]].reset_index(drop=True)
    out = _predict_frame(model_id, raw)
    raw["actual"] = _decode(D["y_test"])
    raw["prediction"] = out["prediction"]
    for k in (out.get("probabilities") or [{}])[0].keys():
        raw[f"prob_{k}"] = [p[k] for p in out["probabilities"]]
    return dumps({"csv": raw.to_csv(index=False)})


# -------------------------------------------------------------- code runner
def _register(name, estimator, cv=True):
    """Fit a new estimator (wrapped in the current preprocessing) and add it to the leaderboard."""
    D, cfg = S["D"], S["cfg"]
    pipe = estimator if isinstance(estimator, Pipeline) else Pipeline([("prep", build_preprocessor(cfg, D["task"]))] + build_fs_steps(cfg, D["task"], D["K"]) + [("model", estimator)])
    mid = "custom_" + "".join(ch if ch.isalnum() else "_" for ch in name.lower())[:40]
    cvD = dict(D)
    if not cv:
        cvD["cv"] = None
    r, fitted = train_one(mid, name, "custom", lambda: clone(pipe), cfg, cvD, family="Custom", params={k: str(v) for k, v in pipe.steps[-1][1].get_params(deep=False).items()})
    S["results"][mid] = r
    S["models"][mid] = fitted
    S["_changed"] = True
    return {k: r.get(k) for k in ("id", "name", "score", "score_source")} | {"test": r["test"]}


@safe
def run_code(code):
    ns = S.setdefault("_ns", {})
    D = S.get("D") or {}
    ns.update({"np": np, "pd": pd, "S": S, "df": S.get("df"), "models": S.get("models", {}), "results": S.get("results", {}),
               "X_train": D.get("X_train"), "X_val": D.get("X_val"), "X_test": D.get("X_test"),
               "y_train": D.get("y_train"), "y_val": D.get("y_val"), "y_test": D.get("y_test"),
               "classes": D.get("classes"), "task": D.get("task"), "best_model_id": S.get("best"),
               "register_model": _register, "make_preprocessor": lambda: build_preprocessor(S["cfg"], D["task"]),
               "predict": lambda mid, rows: json.loads(predict(mid, json.dumps(rows))), "compute_metrics": lambda y, p: compute_metrics(D["task"], y, p, K=D.get("K", 2))})
    S["_changed"] = False
    buf = io.StringIO()
    result = None
    figs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.close("all")
        ns["plt"] = plt
    except Exception:
        plt = None
    err = None
    info = None
    try:
        tree = ast.parse(code, mode="exec")
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(tree, "<assistant>", "exec"), ns)
            if last is not None:
                result = eval(compile(ast.Expression(last.value), "<assistant>", "eval"), ns)
    except Exception as e:  # noqa: BLE001
        err = traceback.format_exc(limit=4)[-2500:]
        info = _explain_exception(e, code, ns)
    if plt is not None:
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
            figs.append(base64.b64encode(b.getvalue()).decode())
        plt.close("all")
    table = None
    if isinstance(result, pd.Series):
        result = result.to_frame()
    if isinstance(result, pd.DataFrame):
        t = result.head(100)
        table = {"columns": [str(c) for c in t.columns], "index": [str(i) for i in t.index], "rows": t.astype(object).where(t.notna(), None).values.tolist()}
        result_text = f"DataFrame {result.shape[0]}×{result.shape[1]}"
    else:
        result_text = None if result is None else repr(result)[:4000]
    return dumps({"stdout": buf.getvalue()[-8000:], "result": result_text, "table": table, "figures": figs, "exception": err, "error": None,
                  "error_info": info, "warnings": lint_code(code), "results_changed": bool(S.get("_changed"))})


def _explain_exception(e, code, ns):
    """Locate the failing line and give a concrete hint, so the AI (or user) can fix the code."""
    tb = traceback.extract_tb(e.__traceback__)
    line = next((f.lineno for f in reversed(tb) if f.filename == "<assistant>"), None)
    if isinstance(e, SyntaxError):
        line = e.lineno
    lines = code.splitlines()
    msg = str(e)
    names = sorted(k for k in ns if not k.startswith("_") and k not in ("__builtins__",))
    cols = list(S["df"].columns) if S.get("df") is not None else []
    feats = list(S["frame_spec"]["features"]) if S.get("frame_spec") else []
    hint = "Read the message and the failing line, fix the root cause and run again."
    if "NoneType" in msg and "D" not in S:
        hint = "X_train, y_train, models and results only exist after a pipeline run. Run the pipeline first (or use `df`, the raw data)."
    elif isinstance(e, NameError):
        hint = f"That name is not defined. Available names: {', '.join(names[:60])}."
    elif isinstance(e, KeyError):
        hint = f"Key {msg} not found. Data columns: {', '.join(cols[:80])}. Model ids: {', '.join(list(S.get('models', {}))[:40])}."
    elif isinstance(e, (ModuleNotFoundError, ImportError)):
        hint = "That package is not available in the browser engine. Use numpy, pandas, scipy, scikit-learn, xgboost, lightgbm, matplotlib or optuna."
    elif isinstance(e, SyntaxError):
        hint = "Python syntax error: check brackets, quotes, colons and indentation on that line."
    elif "could not convert string to float" in msg or "dtype('O')" in msg:
        hint = "Raw data has text columns. Use a fitted pipeline from `models[...]` (it encodes categoricals), or wrap your estimator with register_model() / make_preprocessor()."
    elif "is not fitted" in msg:
        hint = "Fit the estimator on X_train, y_train first, or use an already fitted pipeline from `models`."
    elif "features" in msg and ("expecting" in msg or "has" in msg):
        hint = f"Feature mismatch. Fitted pipelines in `models` expect the raw input columns: {', '.join(feats[:60])}."
    elif "Found input variables with inconsistent numbers of samples" in msg:
        hint = "X and y have different lengths; pair X_train with y_train, X_val with y_val and X_test with y_test."
    elif "Unknown label type" in msg or "continuous" in msg:
        hint = "Wrong target type for this estimator: use a classifier for class labels and a regressor for continuous targets."
    return {"type": type(e).__name__, "message": msg[:600], "line": line, "code_line": lines[line - 1].strip() if line and 0 < line <= len(lines) else None, "hint": hint}


LINT_RULES = [
    (r"\.fit(_transform)?\(\s*X_test", "Fits on the test set. That leaks test data into training; fit on X_train only."),
    (r"\.fit(_transform)?\([^)]*\by_test\b", "Uses y_test for fitting. The test labels must only be used for the final evaluation."),
    (r"\.fit(_transform)?\(\s*df\b", "Fits on the full dataset (df) before splitting. Preprocessing learned on all rows leaks information; fit on X_train or use a Pipeline."),
    (r"\.fit(_transform)?\(\s*X_val", "Fits on the validation set; keep it for model selection only."),
]


def lint_code(code):
    """Cheap static review for common conceptual ML mistakes in user / AI code."""
    import re
    out = []
    for pat, msg in LINT_RULES:
        if re.search(pat, code):
            out.append(msg)
    task = (S.get("D") or {}).get("task")
    if task == "regression" and re.search(r"\b(accuracy_score|f1_score|roc_auc_score|confusion_matrix)\b", code):
        out.append("Uses classification metrics on a regression task.")
    if task == "classification" and re.search(r"\b(r2_score|mean_squared_error|mean_absolute_error)\b", code):
        out.append("Uses regression metrics on a classification task.")
    if re.search(r"\bfor\b[\s\S]*X_test[\s\S]*\b(max|argmax|sort|sorted|best)\b", code) and "X_val" not in code and "cross_val" not in code:
        out.append("Looks like models are compared or tuned on the test set. Selecting by test score overfits it; use cross-validation on X_train or X_val.")
    if re.search(r"\.predict\(\s*X_train\s*\)", code) and re.search(r"(score|accuracy|r2|error)", code) and "X_test" not in code and "X_val" not in code:
        out.append("Evaluates only on training data, which is optimistic; also report X_val / X_test scores.")
    return out


@safe
def preview(start=0, n=50):
    df = S["df"]
    part = df.iloc[int(start):int(start) + int(n)]
    return dumps({"columns": list(df.columns), "rows": part.astype(object).where(part.notna(), None).values.tolist(), "total": len(df)})


@safe
def frame_csv():
    return dumps({"csv": S["df"].to_csv(index=False), "name": S.get("name")})


@safe
def get_profile():
    return dumps({"profile": profile(S["df"]), "name": S.get("name")})


PREP_KEYS = ("dataset", "split", "cv", "preprocess", "fe", "fs", "advanced")


@safe
def retrain_models(config_json, ids_json):
    """Retrain only some models on the existing splits (fast fix-and-verify loop)."""
    if "D" not in S:
        raise ValueError("Run the full pipeline first.")
    cfg, ids = json.loads(config_json), json.loads(ids_json)
    old = S["cfg"]
    changed = [k for k in PREP_KEYS if json.dumps(old.get(k), sort_keys=True) != json.dumps(cfg.get(k), sort_keys=True)]
    if changed:
        raise ValueError(f"These settings changed since the last run: {', '.join(changed)}. Run the full pipeline instead.")
    D, results = S["D"], S["results"]
    specs = {s["id"]: s for s in cfg.get("models") or []}
    S["cfg"] = {**old, "models": cfg.get("models"), "eval": cfg.get("eval", old.get("eval"))}
    report = []
    for mid in ids:
        spec = specs.get(mid)
        if not spec:
            report.append({"id": mid, "status": "error", "error": "No such model in the current pipeline."})
            continue
        if D["task"] not in (spec.get("tasks") or [D["task"]]):
            report.append({"id": mid, "status": "error", "error": f"{spec['name']} does not support {D['task']}."})
            continue
        progress("model", f"Retraining {spec['name']}", ids.index(mid) + 1, len(ids), model=mid)
        try:
            r, pipe = train_one(mid, spec["name"], spec["key"], lambda s=spec: build_pipeline(S["cfg"], s, D["task"], D["K"]), S["cfg"], D, family=spec.get("family", ""), params=spec.get("params"), spec=spec)
            results[mid] = r
            S["models"][mid] = pipe
            if (S["cfg"].get("eval") or {}).get("permutation", "none") != "none":
                try:
                    r["permutation"] = _permutation(mid, 3)
                except Exception:
                    pass
            report.append({"id": mid, "name": spec["name"], "status": "ok", "score": r["score"], "test": r["test"].get(D["primary"])})
            log(f"Retrained {spec['name']}: {D['primary']} {_fmt(r['score'])}.")
        except Exception as e:  # noqa: BLE001
            results[mid] = {"id": mid, "name": spec["name"], "key": spec["key"], "family": spec.get("family", ""), "status": "error", "error": f"{type(e).__name__}: {e}"}
            report.append({"id": mid, "name": spec["name"], "status": "error", "error": f"{type(e).__name__}: {e}"})
            log(f"Retraining {spec['name']} failed: {e}", "error")
    ranked = _ranked(results, D["primary"])
    if ranked:
        S["best"] = ranked[0]["id"]
    out = summary()
    out["retrained"] = report
    return dumps(out)


# ------------------------------------------------------------ pipeline doctor
SCALE_SENSITIVE = {"knn", "svc", "nusvc", "linear_svc", "svr", "nusvr", "linear_svr", "mlp", "logreg", "sgd_clf", "sgd_reg", "perceptron", "pa_clf", "pa_reg",
                   "kernel_ridge", "gp_clf", "gp_reg", "ridge_clf", "ridge", "lasso", "elasticnet", "huber", "nearest_centroid", "lda", "qda"}
OVERFIT_FIX = {"rf": {"min_samples_leaf": 5, "max_depth": 12}, "et": {"min_samples_leaf": 5, "max_depth": 12}, "dt": {"max_depth": 6, "min_samples_leaf": 10},
               "extra_tree": {"max_depth": 6, "min_samples_leaf": 10}, "gb": {"learning_rate": 0.05, "max_depth": 2, "subsample": 0.8},
               "hgb": {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0}, "xgb": {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5},
               "lgbm": {"num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 40, "reg_lambda": 5}, "mlp": {"alpha": 0.01, "early_stopping": True},
               "knn": {"n_neighbors": 15}, "svc": {"C": 0.3}, "svr": {"C": 0.3}, "logreg": {"C": 0.3}, "bagging": {"max_samples": 0.7}, "adaboost": {"learning_rate": 0.1}}
CLS_ONLY_METRICS = {"accuracy", "balanced_accuracy", "f1", "f1_weighted", "precision", "recall", "roc_auc", "average_precision", "mcc", "log_loss"}
REG_ONLY_METRICS = {"r2", "rmse", "mae", "mape", "medae", "explained_variance"}


def _error_hint(msg):
    m = msg.lower()
    if "not available" in m or "needs the" in m:
        return "The library did not load in this engine. Remove the block or use Hist Gradient Boosting instead."
    if "nan" in m and ("input" in m or "contain" in m):
        return "The model received missing values. Choose an imputation strategy in Preprocessing."
    if "negative" in m:
        return "This model needs non-negative inputs. Use MinMax scaling or another model."
    if "solver" in m or "penalty" in m:
        return "Incompatible solver/penalty. Use solver lbfgs with the l2 penalty, or saga for l1/elasticnet."
    if "n_neighbors" in m or "n_samples_fit" in m:
        return "n_neighbors is larger than the number of training rows; lower it."
    if "least populated class" in m or "only one class" in m or "n_splits" in m:
        return "A class is too small for this split / CV setting. Use fewer folds or disable stratification."
    if "converge" in m:
        return "The optimiser did not converge. Increase max_iter or scale the features."
    if "memory" in m:
        return "Out of memory. Lower the training rows in Settings or use a lighter model."
    if "__sklearn_tags__" in m:
        return "Library version clash (patched automatically); retrain the model."
    return "Read the error, change the hyperparameters in the model block and retrain it."


@safe
def diagnose(config_json):
    """Check the pipeline for code-level and conceptual ML problems. Every issue can carry machine-applicable fixes."""
    cfg = json.loads(config_json)
    issues = []

    def add(sev, code, title, detail, fix=None, model=None):
        issues.append({"id": f"{code}{'_' + str(model) if model else ''}", "severity": sev, "code": code, "title": title, "detail": detail, "fix": fix, "model": model})

    df = S.get("df")
    if df is None:
        add("error", "no_data", "No dataset loaded", "Upload a file or load a sample dataset.")
        return dumps({"issues": issues})
    D = cfg.get("dataset") or {}
    target = D.get("target")
    if not target or target not in df.columns:
        add("error", "no_target", "Target column is missing", f"Choose a target in the Dataset block. Columns: {', '.join(map(str, df.columns[:30]))}.", [{"block": "dataset", "settings": {"target": profile(df)["target_guess"]}}])
        return dumps({"issues": issues})
    n = len(df)
    y = df[target]
    exclude = set(D.get("exclude") or [])
    feats = [c for c in df.columns if c != target and c not in exclude and c != D.get("group_col")]
    ykind = col_kind(y)
    yv = pd.to_numeric(y, errors="coerce")
    nun = int(y.nunique())
    task = D.get("task", "auto")
    resolved = task if task != "auto" else ("classification" if ykind != "numeric" or (nun <= 20 and np.all(np.mod(yv.dropna(), 1) == 0)) else "regression")
    if y.isna().any():
        add("info", "target_missing", f"{int(y.isna().sum())} rows have no target", "They are dropped before training.")
    if task == "regression" and ykind != "numeric":
        add("error", "task_mismatch", "Regression on a non-numeric target", f"'{target}' contains text labels, so it must be classification.", [{"block": "dataset", "settings": {"task": "classification"}}])
    if task == "classification" and ykind == "numeric" and nun > 50:
        add("error", "task_mismatch", "Classification on a continuous target", f"'{target}' has {nun} distinct numeric values; that is a regression problem.", [{"block": "dataset", "settings": {"task": "regression"}}])
    if task == "auto" and resolved == "classification" and ykind == "numeric" and nun > 5:
        add("info", "task_auto", f"'{target}' is treated as classification", f"It has only {nun} integer values. If it is a count or a score, set the task to regression.", [{"block": "dataset", "settings": {"task": "regression"}}])
    if resolved == "classification" and nun > 50:
        add("error", "too_many_classes", "Too many classes", f"The target has {nun} classes (max 50).", [{"block": "dataset", "settings": {"task": "regression"}}] if ykind == "numeric" else None)
    # ---- leakage
    for f in (cfg.get("fe") or {}).get("custom_features") or []:
        import re
        if re.search(r"(?<![\w])" + re.escape(str(target)) + r"(?![\w])", str(f.get("expr", ""))):
            add("error", "leak_custom", f"Custom feature '{f.get('name')}' uses the target", f"'{f.get('expr')}' is computed from '{target}', so the model would see the answer (target leakage).", [{"block": "fe", "remove_feature": f.get("name")}])
    prof = {c["name"]: c for c in profile(df)["cols"]}
    samp = df.iloc[sample_idx(n, 20000, 13)]
    ys = samp[target]
    ycode = pd.to_numeric(ys, errors="coerce") if resolved == "regression" else pd.Series(pd.Categorical(ys.astype(str)).codes, index=ys.index).astype(float)
    binary = resolved == "classification" and nun == 2
    for c in feats:
        s = samp[c]
        if s.astype(str).equals(ys.astype(str)):
            add("error", "leak_copy", f"'{c}' is a copy of the target", "Remove it from the features.", [{"block": "dataset", "add_exclude": [c]}], model=None)
            continue
        if prof.get(c, {}).get("id_like"):
            add("warning", "id_feature", f"'{c}' looks like an ID column", "Row identifiers carry no signal and let models memorise rows. Exclude it.", [{"block": "dataset", "add_exclude": [c]}])
            continue
        if pd.api.types.is_numeric_dtype(s) and (resolved == "regression" or binary) and s.nunique() > 2:
            ok = s.notna() & ycode.notna()
            if ok.sum() > 10 and s[ok].std() > 0 and ycode[ok].std() > 0:
                r = float(np.corrcoef(s[ok], ycode[ok])[0, 1])
                if abs(r) >= 0.97:
                    add("warning", "leak_corr", f"'{c}' is almost identical to the target (r = {r:.3f})", "Such a strong relation usually means the column is derived from the target or only known afterwards (leakage). Exclude it unless it is truly available at prediction time.", [{"block": "dataset", "add_exclude": [c]}])
        elif resolved == "classification" and not pd.api.types.is_numeric_dtype(s) and 1 < s.nunique() < 0.5 * len(s):
            purity = samp.groupby(s.astype(str))[target].agg(lambda v: v.astype(str).value_counts(normalize=True).iloc[0]).mean()
            if purity > 0.999:
                add("warning", "leak_purity", f"'{c}' determines the target exactly", "Every value of this column maps to one class. Check that it is not derived from the target.", [{"block": "dataset", "add_exclude": [c]}])
    # ---- split / validation
    sp, cv = cfg.get("split") or {}, cfg.get("cv") or {}
    ts, vs, strat = float(sp.get("test_size", 0.2)), float(sp.get("val_size", 0.1)), cv.get("strategy", "stratified_kfold")
    if ts + vs >= 0.6:
        add("warning", "small_train", "Very small training set", f"Test {ts:.0%} + validation {vs:.0%} leaves only {1 - ts - vs:.0%} for training.", [{"block": "split", "settings": {"test_size": 0.2, "val_size": 0.1}}])
    if strat == "none" and vs <= 0:
        add("error", "select_on_test", "Models are selected on the test set", "Without cross-validation or a validation set, the leaderboard ranks models by test score, so the test score is no longer an unbiased estimate.", [{"block": "split", "settings": {"cv_strategy": "stratified_kfold" if resolved == "classification" else "kfold", "folds": 5}}])
    if D.get("time_col") and sp.get("shuffle", True) and strat != "time_series":
        add("warning", "time_shuffle", "Time data is shuffled", "With a time column, shuffled splits train on the future and test on the past. Use time-ordered splits.", [{"block": "split", "settings": {"shuffle": False, "cv_strategy": "time_series"}}])
    if strat == "group_kfold" and not D.get("group_col"):
        add("error", "group_missing", "Group K-fold without a group column", "Set a group column in the Dataset block (for example a patient or customer id) or choose another CV.", [{"block": "split", "settings": {"cv_strategy": "stratified_kfold" if resolved == "classification" else "kfold"}}])
    if resolved == "regression" and strat in ("stratified_kfold", "repeated_stratified_kfold", "stratified_shuffle_split"):
        add("info", "strat_regression", "Stratified CV on a regression task", "Stratification needs classes, so plain K-fold is used.", [{"block": "split", "settings": {"cv_strategy": "kfold"}}])
    counts = y.astype(str).value_counts() if resolved == "classification" else None
    folds = int(cv.get("folds", 5))
    if counts is not None and strat != "none" and counts.min() * (1 - ts - vs) < folds:
        add("warning", "tiny_class", f"Class '{counts.idxmin()}' is too small for {folds}-fold CV", f"It has {int(counts.min())} rows in total.", [{"block": "split", "settings": {"folds": max(2, int(counts.min() * (1 - ts - vs)))}}])
    # ---- imbalance / metric
    P, ev = cfg.get("preprocess") or {}, cfg.get("eval") or {}
    pm = ev.get("primary_metric", "auto")
    if counts is not None:
        share = counts.min() / counts.sum()
        if share < 0.2 and pm == "accuracy":
            add("warning", "accuracy_imbalanced", f"Accuracy on imbalanced classes (minority {share:.0%})", "A model predicting only the majority class already scores high accuracy. Rank by F1 or balanced accuracy and weight the classes.", [{"block": "eval", "settings": {"primary_metric": "f1" if nun == 2 else "balanced_accuracy"}}, {"block": "preprocess", "settings": {"imbalance": "class_weight"}}])
        elif share < 0.2 and P.get("imbalance", "none") == "none":
            add("info", "imbalanced", f"Imbalanced classes (minority {share:.0%})", "Consider class weights or SMOTE in Preprocessing.", [{"block": "preprocess", "settings": {"imbalance": "class_weight"}}])
        if P.get("imbalance") == "smote" and counts.min() < 6:
            add("warning", "smote_tiny", "SMOTE with a tiny class", "SMOTE needs several examples per class; use random oversampling.", [{"block": "preprocess", "settings": {"imbalance": "random_over"}}])
    if resolved == "regression" and P.get("imbalance", "none") != "none":
        add("warning", "imbalance_regression", "Class-imbalance handling on a regression task", "It only applies to classification and is ignored.", [{"block": "preprocess", "settings": {"imbalance": "none"}}])
    if pm != "auto" and ((resolved == "classification" and pm in REG_ONLY_METRICS) or (resolved == "regression" and pm in CLS_ONLY_METRICS)):
        add("error", "metric_task", f"Metric '{pm}' does not fit a {resolved} task", "The ranking metric must match the task.", [{"block": "eval", "settings": {"primary_metric": "auto"}}])
    # ---- preprocessing / FE / FS
    keys = {m.get("key") for m in cfg.get("models") or []}
    sens = sorted(keys & SCALE_SENSITIVE)
    if P.get("scaling", "standard") == "none" and sens:
        add("warning", "no_scaling", "No feature scaling for scale-sensitive models", f"{', '.join(sens)} depend on feature scale (distances, gradients, regularisation). Use StandardScaler.", [{"block": "preprocess", "settings": {"scaling": "standard"}}])
    if P.get("num_impute") == "drop":
        lost = float(df[feats].isna().any(axis=1).mean()) if feats else 0
        if lost > 0.3:
            add("warning", "drop_rows", f"Dropping rows with missing values loses {lost:.0%} of the data", "Impute instead.", [{"block": "preprocess", "settings": {"num_impute": "median"}}])
    numc = [c for c in feats if prof.get(c, {}).get("kind") == "numeric"]
    FE, FS = cfg.get("fe") or {}, cfg.get("fs") or {}
    deg = int(FE.get("polynomial", 0) or 0)
    pc = len(FE.get("poly_columns") or []) or len(numc)
    if deg >= 2:
        nfeat = math.comb(pc + deg, deg) - 1
        if nfeat > 400:
            add("warning", "poly_explosion", f"Polynomial features create about {nfeat} columns", "That slows training and invites overfitting. Use degree 2 on a few chosen columns.", [{"block": "fe", "settings": {"polynomial": 2, "poly_columns": numc[:5]}}])
    if resolved == "regression" and FS.get("method") == "kbest_chi2":
        add("warning", "chi2_regression", "chi² selection on a regression task", "chi² only works for classification (F-test is used instead).", [{"block": "fs", "settings": {"method": "kbest_f"}}])
    if resolved == "regression" and FS.get("reduction") == "lda":
        add("error", "lda_regression", "LDA projection on a regression task", "LDA needs classes, so it is skipped. Use PCA.", [{"block": "fs", "settings": {"reduction": "pca"}}])
    if FS.get("method") in ("sfs_forward", "sfs_backward") and len(feats) > 25:
        add("info", "sfs_slow", "Sequential feature selection is slow with many columns", "It refits the model many times per CV fold. Mutual information or tree importance is much faster.", [{"block": "fs", "settings": {"method": "kbest_mi"}}])
    if FS.get("reduction") in ("pca", "svd", "ica", "kernel_pca") and keys & {"rf", "et", "xgb", "lgbm", "hgb", "gb", "dt"}:
        add("info", "pca_trees", "Dimensionality reduction before tree models", "Trees handle raw features well, and PCA hides which inputs matter.")
    # ---- models / tuning
    if not cfg.get("models"):
        add("error", "no_models", "No models in the pipeline", "Add model blocks or a Model Zoo.", [{"add_block": "zoo"}])
    for m in cfg.get("models") or []:
        if resolved not in (m.get("tasks") or [resolved]) and "__" not in m["id"]:
            add("warning", "model_task", f"{m['name']} does not support {resolved}", "It will be skipped.", [{"block": m["id"], "remove": True}], model=m["id"])
    T = cfg.get("tuning") or {}
    if T.get("enabled") and T.get("method") in ("optuna", "bayesian") and int(T.get("n_iter", 20)) < 8:
        add("info", "few_trials", "Very few tuning trials", "Bayesian optimisation needs about 15+ trials to beat random search.", [{"block": "tuning", "settings": {"n_iter": 20}}])
    # ---- results (only for the same target)
    R = S.get("results")
    if R and S.get("cfg", {}).get("dataset", {}).get("target") == target and "D" in S:
        prim = S["D"]["primary"]
        lower = prim in LOWER_BETTER
        base = next((r for r in R.values() if r.get("baseline") and r["status"] == "ok"), None)
        ranked = _ranked(R, prim)
        for r in R.values():
            if r["status"] != "ok":
                add("error", "model_failed", f"{r['name']} failed", f"{r.get('error', '')[:400]} — {_error_hint(r.get('error', ''))}", None, model=r["id"])
        for r in ranked[:5]:
            gap = r.get("overfit_gap")
            if gap is not None and gap > (0.1 if S["D"]["task"] == "classification" else 0.15) and not r.get("ensemble"):
                fx = OVERFIT_FIX.get(r["key"])
                add("warning", "overfit", f"{r['name']} overfits (train − test gap {gap:.3f})", "It fits the training data much better than new data. Regularise it or give it less capacity.",
                    [{"block": r.get("tuned_from") or r["id"], "settings": fx}] if fx and "__" not in (r.get("tuned_from") or r["id"]) else None, model=r["id"])
            t = (r.get("test") or {}).get(prim)
            if t is not None and not lower and t >= 0.999 and prim in ("accuracy", "r2", "roc_auc", "f1", "balanced_accuracy"):
                add("warning", "too_perfect", f"{r['name']} scores a perfect {t:.4f} on the test set", "Perfect scores on real data usually mean leakage: a feature that encodes the target, duplicates across splits, or an ID. Check the leakage warnings above.", model=r["id"])
        if base and ranked:
            b, s = base.get("score"), ranked[0].get("score")
            if b is not None and s is not None and (s <= b + 1e-3 if not lower else s >= b - 1e-3):
                add("error", "no_better_than_baseline", "The best model is no better than guessing", f"Baseline {prim} {b:.4f} vs best {s:.4f}. The features may carry no signal, the target may be wrong, or preprocessing may remove the useful columns.")
        for r in ranked[:3]:
            c = (r.get("cv") or {}).get(prim)
            if c and c["std"] > 0.05 and not lower:
                add("info", "cv_unstable", f"{r['name']} is unstable across CV folds (± {c['std']:.3f})", "Scores vary a lot between folds: more data, repeated CV or a simpler model will give more reliable results.", model=r["id"])
        for r in R.values():
            tu = r.get("tuning")
            if tu and tu.get("before") is not None and tu.get("after") is not None and ((tu["after"] <= tu["before"]) if not lower else (tu["after"] >= tu["before"])):
                add("info", "tuning_no_gain", f"Tuning did not improve {r['name'].replace(' (tuned)', '')}", "Widen the search space or run more trials.", [{"block": "tuning", "settings": {"n_iter": max(30, int(T.get("n_iter", 20)) * 2)}}], model=r["id"])
    order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: order[i["severity"]])
    return dumps({"issues": issues, "task": resolved})


@safe
def get_summary():
    return dumps(summary())


@safe
def model_info(model_id):
    r = S["results"][model_id]
    return dumps({"result": r, "pipeline": str(S["models"].get(model_id, ""))})


# --------------------------------------------------------------------- export
@safe
def export_model(model_id):
    import joblib
    mid = model_id or S["best"]
    art = {"pipeline": S["models"][mid], "frame_spec": S["frame_spec"], "task": S["D"]["task"], "classes": S["D"]["classes"],
           "name": S["results"][mid]["name"], "features": list(S["df"].columns), "sklearn": __import__("sklearn").__version__}
    b = io.BytesIO()
    joblib.dump(art, b, compress=3)
    return dumps({"b64": base64.b64encode(b.getvalue()).decode(), "name": S["results"][mid]["name"]})


def load_artifact(path):
    import joblib
    return joblib.load(path)


def predict_with_artifact(art, rows):
    raw = pd.DataFrame(rows if isinstance(rows, list) else [rows])
    for c in art["features"]:
        if c not in raw.columns:
            raw[c] = np.nan
    X = apply_frame(raw, art["frame_spec"])
    pred = art["pipeline"].predict(X)
    out = {"prediction": [art["classes"][int(v)] for v in pred] if art["classes"] else [float(v) for v in pred]}
    if art["classes"] and hasattr(art["pipeline"], "predict_proba"):
        pr = art["pipeline"].predict_proba(X)
        out["probabilities"] = [{c: float(pr[i, j]) for j, c in enumerate(art["classes"])} for i in range(len(X))]
    return out


@safe
def versions():
    out = {"python": sys.version.split()[0], "browser": IN_BROWSER}
    for m in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "matplotlib", "optuna"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:
            out[m] = None
    return dumps(out)
