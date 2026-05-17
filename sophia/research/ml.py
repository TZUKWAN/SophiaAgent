"""Machine learning experiment engine.

Pure-computation wrapper around scikit-learn, xgboost, lightgbm, optuna,
shap, and flaml.  All public methods accept ``args: dict`` (from tool
dispatch) and return ``str`` (JSON).  Optional dependencies are handled
gracefully.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sophia.research.seed import GlobalSeed

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
try:
    from sklearn.model_selection import (
        cross_val_score, StratifiedKFold, KFold,
        GridSearchCV, RandomizedSearchCV,
        learning_curve as sk_learning_curve,
    )
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        VotingClassifier, VotingRegressor,
        BaggingClassifier, BaggingRegressor,
        StackingClassifier, StackingRegressor,
    )
    from sklearn.svm import SVC, SVR
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
        confusion_matrix as sk_confusion_matrix,
        classification_report,
    )
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import flaml
    HAS_FLAML = True
except ImportError:
    HAS_FLAML = False

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json(result: dict) -> str:
    """Serialize *result* to a JSON string, converting non-serializable types."""
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return _convert(obj.tolist())
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj
    return json.dumps(_convert(result), ensure_ascii=False, indent=2)


def _model_from_name(model_name: str, task: str, params: Optional[dict] = None,
                     random_state: Optional[int] = None):
    """Instantiate a sklearn-compatible model from a string name."""
    params = params or {}
    if random_state is None:
        random_state = GlobalSeed.get_or_default(42)
    rs = params.pop("random_state", random_state)
    # Filter params to only include valid constructor args
    p = {k: v for k, v in params.items()}

    model_map = {
        "logistic": (LogisticRegression, {"max_iter": 1000, "random_state": rs}),
        "random_forest": (
            RandomForestClassifier if task == "classification" else RandomForestRegressor,
            {"random_state": rs, "n_estimators": 100},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor,
            {"random_state": rs, "n_estimators": 100},
        ),
        "decision_tree": (
            DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor,
            {"random_state": rs},
        ),
        "svm": (SVC if task == "classification" else SVR, {"random_state": rs}),
        "linear": (LinearRegression, {}),
        "kmeans": (KMeans, {"random_state": rs, "n_clusters": p.pop("n_clusters", 8)}),
    }

    if model_name == "xgboost":
        if not HAS_XGBOOST:
            return None, "xgboost is not installed"
        if task == "classification":
            cls = xgb.XGBClassifier
        else:
            cls = xgb.XGBRegressor
        default_p = {"random_state": rs, "use_label_encoder": False, "verbosity": 0}
        default_p.update(p)
        return cls(**default_p), None

    if model_name == "lightgbm":
        if not HAS_LIGHTGBM:
            return None, "lightgbm is not installed"
        if task == "classification":
            cls = lgb.LGBMClassifier
        else:
            cls = lgb.LGBMRegressor
        default_p = {"random_state": rs, "verbose": -1}
        default_p.update(p)
        return cls(**default_p), None

    if model_name not in model_map:
        return None, f"Unknown model '{model_name}'"

    cls, default_params = model_map[model_name]
    # Merge user params
    merged = dict(default_params)
    merged.update(p)
    try:
        return cls(**merged), None
    except TypeError as e:
        # If invalid params, try without them
        return cls(**default_params), None


def _compute_metrics(y_true, y_pred, y_score=None, task="classification"):
    """Compute evaluation metrics."""
    metrics: Dict[str, Any] = {}

    if task == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["confusion_matrix"] = sk_confusion_matrix(y_true, y_pred).tolist()

        if y_score is not None:
            try:
                n_classes = len(set(y_true))
                if n_classes == 2:
                    metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
                else:
                    metrics["auc_roc"] = float(roc_auc_score(
                        y_true, y_score, multi_class="ovr", average="weighted",
                    ))
            except (ValueError, TypeError):
                metrics["auc_roc"] = None
        try:
            metrics["classification_report"] = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0,
            )
        except Exception:
            pass

    elif task == "regression":
        mse = float(mean_squared_error(y_true, y_pred))
        metrics["mse"] = mse
        metrics["rmse"] = float(math.sqrt(mse))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    elif task == "clustering":
        # For clustering, return the labels
        metrics["labels"] = y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)
        metrics["n_clusters_found"] = len(set(y_pred))

    return metrics


# ======================================================================
# MLEngine
# ======================================================================

class MLEngine:
    """Machine learning experimentation engine."""

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------

    def preprocess(self, args: dict) -> str:
        """Data preprocessing.

        Parameters
        ----------
        args : dict
            X : list of lists
            y : list
            operations : list of {type: str, params: dict}
                Supported: 'standardize', 'normalize', 'encode_labels',
                'pca' (params: n_components), 'select_k_best' (params: k),
                'impute_mean', 'impute_median'

        Returns
        -------
        str
            JSON with transformed shapes and operations applied.
        """
        if not HAS_SKLEARN:
            return _json({"error": "scikit-learn is required for preprocessing."})

        X_raw = args.get("X")
        y_raw = args.get("y")
        operations = args.get("operations", [])

        if X_raw is None:
            return _json({"error": "X is required."})

        try:
            X = np.asarray(X_raw, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        except (TypeError, ValueError):
            return _json({"error": "X must contain numeric values."})

        y = None
        if y_raw is not None:
            y = np.asarray(y_raw)

        applied = []
        original_shape = X.shape

        for op in operations:
            op_type = op.get("type", "")
            op_params = op.get("params", {})

            if op_type == "standardize":
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                applied.append({"operation": "standardize", "params": op_params})

            elif op_type == "normalize":
                scaler = MinMaxScaler()
                X = scaler.fit_transform(X)
                applied.append({"operation": "normalize", "params": op_params})

            elif op_type == "encode_labels":
                if y is not None:
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    applied.append({
                        "operation": "encode_labels",
                        "classes": le.classes_.tolist(),
                    })
                else:
                    applied.append({"operation": "encode_labels", "skipped": "no y provided"})

            elif op_type == "pca":
                n_components = op_params.get("n_components", min(X.shape))
                n_components = min(n_components, min(X.shape))
                pca = PCA(n_components=n_components)
                X = pca.fit_transform(X)
                explained_var = pca.explained_variance_ratio_.tolist()
                applied.append({
                    "operation": "pca",
                    "n_components": int(n_components),
                    "explained_variance_ratio": explained_var,
                })

            elif op_type == "select_k_best":
                if y is None:
                    applied.append({"operation": "select_k_best", "skipped": "no y provided"})
                    continue
                k = op_params.get("k", min(10, X.shape[1]))
                k = min(k, X.shape[1])
                selector = SelectKBest(f_classif, k=k)
                X = selector.fit_transform(X, y)
                applied.append({
                    "operation": "select_k_best",
                    "k": int(k),
                    "selected_features": selector.get_support(indices=True).tolist(),
                })

            elif op_type == "impute_mean":
                imputer = SimpleImputer(strategy="mean")
                X = imputer.fit_transform(X)
                applied.append({"operation": "impute_mean"})

            elif op_type == "impute_median":
                imputer = SimpleImputer(strategy="median")
                X = imputer.fit_transform(X)
                applied.append({"operation": "impute_median"})

            else:
                applied.append({"operation": op_type, "error": "unknown operation"})

        result: Dict[str, Any] = {
            "original_shape": list(original_shape),
            "transformed_shape": list(X.shape),
            "operations_applied": applied,
        }
        if y is not None:
            result["y_shape"] = list(y.shape) if hasattr(y, "shape") else len(y)

        return _json(result)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self, args: dict) -> str:
        """Train ML model.

        Parameters
        ----------
        args : dict
            X_train, y_train, X_test, y_test,
            model, task, params, random_state

        Returns
        -------
        str
            JSON with model info, train/test scores, predictions.
        """
        if not HAS_SKLEARN:
            return _json({"error": "scikit-learn is required."})

        X_train_raw = args.get("X_train")
        y_train_raw = args.get("y_train")
        X_test_raw = args.get("X_test")
        y_test_raw = args.get("y_test")
        model_name: str = str(args.get("model", "random_forest")).lower()
        task: str = str(args.get("task", "classification")).lower()
        params = dict(args.get("params", {}))
        random_state: int = int(args.get("random_state", GlobalSeed.get_or_default(42)))

        if X_train_raw is None or y_train_raw is None:
            return _json({"error": "X_train and y_train are required."})

        try:
            X_train = np.asarray(X_train_raw, dtype=np.float64)
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
            y_train = np.asarray(y_train_raw)
        except (TypeError, ValueError):
            return _json({"error": "X_train / y_train must contain numeric values."})

        has_test = X_test_raw is not None and y_test_raw is not None
        X_test = None
        y_test = None
        if has_test:
            try:
                X_test = np.asarray(X_test_raw, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                y_test = np.asarray(y_test_raw)
            except (TypeError, ValueError):
                return _json({"error": "X_test / y_test must contain numeric values."})

        # Instantiate model
        model, err = _model_from_name(model_name, task, params, random_state)
        if err:
            return _json({"error": err})

        # Suppress warnings during training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        train_metrics = _compute_metrics(y_train, y_train_pred, task=task)

        result: Dict[str, Any] = {
            "model_type": model_name,
            "task": task,
            "params": params,
            "train_score": train_metrics,
            "train_shape": [X_train.shape[0], X_train.shape[1]],
        }

        if has_test:
            y_test_pred = model.predict(X_test)
            # Try to get probability scores for classification
            y_score = None
            if task == "classification" and hasattr(model, "predict_proba"):
                try:
                    y_score = model.predict_proba(X_test)
                except Exception:
                    y_score = None
            elif task == "classification" and hasattr(model, "decision_function"):
                try:
                    y_score = model.decision_function(X_test)
                except Exception:
                    y_score = None

            test_metrics = _compute_metrics(y_test, y_test_pred, y_score=y_score, task=task)
            result["test_score"] = test_metrics
            result["test_shape"] = [X_test.shape[0], X_test.shape[1]]
            result["predictions_sample"] = y_test_pred[:10].tolist()
        else:
            result["predictions_sample"] = y_train_pred[:10].tolist()

        return _json(result)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, args: dict) -> str:
        """Evaluate model predictions.

        Parameters
        ----------
        args : dict
            y_true, y_pred, y_score (for ROC), task

        Returns
        -------
        str
            JSON with evaluation metrics.
        """
        if not HAS_SKLEARN:
            return _json({"error": "scikit-learn is required."})

        y_true_raw = args.get("y_true")
        y_pred_raw = args.get("y_pred")
        y_score_raw = args.get("y_score")
        task: str = str(args.get("task", "classification")).lower()

        if y_true_raw is None or y_pred_raw is None:
            return _json({"error": "y_true and y_pred are required."})

        y_true = np.asarray(y_true_raw)
        y_pred = np.asarray(y_pred_raw)
        y_score = np.asarray(y_score_raw) if y_score_raw is not None else None

        if len(y_true) != len(y_pred):
            return _json({"error": "y_true and y_pred must have the same length."})

        metrics = _compute_metrics(y_true, y_pred, y_score=y_score, task=task)
        metrics["task"] = task
        metrics["n_samples"] = len(y_true)

        return _json(metrics)

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def crossval(self, args: dict) -> str:
        """Cross-validation.

        Parameters
        ----------
        args : dict
            X, y, model, task, cv, stratified, params

        Returns
        -------
        str
            JSON with mean/std scores and per-fold scores.
        """
        if not HAS_SKLEARN:
            return _json({"error": "scikit-learn is required."})

        X_raw = args.get("X")
        y_raw = args.get("y")
        model_name: str = str(args.get("model", "random_forest")).lower()
        task: str = str(args.get("task", "classification")).lower()
        cv: int = int(args.get("cv", 5))
        stratified: bool = bool(args.get("stratified", True))
        params = dict(args.get("params", {}))
        random_state: int = int(args.get("random_state", GlobalSeed.get_or_default(42)))

        if X_raw is None or y_raw is None:
            return _json({"error": "X and y are required."})

        X = np.asarray(X_raw, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y_raw)

        model, err = _model_from_name(model_name, task, params, random_state)
        if err:
            return _json({"error": err})

        # Choose scoring
        if task == "classification":
            scoring = "accuracy"
        elif task == "regression":
            scoring = "r2"
        else:
            scoring = None

        # Choose CV splitter
        if task == "classification" and stratified:
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)

        return _json({
            "model": model_name,
            "task": task,
            "scoring": scoring,
            "cv": cv,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "fold_scores": scores.tolist(),
        })

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def tune(self, args: dict) -> str:
        """Hyperparameter tuning.

        Parameters
        ----------
        args : dict
            X, y, model, task, param_grid, method, cv, n_iter, scoring

        Returns
        -------
        str
            JSON with best_params, best_score, results summary.
        """
        if not HAS_SKLEARN:
            return _json({"error": "scikit-learn is required."})

        X_raw = args.get("X")
        y_raw = args.get("y")
        model_name: str = str(args.get("model", "random_forest")).lower()
        task: str = str(args.get("task", "classification")).lower()
        param_grid = args.get("param_grid", {})
        method: str = str(args.get("method", "grid")).lower()
        cv: int = int(args.get("cv", 5))
        n_iter: int = int(args.get("n_iter", 20))
        scoring = args.get("scoring")
        random_state: int = int(args.get("random_state", GlobalSeed.get_or_default(42)))

        if X_raw is None or y_raw is None:
            return _json({"error": "X and y are required."})

        X = np.asarray(X_raw, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y_raw)

        if not param_grid:
            return _json({"error": "param_grid is required for tuning."})

        # Base model (without the grid params)
        base_model, err = _model_from_name(model_name, task, {}, random_state)
        if err:
            return _json({"error": err})

        # Default scoring
        if scoring is None:
            scoring = "accuracy" if task == "classification" else "r2"

        if task == "classification":
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        # ---- Optuna path ----
        if method == "optuna" and HAS_OPTUNA:
            return self._tune_optuna(X, y, base_model, model_name, task, param_grid,
                                     cv, scoring, random_state)

        # ---- Sklearn path ----
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if method == "grid":
                searcher = GridSearchCV(
                    base_model, param_grid, cv=cv_splitter,
                    scoring=scoring, n_jobs=-1, refit=True,
                )
            elif method == "random":
                searcher = RandomizedSearchCV(
                    base_model, param_grid, cv=cv_splitter,
                    scoring=scoring, n_iter=n_iter, n_jobs=-1,
                    random_state=random_state, refit=True,
                )
            else:
                return _json({"error": f"Unknown method '{method}'. Use 'grid', 'random', or 'optuna'."})

            searcher.fit(X, y)

        # Build summary of all results
        all_results = []
        for i in range(len(searcher.cv_results_["mean_test_score"])):
            all_results.append({
                "params": searcher.cv_results_["params"][i],
                "mean_score": float(searcher.cv_results_["mean_test_score"][i]),
                "std_score": float(searcher.cv_results_["std_test_score"][i]),
                "rank": int(searcher.cv_results_["rank_test_score"][i]),
            })

        return _json({
            "model": model_name,
            "method": method,
            "scoring": scoring,
            "best_params": searcher.best_params_,
            "best_score": float(searcher.best_score_),
            "n_candidates": len(all_results),
            "top_5_results": sorted(all_results, key=lambda x: x["rank"])[:5],
        })

    def _tune_optuna(self, X, y, base_model, model_name, task, param_grid,
                     cv, scoring, random_state):
        """Hyperparameter tuning with Optuna."""
        import copy

        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=random_state))

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state) \
            if task == "classification" else KFold(n_splits=cv, shuffle=True, random_state=random_state)

        def objective(trial):
            params = {}
            for param_name, param_spec in param_grid.items():
                if isinstance(param_spec, dict):
                    ptype = param_spec.get("type", "float")
                    if ptype == "float":
                        params[param_name] = trial.suggest_float(
                            param_name, param_spec["low"], param_spec["high"],
                            log=param_spec.get("log", False),
                        )
                    elif ptype == "int":
                        params[param_name] = trial.suggest_int(
                            param_name, param_spec["low"], param_spec["high"],
                            log=param_spec.get("log", False),
                        )
                    elif ptype == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_spec["choices"],
                        )
                elif isinstance(param_spec, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_spec)
                elif isinstance(param_spec, tuple) and len(param_spec) == 2:
                    params[param_name] = trial.suggest_float(param_name, param_spec[0], param_spec[1])

            model = copy.deepcopy(base_model)
            model.set_params(**params)

            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)
            return float(np.mean(scores))

        study.optimize(objective, n_trials=20, show_progress_bar=False)

        return _json({
            "model": model_name,
            "method": "optuna",
            "scoring": scoring,
            "best_params": study.best_params,
            "best_score": float(study.best_value),
            "n_trials": len(study.trials),
        })

    # ------------------------------------------------------------------
    # Compare models
    # ------------------------------------------------------------------

    def compare(self, args: dict) -> str:
        """Compare multiple models.

        Parameters
        ----------
        args : dict
            X, y, models (list of str), task, cv, params

        Returns
        -------
        str
            JSON with model comparison table, best model, statistical test.
        """
        if not HAS_SKLEARN:
            return _json({"error": "scikit-learn is required."})

        X_raw = args.get("X")
        y_raw = args.get("y")
        model_names: List[str] = args.get("models", ["logistic", "random_forest", "decision_tree"])
        task: str = str(args.get("task", "classification")).lower()
        cv: int = int(args.get("cv", 5))
        params = args.get("params", {})
        random_state: int = int(args.get("random_state", GlobalSeed.get_or_default(42)))

        if X_raw is None or y_raw is None:
            return _json({"error": "X and y are required."})

        X = np.asarray(X_raw, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y_raw)

        scoring = "accuracy" if task == "classification" else "r2"

        if task == "classification":
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        comparison = []
        all_fold_scores: Dict[str, List[float]] = {}

        for name in model_names:
            model, err = _model_from_name(
                name, task, dict(params.get(name, {})), random_state,
            )
            if err:
                comparison.append({"model": name, "error": err})
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)

            all_fold_scores[name] = scores.tolist()
            comparison.append({
                "model": name,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "fold_scores": scores.tolist(),
            })

        # Sort by mean_score descending
        valid_results = [c for c in comparison if "error" not in c]
        valid_results.sort(key=lambda x: x["mean_score"], reverse=True)

        best_model = valid_results[0]["model"] if valid_results else None

        # Statistical comparison (paired t-test) between top 2 models
        stat_comparison = None
        if len(valid_results) >= 2 and HAS_SCIPY:
            name1 = valid_results[0]["model"]
            name2 = valid_results[1]["model"]
            s1 = np.array(all_fold_scores[name1])
            s2 = np.array(all_fold_scores[name2])
            t_stat, p_val = sp_stats.ttest_rel(s1, s2)
            stat_comparison = {
                "test": "paired t-test",
                "model_1": name1,
                "model_2": name2,
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant_at_005": float(p_val) < 0.05,
            }

        return _json({
            "comparison": comparison,
            "ranking": [c["model"] for c in valid_results],
            "best_model": best_model,
            "scoring": scoring,
            "cv": cv,
            "statistical_comparison": stat_comparison,
        })

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self, args: dict) -> str:
        """Feature importance / SHAP.

        Parameters
        ----------
        args : dict
            X, y, model, task, method ('builtin'|'shap'|'permutation'),
            feature_names, params

        Returns
        -------
        str
            JSON with importance scores per feature, top features.
        """
        if not HAS_SKLEARN:
            return _json({"error": "scikit-learn is required."})

        X_raw = args.get("X")
        y_raw = args.get("y")
        model_name: str = str(args.get("model", "random_forest")).lower()
        task: str = str(args.get("task", "classification")).lower()
        method: str = str(args.get("method", "builtin")).lower()
        feature_names = args.get("feature_names")
        params = dict(args.get("params", {}))
        random_state: int = int(args.get("random_state", GlobalSeed.get_or_default(42)))

        if X_raw is None or y_raw is None:
            return _json({"error": "X and y are required."})

        X = np.asarray(X_raw, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y_raw)

        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            return _json({"error": "feature_names count does not match number of features."})

        # Train model
        model, err = _model_from_name(model_name, task, params, random_state)
        if err:
            return _json({"error": err})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)

        importance_scores: Dict[str, float] = {}

        if method == "shap" and HAS_SHAP:
            try:
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                if mean_abs_shap.ndim > 1:
                    mean_abs_shap = mean_abs_shap.mean(axis=1)
                for i, name in enumerate(feature_names):
                    importance_scores[name] = float(mean_abs_shap[i])
                method_used = "shap"
            except Exception as e:
                # Fall back to builtin
                method_used = "builtin (shap failed)"
                importance_scores = self._builtin_importance(model, feature_names)

        elif method == "permutation":
            from sklearn.inspection import permutation_importance
            scoring = "accuracy" if task == "classification" else "r2"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                perm_result = permutation_importance(
                    model, X, y, n_repeats=10, random_state=random_state,
                    scoring=scoring,
                )
            for i, name in enumerate(feature_names):
                importance_scores[name] = float(perm_result.importances_mean[i])
            method_used = "permutation"

        else:
            importance_scores = self._builtin_importance(model, feature_names)
            method_used = "builtin"

        # Sort by importance
        sorted_importance = sorted(
            importance_scores.items(), key=lambda x: abs(x[1]), reverse=True,
        )

        return _json({
            "model": model_name,
            "method": method_used,
            "importance": dict(sorted_importance),
            "top_features": sorted_importance[:10],
            "n_features": n_features,
        })

    def _builtin_importance(self, model, feature_names):
        """Get feature_importances_ or coef_ from a trained model."""
        scores = {}
        if hasattr(model, "feature_importances_"):
            for i, name in enumerate(feature_names):
                scores[name] = float(model.feature_importances_[i])
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            else:
                coef = np.abs(coef)
            for i, name in enumerate(feature_names):
                scores[name] = float(coef[i])
        else:
            for name in feature_names:
                scores[name] = 0.0
        return scores

    # ------------------------------------------------------------------
    # AutoML
    # ------------------------------------------------------------------

    def automl(self, args: dict) -> str:
        """AutoML using FLAML.

        Parameters
        ----------
        args : dict
            X, y, task, time_budget, metric, cv, estimators

        Returns
        -------
        str
            JSON with best_model, best_params, best_score, time_used.
        """
        X_raw = args.get("X")
        y_raw = args.get("y")
        task: str = str(args.get("task", "classification")).lower()
        time_budget: int = int(args.get("time_budget", 60))
        metric = args.get("metric")
        cv: int = int(args.get("cv", 5))
        estimators = args.get("estimators")

        if X_raw is None or y_raw is None:
            return _json({"error": "X and y are required."})

        X = np.asarray(X_raw, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y_raw)

        # ---- FLAML path ----
        if HAS_FLAML:
            from flaml import AutoML
            automl_model = AutoML()

            settings = {
                "time_budget": time_budget,
                "task": "classification" if task == "classification" else "regression",
                "n_jobs": -1,
                "cv": cv,
                "seed": 42,
                "verbose": 0,
            }
            if metric:
                settings["metric"] = metric
            if estimators:
                settings["estimator_list"] = estimators

            start_time = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                automl_model.fit(X, y, **settings)
            elapsed = time.time() - start_time

            return _json({
                "best_model": automl_model.best_estimator,
                "best_params": automl_model.best_config,
                "best_score": float(automl_model.best_loss) if automl_model.best_loss is not None else None,
                "time_used": round(elapsed, 2),
                "method": "flaml",
            })

        # ---- Fallback: run compare and return best ----
        default_models = ["logistic", "random_forest", "gradient_boosting", "decision_tree"]
        if task == "regression":
            default_models = ["linear", "random_forest", "gradient_boosting", "decision_tree"]
        if HAS_XGBOOST:
            default_models.append("xgboost")

        compare_result = json.loads(self.compare({
            "X": X.tolist(), "y": y.tolist(),
            "models": default_models, "task": task, "cv": cv,
        }))

        return _json({
            "best_model": compare_result.get("best_model"),
            "best_score": compare_result.get("best_score"),
            "comparison": compare_result.get("comparison"),
            "method": "compare_fallback (flaml not installed)",
        })

    # ------------------------------------------------------------------
    # Learning curve
    # ------------------------------------------------------------------

    def learning_curve(self, args: dict) -> str:
        """Learning curve.

        Parameters
        ----------
        args : dict
            X, y, model, task, cv, train_sizes, params

        Returns
        -------
        str
            JSON with train_sizes, train_scores, val_scores.
        """
        if not HAS_SKLEARN:
            return _json({"error": "scikit-learn is required."})

        X_raw = args.get("X")
        y_raw = args.get("y")
        model_name: str = str(args.get("model", "random_forest")).lower()
        task: str = str(args.get("task", "classification")).lower()
        cv: int = int(args.get("cv", 5))
        train_sizes_raw = args.get("train_sizes")
        params = dict(args.get("params", {}))
        random_state: int = int(args.get("random_state", GlobalSeed.get_or_default(42)))

        if X_raw is None or y_raw is None:
            return _json({"error": "X and y are required."})

        X = np.asarray(X_raw, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y_raw)

        if train_sizes_raw is None:
            train_sizes_raw = np.linspace(0.1, 1.0, 10)
        else:
            train_sizes_raw = np.asarray(train_sizes_raw)

        model, err = _model_from_name(model_name, task, params, random_state)
        if err:
            return _json({"error": err})

        scoring = "accuracy" if task == "classification" else "r2"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_sizes_abs, train_scores, val_scores = sk_learning_curve(
                model, X, y,
                train_sizes=train_sizes_raw,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=random_state,
            )

        return _json({
            "model": model_name,
            "scoring": scoring,
            "train_sizes": train_sizes_abs.tolist(),
            "train_scores_mean": train_scores.mean(axis=1).tolist(),
            "train_scores_std": train_scores.std(axis=1).tolist(),
            "val_scores_mean": val_scores.mean(axis=1).tolist(),
            "val_scores_std": val_scores.std(axis=1).tolist(),
        })

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def ensemble(self, args: dict) -> str:
        """Ensemble methods.

        Parameters
        ----------
        args : dict
            X_train, y_train, X_test, y_test,
            method ('voting'|'bagging'|'stacking'),
            base_models (list of str), meta_model (str, for stacking),
            task, params

        Returns
        -------
        str
            JSON with ensemble_score, individual_scores, improvement.
        """
        if not HAS_SKLEARN:
            return _json({"error": "scikit-learn is required."})

        X_train_raw = args.get("X_train")
        y_train_raw = args.get("y_train")
        X_test_raw = args.get("X_test")
        y_test_raw = args.get("y_test")
        method: str = str(args.get("method", "voting")).lower()
        base_model_names: List[str] = args.get("base_models", ["logistic", "random_forest", "decision_tree"])
        meta_model_name: str = str(args.get("meta_model", "logistic")).lower()
        task: str = str(args.get("task", "classification")).lower()
        params = args.get("params", {})
        random_state: int = int(args.get("random_state", GlobalSeed.get_or_default(42)))

        if X_train_raw is None or y_train_raw is None:
            return _json({"error": "X_train and y_train are required."})

        X_train = np.asarray(X_train_raw, dtype=np.float64)
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        y_train = np.asarray(y_train_raw)

        has_test = X_test_raw is not None and y_test_raw is not None
        X_test = y_test = None
        if has_test:
            X_test = np.asarray(X_test_raw, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            y_test = np.asarray(y_test_raw)

        # Build base estimators
        estimators = []
        for name in base_model_names:
            model, err = _model_from_name(name, task, {}, random_state)
            if not err:
                estimators.append((name, model))

        if len(estimators) < 2:
            return _json({"error": "At least 2 valid base models are required for ensemble."})

        # Build ensemble
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if method == "voting":
                if task == "classification":
                    ensemble_model = VotingClassifier(
                        estimators=estimators, voting="hard",
                    )
                else:
                    ensemble_model = VotingRegressor(estimators=estimators)

            elif method == "bagging":
                # Use first base model as the base for bagging
                base = estimators[0][1]
                if task == "classification":
                    ensemble_model = BaggingClassifier(
                        estimator=base, n_estimators=10, random_state=random_state,
                    )
                else:
                    ensemble_model = BaggingRegressor(
                        estimator=base, n_estimators=10, random_state=random_state,
                    )

            elif method == "stacking":
                meta_model, err = _model_from_name(meta_model_name, task, {}, random_state)
                if err:
                    return _json({"error": f"Meta model error: {err}"})
                if task == "classification":
                    ensemble_model = StackingClassifier(
                        estimators=estimators,
                        final_estimator=meta_model,
                        cv=5,
                    )
                else:
                    ensemble_model = StackingRegressor(
                        estimators=estimators,
                        final_estimator=meta_model,
                        cv=5,
                    )
            else:
                return _json({"error": f"Unknown ensemble method '{method}'."})

            ensemble_model.fit(X_train, y_train)

        # Evaluate ensemble
        scoring_fn = accuracy_score if task == "classification" else r2_score

        if has_test:
            ensemble_pred = ensemble_model.predict(X_test)
            ensemble_score = float(scoring_fn(y_test, ensemble_pred))
        else:
            ensemble_pred = ensemble_model.predict(X_train)
            ensemble_score = float(scoring_fn(y_train, ensemble_pred))

        # Evaluate individual models
        individual_scores = {}
        for name, model in estimators:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            if has_test:
                pred = model.predict(X_test)
                score = float(scoring_fn(y_test, pred))
            else:
                pred = model.predict(X_train)
                score = float(scoring_fn(y_train, pred))
            individual_scores[name] = score

        # Improvement over best individual
        best_individual = max(individual_scores.values()) if individual_scores else 0
        improvement = ensemble_score - best_individual

        return _json({
            "method": method,
            "task": task,
            "base_models": base_model_names,
            "ensemble_score": ensemble_score,
            "individual_scores": individual_scores,
            "best_individual_score": best_individual,
            "improvement": round(improvement, 6),
        })
