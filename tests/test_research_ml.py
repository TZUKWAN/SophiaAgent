"""Tests for MLEngine: preprocess, train, evaluate, crossval, tune, compare,
feature_importance, automl, learning_curve, ensemble.

Uses sklearn's built-in datasets (make_classification, make_regression, load_iris).
"""
import json
import math

import numpy as np
import pytest

from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split

from sophia.research.ml import (
    MLEngine, HAS_SKLEARN, HAS_XGBOOST, HAS_LIGHTGBM, HAS_OPTUNA,
    HAS_SHAP, HAS_FLAML, HAS_SCIPY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(result: str) -> dict:
    return json.loads(result)


@pytest.fixture
def engine():
    return MLEngine()


@pytest.fixture
def classification_data():
    """Standard classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, n_classes=2, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )
    return {
        "X": X.tolist(), "y": y.tolist(),
        "X_train": X_train.tolist(), "y_train": y_train.tolist(),
        "X_test": X_test.tolist(), "y_test": y_test.tolist(),
    }


@pytest.fixture
def regression_data():
    """Standard regression dataset."""
    X, y = make_regression(
        n_samples=200, n_features=5, n_informative=3,
        noise=10.0, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    return {
        "X": X.tolist(), "y": y.tolist(),
        "X_train": X_train.tolist(), "y_train": y_train.tolist(),
        "X_test": X_test.tolist(), "y_test": y_test.tolist(),
    }


@pytest.fixture
def iris_data():
    """Iris dataset for multiclass classification."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target,
    )
    return {
        "X": iris.data.tolist(), "y": iris.target.tolist(),
        "X_train": X_train.tolist(), "y_train": y_train.tolist(),
        "X_test": X_test.tolist(), "y_test": y_test.tolist(),
    }


# ===================================================================
# Preprocess tests
# ===================================================================

class TestPreprocess:

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_standardize(self, engine, classification_data):
        res = _parse(engine.preprocess({
            "X": classification_data["X"],
            "operations": [{"type": "standardize"}],
        }))
        assert "transformed_shape" in res
        assert res["original_shape"] == [200, 10]
        assert res["transformed_shape"] == [200, 10]
        assert len(res["operations_applied"]) == 1
        assert res["operations_applied"][0]["operation"] == "standardize"

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_normalize(self, engine, classification_data):
        res = _parse(engine.preprocess({
            "X": classification_data["X"],
            "operations": [{"type": "normalize"}],
        }))
        assert res["operations_applied"][0]["operation"] == "normalize"

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_pca(self, engine, classification_data):
        res = _parse(engine.preprocess({
            "X": classification_data["X"],
            "operations": [{"type": "pca", "params": {"n_components": 3}}],
        }))
        assert res["transformed_shape"] == [200, 3]
        pca_info = res["operations_applied"][0]
        assert pca_info["operation"] == "pca"
        assert pca_info["n_components"] == 3
        assert len(pca_info["explained_variance_ratio"]) == 3

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_select_k_best(self, engine, classification_data):
        res = _parse(engine.preprocess({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "operations": [{"type": "select_k_best", "params": {"k": 5}}],
        }))
        assert res["transformed_shape"][1] == 5

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_multiple_operations(self, engine, classification_data):
        res = _parse(engine.preprocess({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "operations": [
                {"type": "standardize"},
                {"type": "pca", "params": {"n_components": 5}},
            ],
        }))
        assert len(res["operations_applied"]) == 2
        assert res["transformed_shape"] == [200, 5]

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_impute_mean(self, engine):
        X = [[1.0, 2.0], [float("nan"), 4.0], [5.0, float("nan")]]
        res = _parse(engine.preprocess({
            "X": X,
            "operations": [{"type": "impute_mean"}],
        }))
        assert res["operations_applied"][0]["operation"] == "impute_mean"

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_missing_X_error(self, engine):
        res = _parse(engine.preprocess({"operations": []}))
        assert "error" in res

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_encode_labels(self, engine):
        res = _parse(engine.preprocess({
            "X": [[1, 2], [3, 4]],
            "y": ["cat", "dog"],
            "operations": [{"type": "encode_labels"}],
        }))
        assert any(op["operation"] == "encode_labels" for op in res["operations_applied"])


# ===================================================================
# Train tests
# ===================================================================

class TestTrain:

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_random_forest_classification(self, engine, classification_data):
        res = _parse(engine.train({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "X_test": classification_data["X_test"],
            "y_test": classification_data["y_test"],
            "model": "random_forest",
            "task": "classification",
        }))
        assert res["model_type"] == "random_forest"
        assert "train_score" in res
        assert "test_score" in res
        assert res["test_score"]["accuracy"] > 0.5

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_logistic_regression(self, engine, classification_data):
        res = _parse(engine.train({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "X_test": classification_data["X_test"],
            "y_test": classification_data["y_test"],
            "model": "logistic",
            "task": "classification",
        }))
        assert res["model_type"] == "logistic"
        assert "train_score" in res

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_linear_regression(self, engine, regression_data):
        res = _parse(engine.train({
            "X_train": regression_data["X_train"],
            "y_train": regression_data["y_train"],
            "X_test": regression_data["X_test"],
            "y_test": regression_data["y_test"],
            "model": "linear",
            "task": "regression",
        }))
        assert res["model_type"] == "linear"
        assert "r2" in res["test_score"]

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_decision_tree(self, engine, classification_data):
        res = _parse(engine.train({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "X_test": classification_data["X_test"],
            "y_test": classification_data["y_test"],
            "model": "decision_tree",
            "task": "classification",
        }))
        assert res["model_type"] == "decision_tree"

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_svm(self, engine, classification_data):
        res = _parse(engine.train({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "X_test": classification_data["X_test"],
            "y_test": classification_data["y_test"],
            "model": "svm",
            "task": "classification",
        }))
        assert res["model_type"] == "svm"

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_gradient_boosting(self, engine, classification_data):
        res = _parse(engine.train({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "X_test": classification_data["X_test"],
            "y_test": classification_data["y_test"],
            "model": "gradient_boosting",
            "task": "classification",
        }))
        assert res["model_type"] == "gradient_boosting"

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_train_without_test(self, engine, classification_data):
        res = _parse(engine.train({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "model": "random_forest",
            "task": "classification",
        }))
        assert "train_score" in res
        assert "test_score" not in res

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_missing_training_data_error(self, engine):
        res = _parse(engine.train({"model": "random_forest"}))
        assert "error" in res

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_kmeans_clustering(self, engine, classification_data):
        res = _parse(engine.train({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "model": "kmeans",
            "task": "clustering",
        }))
        assert res["model_type"] == "kmeans"

    @pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
    def test_xgboost(self, engine, classification_data):
        res = _parse(engine.train({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "X_test": classification_data["X_test"],
            "y_test": classification_data["y_test"],
            "model": "xgboost",
            "task": "classification",
        }))
        assert res["model_type"] == "xgboost"


# ===================================================================
# Evaluate tests
# ===================================================================

class TestEvaluate:

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_classification_metrics(self, engine):
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 0, 1, 1, 1]
        res = _parse(engine.evaluate({
            "y_true": y_true, "y_pred": y_pred, "task": "classification",
        }))
        assert "accuracy" in res
        assert "f1" in res
        assert "precision" in res
        assert "recall" in res
        assert "confusion_matrix" in res
        assert res["n_samples"] == 8

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_regression_metrics(self, engine):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 5.2]
        res = _parse(engine.evaluate({
            "y_true": y_true, "y_pred": y_pred, "task": "regression",
        }))
        assert "mse" in res
        assert "rmse" in res
        assert "mae" in res
        assert "r2" in res
        assert res["r2"] > 0.9

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_perfect_predictions(self, engine):
        y = [0, 1, 0, 1, 0, 1]
        res = _parse(engine.evaluate({
            "y_true": y, "y_pred": y, "task": "classification",
        }))
        assert res["accuracy"] == 1.0
        assert res["f1"] == 1.0

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_missing_predictions_error(self, engine):
        res = _parse(engine.evaluate({"y_true": [1, 2, 3]}))
        assert "error" in res

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_length_mismatch_error(self, engine):
        res = _parse(engine.evaluate({
            "y_true": [1, 2, 3], "y_pred": [1, 2], "task": "classification",
        }))
        assert "error" in res


# ===================================================================
# Cross-validation tests
# ===================================================================

class TestCrossval:

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_classification_cv(self, engine, classification_data):
        res = _parse(engine.crossval({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "model": "random_forest",
            "task": "classification",
            "cv": 5,
        }))
        assert res["model"] == "random_forest"
        assert res["cv"] == 5
        assert len(res["fold_scores"]) == 5
        assert res["mean_score"] > 0.5
        assert res["std_score"] >= 0

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_regression_cv(self, engine, regression_data):
        res = _parse(engine.crossval({
            "X": regression_data["X"],
            "y": regression_data["y"],
            "model": "linear",
            "task": "regression",
            "cv": 3,
        }))
        assert res["scoring"] == "r2"
        assert len(res["fold_scores"]) == 3

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_missing_data_error(self, engine):
        res = _parse(engine.crossval({"model": "random_forest"}))
        assert "error" in res


# ===================================================================
# Tune tests
# ===================================================================

class TestTune:

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_grid_search(self, engine, classification_data):
        res = _parse(engine.tune({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "model": "random_forest",
            "task": "classification",
            "param_grid": {"n_estimators": [10, 50], "max_depth": [3, 5]},
            "method": "grid",
            "cv": 3,
        }))
        assert "best_params" in res
        assert "best_score" in res
        assert res["method"] == "grid"
        assert res["n_candidates"] == 4  # 2 x 2

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_random_search(self, engine, classification_data):
        res = _parse(engine.tune({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "model": "random_forest",
            "task": "classification",
            "param_grid": {"n_estimators": [10, 50, 100], "max_depth": [3, 5, 10, None]},
            "method": "random",
            "cv": 3,
            "n_iter": 5,
        }))
        assert "best_params" in res
        assert res["method"] == "random"

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_missing_param_grid_error(self, engine, classification_data):
        res = _parse(engine.tune({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "model": "random_forest",
        }))
        assert "error" in res


# ===================================================================
# Compare tests
# ===================================================================

class TestCompare:

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_compare_models(self, engine, classification_data):
        res = _parse(engine.compare({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "models": ["logistic", "random_forest", "decision_tree"],
            "task": "classification",
            "cv": 3,
        }))
        assert "comparison" in res
        assert "best_model" in res
        assert "ranking" in res
        assert len(res["ranking"]) == 3

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_compare_regression(self, engine, regression_data):
        res = _parse(engine.compare({
            "X": regression_data["X"],
            "y": regression_data["y"],
            "models": ["linear", "random_forest"],
            "task": "regression",
            "cv": 3,
        }))
        assert res["scoring"] == "r2"
        assert len(res["comparison"]) == 2

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_statistical_comparison(self, engine, classification_data):
        res = _parse(engine.compare({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "models": ["logistic", "random_forest"],
            "task": "classification",
            "cv": 5,
        }))
        if HAS_SCIPY and res.get("statistical_comparison"):
            sc = res["statistical_comparison"]
            assert "t_statistic" in sc
            assert "p_value" in sc


# ===================================================================
# Feature importance tests
# ===================================================================

class TestFeatureImportance:

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_builtin_importance(self, engine, classification_data):
        res = _parse(engine.feature_importance({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "model": "random_forest",
            "task": "classification",
            "feature_names": [f"feat_{i}" for i in range(10)],
        }))
        assert "importance" in res
        assert len(res["importance"]) == 10
        assert res["method"] == "builtin"

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_top_features(self, engine, classification_data):
        res = _parse(engine.feature_importance({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "model": "random_forest",
            "task": "classification",
        }))
        assert "top_features" in res
        assert len(res["top_features"]) <= 10

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_linear_model_coef_importance(self, engine, classification_data):
        res = _parse(engine.feature_importance({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "model": "logistic",
            "task": "classification",
        }))
        assert "importance" in res
        # Logistic regression uses coef_ instead of feature_importances_
        assert len(res["importance"]) == 10

    @pytest.mark.skipif(not HAS_SKLEARN or not HAS_SHAP, reason="sklearn and shap required")
    def test_shap_importance(self, engine, classification_data):
        # Use smaller subset for SHAP speed
        X_small = classification_data["X"][:50]
        y_small = classification_data["y"][:50]
        res = _parse(engine.feature_importance({
            "X": X_small,
            "y": y_small,
            "model": "random_forest",
            "task": "classification",
            "method": "shap",
        }))
        assert "importance" in res


# ===================================================================
# Learning curve tests
# ===================================================================

class TestLearningCurve:

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_learning_curve(self, engine, classification_data):
        res = _parse(engine.learning_curve({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "model": "random_forest",
            "task": "classification",
            "cv": 3,
        }))
        assert "train_sizes" in res
        assert "train_scores_mean" in res
        assert "val_scores_mean" in res
        assert len(res["train_sizes"]) == 10  # default linspace

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_custom_train_sizes(self, engine, classification_data):
        res = _parse(engine.learning_curve({
            "X": classification_data["X"],
            "y": classification_data["y"],
            "model": "decision_tree",
            "task": "classification",
            "cv": 3,
            "train_sizes": [0.3, 0.6, 1.0],
        }))
        assert len(res["train_sizes"]) == 3

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_regression_learning_curve(self, engine, regression_data):
        res = _parse(engine.learning_curve({
            "X": regression_data["X"],
            "y": regression_data["y"],
            "model": "linear",
            "task": "regression",
            "cv": 3,
        }))
        assert res["scoring"] == "r2"


# ===================================================================
# Ensemble tests
# ===================================================================

class TestEnsemble:

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_voting_classifier(self, engine, classification_data):
        res = _parse(engine.ensemble({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "X_test": classification_data["X_test"],
            "y_test": classification_data["y_test"],
            "method": "voting",
            "base_models": ["logistic", "random_forest", "decision_tree"],
            "task": "classification",
        }))
        assert "ensemble_score" in res
        assert "individual_scores" in res
        assert len(res["individual_scores"]) == 3

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_bagging_classifier(self, engine, classification_data):
        res = _parse(engine.ensemble({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "X_test": classification_data["X_test"],
            "y_test": classification_data["y_test"],
            "method": "bagging",
            "base_models": ["random_forest", "decision_tree"],
            "task": "classification",
        }))
        assert "ensemble_score" in res

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_stacking_classifier(self, engine, classification_data):
        res = _parse(engine.ensemble({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "X_test": classification_data["X_test"],
            "y_test": classification_data["y_test"],
            "method": "stacking",
            "base_models": ["logistic", "random_forest"],
            "meta_model": "logistic",
            "task": "classification",
        }))
        assert "ensemble_score" in res

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_ensemble_improvement(self, engine, classification_data):
        res = _parse(engine.ensemble({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "X_test": classification_data["X_test"],
            "y_test": classification_data["y_test"],
            "method": "voting",
            "base_models": ["logistic", "random_forest", "decision_tree"],
            "task": "classification",
        }))
        assert "improvement" in res
        assert "best_individual_score" in res

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_invalid_method_error(self, engine, classification_data):
        res = _parse(engine.ensemble({
            "X_train": classification_data["X_train"],
            "y_train": classification_data["y_train"],
            "X_test": classification_data["X_test"],
            "y_test": classification_data["y_test"],
            "method": "unknown_ensemble",
            "base_models": ["logistic"],
            "task": "classification",
        }))
        assert "error" in res

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_missing_training_data_error(self, engine):
        res = _parse(engine.ensemble({
            "method": "voting",
            "base_models": ["logistic", "random_forest"],
            "task": "classification",
        }))
        assert "error" in res
