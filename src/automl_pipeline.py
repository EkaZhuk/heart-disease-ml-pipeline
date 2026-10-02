"""AutoML-пайплайн для предсказания сердечных заболеваний."""

import warnings
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from src.config import config
from src.logger import setup_logger

warnings.filterwarnings("ignore")

logger = setup_logger(__name__)


class CustomAutoML:
    """Кастомный AutoML с автоматическим подбором моделей и гиперпараметров."""

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = (
            random_state if random_state is not None else config.random_state
        )
        self.models: dict[str, dict[str, Any]] = {}
        self.best_model: Any = None
        self.best_model_name: str | None = None
        self.metrics: dict[str, float] = {}
        self.cv = StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

    def _get_models_and_params(self) -> dict[str, dict[str, Any]]:
        """Определение моделей и сеток гиперпараметров.

        Returns:
            Словарь вида {имя_модели: {"model": estimator, "params": dict}}.
        """
        return {
            "RandomForest": {
                "model": RandomForestClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "GradientBoosting": {
                "model": GradientBoostingClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                },
            },
            "LogisticRegression": {
                "model": LogisticRegression(
                    random_state=self.random_state, max_iter=1000
                ),
                "params": {
                    "C": [0.1, 1, 10],
                    "penalty": ["l2"],
                    "solver": ["lbfgs", "liblinear"],
                },
            },
            "SVM": {
                "model": SVC(probability=True, random_state=self.random_state),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"],
                },
            },
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        verbose: bool = True,
    ) -> Any:
        """Обучение всех моделей с подбором гиперпараметров.

        Args:
            X_train: Обучающие признаки.
            y_train: Обучающие метки.
            verbose: Печатать ли прогресс.

        Returns:
            Лучшая обученная модель.
        """
        models_dict = self._get_models_and_params()
        best_score = 0.0

        if verbose:
            logger.info("=" * 60)
            logger.info("ЗАПУСК AUTOML: Автоматический подбор моделей")
            logger.info("=" * 60)

        for name, model_config in models_dict.items():
            if verbose:
                logger.info("Обучение модели: %s", name)
                logger.info("-" * 40)

            grid_search = GridSearchCV(
                model_config["model"],
                model_config["params"],
                cv=self.cv,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
            )

            grid_search.fit(X_train, y_train)

            self.models[name] = {
                "best_model": grid_search.best_estimator_,
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
            }

            if verbose:
                logger.info("   Лучшие параметры: %s", grid_search.best_params_)
                logger.info("   Лучший CV score: %.4f", grid_search.best_score_)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_model_name = name

        if verbose:
            logger.info("Лучшая модель: %s", self.best_model_name)
            logger.info("   CV Accuracy: %.4f", best_score)
            logger.info("=" * 60)

        return self.best_model

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: pd.Series,
        save_plots: bool = True,
        X_train: np.ndarray | None = None,
        y_train: pd.Series | None = None,
    ) -> dict[str, float]:
        """Оценка модели на тестовых данных + кросс-валидация.

        Args:
            X_test: Тестовые признаки.
            y_test: Тестовые метки.
            save_plots: Сохранять ли графики.
            X_train: Обучающие признаки (для CV).
            y_train: Обучающие метки (для CV).

        Returns:
            Словарь с метриками.

        Raises:
            ValueError: Если модель не обучена.
        """
        if self.best_model is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit()")

        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        metrics: dict[str, float] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        if X_train is not None and y_train is not None:
            cv_scores = cross_val_score(
                self.best_model,
                X_train,
                y_train,
                cv=self.cv,
                scoring="accuracy",
                n_jobs=-1,
            )
            metrics["cv_accuracy_mean"] = cv_scores.mean()
            metrics["cv_accuracy_std"] = cv_scores.std()

        self.metrics = metrics

        logger.info("МЕТРИКИ КАЧЕСТВА МОДЕЛИ")
        logger.info("=" * 40)
        for metric, value in metrics.items():
            logger.info("%-20s: %.4f", metric.upper(), value)

        if save_plots:
            self._plot_confusion_matrix(y_test, y_pred)
            self._plot_feature_importance()
            self._plot_roc_curve(y_test, y_pred_proba)

        return metrics

    def _plot_confusion_matrix(self, y_test: pd.Series, y_pred: np.ndarray) -> None:
        """Построение матрицы ошибок.

        Args:
            y_test: Истинные метки.
            y_pred: Предсказанные метки.
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {self.best_model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        config.ensure_dirs()
        plt.savefig(config.plots_dir / "confusion_matrix.png", dpi=100)
        plt.close()
        logger.info(
            "Confusion matrix сохранена в '%s'",
            config.plots_dir / "confusion_matrix.png",
        )

    def _plot_feature_importance(self) -> None:
        """Построение графика важности признаков (если модель их имеет)."""
        if self.best_model is not None and hasattr(
            self.best_model, "feature_importances_"
        ):
            plt.figure(figsize=(10, 6))
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]

            plt.bar(range(len(indices)), importances[indices])
            plt.title(f"Top 10 Feature Importances - {self.best_model_name}")
            plt.xlabel("Feature Index")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.savefig(config.plots_dir / "feature_importance.png", dpi=100)
            plt.close()
            logger.info(
                "Feature importance сохранена в '%s'",
                config.plots_dir / "feature_importance.png",
            )

    def _plot_roc_curve(self, y_test: pd.Series, y_pred_proba: np.ndarray) -> None:
        """Построение ROC-кривой.

        Args:
            y_test: Истинные метки.
            y_pred_proba: Вероятности положительного класса.
        """
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {self.metrics['roc_auc']:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {self.best_model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.plots_dir / "roc_curve.png", dpi=100)
        plt.close()
        logger.info(
            "ROC curve сохранена в '%s'",
            config.plots_dir / "roc_curve.png",
        )

    def save_model(self, filepath: str | None = None) -> None:
        """Сохранение лучшей модели.

        Args:
            filepath: Путь для сохранения. Если None — берётся из конфига.
        """
        config.ensure_dirs()
        filepath = filepath or str(config.models_dir / "heart_disease_model.pkl")
        joblib.dump(self.best_model, filepath)
        logger.info("Модель сохранена в '%s'", filepath)

    def compare_all_models(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Сравнение всех моделей.

        Args:
            X_train: Обучающие признаки (не используется, для единообразия API).
            y_train: Обучающие метки (не используется).
            X_test: Тестовые признаки.
            y_test: Тестовые метки.

        Returns:
            DataFrame с метриками всех моделей.
        """
        results: list[dict[str, Any]] = []

        logger.info("СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
        logger.info("=" * 60)

        for name, model_info in self.models.items():
            model = model_info["best_model"]
            y_pred = model.predict(X_test)

            results.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred),
                    "Recall": recall_score(y_test, y_pred),
                    "F1-Score": f1_score(y_test, y_pred),
                }
            )

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("Accuracy", ascending=False)
        logger.info("\n%s", results_df.to_string(index=False))
        logger.info("=" * 60)

        return results_df


def run_automl_pipeline(
    data_path: str | None = None,
) -> tuple[CustomAutoML, dict[str, float]]:
    """Запуск полного AutoML пайплайна.

    Args:
        data_path: Путь к CSV с данными. Если None — берётся из конфига.

    Returns:
        Кортеж (обученный AutoML, метрики).
    """
    data_path = data_path or str(config.data_path)

    logger.info("HEART DISEASE PREDICTION - AUTOML PIPELINE")
    logger.info(" Загрузка данных из %s...", data_path)

    from src.data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor(data_path)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()

    logger.info("Запуск автоматизированного обучения...")
    automl = CustomAutoML()
    automl.fit(X_train, y_train)

    metrics = automl.evaluate(X_test, y_test, X_train=X_train, y_train=y_train)

    results_df = automl.compare_all_models(X_train, y_train, X_test, y_test)

    automl.save_model()

    config.ensure_dirs()
    results_df.to_csv(config.results_dir / "model_comparison.csv", index=False)

    logger.info("AutoML пайплайн успешно завершён!")

    return automl, metrics


if __name__ == "__main__":
    run_automl_pipeline()
