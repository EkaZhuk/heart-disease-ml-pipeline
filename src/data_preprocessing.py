"""Модуль предобработки данных для ML-пайплайна."""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import config
from src.logger import setup_logger

logger = setup_logger(__name__)


class DataPreprocessor:
    """Класс для загрузки, очистки и подготовки данных."""

    def __init__(self, file_path: str | None = None) -> None:
        self.file_path = file_path or str(config.data_path)
        self.scaler = StandardScaler()
        self.feature_columns: list[str] | None = None
        self.target_column = config.target_column

    def load_data(self) -> pd.DataFrame:
        """Загрузка данных из CSV файла.

        Returns:
            DataFrame с загруженными данными.

        Raises:
            FileNotFoundError: Если файл не найден.
        """
        try:
            df = pd.read_csv(self.file_path)
            logger.info("Данные загружены успешно. Размер: %s", df.shape)
            return df
        except FileNotFoundError:
            logger.error("Файл %s не найден", self.file_path)
            raise
        except Exception as e:
            logger.error("Ошибка при загрузке данных: %s", e)
            raise

    def basic_eda(self, df: pd.DataFrame) -> pd.DataFrame:
        """Базовый анализ данных (EDA).

        Args:
            df: Исходный DataFrame.

        Returns:
            Тот же DataFrame (для цепочки вызовов).
        """
        logger.info("=== Базовый анализ данных ===")
        logger.info("Размер датасета: %s", df.shape)
        logger.info("Колонки датасета: %s", df.columns.tolist())
        logger.info("Типы данных:\n%s", df.dtypes)
        logger.info("Пропущенные значения:\n%s", df.isnull().sum())
        logger.info("Статистическое описание:\n%s", df.describe())

        if self.target_column in df.columns:
            target_dist = df[self.target_column].value_counts()
            logger.info("Распределение целевой переменной:\n%s", target_dist)

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропущенных значений.

        Числовые — медианой, категориальные — модой.

        Args:
            df: DataFrame с возможными пропусками.

        Returns:
            DataFrame без пропусков.
        """
        if df.isnull().sum().sum() > 0:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col] = df[col].fillna(df[col].median())

            categorical_columns = df.select_dtypes(include=["object"]).columns
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float | None = None,
        random_state: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Разделение данных на обучающую и тестовую выборки.

        Args:
            df: Исходный DataFrame.
            test_size: Доля тестовой выборки. Если None — берётся из конфига.
            random_state: Seed для воспроизводимости. Если None — из конфига.

        Returns:
            Кортеж (X_train, X_test, y_train, y_test).
        """
        test_size = test_size if test_size is not None else config.test_size
        random_state = random_state if random_state is not None else config.random_state

        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        self.feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info("Размер обучающей выборки: %s", X_train.shape)
        logger.info("Размер тестовой выборки: %s", X_test.shape)

        return X_train, X_test, y_train, y_test

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Масштабирование признаков.

        Args:
            X_train: Обучающие признаки.
            X_test: Тестовые признаки.

        Returns:
            Кортеж (X_train_scaled, X_test_scaled).
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        config.ensure_dirs()
        joblib.dump(self.scaler, config.models_dir / "scaler.pkl")

        return X_train_scaled, X_test_scaled

    def preprocess_pipeline(
        self,
    ) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """Полный пайплайн предобработки данных.

        Returns:
            Кортеж (X_train_scaled, X_test_scaled, y_train, y_test).
        """
        config.ensure_dirs()

        df = self.load_data()
        self.basic_eda(df)
        df = self.handle_missing_values(df)
        X_train, X_test, y_train, y_test = self.split_data(df)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test
