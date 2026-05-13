import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.automl_pipeline import CustomAutoML
from src.data_preprocessing import DataPreprocessor

class TestCustomAutoML:
    
    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        return X, y
    
    @pytest.fixture
    def automl(self):
        return CustomAutoML()
    
    def test_initialization(self, automl):
        """Тест инициализации AutoML"""
        assert automl.random_state == 42
        assert automl.best_model is None
        assert automl.best_model_name is None
    
    def test_get_models(self, automl):
        """Тест получения списка моделей"""
        models = automl._get_models_and_params()
        assert len(models) > 0
        assert 'RandomForest' in models
    
    def test_fit(self, automl, sample_data):
        """Тест обучения моделей"""
        X, y = sample_data
        best_model = automl.fit(X, y, verbose=False)
        assert best_model is not None
        assert automl.best_model is not None
    
    def test_evaluate(self, automl, sample_data):
        """Тест оценки модели"""
        X, y = sample_data
        automl.fit(X, y, verbose=False)
        
        # Разделяем данные для теста
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        automl.fit(X_train, y_train, verbose=False)
        metrics = automl.evaluate(X_test, y_test, save_plots=False)
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1