import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessing:
    
    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor('data/heart.csv')
    
    def test_load_data(self, preprocessor):
        """Тест загрузки данных"""
        df = preprocessor.load_data()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'target' in df.columns
    
    def test_handle_missing_values(self, preprocessor):
        """Тест обработки пропущенных значений"""
        df = preprocessor.load_data()
        df_clean = preprocessor.handle_missing_values(df)
        assert df_clean.isnull().sum().sum() == 0
    
    def test_split_data(self, preprocessor):
        """Тест разделения данных"""
        df = preprocessor.load_data()
        X_train, X_test, y_train, y_test = preprocessor.split_data(df)
        
        assert len(X_train) > len(X_test)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_scale_features(self, preprocessor):
        """Тест масштабирования признаков"""
        df = preprocessor.load_data()
        X_train, X_test, y_train, y_test = preprocessor.split_data(df)
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        assert np.abs(np.mean(X_train_scaled)) < 1e-10
        assert np.abs(np.std(X_train_scaled) - 1) < 0.1