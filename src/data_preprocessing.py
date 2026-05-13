import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DataPreprocessor:
    def __init__(self, file_path='data/heart.csv'):
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'target'
        
    def load_data(self):
        """Загрузка данных из CSV файла"""
        try:
            df = pd.read_csv(self.file_path)
            print(f"Данные загружены успешно. Размер: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Ошибка: Файл {self.file_path} не найден")
            raise
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            raise
    
    def basic_eda(self, df):
        """Базовый анализ данных"""
        print("\n=== Базовый анализ данных ===")
        print(f"Размер датасета: {df.shape}")
        print(f"\nКолонки датасета: {df.columns.tolist()}")
        print(f"\nТипы данных:\n{df.dtypes}")
        print(f"\nПропущенные значения:\n{df.isnull().sum()}")
        print(f"\nСтатистическое описание:\n{df.describe()}")
        
        if self.target_column in df.columns:
            target_dist = df[self.target_column].value_counts()
            print(f"\nРаспределение целевой переменной:\n{target_dist}")
        
        return df
    
    def handle_missing_values(self, df):
        """Обработка пропущенных значений"""
        if df.isnull().sum().sum() > 0:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col].fillna(df[col].median(), inplace=True)
            
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Разделение данных на обучающую и тестовую выборки"""
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        
        self.feature_columns = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Размер обучающей выборки: {X_train.shape}")
        print(f"Размер тестовой выборки: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Масштабирование признаков"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        return X_train_scaled, X_test_scaled
    
    def preprocess_pipeline(self):
        """Полный пайплайн предобработки данных"""
        os.makedirs('models', exist_ok=True)
        
        df = self.load_data()
        self.basic_eda(df)
        df = self.handle_missing_values(df)
        X_train, X_test, y_train, y_test = self.split_data(df)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test