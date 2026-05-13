# src/automl_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Добавляем корневую папку проекта в путь для импортов
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class CustomAutoML:
    """
    Кастомный AutoML с автоматическим подбором моделей и гиперпараметров
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
    def _get_models_and_params(self):
        """Определение моделей и сеток гиперпараметров"""
        return {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=self.random_state),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
    
    def fit(self, X_train, y_train, verbose=True):
        """Обучение всех моделей с подбором гиперпараметров"""
        models_dict = self._get_models_and_params()
        best_score = 0
        
        if verbose:
            print("="*60)
            print("ЗАПУСК AUTOML: Автоматический подбор моделей")
            print("="*60)
        
        for name, config in models_dict.items():
            if verbose:
                print(f"Обучение модели: {name}")
                print("-" * 40)
            
            # GridSearchCV для автоматического подбора гиперпараметров
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=self.cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            # Обучение
            grid_search.fit(X_train, y_train)
            
            # Сохраняем лучшую модель
            self.models[name] = {
                'best_model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
            if verbose:
                print(f"   Лучшие параметры: {grid_search.best_params_}")
                print(f"   Лучший CV score: {grid_search.best_score_:.4f}")
            
            # Обновляем лучшую модель
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_model_name = name
        
        if verbose:
            print(f"Лучшая модель: {self.best_model_name}")
            print(f"   CV Accuracy: {best_score:.4f}")
            print("="*60)
        
        return self.best_model
    
    def evaluate(self, X_test, y_test, save_plots=True):
        """Оценка модели на тестовых данных"""
        if self.best_model is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit()")
        
        # Предсказания
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Расчёт метрик
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.metrics = metrics
        
        print("МЕТРИКИ КАЧЕСТВА МОДЕЛИ")
        print("="*40)
        for metric, value in metrics.items():
            print(f"{metric.upper():<15}: {value:.4f}")
        
        # Визуализации
        if save_plots:
            self._plot_confusion_matrix(y_test, y_pred)
            self._plot_feature_importance()
            self._plot_roc_curve(y_test, y_pred_proba)
        
        return metrics
    
    def _plot_confusion_matrix(self, y_test, y_pred):
        """Построение матрицы ошибок"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/confusion_matrix.png', dpi=100)
        plt.close()
        print("Confusion matrix сохранена в 'plots/confusion_matrix.png'")
    
    def _plot_feature_importance(self):
        """Построение графика важности признаков"""
        if hasattr(self.best_model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            plt.bar(range(len(indices)), importances[indices])
            plt.title(f'Top 10 Feature Importances - {self.best_model_name}')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig('plots/feature_importance.png', dpi=100)
            plt.close()
            print("Feature importance сохранена в 'plots/feature_importance.png'")
    
    def _plot_roc_curve(self, y_test, y_pred_proba):
        """Построение ROC-кривой"""
        from sklearn.metrics import roc_curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.metrics["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.best_model_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/roc_curve.png', dpi=100)
        plt.close()
        print("ROC curve сохранена в 'plots/roc_curve.png'")
    
    def save_model(self, filepath='models/best_model.pkl'):
        """Сохранение лучшей модели"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.best_model, filepath)
        print(f"Модель сохранена в '{filepath}'")
    
    def compare_all_models(self, X_train, y_train, X_test, y_test):
        """Сравнение всех моделей"""
        results = []
        
        print("СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
        print("="*60)
        
        for name, model_info in self.models.items():
            model = model_info['best_model']
            y_pred = model.predict(X_test)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred)
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        print(results_df.to_string(index=False))
        print("="*60)
        
        return results_df

def run_automl_pipeline(data_path='data/heart.csv'):
    """Запуск полного AutoML пайплайна"""

    print("HEART DISEASE PREDICTION - AUTOML PIPELINE")
    
    
    # Загрузка данных
    print(" Загрузка данных...")
    df = pd.read_csv(data_path)
    print(f" Размер датасета: {df.shape}")
    
    # Подготовка данных
    from src.data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor(data_path)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    # AutoML
    print("Запуск автоматизированного обучения...")
    automl = CustomAutoML()
    best_model = automl.fit(X_train, y_train)
    
    # Оценка
    metrics = automl.evaluate(X_test, y_test)
    
    # Сравнение моделей
    results_df = automl.compare_all_models(X_train, y_train, X_test, y_test)
    
    # Сохранение модели
    automl.save_model('models/heart_disease_model.pkl')
    
    # Сохранение метрик
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/model_comparison.csv', index=False)
    
    print("AutoML пайплайн успешно завершён!")
    
    return automl, metrics

if __name__ == "__main__":
    run_automl_pipeline()