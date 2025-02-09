import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any
import joblib
from datetime import datetime

class BoutiqueModel:
    def __init__(self):
        self.sales_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_time_series_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка данных для прогнозирования временных рядов
        """
        # Убедимся, что дата в правильном формате
        df['date'] = pd.to_datetime(df['date'])
        
        # Календарные признаки
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Создаем лаги продаж (7 дней и 30 дней)
        for i in range(1, 8):
            df[f'sales_lag_{i}d'] = df['total_amount'].shift(i)
            
        # Добавляем скользящие средние
        df['sales_ma_7d'] = df['total_amount'].rolling(window=7).mean()
        df['sales_ma_30d'] = df['total_amount'].rolling(window=30).mean()
        
        # Добавляем стандартное отклонение за последние 7 дней
        df['sales_std_7d'] = df['total_amount'].rolling(window=7).std()
        
        # Удаляем строки с NaN после создания лагов
        df = df.dropna()
        
        # Определяем признаки для модели
        self.feature_columns = [
            'year', 'month', 'day_of_month', 'day_of_week', 'week_of_year',
            'is_weekend', 'sales_ma_7d', 'sales_ma_30d', 'sales_std_7d'
        ] + [f'sales_lag_{i}d' for i in range(1, 8)]
        
        X = df[self.feature_columns]
        y = df['total_amount']
        
        # Масштабируем признаки
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def train_sales_model(self, daily_metrics: pd.DataFrame):
        """Обучение модели прогнозирования продаж"""
        print("Preparing data...")
        X, y = self.prepare_time_series_data(daily_metrics)
        
        # Разделяем данные на обучающую и тестовую выборки
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print("Training model...")
        # Обновленные параметры LightGBM
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 20,  # уменьшаем для борьбы с переобучением
            'learning_rate': 0.03,  # уменьшаем для лучшей обобщающей способности
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,  # добавляем бэггинг
            'bagging_freq': 5,
            'n_estimators': 1000,
            'min_child_samples': 20,  # минимальное число наблюдений в листе
            'reg_alpha': 0.1,  # L1 регуляризация
            'reg_lambda': 0.1,  # L2 регуляризация
        }
        
        # Обучаем модель
        self.sales_model = lgb.LGBMRegressor(**params)
        self.sales_model.fit(
            X_train, 
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=['rmse', 'mae'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # Оцениваем модель
        train_predictions = self.sales_model.predict(X_train)
        test_predictions = self.sales_model.predict(X_test)
        
        print("\nModel Evaluation:")
        print("Training Set Metrics:")
        print(self.evaluate_model(y_train, train_predictions))
        print("\nTest Set Metrics:")
        print(self.evaluate_model(y_test, test_predictions))
        
        # Выводим важность признаков
        self.print_feature_importance()

    def cross_validate_model(self, X, y, n_splits=5):
        """Кросс-валидация модели"""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = {
            'rmse': [],
            'mae': [],
            'r2': [],
            'mape': []
        }
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.sales_model.fit(
                X_train, 
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=['rmse', 'mae'],
                callbacks=[lgb.early_stopping(stopping_rounds=30)],
            )
            
            y_pred = self.sales_model.predict(X_val)
            metrics = self.evaluate_model(y_val, y_pred)
            
            for metric, value in metrics.items():
                scores[metric.lower()].append(value)
        
        print("\nCross-validation scores:")
        for metric, values in scores.items():
            print(f"{metric.upper()}: {np.mean(values):.2f} (+/- {np.std(values):.2f})")
        
    def predict_sales(self, features: pd.DataFrame) -> np.ndarray:
        """Прогноз продаж"""
        if self.sales_model is None:
            raise ValueError("Model not trained yet")
            
        # Подготовка данных
        features = self.scaler.transform(features[self.feature_columns])
        return self.sales_model.predict(features)
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Оценка качества модели"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': mape
        }
    
    def print_feature_importance(self):
        """Вывод важности признаков"""
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.sales_model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance.head(10))
    
    def save_model(self, path: str):
        """Сохранение модели"""
        model_data = {
            'model': self.sales_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Загрузка модели"""
        model_data = joblib.load(path)
        self.sales_model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']

def main():
    # Загрузка данных
    print("Loading data...")
    daily_metrics = pd.read_csv('data/processed/daily_metrics.csv')
    
    # Инициализация и обучение моделей
    model = BoutiqueModel()
    model.train_sales_model(daily_metrics)

    model.evaluate_model(daily_metrics['total_amount'], model.predict_sales(daily_metrics))
    model.cross_validate_model(daily_metrics[model.feature_columns].values, daily_metrics['total_amount'].values)
    
    # Сохранение модели
    print("\nSaving model...")
    model.save_model('models/sales_forecast_model.joblib')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
