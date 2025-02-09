import pandas as pd
import numpy as np
from typing import Dict, List

class DataProcessor:
    def __init__(self, raw_data_path: str):
        self.raw_data = pd.read_csv(raw_data_path)
        self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
        
    def create_daily_metrics(self) -> pd.DataFrame:
        daily = self.raw_data.groupby('date').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'item_type': lambda x: x.value_counts().index[0]
        }).reset_index()
        
        # Add average order value
        daily['average_order_value'] = daily['total_amount'] / daily['quantity']
        
        return daily
        
    def create_product_metrics(self) -> pd.DataFrame:
        product = self.raw_data.groupby(['item_name', 'item_type']).agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'price': ['mean', 'min', 'max']
        }).reset_index()
        
        # Add revenue share
        total_revenue = self.raw_data['total_amount'].sum()
        product['revenue_share'] = product['total_amount'] / total_revenue
        
        return product
        
    def create_seasonal_metrics(self) -> pd.DataFrame:
        seasonal = self.raw_data.groupby('season').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'price': 'mean'
        }).reset_index()
        
        return seasonal
        
    def create_price_metrics(self) -> pd.DataFrame:
        price = self.raw_data.groupby(['item_name', 'season']).agg({
            'price': ['mean', 'std', 'min', 'max', 'median']
        }).reset_index()
        
        return price
        
    def process_all(self, output_path: str):
        """Process all metrics and save to specified directory"""
        metrics = {
            'daily_metrics.csv': self.create_daily_metrics(),
            'product_metrics.csv': self.create_product_metrics(),
            'seasonal_metrics.csv': self.create_seasonal_metrics(),
            'price_metrics.csv': self.create_price_metrics()
        }
        
        for filename, df in metrics.items():
            df.to_csv(f"{output_path}/{filename}", index=False)
