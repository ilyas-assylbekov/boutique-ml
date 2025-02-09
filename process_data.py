from src.data.data_processing import DataProcessor

processor = DataProcessor('data/raw/sales_data.csv')
processor.process_all('data/processed')