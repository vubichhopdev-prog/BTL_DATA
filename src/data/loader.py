# File: src/data/loader.py
import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Đọc dữ liệu từ file csv"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Không tìm thấy file tại: {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        print(f"-> Đã load dữ liệu: {df.shape}")
        return df

    def discretize_data(self, df, cols):
        """Rời rạc hóa dữ liệu (Binning) cho bài toán luật kết hợp"""
        df_bin = pd.DataFrame()
        for col in cols:
            # Chia thành 3 khoảng: Low, Medium, High
            df_bin[f'{col}_Bin'] = pd.qcut(df[col], q=3, labels=['Low', 'Medium', 'High'])
        
        # Mapping cột lỗi
        if 'Machine failure' in df.columns:
            df_bin['Status'] = df['Machine failure'].map({0: 'Normal', 1: 'Failure'})
            
        return df_bin