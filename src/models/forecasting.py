# File: src/models/forecasting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ToolWearPredictor:
    def __init__(self):
        # Dùng Random Forest Regressor để dự đoán số thực (độ mòn dao)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def prepare_data(self, df):
        """Chuẩn bị dữ liệu chuỗi thời gian (Time-series)"""
        print("-> Đang chuẩn bị dữ liệu Hồi quy chuỗi thời gian...")
        
        # 1. Coi UDI là thời gian, sắp xếp lại theo thứ tự UDI
        df_time = df.sort_values('UDI').copy()
        
        # 2. Tạo Lag feature (Lấy độ mòn dao của chu kỳ liền trước làm feature)
        df_time['Tool_wear_lag1'] = df_time['Tool wear [min]'].shift(1)
        
        # Bỏ dòng đầu tiên bị NaN do lệnh shift
        df_time = df_time.dropna()

        # Chọn đặc trưng (Features)
        features = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool_wear_lag1']
        target = 'Tool wear [min]'

        X = df_time[features]
        y = df_time[target]

        # 3. CHIA TRAIN/TEST THEO THỨ TỰ (KHÔNG SHUFFLE) - Yêu cầu bắt buộc của Rubric
        split_idx = int(len(df_time) * 0.8) # 80% Train, 20% Test
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        print("-> Đang huấn luyện mô hình dự đoán Tool wear...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Đánh giá và vẽ biểu đồ"""
        y_pred = self.model.predict(X_test)
        
        # Tính MAE và RMSE
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print("\n=== KẾT QUẢ DỰ ĐOÁN ĐỘ MÒN DAO (HỒI QUY) ===")
        print(f"Mean Absolute Error (MAE): {mae:.4f} phút")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f} phút")

        # Vẽ biểu đồ so sánh 150 điểm test đầu tiên cho dễ nhìn
        plt.figure(figsize=(12, 5))
        plt.plot(y_test.values[:150], label='Độ mòn Thực tế', color='blue', alpha=0.7)
        plt.plot(y_pred[:150], label='Độ mòn Dự đoán', color='red', linestyle='--', alpha=0.8)
        
        plt.title('Dự báo Độ mòn dao theo thời gian (150 chu kỳ test)')
        plt.xlabel('Thời gian (Sequence)')
        plt.ylabel('Độ mòn dao [phút]')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()