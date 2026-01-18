# File: src/models/supervised.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

class FailurePredictor:
    def __init__(self):
        # Sử dụng Random Forest với class_weight='balanced' để xử lý dữ liệu mất cân bằng
        self.model = RandomForestClassifier(n_estimators=100, 
                                            class_weight='balanced', 
                                            random_state=42)
        self.feature_cols = []

    def prepare_data(self, df, target_col='Machine failure'):
        """Chia dữ liệu train/test"""
        # Loại bỏ các cột không dùng để train (như ID, cột target)
        drop_cols = ['UDI', 'Product ID', 'Type', target_col]
        # Nếu đã chạy binning ở bước trước, loại bỏ luôn các cột _Bin để tránh nhiễu
        drop_cols += [c for c in df.columns if '_Bin' in c or 'Cluster' in c]
        
        # Chỉ giữ lại các cột có trong DataFrame
        drop_cols = [c for c in drop_cols if c in df.columns]
        
        X = df.drop(columns=drop_cols)
        y = df[target_col]
        self.feature_cols = X.columns.tolist()
        
        # Chia train/test tỉ lệ 80-20, stratify=y để giữ nguyên tỉ lệ lỗi trong cả 2 tập
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train(self, X_train, y_train):
        """Huấn luyện mô hình"""
        print("-> Đang huấn luyện mô hình Random Forest...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Đánh giá mô hình và vẽ Confusion Matrix"""
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        print("\n=== BÁO CÁO ĐÁNH GIÁ (CLASSIFICATION REPORT) ===")
        print(classification_report(y_test, y_pred))
        
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

        # Vẽ Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix (Ma trận nhầm lẫn)')
        plt.xlabel('Dự đoán')
        plt.ylabel('Thực tế')
        plt.show()

    def feature_importance(self):
        """Xem đặc trưng nào quan trọng nhất gây ra lỗi"""
        importances = self.model.feature_importances_
        indices = pd.Series(importances, index=self.feature_cols).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=indices.values, y=indices.index, palette='viridis')
        plt.title("Các yếu tố ảnh hưởng nhất đến Lỗi Máy")
        plt.xlabel("Mức độ quan trọng")
        plt.show()