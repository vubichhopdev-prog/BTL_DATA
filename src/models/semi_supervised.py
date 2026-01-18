# File: src/models/semi_supervised.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

class SelfTrainingModule:
    def __init__(self, base_model=None, threshold=0.9, max_iter=10):
        """
        threshold: Ngưỡng tự tin (chỉ lấy nhãn giả nếu xác suất > 90%)
        """
        self.base_model = base_model if base_model else RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.threshold = threshold
        self.max_iter = max_iter
        self.model = None

    def mask_labels(self, y, ratio=0.1):
        """
        Giả lập mất nhãn: Chỉ giữ lại ratio% (ví dụ 10%) nhãn, còn lại gán -1 (Unlabeled)
        """
        n_samples = len(y)
        n_labeled = int(n_samples * ratio)
        
        # Chọn ngẫu nhiên chỉ số để giữ nhãn
        indices = np.random.permutation(n_samples)
        labeled_idx = indices[:n_labeled]
        unlabeled_idx = indices[n_labeled:]
        
        y_masked = y.copy()
        # Gán -1 cho các phần tử không được chọn (coi như mất nhãn)
        if isinstance(y, pd.Series):
            y_masked.iloc[unlabeled_idx] = -1
        else:
            y_masked[unlabeled_idx] = -1
            
        return y_masked

    def fit(self, X, y_masked):
        """Quy trình Self-Training"""
        self.model = clone(self.base_model)
        
        # Tách dữ liệu có nhãn (Labeled) và chưa nhãn (Unlabeled -1)
        X_labeled = X[y_masked != -1]
        y_labeled = y_masked[y_masked != -1]
        
        X_unlabeled = X[y_masked == -1]
        
        print(f"Bắt đầu Self-Training với {len(X_labeled)} mẫu có nhãn...")
        
        for i in range(self.max_iter):
            # 1. Huấn luyện trên tập có nhãn hiện tại
            self.model.fit(X_labeled, y_labeled)
            
            if len(X_unlabeled) == 0:
                break
                
            # 2. Dự đoán xác suất trên tập chưa nhãn
            probs = self.model.predict_proba(X_unlabeled)
            pred_labels = self.model.predict(X_unlabeled)
            max_probs = probs.max(axis=1)
            
            # 3. Lọc những mẫu có độ tin cậy cao hơn ngưỡng (Threshold)
            high_conf_idx = np.where(max_probs >= self.threshold)[0]
            
            if len(high_conf_idx) == 0:
                print(f"Iter {i+1}: Không tìm thấy nhãn giả nào đủ tin cậy.")
                break
                
            # 4. Thêm nhãn giả (Pseudo-labels) vào tập huấn luyện
            X_pseudo = X_unlabeled.iloc[high_conf_idx]
            y_pseudo = pred_labels[high_conf_idx]
            
            X_labeled = pd.concat([X_labeled, X_pseudo])
            y_labeled = pd.concat([y_labeled, pd.Series(y_pseudo)])
            
            # Loại bỏ mẫu đã gán nhãn khỏi tập Unlabeled
            X_unlabeled = X_unlabeled.drop(X_unlabeled.index[high_conf_idx])
            
            print(f"Iter {i+1}: Đã thêm {len(X_pseudo)} nhãn giả (Pseudo-labels). Tổng mẫu train: {len(X_labeled)}")
            
        print("-> Hoàn tất Self-Training.")
        return self

    def predict(self, X):
        return self.model.predict(X)