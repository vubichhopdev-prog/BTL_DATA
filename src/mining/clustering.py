# File: src/mining/clustering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class ClusterMiner:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()

    def preprocess(self, df, cols):
        """Chuẩn hóa dữ liệu (Standardization) - Bắt buộc cho K-Means"""
        self.feature_cols = cols
        X = df[cols].copy()
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def find_optimal_k(self, X_scaled, k_range=range(2, 7)):
        """Trả về dữ liệu để vẽ biểu đồ Elbow & Silhouette"""
        inertia = []
        sil_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
            sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        return inertia, sil_scores

    def fit_predict(self, df, X_scaled):
        """Phân cụm và gán nhãn vào DataFrame"""
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = self.model.fit_predict(X_scaled)
        
        # Thêm cột Cluster vào dữ liệu gốc
        result_df = df.copy()
        result_df['Cluster'] = clusters
        return result_df

    def get_pca_2d(self, X_scaled):
        """Giảm chiều dữ liệu xuống 2D để vẽ hình"""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        return X_pca

    def profile_clusters(self, df_clustered):
        """Tính giá trị trung bình các đặc trưng theo từng cụm"""
        # Groupby theo Cluster và tính trung bình các cột đặc trưng
        profile = df_clustered.groupby('Cluster')[self.feature_cols].mean()
        
        # Tính thêm tỷ lệ lỗi (nếu có cột Machine failure)
        if 'Machine failure' in df_clustered.columns:
            profile['Failure_Rate_%'] = df_clustered.groupby('Cluster')['Machine failure'].mean() * 100
            profile['Count'] = df_clustered.groupby('Cluster')['Machine failure'].count()
            
        return profile