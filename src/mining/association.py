# File: src/mining/association.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

class AssociationMiner:
    def __init__(self, min_support=0.005, min_lift=1.2):
        self.min_support = min_support
        self.min_lift = min_lift

    def mine_rules(self, df_bin):
        """Chạy thuật toán tìm luật"""
        # One-hot encoding
        df_ohe = pd.get_dummies(df_bin).astype(bool)
        
        # Apriori
        print(f"-> Đang chạy Apriori (min_support={self.min_support})...")
        frequent_itemsets = apriori(df_ohe, min_support=self.min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            print("Không tìm thấy tập phổ biến.")
            return pd.DataFrame()

        # Sinh luật
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=self.min_lift)
        return rules

    def filter_failure_rules(self, rules):
        """Chỉ lấy các luật dẫn đến lỗi máy"""
        if rules.empty: return rules
        
        # Lọc luật có vế phải là Failure
        failure_rules = rules[rules['consequents'].apply(lambda x: 'Status_Failure' in str(x))]
        return failure_rules.sort_values(by='lift', ascending=False)