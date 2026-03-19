# Bài Tập Lớn Data Mining — Đề 16: Phân tích lỗi sản xuất & dự đoán lỗi 

Dự án triển khai một pipeline khai phá dữ liệu hoàn chỉnh trên bộ dữ liệu AI4I 2020 về bảo trì dự đoán, gồm: luật kết hợp (Apriori), phân cụm (K-Means + PCA), phân lớp dự đoán lỗi (imbalanced), học bán giám sát (Self-Training) khi thiếu nhãn, và dự báo hồi quy độ mòn dao theo chuỗi thời gian. Dữ liệu có 10.000 mẫu và 14 cột, không có missing values theo mô tả chính thức, đồng thời quan sát EDA xác nhận không có dòng trùng lặp. Tuy nhiên, tỷ lệ lỗi rất thấp (~3,39%) khiến bài toán phân lớp mất cân bằng mạnh, nên PR-AUC/F1 được ưu tiên hơn accuracy. Kết quả thực nghiệm cho thấy các luật kết hợp có Lift cao (tối đa 8,168431), phân cụm K=3 xuất hiện một cụm có Failure_Rate_% cao nhất 5,241730%, mô hình phân lớp đạt ROC-AUC 0,9842 và F1-score 0,9774 trên tập test; bài toán forecasting đạt MAE 4,3920 phút và RMSE 21,8645 phút.

**Mục lục (gợi ý)**  
1) Bối cảnh và mục tiêu  
2) Dữ liệu và từ điển dữ liệu  
3) Quy trình khai phá dữ liệu  
4) Cấu trúc repository và mô tả module `src/`  
5) Tiền xử lý và feature engineering  
6) Kết quả thực nghiệm theo từng module  
7) Hướng dẫn chạy repo và reproducibility + checklist trước khi nộp  
8) Tài liệu tham khảo  

**Bảng tóm tắt kết quả tổng quan (từ notebook/ảnh thực nghiệm)**

| Hạng mục | Metric chính | Kết quả | Ghi chú |
|---|---|---:|---|
| Dữ liệu | Số mẫu × số cột | 10.000 × 14 | Theo mô tả chính thức của dataset |
| Dữ liệu | Tỷ lệ lỗi `Machine failure=1` | **3,39%** | Imbalanced mạnh (EDA) |
| Association Rules | Lift (top 1) | **8,168431** | Support **()** |
| Clustering | K chọn để chạy | **3** | Theo Elbow plot (inertia) |
| Clustering | Failure_Rate_% cao nhất | **5,241730% (Cluster 0)** | Profiling cụm |
| Classification | ROC-AUC | **0,9842** | PR-AUC **()** |
| Classification | F1-score (pos=1) | **0,9774** | Từ log notebook |
| Semi-supervised | F1 (10% nhãn) | **0,7965 vs 0,7857** | Self-training không cải thiện |
| Forecasting | MAE / RMSE | **4,3920 / 21,8645** | SMAPE **()** |

## 1. Bối cảnh và mục tiêu

**Môn học:** Khai Phá Dữ Liệu (Data Mining)  
**Giảng viên:** entity["people","ThS. Lê Thị Thùy Trang","lecturer, data mining"]  
**Đề tài:** Đề 16 — Phân tích lỗi sản xuất & dự đoán lỗi (AI4I 2020 Predictive Maintenance)

Trong công nghiệp 4.0, bảo trì dự đoán (predictive maintenance) giúp giảm downtime và chi phí bảo trì đột xuất bằng cách phát hiện sớm rủi ro hỏng hóc từ dữ liệu vận hành/cảm biến. Bộ dữ liệu AI4I 2020 là một dataset tổng hợp (synthetic) mô phỏng dữ liệu bảo trì dự đoán trong công nghiệp, được công bố vì dữ liệu thực tế thường khó thu thập và khó chia sẻ. citeturn2view1

**Mục tiêu dự án và kỹ thuật sử dụng**

| STT | Mục tiêu | Kỹ thuật |
|---:|---|---|
| 1 | Tìm mẫu kết hợp/điều kiện vận hành dẫn đến lỗi | Apriori + Association Rules |
| 2 | Phân cụm trạng thái máy theo hành vi vận hành và nguy cơ | K-Means + PCA |
| 3 | Phân lớp lỗi/không lỗi trong điều kiện mất cân bằng | Random Forest, Decision Tree |
| 4 | Thực nghiệm học bán giám sát khi thiếu nhãn | SelfTrainingClassifier |
| 5 | Dự báo hồi quy độ mòn dao theo thời gian | RandomForestRegressor (chia theo thứ tự, không shuffle) |

**Tiêu chí thành công**

| Bài toán | Metric chính | Lý do lựa chọn |
|---|---|---|
| Classification (imbalanced) | **PR-AUC** (Average Precision) + **F1-score** | Precision–Recall hữu ích khi lớp dương hiếm; AP/PR-AUC tóm tắt đường PR. citeturn0search11turn0search3 |
| Clustering | Silhouette Score, Davies–Bouldin, Elbow (inertia) | Silhouette càng gần 1 càng tốt; Davies–Bouldin càng thấp càng tốt; inertia là mục tiêu tối ưu của KMeans. citeturn1search1turn1search5turn1search0 |
| Association Rules | Lift (lọc > 1,5), Confidence, Support | Lift/Confidence là metric chuẩn để đánh giá “độ thú vị” của luật. citeturn0search1turn0search5 |
| Forecasting/Regression | MAE, RMSE, SMAPE | MAE/RMSE là metric phổ biến cho hồi quy; SMAPE là sai số % đối xứng. citeturn5search0turn5search9 |

## 2. Dữ liệu và từ điển dữ liệu

**Nguồn dữ liệu.** Dataset AI4I 2020 được cung cấp bởi entity["organization","UCI Machine Learning Repository","uci dataset portal"], gồm 10.000 dòng và 14 biến; tác vụ: classification và regression; và được ghi nhận “không có missing values” trong metadata. citeturn2view1turn2view0

**Thông tin dataset**

| Thuộc tính | Giá trị |
|---|---|
| Tên dataset | AI4I 2020 Predictive Maintenance Dataset |
| Số dòng × số cột | 10.000 × 14 citeturn2view1 |
| Định dạng | Multivariate, time-series citeturn2view1 |
| Tác vụ | Classification, Regression citeturn2view1 |
| File (tham chiếu repo) | `data/raw/ai4i2020.csv` |
| Ghi chú | `Machine failure` tổng hợp từ 5 mode lỗi (TWF/HDF/PWF/OSF/RNF). citeturn2view1 |

**Data dictionary (14 cột)**  
(Bảng dưới bám theo “Variables/Features table” của UCI.)

| Cột | Vai trò | Kiểu | Đơn vị | Missing |
|---|---|---:|---:|---:|
| UID | ID | Integer | – | No citeturn2view0turn2view1 |
| Product ID | ID | Categorical | – | No citeturn2view0turn2view1 |
| Type | Feature | Categorical | – | No citeturn2view0turn2view1 |
| Air temperature | Feature | Continuous | K | No citeturn2view0turn2view1 |
| Process temperature | Feature | Continuous | K | No citeturn2view0turn2view1 |
| Rotational speed | Feature | Integer | rpm | No citeturn2view0turn2view1 |
| Torque | Feature | Continuous | Nm | No citeturn2view0turn2view1 |
| Tool wear | Feature | Integer | min | No citeturn2view0turn2view1 |
| Machine failure | Target | Integer | – | No citeturn2view0turn2view1 |
| TWF | Target | Integer | – | No citeturn2view0turn2view1 |
| HDF | Target | Integer | – | No citeturn2view0turn2view1 |
| PWF | Target | Integer | – | No citeturn2view0turn2view1 |
| OSF | Target | Integer | – | No citeturn2view0turn2view1 |
| RNF | Target | Integer | – | No citeturn2view0turn2view1 |

**Định nghĩa các mode lỗi (tóm tắt, theo mô tả UCI)**  
Phần này quan trọng để diễn giải business insight và kiểm soát “data leakage”.

| Mode lỗi | Điều kiện kích hoạt (tóm tắt) |
|---|---|
| TWF | Tool wear đạt ngưỡng ngẫu nhiên 200–240 phút → dao được thay hoặc bị fail. citeturn2view1 |
| HDF | Chênh lệch nhiệt độ (air–process) < 8,6K **và** speed < 1380 rpm. citeturn2view1 |
| PWF | Công suất tính từ torque × tốc độ (rad/s) < 3500W **hoặc** > 9000W. citeturn2view1 |
| OSF | Tool wear × torque vượt ngưỡng (L: 11.000; M: 12.000; H: 13.000 minNm). citeturn2view1 |
| RNF | Xác suất lỗi ngẫu nhiên 0,1% mỗi process. citeturn2view1 |

**Lưu ý quan trọng về nhãn `Machine failure`.** UCI mô tả `Machine failure = 1` nếu *ít nhất một* mode lỗi phía trên xảy ra; do đó nếu dùng các cột TWF/HDF/PWF/OSF/RNF làm feature để dự đoán `Machine failure`, mô hình có thể tăng điểm mạnh vì đã “nhìn thấy” nguyên nhân trực tiếp. citeturn2view1

## 3. Quy trình khai phá dữ liệu

Pipeline tổng thể (tương tự mẫu Đề 13, nhưng điều chỉnh theo bài toán AI4I 2020):

```text
┌──────────────┐   ┌──────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐   ┌──────────────────┐
│ Data Source   │→  │ EDA + Cleaning   │→  │ Feature Engineering │→  │ Mining / Modeling     │→  │ Evaluation       │
│ ai4i2020.csv  │   │ (drop/encode)    │   │ (binning/scale)     │   │ (rules/cluster/ML)    │   │ (metrics/plots)  │
└──────────────┘   └──────────────────┘   └─────────────────────┘   └──────────────────────┘   └──────────────────┘
```

**Tóm tắt các nhánh kỹ thuật**
- **Association Rules:** Apriori để tìm frequent itemsets theo ngưỡng support, sau đó sinh luật và đánh giá bằng confidence/lift. citeturn3search2turn0search5  
- **Clustering:** KMeans (tối ưu inertia) + PCA 2D để trực quan hóa phân nhóm. citeturn1search0turn3search0  
- **Classification:** Random Forest/Decision Tree; báo cáo precision/recall/F1 + ROC-AUC; bổ sung PR-AUC bằng Average Precision khi dữ liệu lệch lớp. citeturn4search1turn0search3  
- **Semi-supervised:** SelfTrainingClassifier lặp gán pseudo-label theo ngưỡng xác suất và dừng khi hết lượt hoặc không thêm được nhãn mới. citeturn0search2turn0search10  
- **Forecasting (Regression):** RandomForestRegressor dự đoán `Tool wear [min]` theo thứ tự thời gian; đánh giá MAE/RMSE/SMAPE. citeturn4search2turn5search0  

## 4. Cấu trúc repository và mô tả module `src/`

**Cấu trúc thư mục (dạng chuẩn để nộp, bám theo repo thực tế trong ảnh)**

```text
BTL_DATA/
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   └── processed/
│       └── ai4i2020_processed.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining.ipynb
│   ├── 04_modeling.ipynb
│   ├── 04b_semi_supervised.ipynb
│   └── 05_evaluation_report.ipynb
├── src/
│   ├── data/
│   │   └── loader.py
│   ├── mining/
│   │   ├── association.py
│   │   └── clustering.py
│   └── models/
│       ├── supervised.py
│       ├── semi_supervised.py
│       └── forecasting.py
├── outputs/
│   ├── figures/
│   └── tables/
├── requirements.txt
└── blog.md
```

**Vai trò các module chính (tóm tắt theo luồng notebook)**

| Module | Thành phần | Trách nhiệm |
|---|---|---|
| `src/data/loader.py` | `DataLoader` | Tải dữ liệu (`load_data`), rời rạc hóa dữ liệu (`discretize_data`) cho Apriori |
| `src/mining/association.py` | `AssociationMiner` | Chạy Apriori và sinh luật; lọc luật liên quan “Status_Failure” theo lift/confidence citeturn3search2turn0search5 |
| `src/mining/clustering.py` | `ClusterMiner` | Chuẩn hóa (scaling), dò K tối ưu (Elbow/silhouette), fit KMeans, profiling cụm, PCA 2D citeturn1search0turn3search0turn3search1 |
| `src/models/supervised.py` | `FailurePredictor` | Chuẩn bị train/test, train RandomForest/DecisionTree, xuất classification report + confusion matrix + ROC-AUC citeturn4search1turn1search2turn1search3 |
| `src/models/semi_supervised.py` | `SelfTrainingModule` | Mask nhãn theo tỷ lệ, chạy self-training theo threshold/max_iter citeturn0search2turn0search10 |
| `src/models/forecasting.py` | `ToolWearPredictor` | Chia tập theo thứ tự thời gian (no shuffle), dự đoán `Tool wear`, đánh giá MAE/RMSE/SMAPE citeturn4search2turn5search0 |

## 5. Tiền xử lý và feature engineering

### Tiền xử lý (preprocessing) — các thao tác đã thực hiện

Các bước dưới đây lấy trực tiếp từ notebook `02_preprocess_feature.ipynb` (ảnh code/console đã cung cấp).

| Bước | Thao tác | Cột liên quan | Output |
|---|---|---|---|
| Đọc dữ liệu | `pd.read_csv('../data/raw/ai4i2020.csv')` | – | `df_raw` |
| Drop cột định danh | `drop(['UDI','Product ID'])` | `UDI`, `Product ID` | `df_clean` |
| Encode biến phân loại | `Type: {'L':0,'M':1,'H':2}` | `Type` | `Type` dạng số |
| Lưu dữ liệu sạch | `to_csv('../data/processed/ai4i2020_processed.csv')` | – | File processed |

**Giải thích ngắn:**  
- `UDI` và `Product ID` là các ID, thường không mang ý nghĩa dự đoán và có thể gây overfitting/ghi nhớ.  
- `Type` là categorical nên được mã hóa dạng số để mô hình học máy xử lý dễ hơn.

### Feature engineering theo từng nhánh (rules/cluster/classification/forecasting)

**Danh sách sensor chính được sử dụng lặp lại trong nhiều module (theo notebook):**

| Nhóm | Cột |
|---|---|
| Sensor liên tục/chính | `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]` |
| Label | `Machine failure` |
| Mode lỗi | `TWF`, `HDF`, `PWF`, `OSF`, `RNF` |

**Thiết kế feature theo từng bài toán**

| Nhánh | Feature engineering | Ghi chú kỹ thuật |
|---|---|---|
| Association Rules | Rời rạc hóa (binning) 5 sensor thành `Bin_Low/Medium/High`, tạo token trạng thái lỗi | Apriori của mlxtend triệu hồi frequent itemsets theo support và trả về `['support','itemsets']`. citeturn3search6turn0search5 |
| Clustering | Chuẩn hóa dữ liệu bằng scaling trước KMeans | chuẩn hóa giúp các biến cùng thang đo; StandardScaler đưa dữ liệu về mean=0, variance=1. citeturn3search1 |
| Classification | Chia train/test (20% test theo notebook), training RandomForest | Cần bổ sung PR-AUC bằng `average_precision_score`. citeturn0search3turn3search3 |
| Semi-supervised | Che nhãn theo tỷ lệ; self-training thêm pseudo-label theo threshold | Dừng khi đạt `max_iter` hoặc không thêm được nhãn mới. citeturn0search2turn0search10 |
| Forecasting (Regression) | Split theo thứ tự thời gian (no shuffle) để dự báo `Tool wear` | `train_test_split` có tham số `shuffle`; với chuỗi thời gian nên giữ thứ tự. citeturn3search3 |

### Tham số cấu hình (mẫu `configs/params.yaml`)

Repo hiện tại trong ảnh chưa thể hiện file YAML, nhưng để bám sát “chuẩn Đề 13” và tăng reproducibility, có thể thêm `configs/params.yaml` như mẫu sau (có placeholder để bạn điền theo code cuối):

```yaml
seed: 42

paths:
  raw_data: data/raw/ai4i2020.csv
  processed_data: data/processed/ai4i2020_processed.csv
  outputs_dir: outputs
  figures_dir: outputs/figures
  tables_dir: outputs/tables

preprocessing:
  drop_cols: ["UDI", "Product ID"]
  type_mapping: {"L": 0, "M": 1, "H": 2}

association:
  cols_bin:
    - "Air temperature [K]"
    - "Process temperature [K]"
    - "Rotational speed [rpm]"
    - "Torque [Nm]"
    - "Tool wear [min]"
  min_support: 0.005
  min_lift: 1.5
  top_k_rules: 10

clustering:
  cols_obs:
    - "Air temperature [K]"
    - "Process temperature [K]"
    - "Rotational speed [rpm]"
    - "Torque [Nm]"
    - "Tool wear [min]"
  k_min: 2
  k_max: 6
  selected_k: 3

classification:
  target_col: "Machine failure"
  test_size: 0.2
  random_state: 42
  models: ["decision_tree", "random_forest"]
  random_forest:
    n_estimators: <INSERT_N_ESTIMATORS>
    max_depth: <INSERT_MAX_DEPTH_OR_NULL>
    class_weight: <INSERT_CLASS_WEIGHT_OR_NULL>

semi_supervised:
  labeled_percents: [10, 20, 30]
  threshold: 0.85
  max_iter: 5

forecasting:
  target_col: "Tool wear [min]"
  train_size: 7999
  test_size: 2000
  no_shuffle: true
```


## 6. Kết quả thực nghiệm theo từng module

### Kết quả EDA (khám phá dữ liệu)

**Tóm tắt chất lượng dữ liệu (từ notebook):**

| Hạng mục | Kết quả |
|---|---:|
| Kích thước DataFrame | 10.000 × 14 |
| Missing values | 0 |
| Duplicates | 0 |
| Tỷ lệ lỗi `Machine failure=1` | **3,39%** |

**Các tương quan nổi bật (đọc từ heatmap trong notebook)**  
(Nhằm minh họa quan hệ giữa sensor và nhãn.)

| Cặp biến | Hệ số tương quan |
|---|---:|
| Air temperature ↔ Process temperature | **0,88** |
| Rotational speed ↔ Torque | **-0,88** |
| Machine failure ↔ HDF | **0,58** |
| Machine failure ↔ OSF | **0,53** |
| Machine failure ↔ PWF | **0,52** |

**Nhận xét:**  
- Tương quan mạnh giữa `Air temperature` và `Process temperature` là hợp lý vì UCI mô tả `Process temperature` được tạo từ `Air temperature + 10K` và biến thiên theo random walk. citeturn2view1  
- `Machine failure` tương quan đáng kể với các mode lỗi (HDF/PWF/OSF), phù hợp với định nghĩa `Machine failure` là nhãn tổng hợp từ các mode lỗi. citeturn2view1

### Kết quả Association Rules (Apriori)

**Thiết lập (theo notebook):**  
- Binning các sensor: `Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, `Tool wear` → `Bin_Low/Medium/High`.  
- Chạy Apriori với `min_support=0.005` và lọc luật `lift >= 1.5`. (Apriori tìm frequent itemsets theo ngưỡng support; association_rules sinh luật và tính confidence/lift.) citeturn3search2turn0search5  

**Top 10 luật liên quan lỗi (`Status_Failure`)**  

| # | Antecedents (rút gọn như notebook, cần thay bằng bản đầy đủ khi có CSV) | Consequents (rút gọn) | Support | Confidence | Lift |
|---:|---|---|---:|---:|---:|
| 1 | Rotational speed_Bin_Low, Process tempe… | Torque_Bin_High, Status_Failure | () `<INSERT_SUPPORT_1>` | **0,216463** | **8,168431** |
| 2 | Process temperature_Bin_Medium, Air tempe… | Rotational speed_Bin_Low, Status_Failure | () `<INSERT_SUPPORT_2>` | **0,213855** | **8,039678** |
| 3 | Rotational speed_Bin_Low, Process tempe… | Status_Failure | () `<INSERT_SUPPORT_3>` | **0,268939** | **7,933315** |
| 4 | Air temperature_Bin_High, Torque_Bin… | Rotational speed_Bin_Low, Process tempe… | () `<INSERT_SUPPORT_4>` | **0,076487** | **7,498750** |
| 5 | Rotational speed_Bin_Low, Air temperatu… | Process temperature_Bin_High, Status_Fail… | () `<INSERT_SUPPORT_5>` | **0,095972** | **7,382428** |
| 6 | Rotational speed_Bin_Low, Air temperatu… | Torque_Bin_High, Process temperature… | () `<INSERT_SUPPORT_6>` | **0,076705** | **7,375437** |
| 7 | Air temperature_Bin_High, Tool wear… | Rotational speed_Bin_Low, Status_Failure | () `<INSERT_SUPPORT_7>` | **0,190202** | **7,150441** |
| 8 | Rotational speed_Bin_Low, Air temperatu… | Torque_Bin_High, Status_Failure | () `<INSERT_SUPPORT_8>` | **0,189112** | **7,136292** |
| 9 | Rotational speed_Bin_Low, Air temperatu… | Status_Failure | () `<INSERT_SUPPORT_9>` | **0,235714** | **6,953224** |
| 10 | Process temperature_Bin_Medium, Air tempe… | Status_Failure | () `<INSERT_SUPPORT_10>` | **0,225610** | **6,655155** |

- Lift cao (≈6,66–8,17) cho thấy các tổ hợp điều kiện vận hành (đặc biệt liên quan speed thấp và torque cao) làm tăng xác suất lỗi so với mức nền. Khái niệm confidence/lift là metric chuẩn khi đánh giá luật kết hợp. citeturn0search1turn0search5  

### Kết quả Clustering (K-Means + PCA)

**Thiết lập:**  
- Chuẩn hóa dữ liệu trước KMeans (khuyến nghị StandardScaler). citeturn3search1  
- Dò K theo Elbow bằng inertia (K=2→6 trong notebook). Inertia là tiêu chí tối ưu của KMeans. citeturn1search0  
- Trực quan PCA 2D: PCA là phép giảm chiều tuyến tính bằng SVD để chiếu dữ liệu xuống không gian thấp chiều. citeturn3search0  

**Bảng scores theo K (placeholder vì ảnh không cung cấp số inertia/silhouette cụ thể)**

| K | Inertia | Silhouette Score | Davies–Bouldin |
|---:|---:|---:|---:|
| 2 | () `<INSERT_INERTIA_K2>` | () `<INSERT_SIL_K2>` | () `<INSERT_DB_K2>` |
| 3 | () `<INSERT_INERTIA_K3>` | () `<INSERT_SIL_K3>` | () `<INSERT_DB_K3>` |
| 4 | () `<INSERT_INERTIA_K4>` | () `<INSERT_SIL_K4>` | () `<INSERT_DB_K4>` |
| 5 | () `<INSERT_INERTIA_K5>` | () `<INSERT_SIL_K5>` | () `<INSERT_DB_K5>` |
| 6 | () `<INSERT_INERTIA_K6>` | () `<INSERT_SIL_K6>` | () `<INSERT_DB_K6>` |

Silhouette tốt nhất gần 1 và kém nhất -1; Davies–Bouldin càng thấp càng tốt. citeturn1search1turn1search5  

**Profiling cụm (K=3) — số liệu có trong ảnh notebook**

| Cluster | Air temperature [K] | Process temperature [K] | Rotational speed [rpm] | Torque [Nm] | Tool wear [min] | Failure_Rate_% | Count |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 301,723308 | 311,246590 | 1469,807379 | 43,643333 | 110,358270 | **5,241730** | 3930 |
| 1 | 298,264252 | 308,741524 | 1474,144884 | 43,264600 | 104,624844 | 2,165795 | 4017 |
| 2 | 300,121383 | 310,103166 | **1797,261568** | **26,574233** | 109,850950 | 2,240623 | 2053 |

**Nhận xét:**  
- Cụm 0 có Failure_Rate_% cao nhất (5,241730%) và đồng thời có nhiệt độ/độ mòn dao trung bình cao → là **cụm nguy cơ** để ưu tiên giám sát/bảo trì.  
- PCA 2D cho thấy 3 vùng cụm tách biệt tương đối rõ trong mặt phẳng (phục vụ diễn giải trực quan). citeturn3search0  

### Kết quả Classification (Supervised)

**Thiết lập:** train 8000, test 2000 (theo notebook).  
Random Forest là meta-estimator fit nhiều cây trên các mẫu con và dùng averaging để cải thiện độ chính xác và kiểm soát overfitting. citeturn4search1  

**Bảng metrics đầy đủ (yêu cầu: precision/recall/f1/PR-AUC/ROC-AUC)**  
PR-AUC cần tính thêm từ xác suất dự đoán (`predict_proba`) bằng `average_precision_score` (AP). citeturn0search3  

| Model | Precision (pos=1) | Recall (pos=1) | F1 (pos=1) | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest (best) | **1,0000** | **0,9559** | **0,9774** | **0,9842** | () `<INSERT_PR_AUC_RF>` |
| Decision Tree | () `<INSERT_PREC_DT>` | () `<INSERT_REC_DT>` | () `<INSERT_F1_DT>` | () `<INSERT_ROC_DT>` | () `<INSERT_PR_DT>` |

**Confusion Matrix (Random Forest)**  
(Confusion matrix là bảng C với C[i,j] = số mẫu thuộc lớp thật i nhưng dự đoán j.) citeturn1search3  

|  | Pred=0 | Pred=1 |
|---|---:|---:|
| Actual=0 | **1932** | **0** |
| Actual=1 | **3** | **65** |

- Precision đạt 1,0 vì FP=0 (không cảnh báo nhầm), Recall ~0,956 vì FN=3 (bỏ sót 3 ca lỗi). Công thức precision = tp/(tp+fp) là định nghĩa chuẩn của scikit-learn. citeturn0search19  
- Do dữ liệu lệch lớp, nên bổ sung PR-AUC/Average Precision vì PR-curve được khuyến nghị khi classes rất imbalanced. citeturn0search11turn0search3  
- **Điểm cần minh bạch:** nếu feature set có bao gồm các cột TWF/HDF/PWF/OSF/RNF, kết quả có thể cao do `Machine failure` được định nghĩa từ chính các mode lỗi đó; cần ghi rõ hoặc chạy thêm phiên bản “sensor-only”. citeturn2view1  

### Kết quả Semi-supervised (Self-Training)

SelfTrainingClassifier lặp: dự đoán pseudo-label cho dữ liệu chưa gán nhãn, thêm những mẫu đủ tự tin, và dừng khi hết lượt hoặc không thêm được nhãn mới. citeturn0search2turn0search10  

**Thiết lập (theo notebook):**  
- Giữ 10% nhãn trên tập train (8000) → 800 có nhãn, 7200 bị ẩn nhãn.  
- `threshold=0.85`, `max_iter=5`.

**Bảng so sánh supervised vs self-training theo % nhãn**  
(Ảnh chỉ có kết quả cho 10%. Các mức 20/30 đặt placeholder theo yêu cầu.)

| % nhãn giữ lại | Phương pháp | F1-score | Pseudo coverage | Pseudo accuracy |
|---:|---|---:|---:|---:|
| 10% | supervised_only | **0,7965** | – | – |
| 10% | self_training | **0,7857** | **97,32%** (từ log: 7007/7200) | () `<INSERT_PSEUDO_ACC_10>` |
| 20% | supervised_only | () `<INSERT_F1_SUP_20>` | – | – |
| 20% | self_training | () `<INSERT_F1_ST_20>` | () `<INSERT_COV_20>` | () `<INSERT_ACC_20>` |
| 30% | supervised_only | () `<INSERT_F1_SUP_30>` | – | – |
| 30% | self_training | () `<INSERT_F1_ST_30>` | () `<INSERT_COV_30>` | () `<INSERT_ACC_30>` |

**Nhận xét:** Trong thực nghiệm 10% nhãn, self-training không cải thiện F1 so với supervised-only. Điều này có thể xảy ra nếu pseudo-label không cung cấp thêm thông tin hữu ích hoặc nếu mô hình nền đã học được cấu trúc tốt từ phần nhãn ít.

### Kết quả Forecasting/Regression (Tool wear)

RandomForestRegressor là meta-estimator fit nhiều cây hồi quy và trung bình hóa dự đoán để cải thiện ổn định. citeturn4search2  
MAE/MAPE là metric chuẩn; SMAPE thường được định nghĩa theo công thức đối xứng. citeturn5search9turn5search0  

**Thiết lập (theo notebook):** train 7999 dòng, test 2000 dòng; chia theo thứ tự thời gian (no shuffle).

**Bảng metrics (theo notebook + placeholder)**

| Model | MAE (phút) | RMSE (phút) | SMAPE |
|---|---:|---:|---:|
| Random Forest Regressor | **4,3920** | **21,8645** | () `<INSERT_SMAPE>` |

**Placeholder biểu đồ dự báo**  
(Để repo “đúng chuẩn nộp”, nên xuất hình vào outputs.)

```text
outputs/figures/forecast_tool_wear.png  (placeholder)
```

### Đề xuất 3 biểu đồ PNG cần xuất vào `outputs/figures/` (để đính kèm bài nộp)

| Tên file | Mô tả | Dùng trong mục |
|---|---|---|
| `class_distribution.png` | Phân phối `Machine failure` (0/1) để minh họa imbalance 3,39% | EDA |
| `association_top_rules.png` | Bar chart Top-K rules theo Lift (lọc consequent = Status_Failure) | Association Rules |
| `clustering_pca_k3.png` | Scatter PCA 2D tô màu theo cụm (K=3) | Clustering |

### Đề xuất hành động (business insights) và hướng phát triển kỹ thuật

**Business insights (đề xuất hành động)**
- Ưu tiên giám sát và bảo trì cho **Cluster 0** (Failure_Rate_% cao nhất 5,241730%), vì đây là nhóm có tín hiệu rủi ro rõ ràng trong profiling.
- Dựa trên luật kết hợp (Lift cao), thiết lập rule-based alert ở tầng vận hành: khi xuất hiện tổ hợp điều kiện theo các rules top (đặc biệt liên quan speed thấp/torque cao), thực hiện kiểm tra nhanh (quick check) trước khi tiếp tục chạy.
- Dùng dự báo Tool wear để lập lịch thay dao: tránh thay quá sớm (lãng phí) và tránh thay quá muộn (tăng xác suất lỗi).

**Hướng phát triển kỹ thuật**
- **Kiểm soát leakage:** chạy thêm một phiên bản classification “sensor-only” (drop TWF/HDF/PWF/OSF/RNF) để phản ánh đúng bài toán dự đoán từ cảm biến; điều này phù hợp với định nghĩa nhãn `Machine failure` trong dataset. citeturn2view1  
- Tối ưu mô hình cho imbalanced: dùng `class_weight='balanced'` (hoặc tính class weight), và báo cáo PR-AUC. Công thức class weight “balanced” được scikit-learn mô tả theo tần suất lớp. citeturn4search12turn0search3  
- Van hóa “reproducible”: gom mọi tham số vào `params.yaml`, lưu output bảng/hình theo chuẩn, cố định seed ở KMeans/RandomForest.

## 7. Hướng dẫn chạy repo và reproducibility

### Cài đặt và chạy

**Cài thư viện**
```bash
pip install -r requirements.txt
```

**Chuẩn bị dữ liệu**
- Tải `ai4i2020.csv` theo nguồn UCI và đặt vào `data/raw/ai4i2020.csv`. citeturn2view1  

**Chạy theo notebook (khuyến nghị theo thứ tự)**
```text
notebooks/01_eda.ipynb
notebooks/02_preprocess_feature.ipynb
notebooks/03_mining.ipynb
notebooks/04_modeling.ipynb
notebooks/04b_semi_supervised.ipynb
notebooks/05_evaluation_report.ipynb
```

### Reproducibility (tái lập kết quả)

- Cố định `seed/random_state` cho các bước có ngẫu nhiên (train/test split, KMeans, RandomForest). `train_test_split` có tham số `random_state` và `shuffle`. citeturn3search3  
- Chuẩn hóa theo đúng “train-only”: fit scaler trên train và transform lên test (để tránh leakage). Scikit-learn khuyến nghị dùng scaler trong pipeline để giảm rủi ro data leaking. citeturn3search22turn3search1  
- Lưu bảng/ảnh kết quả vào `outputs/tables` và `outputs/figures` để nộp cùng blog.md.

**Những điểm cần kiểm tra trước khi nộp (6 mục)**  
1) Điền **support** cho bảng Top luật Apriori từ file `ket_qua_luat_loi.csv` và thay các antecedents/consequents bị “…” bằng chuỗi đầy đủ.  
2) Xác nhận lại confusion matrix đúng **TN=1932, FP=0, FN=3, TP=65** (khớp log notebook).  
3) Tính và điền **PR-AUC** bằng `average_precision_score(y_true, y_proba)` cho Random Forest (và Decision Tree nếu có). citeturn0search3turn0search11  
4) Kiểm tra feature set classification có đang bao gồm TWF/HDF/PWF/OSF/RNF hay không; nếu có, ghi rõ trong báo cáo hoặc chạy thêm bản “sensor-only” để tránh tranh cãi leakage. citeturn2view1  
5) Đính kèm các hình PNG gốc trong `outputs/figures/` và đảm bảo tên file trong blog.md trùng khớp.  
6) Chốt `seed` trong `params.yaml` và `requirements.txt` có phiên bản thư viện ổn định (để máy khác chạy lại ra kết quả tương đương).

## 8. Tài liệu tham khảo

- AI4I 2020 Predictive Maintenance Dataset — trang dataset và mô tả biến/định nghĩa failure modes. citeturn2view1turn2view0  
- Mlxtend Frequent Patterns — Apriori và Association Rules (support, confidence, lift). citeturn3search2turn0search5turn0search1  
- Scikit-learn — KMeans, Silhouette/Davies–Bouldin, PCA, RandomForestClassifier/Regressor, train_test_split, classification_report, confusion_matrix, SelfTrainingClassifier. citeturn1search0turn1search1turn1search5turn3search0turn4search1turn4search2turn3search3turn1search2turn1search3turn0search2  
- Scikit-learn — Average Precision (PR-AUC/AP) và Precision–Recall khi dữ liệu lệch lớp. citeturn0search3turn0search11  
- SMAPE — định nghĩa công thức và diễn giải. citeturn5search0