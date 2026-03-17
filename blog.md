# Bài Tập Lớn Data Mining — Đề 16: Phân Tích Lỗi Sản Xuất & Dự Đoán Lỗi

> **Môn học:** Khai Phá Dữ Liệu (Data Mining)  
> **Giảng viên:** ThS. Lê Thị Thùy Trang  
> **Đề tài:** Đề 16 — Phân tích lỗi sản xuất & Dự đoán lỗi (AI4I 2020 Predictive Maintenance)

---

## 1. Đặt Vấn Đề và Mục Tiêu

### 1.1 Bối cảnh
Trong nền công nghiệp 4.0, bảo trì dự đoán (Predictive Maintenance) đóng vai trò then chốt giúp giảm thiểu thời gian chết (downtime) và chi phí sửa chữa đột xuất. Thay vì bảo trì định kỳ tốn kém hoặc đợi máy hỏng mới sửa, việc phân tích dữ liệu cảm biến giúp doanh nghiệp:
* **Xác định nguyên nhân** và các điều kiện vận hành dẫn đến lỗi máy.
* **Phân nhóm trạng thái** máy móc để có lịch trình bảo trì phù hợp.
* **Xây dựng mô hình dự đoán** sớm các sự cố ngay cả khi thiếu hụt dữ liệu gán nhãn.
* **Dự báo độ hao mòn** của thiết bị (dao cắt) theo thời gian thực để tối ưu chi phí thay thế.

### 1.2 Mục tiêu dự án

| STT | Mục tiêu | Kỹ thuật sử dụng |
|:---:|:---|:---|
| 1 | Tìm mẫu kết hợp, điều kiện vận hành dẫn đến lỗi | **Apriori** (Association Rules) |
| 2 | Phân cụm máy móc theo hành vi và nguy cơ | **K-Means Clustering** & PCA |
| 3 | Phân lớp trạng thái máy lỗi/không lỗi (Imbalanced) | **Random Forest**, Decision Tree |
| 4 | Thực nghiệm học bán giám sát khi thiếu nhãn (10% nhãn) | **Self-Training Classifier** |
| 5 | Dự báo hồi quy chuỗi thời gian độ mòn dao cắt (Tool wear) | **Random Forest Regressor** (No shuffle) |

### 1.3 Tiêu chí thành công
* **PR-AUC và F1-Score** là metric chính cho bài toán phân lớp (do dữ liệu cực kỳ mất cân bằng, lỗi chỉ chiếm ~3.4%).
* **Silhouette Score** và phương pháp Elbow để đánh giá phân cụm (Clustering).
* **MAE, RMSE** để đánh giá sai số của mô hình dự báo hồi quy (Forecasting).
* **Lift > 1.5** và **Confidence cao** cho các luật kết hợp liên quan đến nguyên nhân hỏng máy.

---

## 2. Nguồn Dữ Liệu

### 2.1 Thông tin dataset
* **Tên Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository).
* **Kích thước:** 10,000 dòng x 14 cột.
* **Đặc điểm:** Dữ liệu cảm biến thu thập từ máy phay công nghiệp.

### 2.2 Từ điển dữ liệu (Data Dictionary)

| Thuộc tính | Đơn vị | Phân loại | Mô tả chi tiết |
|:---|:---:|:---:|:---|
| **Air temperature** | [K] | Numeric | Nhiệt độ không khí xung quanh máy. |
| **Process temperature** | [K] | Numeric | Nhiệt độ của quá trình sản xuất. |
| **Rotational speed** | [rpm] | Numeric | Tốc độ quay của trục máy tính bằng vòng/phút. |
| **Torque** | [Nm] | Numeric | Momen xoắn (Lực gồng của máy). |
| **Tool wear** | [min] | Numeric | Thời gian (phút) dao cắt đã bị mài mòn trong quá trình gia công. |
| **Type** | L/M/H | Categorical | Chất lượng sản phẩm (Low/Medium/High). |
| **Machine failure** | 0/1 | Target | Biến mục tiêu (1: Máy lỗi, 0: Bình thường). |
| **TWF, HDF, PWF...** | 0/1 | Binary | Các loại lỗi cụ thể (Tool Wear Failure, Heat Dissipation Failure...). |

---

## 3. Cấu trúc thư mục (Repository Structure)

Dự án được thiết kế theo chuẩn module hóa để đảm bảo khả năng tái lập (reproducible).

```text
BTL_DATA/
├── data/
│   ├── raw/                  # Chứa file ai4i2020.csv gốc
│   └── processed/            # Dữ liệu sau khi làm sạch
├── notebooks/
│   ├── 01_eda.ipynb          # Phân tích khám phá dữ liệu (EDA)
│   ├── 02_preprocess_feature.ipynb  # Tiền xử lý dữ liệu
│   ├── 03_mining.ipynb       # Apriori & K-Means Clustering
│   ├── 04_modeling.ipynb     # Phân lớp (Classification)
│   ├── 04b_semi_supervised.ipynb    # Học bán giám sát (Self-Training)
│   └── 05_evaluation_report.ipynb   # Dự báo hồi quy chuỗi thời gian
├── src/                      # Source code (Class & Functions)
│   ├── data/                 # Module tải dữ liệu
│   └── models/               # Module chứa các mô hình học máy
├── requirements.txt          # Thư viện cần thiết (pandas, scikit-learn...)
└── README.md                 # Tài liệu mô tả dự án