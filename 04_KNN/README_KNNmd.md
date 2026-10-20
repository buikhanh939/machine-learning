# BÀI THỰC HÀNH: K-NEAREST NEIGHBORS (KNN)

## 1. Giới thiệu bài học

Bài thực hành tìm hiểu và áp dụng thuật toán **K-Nearest Neighbors (KNN)** cho bài toán phân loại và hồi quy. Nội dung gồm cách tính khoảng cách, chuẩn hóa dữ liệu, lựa chọn tham số `k`, so sánh mô hình và đánh giá hiệu năng thực thi.

## 2. Thông tin sinh viên

- **Giảng viên:** Nguyễn Thế Anh
- **Sinh viên:** Bùi Xuân Khánh
- **MSSV:** 2474802010178
- **Tên bài học:** K-Nearest Neighbors (KNN)

## 3. Nội dung bài tập

### Bài 1. Tính khoảng cách và dự đoán bằng KNN

- Tính khoảng cách Euclid từ mẫu cần dự đoán đến các mẫu trong tập dữ liệu.
- Chuẩn hóa các đặc trưng trước khi tính khoảng cách.
- Chọn 5 láng giềng gần nhất và dự đoán nhãn theo biểu quyết đa số.

**Mục tiêu:** Hiểu cách KNN tìm láng giềng gần nhất và đưa ra dự đoán.

### Bài 2. So sánh cách xử lý đặc trưng

- Huấn luyện KNN với `k = 7`.
- So sánh mô hình dùng 3 đặc trưng không chuẩn hóa với mô hình dùng 3 đặc trưng đã chuẩn hóa.
- So sánh thêm mô hình dùng 2 đặc trưng đã chuẩn hóa.

**Mục tiêu:** Quan sát ảnh hưởng của chuẩn hóa dữ liệu và lựa chọn đặc trưng đến kết quả dự đoán.

### Bài 3. Chọn k bằng kiểm định chéo

- Thử các giá trị `k` lẻ từ 1 đến 51.
- Sử dụng kiểm định chéo 5 phần (5-fold cross-validation).
- So sánh khoảng cách Euclid và Manhattan.
- Vẽ biểu đồ điểm số theo từng giá trị `k`.

**Mục tiêu:** Chọn tham số `k` phù hợp và tìm hiểu ảnh hưởng của cách đo khoảng cách.

### Bài 4. So sánh mô hình và trọng số láng giềng

- So sánh KNN với Logistic Regression và Gaussian Naive Bayes.
- Kiểm tra KNN với `k = 1`, `13` và `101`.
- So sánh hai cách gán trọng số `uniform` và `distance`.
- Lưu biểu đồ kết quả so sánh.

**Mục tiêu:** Đánh giá KNN so với các mô hình phân loại khác và hiểu tác động của trọng số láng giềng.

### Bài 5. Chọn k cho bài toán hồi quy

- Sử dụng dữ liệu giá nhà.
- Thử `k` từ 1 đến 20 cho `KNeighborsRegressor`.
- Dùng kiểm định chéo 5 phần với `scoring="neg_mean_absolute_error"`.
- Chọn `k` có MAE kiểm định chéo tốt nhất.
- Đánh giá MAE trên tập kiểm tra và so sánh với Linear Regression.
- Lưu biểu đồ kết quả.

**Mục tiêu:** Áp dụng KNN cho hồi quy và sử dụng MAE để đánh giá sai số dự đoán.

### Bài 6. Đo thời gian thực thi

- Tạo các bộ dữ liệu có 1.000, 10.000 và 100.000 mẫu.
- Huấn luyện KNN trên từng bộ dữ liệu.
- Đo thời gian huấn luyện và thời gian dự đoán.
- Vẽ biểu đồ so sánh thời gian thực thi.

**Mục tiêu:** Quan sát ảnh hưởng của kích thước dữ liệu đến thời gian huấn luyện và dự đoán của KNN.

## 4. Cấu trúc thư mục đề xuất

```text
KNN_Bai_1_den_6/
├── main.py
├── README.md
├── requirements.txt
├── data/
│   ├── gia_nha.csv
│   ├── sinh_vien_mo_rong.csv
│   └── hai_vang_trang.csv
└── figures/
    ├── bai3_chon_k.png
    ├── bai4_weights_distance.png
    ├── bai5_chon_k_regression.png
    └── bai6_thoi_gian.png
```

**Lưu ý:** `gia_nha.csv` được dùng cho bài 5. Nếu chương trình sử dụng `sinh_vien_mo_rong.csv` và `hai_vang_trang.csv`, cần đặt hai tệp này trong thư mục `data/`. Tên hình trong `figures/` có thể thay đổi theo tên được khai báo trong mã nguồn.

## 5. Cài đặt và chạy chương trình

Mở Terminal tại thư mục `KNN_Bai_1_den_6/`, sau đó chạy:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Các biểu đồ được chương trình tạo sẽ lưu trong thư mục `figures/`.

## 6. Kết luận

Qua sáu bài thực hành, sinh viên làm quen với cách hoạt động của KNN, cách chuẩn hóa dữ liệu, lựa chọn `k`, đánh giá mô hình phân loại và hồi quy, cũng như đo thời gian thực thi. Kết quả thực nghiệm giúp hiểu rõ hơn các yếu tố ảnh hưởng đến hiệu quả của thuật toán.
