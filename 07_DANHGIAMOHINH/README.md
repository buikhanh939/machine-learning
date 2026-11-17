# Bài thực hành 7: Đánh giá mô hình (Học máy ứng dụng)

**Trường Đại học Văn Lang – Khoa Công nghệ Thông tin**  
**Môn học:** Học máy ứng dụng (Năm học 2026 - 2027)  
**Biên soạn:** ThS. Nguyễn Thái Anh  

---

## 1. Giới thiệu & Mục tiêu

Trong các bài toán phân loại dữ liệu lệch lớp (imbalanced data), chỉ số `accuracy` thông thường dễ gây ngộ nhận. Bài thực hành này giúp nắm vững các kỹ thuật và thước đo đánh giá mô hình một cách toàn diện và khách quan:

1. So sánh mô hình với **mô hình lười** (`DummyClassifier`) để xác định mức sàn.
2. Tự tính toán và đọc hiểu 4 ô của **Ma trận nhầm lẫn** (`TP`, `FP`, `FN`, `TN`).
3. Tính toán và phân tích ý nghĩa của `Precision`, `Recall`, `F1-score` và `classification_report`.
4. Điều chỉnh **ngưỡng quyết định** (threshold) bằng kiểm định chéo trên tập học (`cross_val_predict`) mà không làm rò rỉ dữ liệu tập kiểm tra.
5. Vẽ đường cong **ROC** và sử dụng **AUC** để so sánh hiệu năng tổng quát giữa các thuật toán.
6. Áp dụng kiểm định chéo đa thước đo (`scoring`) và xử lý lệch lớp bằng tham số `class_weight="balanced"`.

---

## 2. Cấu trúc thư mục

Đặt toàn bộ 6 tệp Jupyter Notebook (`.ipynb`) vào thư mục `baitap07/` nằm ngay trong thư mục gốc của bài thực hành:

```text
bai07_danh_gia/
├── data/
│   ├── sinh_vien_bo_hoc.csv    # Dữ liệu 1.200 sinh viên (tỷ lệ bỏ học ~10%)
│   └── gia_nha.csv             # Dữ liệu giá nhà (dùng cho Bài tập 6)
├── baitap07/
│   ├── bai1.ipynb              # Bài tập 1: Tự viết hàm đánh giá
│   ├── bai2.ipynb              # Bài tập 2: Chữ số viết tay
│   ├── bai3.ipynb              # Bài tập 3: Chọn ngưỡng theo yêu cầu
│   ├── bai4.ipynb              # Bài tập 4: Tự vẽ đường ROC
│   ├── bai5.ipynb              # Bài tập 5: Cân bằng lớp cho mô hình khác
│   ├── bai6.ipynb              # Bài tập 6: MAE và RMSE với một lần đoán hỏng
│   └── roc.png                 # Biểu đồ ROC xuất ra từ Bài tập 4
└── README.md                   # Tài liệu hướng dẫn này
```

---

## 3. Bộ dữ liệu sử dụng

* **`data/sinh_vien_bo_hoc.csv`** (1.200 mẫu, 5 cột):
  * `diem_tb`: Điểm trung bình học kỳ trước (thang 10).
  * `so_buoi_vang`: Số buổi vắng trong học kỳ.
  * `tin_chi_no`: Số tín chỉ còn nợ.
  * `gio_lam_them`: Số giờ làm thêm mỗi tuần.
  * `bo_hoc`: Nhãn mục tiêu (`1`: Bỏ học, `0`: Học tiếp).
* **`datasets.load_digits()`**: Bộ dữ liệu chữ số viết tay từ `0` đến `9` có sẵn trong `scikit-learn`.
* **`data/gia_nha.csv`**: Bộ dữ liệu hồi quy giá nhà (đơn vị: tỷ đồng).

---

## 4. Hướng dẫn cài đặt và chạy chương trình

### Bước 1: Kích hoạt môi trường ảo
Mở Terminal / Command Prompt tại thư mục gốc và kích hoạt môi trường Python đã cài đặt từ bài 1 (yêu cầu các thư viện: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `jupyter`):

* **Trên Windows:**
  ```bash
  cd C:\MayHoc
  hocmay\Scripts\activate
  cd bai07_danh_gia
  ```
* **Trên macOS / Linux:**
  ```bash
  source hocmay/bin/activate
  cd bai07_danh_gia
  ```

### Bước 2: Mở và chạy Jupyter Notebook
Chạy lệnh sau tại thư mục gốc `bai07_danh_gia`:
```bash
jupyter notebook
```
Sau đó mở lần lượt các tệp trong thư mục `baitap07/` và chọn **Run All** (chạy toàn bộ các cell từ trên xuống dưới để tránh lỗi `NameError` do chưa chạy cell `import`).

*(Lưu ý: Nếu chạy trực tiếp bên trong thư mục `baitap07/`, hãy đổi đường dẫn đọc tệp CSV trong code từ `"data/..."` thành `"../data/..."` hoặc đường dẫn tuyệt đối tương ứng trên máy).*

---

## 5. Tóm tắt nội dung 6 bài tập

| Tệp | Tên bài tập | Nội dung chính & Kết quả nổi bật |
| :--- | :--- | :--- |
| **`bai1.ipynb`** | Tự viết hàm đánh giá | Viết hàm `danh_gia(y_that, y_doan)` bằng `numpy`, xử lý an toàn phép chia cho `0` khi chạy mô hình lười (`TP + FP = 0`), đối chiếu trùng khớp với `scikit-learn`. |
| **`bai2.ipynb`** | Chữ số viết tay | Đánh giá mô hình `SVC` (RBF kernel) trên 10 chữ số bằng `classification_report`. Phân tích lớp có `recall` thấp nhất (số `8`: `0.9302`), `precision` thấp nhất (số `4`, `7`: `0.9565`) và lý do `macro avg` xấp xỉ `weighted avg`. |
| **`bai3.ipynb`** | Chọn ngưỡng theo yêu cầu | Dùng `cross_val_predict` (5-fold) trên tập học để tìm ngưỡng lớn nhất đạt `recall >= 0.8` (chọn ngưỡng **`0.17`**). Kiểm tra trên tập test đạt `recall = 0.8000` ($24/30$ bạn), `precision = 0.5106`, tổng cộng báo `47` bạn. |
| **`bai4.ipynb`** | Tự vẽ đường ROC | Vẽ đường ROC so sánh **Hồi quy logistic** (`AUC = 0.9449`) và **Rừng ngẫu nhiên** (`AUC = 0.9249`), xuất hình `roc.png` và phân tích các điểm giao cắt. |
| **`bai5.ipynb`** | Cân bằng lớp cho mô hình khác | So sánh `RandomForestClassifier` và `SVC` trước và sau khi thêm `class_weight="balanced"`. `SVC` thay đổi mạnh nhất (`recall` tăng từ `0.5000` lên `0.8333`). |
| **`bai6.ipynb`** | MAE và RMSE với một lần đoán hỏng | Thử nghiệm cộng thêm `5 tỷ` sai số vào 1 căn nhà trên tập kiểm tra `gia_nha.csv`. Chứng minh **RMSE** biến động mạnh nhất (tăng `313.41%`) do cơ chế bình phương sai số. |