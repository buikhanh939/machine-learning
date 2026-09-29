# Bài 2: Hồi quy Logistic (Logistic Regression)

**Tên giáo viên:** Nguyễn Thế Anh  
**Sinh viên:** Bùi Xuân Khánh  
**Mã số sinh viên:** 2474802010178

## Nội dung

Bài tập thực hành **Hồi quy Logistic (Logistic Regression)** để dự đoán sinh viên **qua môn hoặc rớt môn** dựa trên số giờ ôn tập và điểm giữa kỳ.

## Dữ liệu

File dữ liệu:

```text
data/sinh_vien.csv
```

Các thuộc tính chính:

- `gio_on`: Số giờ ôn tập
- `diem_giua_ky`: Điểm giữa kỳ
- `qua_mon`: Kết quả môn học
  - `0`: Rớt môn
  - `1`: Qua môn

## Nội dung thực hành

1. Đọc và khám phá dữ liệu.
2. Xây dựng hàm Sigmoid.
3. Xây dựng mô hình Logistic Regression.
4. Chia dữ liệu thành tập train và test.
5. Đánh giá mô hình bằng:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
6. Thử nghiệm các ngưỡng dự đoán khác nhau.
7. So sánh Precision và Recall giữa lớp qua môn và lớp rớt môn.
8. So sánh mô hình sử dụng một biến và hai biến đầu vào.

## Công nghệ sử dụng

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Cách chạy

Cài đặt các thư viện:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Sau đó chạy các file Python trong thư mục `scripts`.

## Mục tiêu

Hiểu cách hoạt động của **Logistic Regression**, cách sử dụng **Sigmoid**, **Confusion Matrix**, các chỉ số đánh giá mô hình và ảnh hưởng của việc thay đổi ngưỡng dự đoán.
