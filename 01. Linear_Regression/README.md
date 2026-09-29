# Bài tập Hồi quy tuyến tính - Dự đoán giá nhà

## 1. Giới thiệu

Bài tập thực hành hồi quy tuyến tính trên bộ dữ liệu giá nhà.  
Mục tiêu là đọc và phân tích dữ liệu, tìm mối quan hệ giữa các biến, xây dựng mô hình hồi quy tuyến tính bằng công thức và bằng thư viện Scikit-learn, thử nghiệm Gradient Descent và xây dựng hàm dự đoán giá nhà.

## 2. Bộ dữ liệu

Dữ liệu được lưu trong file:

`data/gia_nha.csv`

Bộ dữ liệu gồm các cột:

| Cột | Ý nghĩa |
|---|---|
| `dien_tich` | Diện tích căn nhà, tính bằng m² |
| `so_phong` | Số phòng |
| `tuoi_nha` | Tuổi của căn nhà, tính bằng năm |
| `gia` | Giá nhà, tính bằng tỷ đồng |

## 3. Nội dung các bài

### Bài 1 - Phân tích dữ liệu

Đọc file `gia_nha.csv` bằng Pandas và kiểm tra tổng quan bộ dữ liệu.

Nội dung chính:
- Đọc dữ liệu.
- Kiểm tra số dòng và số cột.
- Xem các dòng dữ liệu đầu tiên.
- Kiểm tra tên các cột.
- Thống kê nhanh diện tích và giá.
- Kiểm tra dữ liệu bị thiếu.
- Lọc các căn có diện tích lớn hơn 100 m² và tính giá trung bình của nhóm này.

### Bài 2 - Biểu đồ số phòng và giá nhà

Sử dụng Matplotlib để vẽ biểu đồ phân tán giữa:

- Trục X: `so_phong`
- Trục Y: `gia`

Biểu đồ được lưu thành file:

`bai2.png`

Biểu đồ giúp quan sát xu hướng giữa số phòng và giá nhà.

### Bài 3 - Hồi quy tuyến tính bằng công thức

Sử dụng phương pháp bình phương tối thiểu để tính mô hình hồi quy tuyến tính.

Ở bài này:
- Biến đầu vào `x`: `tuoi_nha`
- Biến mục tiêu `y`: `gia`

Công thức được sử dụng:

```text
w = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)

b = ȳ - w*x̄
```

Mô hình:

```text
gia = w * tuoi_nha + b
```

Kết quả gồm hệ số góc `w` và hệ số chặn `b`.

### Bài 4 - Hồi quy tuyến tính bằng Scikit-learn

Sử dụng `LinearRegression` của Scikit-learn.

Các biến đầu vào:

```text
dien_tich
so_phong
```

Biến mục tiêu:

```text
gia
```

Dữ liệu được chia thành:
- 80% dữ liệu huấn luyện.
- 20% dữ liệu kiểm tra.

Sau đó sử dụng chỉ số `R²` để đánh giá mô hình.

### Bài 5 - Gradient Descent

Xây dựng thuật toán Gradient Descent để tìm các tham số `w` và `b` của mô hình hồi quy tuyến tính.

Dữ liệu diện tích được chuẩn hóa trước khi huấn luyện.

Các thông số chính:

```text
so_vong = 200
```

Tốc độ học `toc_do_hoc` được thay đổi để quan sát ảnh hưởng đến quá trình hội tụ.

Các trường hợp được thử nghiệm gồm:

```text
toc_do_hoc = 0.001
toc_do_hoc = 1.02
```

MSE tại vòng lặp thứ 200 được sử dụng để so sánh kết quả.

### Bài 6 - Hàm dự đoán giá nhà

Xây dựng hàm:

```python
du_doan_gia(dien_tich)
```

Hàm nhận diện tích căn nhà và trả về giá dự đoán theo mô hình:

```text
gia = 0.078367 * dien_tich + 0.401752
```

Hàm được kiểm tra với:

```text
60 m²
80 m²
200 m²
```

Nếu diện tích nằm ngoài phạm vi dữ liệu huấn luyện từ `35.5` đến `117.5` m², chương trình sẽ in cảnh báo.

## 4. Công nghệ và thư viện

Dự án sử dụng:

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab / Jupyter Notebook

## 5. Cấu trúc thư mục

```text
baitap01/
│
├── README.md
├── bai_tap.ipynb
│
└── data/
    └── gia_nha.csv
```

Nếu tách từng bài thành file Python:

```text
baitap01/
│
├── README.md
├── bai1.py
├── bai2.py
├── bai3.py
├── bai4.py
├── bai5.py
├── bai6.py
│
└── data/
    └── gia_nha.csv
```

## 6. Cài đặt thư viện

Nếu chạy trên Google Colab, có thể cài các thư viện bằng:

```python
!pip install -q numpy pandas scikit-learn matplotlib
```

## 7. Cách chạy

Mở file `.ipynb` bằng Google Colab.

Sau khi mở Notebook:

**Runtime → Run all**

Các bài sẽ được thực hiện lần lượt từ Bài 1 đến Bài 6.

## 8. Kết quả chính

Các bài tập lần lượt thực hiện quy trình:

```text
Đọc dữ liệu
    ↓
Phân tích dữ liệu
    ↓
Trực quan hóa dữ liệu
    ↓
Tính hồi quy bằng công thức
    ↓
Huấn luyện bằng Scikit-learn
    ↓
Đánh giá mô hình
    ↓
Gradient Descent
    ↓
Dự đoán giá nhà
```

## 9. Tác giả

**Bùi Xuân Khánh**
