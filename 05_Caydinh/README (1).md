# Dự đoán giá nhà bằng Machine Learning

## 1. Giới thiệu

- Tên bài: Dự đoán giá nhà bằng Machine Learning
- Giảng viên: Nguyễn Thế Anh
- Sinh viên: Bùi Xuân Khánh
- MSSV: 2474802010178

Dự án so sánh ba mô hình hồi quy để dự đoán giá nhà từ các đặc trưng:

- `dien_tich`: Diện tích nhà
- `so_phong`: Số phòng
- `tuoi_nha`: Tuổi nhà
- `gia`: Giá nhà (biến mục tiêu)

Các mô hình được sử dụng:
1. Decision Tree Regressor (cây quyết định, độ sâu tối đa bằng 3)
2. Random Forest Regressor (rừng ngẫu nhiên gồm 200 cây)
3. Linear Regression (hồi quy tuyến tính)

## 2. Yêu cầu

- Python 3
- pandas
- numpy
- scikit-learn

Cài đặt các thư viện cần thiết:

```bash
pip install pandas numpy scikit-learn
```

## 3. Dữ liệu

Đặt file `gia_nha.csv` trong thư mục dữ liệu của dự án. File CSV cần có các cột:

```text
dien_tich, so_phong, tuoi_nha, gia
```

Đảm bảo tên cột khớp chính xác với tên trong code. Nếu file nằm ở vị trí khác, hãy cập nhật đường dẫn trong chương trình.

## 4. Cách chạy

1. Mở notebook hoặc file Python chứa chương trình.
2. Đọc dữ liệu từ `gia_nha.csv`.
3. Chia dữ liệu thành tập huấn luyện và tập kiểm tra.
4. Huấn luyện ba mô hình.
5. So sánh MAE và dự đoán giá cho căn nhà mới.

Ví dụ chia dữ liệu trong chương trình:

```python
from sklearn.model_selection import train_test_split

X = df[["dien_tich", "so_phong", "tuoi_nha"]]
y = df["gia"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Lưu ý:** Nếu bài tập trước đã quy định cách chia dữ liệu khác, hãy dùng đúng cách chia đó để so sánh công bằng.

## 5. Kết quả thực nghiệm

Kết quả MAE thu được trên tập kiểm tra:

| Mô hình | MAE |
|---|---:|
| Decision Tree (max_depth=3) | 0.5995 |
| Random Forest (200 cây) | 0.3762 |
| Linear Regression | 0.2373 |

MAE càng thấp thì sai số dự đoán trung bình càng nhỏ. Với kết quả hiện tại, hồi quy tuyến tính có MAE thấp nhất, tiếp theo là Random Forest và cây quyết định.

### Dự đoán nhà có diện tích 250 m²

Với các giá trị đầu vào được dùng trong chương trình là `so_phong = 4` và `tuoi_nha = 10`:

| Mô hình | Giá dự đoán |
|---|---:|
| Decision Tree | 9.4025 |
| Random Forest | 9.1689 |
| Linear Regression | 19.1279 |

Các giá trị dự đoán đang được ghi theo cùng đơn vị với cột `gia` trong dữ liệu.

## 6. Nhận xét

- **Decision Tree:** Dễ diễn giải nhưng có thể bị giới hạn bởi cách chia dữ liệu thành các nút lá. MAE trong thử nghiệm này lớn nhất.
- **Random Forest:** Kết hợp dự đoán của 200 cây nên giảm sai số so với một cây đơn lẻ trong kết quả này.
- **Linear Regression:** Có MAE thấp nhất trên tập kiểm tra, cho thấy mô hình phù hợp nhất trong ba mô hình đối với lần đánh giá này.

### Random Forest có ngoại suy được không?

Random Forest không ngoại suy xu hướng giá theo cách hồi quy tuyến tính có thể làm. Với cây hồi quy, mỗi nút lá dự đoán bằng trung bình giá trị mục tiêu của các mẫu huấn luyện rơi vào nút đó. Rừng lấy trung bình dự đoán của các cây thành phần. Vì vậy, dự đoán thường nằm trong phạm vi giá mục tiêu đã thấy trong dữ liệu huấn luyện, và mô hình không tiếp tục tăng giá tuyến tính khi diện tích vượt khỏi phạm vi đã học.

Hồi quy tuyến tính có thể tạo dự đoán ngoài phạm vi dữ liệu huấn luyện, nhưng dự đoán ngoại suy có thể không chính xác nếu quan hệ thực tế không còn tuyến tính.

## 7. Lưu ý

- Kết quả MAE phụ thuộc vào dữ liệu, cách chia train/test và tiền xử lý.
- Chỉ nên so sánh MAE khi các mô hình được đánh giá trên cùng tập kiểm tra.
- Dự đoán cho nhà 250 m² cần được thận trọng nếu diện tích này nằm ngoài phạm vi diện tích trong dữ liệu huấn luyện.
- README ghi lại kết quả được cung cấp trong lần chạy hiện tại; khi chạy lại với dữ liệu hoặc cách chia khác, các con số có thể thay đổi.
