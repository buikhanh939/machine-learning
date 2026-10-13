# Bài tập 6 bài Naive Bayes

- **Giảng viên:** Nguyễn Thế Anh
- **Sinh viên:** Bùi Xuân Khánh
- **MSSV:** 2474802010178
- **Môn học / bài học:** Machine Learning – Naive Bayes

## Nội dung

1. Huấn luyện mô hình Multinomial Naive Bayes và đánh giá kết quả.
2. Minh họa hạn chế của Accuracy khi dữ liệu mất cân bằng.
3. Phân tích các từ có xu hướng xuất hiện trong tin rác và tin thường.
4. Tự cài đặt Naive Bayes với Laplace smoothing và so sánh với scikit-learn.
5. So sánh MultinomialNB và GaussianNB trên dữ liệu sinh viên.
6. Huấn luyện và đánh giá mô hình với bộ dữ liệu mất cân bằng: 180 tin thường và 20 tin rác.

## Dữ liệu đầu vào

### Tin nhắn
File CSV cần có hai cột:

- `noi_dung`: nội dung tin nhắn.
- `la_rac`: nhãn tin nhắn (`1` là tin rác, `0` là tin thường).

### Sinh viên (Bài 5)
Nếu thực hiện Bài 5, file CSV cần có các cột:

- `gio_on`
- `diem_giua_ky`
- `qua_mon`

## Thư viện cần cài đặt

```bash
python3 -m pip install pandas scikit-learn
```

## Chạy chương trình

```bash
python3 bai_tap_6_bai_naive_bayes.py
```

Nếu đường dẫn file CSV trên máy khác với đường dẫn cấu hình trong script, hãy sửa các biến `DUONG_DAN_TIN_NHAN` và `DUONG_DAN_SINH_VIEN` ở đầu file Python.

## Lưu ý

Điền MSSV chính xác vào phần thông tin sinh viên trước khi nộp bài.
