#bài 1:
import pandas as pd
df = pd.read_csv("data/sinh_vien.csv")

# Nhóm sinh viên có điểm giữa kỳ từ 7 trở lên
nhom_cao = df[df["diem_giua_ky"] >= 7]

# Nhóm sinh viên có điểm giữa kỳ dưới 7
nhom_thap = df[df["diem_giua_ky"] < 7]

print("So ban co diem giua ky tu 7 tro len:", len(nhom_cao))
print("Ty le qua mon cua nhom diem tu 7 tro len:", round(nhom_cao["qua_mon"].mean(), 4))
print("Ty le qua mon cua nhom con lai:", round(nhom_thap["qua_mon"].mean(), 4))
