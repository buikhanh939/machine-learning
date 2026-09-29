#bai tap 3:
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

df = pd.read_csv("data/sinh_vien.csv")

X = df[["gio_on"]]
y = df["qua_mon"]

mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)

w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])

def du_doan(gio):
    z = w * gio + b
    xac_suat = 1 / (1 + np.exp(-z))
    nhan = 1 if xac_suat >= 0.5 else 0

    print(f"So gio on: {gio}")
    print(f"z = {z:.4f}")
    print(f"Xac suat qua mon = {xac_suat:.4f}")
    print(f"Nhãn = {nhan}")
    print()

for gio in [3, 8, 12.89, 18, 26]:
    du_doan(gio)
