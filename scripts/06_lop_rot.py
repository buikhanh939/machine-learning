#bai tap 6
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/sinh_vien.csv")

X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)

y_pred = mo_hinh.predict(X_test)

precision_0 = precision_score(y_test, y_pred, pos_label=0) #Hãy coi lớp 0 (rớt môn) là lớp dương để tính Precision.
recall_0 = recall_score(y_test, y_pred, pos_label=0) #Hãy coi lớp 0 (rớt môn) là lớp dương để tính Recall.

print("Danh gia lop rot mon (lop 0):")
print(f"Precision = {precision_0:.4f}")
print(f"Recall = {recall_0:.4f}")
