# -*- coding: utf-8 -*-
"""
6 BAI TAP NAIVE BAYES
Du lieu tin nhan: data/tin_nhan.csv
Cot: noi_dung, la_rac (1 = tin rac, 0 = tin thuong)

Chay file nay trong thu muc machine-learning hoac sua DUONG_DAN_TIN_NHAN
cho dung voi may cua ban.
"""

from pathlib import Path
from collections import Counter, defaultdict
import math
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# CAU HINH DUONG DAN
# ============================================================

DUONG_DAN_TIN_NHAN = Path(
    "/Users/buixuankhanh/Documents/machine-learning/03_Navie_Bayes/data/tin_nhan.csv"
)

# Neu bai 5 co file du lieu sinh vien, dat file CSV tai duong dan nay.
# File can co cac cot: gio_on, diem_giua_ky, qua_mon.
DUONG_DAN_SINH_VIEN = Path(
    "/Users/buixuankhanh/Documents/machine-learning/03_Navie_Bayes/data/sinh_vien.csv"
)

RANDOM_STATE = 17


def in_tieu_de(ten_bai):
    print("\n" + "=" * 65)
    print(ten_bai)
    print("=" * 65)


def doc_du_lieu_tin_nhan():
    if not DUONG_DAN_TIN_NHAN.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {DUONG_DAN_TIN_NHAN}\n"
            "Hay sua DUONG_DAN_TIN_NHAN o dau file cho dung."
        )

    du_lieu = pd.read_csv(DUONG_DAN_TIN_NHAN)
    cot_can_co = {"noi_dung", "la_rac"}
    if not cot_can_co.issubset(du_lieu.columns):
        raise ValueError("CSV tin nhan phai co hai cot: noi_dung va la_rac.")

    du_lieu = du_lieu.dropna(subset=["noi_dung", "la_rac"]).copy()
    du_lieu["noi_dung"] = du_lieu["noi_dung"].astype(str)
    du_lieu["la_rac"] = du_lieu["la_rac"].astype(int)
    return du_lieu


def chia_va_bien_doi(du_lieu):
    X_huan_luyen, X_kiem_tra, y_huan_luyen, y_kiem_tra = train_test_split(
        du_lieu["noi_dung"],
        du_lieu["la_rac"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=du_lieu["la_rac"],
    )

    # Giu ca tu chi co mot ky tu, vi du: "a", "e".
    bo_dem_tu = CountVectorizer(token_pattern=r"(?u)\b\S+")
    X_huan_luyen_dem = bo_dem_tu.fit_transform(X_huan_luyen)
    X_kiem_tra_dem = bo_dem_tu.transform(X_kiem_tra)

    return (
        X_huan_luyen_dem,
        X_kiem_tra_dem,
        y_huan_luyen,
        y_kiem_tra,
        bo_dem_tu,
    )


# Doc du lieu dung chung cho cac bai 1-4 va bai 6.
df = doc_du_lieu_tin_nhan()


# ============================================================
# BAI 1: HUAN LUYEN MULTINOMIAL NAIVE BAYES VA DANH GIA
# ============================================================

in_tieu_de("BAI 1 - HUAN LUYEN VA DANH GIA MO HINH")

X_train, X_test, y_train, y_test, vectorizer = chia_va_bien_doi(df)

mo_hinh_bai1 = MultinomialNB()
mo_hinh_bai1.fit(X_train, y_train)
y_du_doan_bai1 = mo_hinh_bai1.predict(X_test)

print("Accuracy :", f"{accuracy_score(y_test, y_du_doan_bai1):.4f}")
print("Precision:", f"{precision_score(y_test, y_du_doan_bai1, pos_label=1, zero_division=0):.4f}")
print("Recall   :", f"{recall_score(y_test, y_du_doan_bai1, pos_label=1, zero_division=0):.4f}")
print("F1-score :", f"{f1_score(y_test, y_du_doan_bai1, pos_label=1, zero_division=0):.4f}")
print("Confusion matrix (hang = nhan that, cot = du doan):")
print(confusion_matrix(y_test, y_du_doan_bai1))


# ============================================================
# BAI 2: MINH HOA BAY ACCURACY
# ============================================================

in_tieu_de("BAI 2 - BAY ACCURACY")

# Mo hinh gia: du doan tat ca tin nhan deu la tin thuong (nhan 0).
# Chi so nay giup thay accuracy co the gay hieu nham khi lech lop.
y_du_doan_tat_ca_thuong = [0] * len(y_test)

print("So mau kiem tra:", len(y_test))
print("So tin thuong that:", int((y_test == 0).sum()))
print("So tin rac that:", int((y_test == 1).sum()))
print(
    "Accuracy neu du doan tat ca la tin thuong:",
    f"{accuracy_score(y_test, y_du_doan_tat_ca_thuong):.4f}",
)
print(
    "Recall lop rac cua cach du doan tren:",
    f"{recall_score(y_test, y_du_doan_tat_ca_thuong, pos_label=1, zero_division=0):.4f}",
)
print("Nhan xet: accuracy co the van cao khi du lieu lech lop,")
print("nhung bo loc co the bo sot tat ca tin rac. Can xem them recall.")


# ============================================================
# BAI 3: TIM 5 TU NGHIENG VE TIN RAC VA 5 TU NGHIENG VE TIN THUONG
# ============================================================

in_tieu_de("BAI 3 - PHAN TICH FEATURE_LOG_PROB_")

# Huan luyen lai tren tap huan luyen da vector hoa.
mo_hinh_bai3 = MultinomialNB(alpha=1.0)
mo_hinh_bai3.fit(X_train, y_train)

cac_tu = vectorizer.get_feature_names_out()
chenh_lech_log = (
    mo_hinh_bai3.feature_log_prob_[1] - mo_hinh_bai3.feature_log_prob_[0]
)

chi_so_tu_rac = chenh_lech_log.argsort()[::-1][:5]
chi_so_tu_thuong = chenh_lech_log.argsort()[:5]

print("5 tu nghieng ve tin rac (gia tri duong lon):")
for i in chi_so_tu_rac:
    print(f"{cac_tu[i]}: {chenh_lech_log[i]:.4f}")

print("\n5 tu nghieng ve tin thuong (gia tri am lon):")
for i in chi_so_tu_thuong:
    print(f"{cac_tu[i]}: {chenh_lech_log[i]:.4f}")

print("\nLuu y: day la xu huong theo mo hinh, khong phai tu nao cung")
print("quyet dinh chac chan mot tin nhan la rac hay tin thuong.")


# ============================================================
# BAI 4: TU CAI DAT NAIVE BAYES, LAPlace SMOOTHING VA SO SANH
# ============================================================

in_tieu_de("BAI 4 - TU CAI DAT NAIVE BAYES VA SO SANH")

# Dung van ban train/test goc truoc khi CountVectorizer bien doi.
X_van_ban_train, X_van_ban_test, y_nhan_train, y_nhan_test = train_test_split(
    df["noi_dung"],
    df["la_rac"],
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=df["la_rac"],
)

bo_dem_tu_tu_cai = CountVectorizer(token_pattern=r"(?u)\b\S+")
ma_tran_train = bo_dem_tu_tu_cai.fit_transform(X_van_ban_train)
ma_tran_test = bo_dem_tu_tu_cai.transform(X_van_ban_test)

tu_vung = bo_dem_tu_tu_cai.get_feature_names_out()
so_tu_vung = len(tu_vung)
alpha = 1.0

# Dem tong so lan xuat hien cua moi tu theo tung lop.
dem_tu_theo_lop = {
    0: Counter(),
    1: Counter(),
}
tong_tu_theo_lop = {0: 0, 1: 0}
so_tin_theo_lop = Counter(y_nhan_train.tolist())

for van_ban, nhan in zip(X_van_ban_train, y_nhan_train):
    cac_tu_trong_tin = bo_dem_tu_tu_cai.build_analyzer()(van_ban)
    dem_tu_theo_lop[int(nhan)].update(cac_tu_trong_tin)
    tong_tu_theo_lop[int(nhan)] += len(cac_tu_trong_tin)

tong_tin_train = len(y_nhan_train)
log_xac_suat_tien_nghiem = {
    lop: math.log(so_tin_theo_lop[lop] / tong_tin_train)
    for lop in [0, 1]
}

log_xac_suat_tu = {}
for lop in [0, 1]:
    mau_so = tong_tu_theo_lop[lop] + alpha * so_tu_vung
    log_xac_suat_tu[lop] = {
        tu: math.log((dem_tu_theo_lop[lop][tu] + alpha) / mau_so)
        for tu in tu_vung
    }


def du_doan_nb_tu_cai(van_ban):
    cac_tu = bo_dem_tu_tu_cai.build_analyzer()(van_ban)
    diem_lop = {}

    for lop in [0, 1]:
        diem = log_xac_suat_tien_nghiem[lop]
        for tu in cac_tu:
            if tu in log_xac_suat_tu[lop]:
                diem += log_xac_suat_tu[lop][tu]
        diem_lop[lop] = diem

    return max(diem_lop, key=diem_lop.get)


y_du_doan_tu_cai = [du_doan_nb_tu_cai(v) for v in X_van_ban_test]

mo_hinh_sklearn = MultinomialNB(alpha=alpha)
mo_hinh_sklearn.fit(ma_tran_train, y_nhan_train)
y_du_doan_sklearn = mo_hinh_sklearn.predict(ma_tran_test)

print("Accuracy tu cai dat :", f"{accuracy_score(y_nhan_test, y_du_doan_tu_cai):.4f}")
print("Accuracy sklearn    :", f"{accuracy_score(y_nhan_test, y_du_doan_sklearn):.4f}")
print("Hai ket qua nen rat gan nhau neu dung cung du lieu, tu vung,")
print("alpha, xac suat tien nghiem va cach tinh diem log.")


# ============================================================
# BAI 5: CHON SAI BIEN THE NAIVE BAYES
# ============================================================

in_tieu_de("BAI 5 - SO SANH MULTINOMIALNB VA GAUSSIANNB")

if DUONG_DAN_SINH_VIEN.exists():
    df_sv = pd.read_csv(DUONG_DAN_SINH_VIEN)
    cot_sv = {"gio_on", "diem_giua_ky", "qua_mon"}

    if not cot_sv.issubset(df_sv.columns):
        print("File sinh vien khong co du cac cot gio_on, diem_giua_ky, qua_mon.")
    else:
        df_sv = df_sv.dropna(subset=list(cot_sv)).copy()
        X_sv = df_sv[["gio_on", "diem_giua_ky"]]
        y_sv = df_sv["qua_mon"]

        X_sv_train, X_sv_test, y_sv_train, y_sv_test = train_test_split(
            X_sv,
            y_sv,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_sv,
        )

        mo_hinh_multi_sv = MultinomialNB()
        mo_hinh_multi_sv.fit(X_sv_train, y_sv_train)
        du_doan_multi_sv = mo_hinh_multi_sv.predict(X_sv_test)

        mo_hinh_gauss_sv = GaussianNB()
        mo_hinh_gauss_sv.fit(X_sv_train, y_sv_train)
        du_doan_gauss_sv = mo_hinh_gauss_sv.predict(X_sv_test)

        print("Accuracy MultinomialNB:", f"{accuracy_score(y_sv_test, du_doan_multi_sv):.4f}")
        print("Accuracy GaussianNB   :", f"{accuracy_score(y_sv_test, du_doan_gauss_sv):.4f}")
        print("Nhan xet: gio_on va diem_giua_ky la du lieu so lien tuc.")
        print("GaussianNB thuong phu hop ve mat gia dinh phan phoi hon.")
        print("MultinomialNB duoc thiet ke chu yeu cho du lieu dem/tan suat.")
else:
    print("Chua tim thay file du lieu sinh vien:", DUONG_DAN_SINH_VIEN)
    print("Neu co file sinh_vien.csv, hay dat dung duong dan o dau script.")
    print("Cot can co: gio_on, diem_giua_ky, qua_mon.")


# ============================================================
# BAI 6: DU LIEU MAT CAN BANG NANG - 180 THUONG, 20 RAC
# ============================================================

in_tieu_de("BAI 6 - DU LIEU MAT CAN BANG NANG")

tin_thuong = df[df["la_rac"] == 0]
tin_rac = df[df["la_rac"] == 1].head(20)

if len(tin_thuong) < 180:
    print(f"Can 180 tin thuong, nhung file hien co {len(tin_thuong)} tin thuong.")
elif len(tin_rac) < 20:
    print(f"Can 20 tin rac, nhung file hien co {len(tin_rac)} tin rac.")
else:
    # Lay dung 180 tin thuong dau tien trong nhom va 20 tin rac dau tien.
    tin_thuong = tin_thuong.head(180)
    du_lieu_mat_can_bang = pd.concat(
        [tin_thuong, tin_rac],
        ignore_index=True,
    )

    print("So tin thuong:", int((du_lieu_mat_can_bang["la_rac"] == 0).sum()))
    print("So tin rac:", int((du_lieu_mat_can_bang["la_rac"] == 1).sum()))
    print("Tong so tin:", len(du_lieu_mat_can_bang))

    X_mb_train, X_mb_test, y_mb_train, y_mb_test = train_test_split(
        du_lieu_mat_can_bang["noi_dung"],
        du_lieu_mat_can_bang["la_rac"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=du_lieu_mat_can_bang["la_rac"],
    )

    bo_dem_mb = CountVectorizer(token_pattern=r"(?u)\b\S+")
    X_mb_train = bo_dem_mb.fit_transform(X_mb_train)
    X_mb_test = bo_dem_mb.transform(X_mb_test)

    mo_hinh_mb = MultinomialNB()
    mo_hinh_mb.fit(X_mb_train, y_mb_train)
    y_mb_du_doan = mo_hinh_mb.predict(X_mb_test)

    do_chinh_xac_mb = accuracy_score(y_mb_test, y_mb_du_doan)
    do_day_du_rac_mb = recall_score(
        y_mb_test,
        y_mb_du_doan,
        pos_label=1,
        zero_division=0,
    )

    print(f"Accuracy = {do_chinh_xac_mb:.4f}")
    print(f"Recall lop rac = {do_day_du_rac_mb:.4f}")
    print("Nhan xet: khi du lieu lech lop, accuracy co the cao")
    print("nhung mo hinh van bo sot nhieu tin rac. Can xem recall")
    print("lop rac va cac chi so khac, khong nen chi xem accuracy.")
