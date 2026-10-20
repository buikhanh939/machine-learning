# -*- coding: utf-8 -*-
"""Tong hop KNN bai 1-6. Chay: python3 main.py"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)
SEED = 17

def title(s): print("\n" + "=" * 68 + "\n" + s + "\n" + "=" * 68)
def load(name, cols):
    p = DATA / name
    if not p.exists():
        print(f"[BO QUA] Thieu {p}. Them CSV goc vao thu muc data/ de chay bai nay.")
        return None
    d = pd.read_csv(p)
    missing = set(cols) - set(d.columns)
    if missing:
        print(f"[BO QUA] {name} thieu cot: {sorted(missing)}")
        return None
    return d.dropna(subset=cols).copy()
def split(X, y):
    return train_test_split(X, y, test_size=.25, random_state=SEED, stratify=y)

def bai1():
    title("BAI 1 - KNN THU CONG: KHOANG CACH EUCLID")
    d = load("sinh_vien_mo_rong.csv", ["gio_on","diem_giua_ky","chi_tieu","qua_mon"])
    if d is None: return
    X = d[["gio_on","diem_giua_ky"]].to_numpy(float); y = d["qua_mon"].to_numpy()
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1; X = (X-mu)/sd
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.25,random_state=SEED,stratify=y)
    dist = np.sqrt(((Xtr-Xte[0])**2).sum(axis=1)); ix = np.argsort(dist)[:5]
    labels, counts = np.unique(ytr[ix], return_counts=True)
    print("5 lang gieng gan nhat:")
    for j, i in enumerate(ix,1): print(f"{j}. distance={dist[i]:.4f}, nhan={ytr[i]}")
    print("Nhan du doan thu cong:", labels[np.argmax(counts)], "| Nhan that:", yte[0])

def bai2():
    title("BAI 2 - ANH HUONG CUA CHUAN HOA")
    d = load("sinh_vien_mo_rong.csv", ["gio_on","diem_giua_ky","chi_tieu","qua_mon"])
    if d is None: return
    y=d.qua_mon; X=d[["gio_on","diem_giua_ky","chi_tieu"]]; Xtr,Xte,ytr,yte=split(X,y)
    three=["gio_on","diem_giua_ky","chi_tieu"]; two=["gio_on","diem_giua_ky"]
    def score(model, cols):
        model.fit(Xtr[cols],ytr); return accuracy_score(yte,model.predict(Xte[cols]))
    a=score(KNeighborsClassifier(n_neighbors=7),three)
    b=score(make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7)),three)
    c=score(make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7)),two)
    print(f"Ty le qua mon tap kiem tra: {yte.mean():.4f}")
    print(f"Ba cot, khong chuan hoa: {a:.4f}\nBa cot, co chuan hoa: {b:.4f}\nHai cot, co chuan hoa: {c:.4f}")

def bai3():
    title("BAI 3 - CHON K BANG CV 5-FOLD, EUCLIDEAN VA MANHATTAN")
    d=load("hai_vang_trang.csv",["x1","x2","nhan"])
    if d is None: return
    X,y=d[["x1","x2"]],d.nhan; Xtr,Xte,ytr,yte=split(X,y); ks=list(range(1,52,2)); out={}
    for metric in ["euclidean","manhattan"]:
        scores=[cross_val_score(KNeighborsClassifier(n_neighbors=k,metric=metric),Xtr,ytr,cv=5,scoring="accuracy").mean() for k in ks]
        k=ks[int(np.argmax(scores))]; model=KNeighborsClassifier(n_neighbors=k,metric=metric).fit(Xtr,ytr)
        out[metric]=scores; print(f"{metric}: k tot nhat={k}, CV={max(scores):.4f}, test={accuracy_score(yte,model.predict(Xte)):.4f}")
    plt.figure(figsize=(8,5))
    for m,s in out.items(): plt.plot(ks,s,marker="o",label=m)
    plt.xlabel("k"); plt.ylabel("Accuracy CV 5-fold"); plt.title("Bai 3 - Chon k"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"bai3_chon_k.png",dpi=150); plt.close()

def bai4():
    title("BAI 4 - SO SANH MO HINH VA KNN CO TRONG SO")
    d=load("hai_vang_trang.csv",["x1","x2","nhan"])
    if d is None: return
    X,y=d[["x1","x2"]],d.nhan; Xtr,Xte,ytr,yte=split(X,y)
    models={"LogisticRegression":LogisticRegression(max_iter=1000),"GaussianNB":GaussianNB(),"KNN k=13":KNeighborsClassifier(n_neighbors=13)}
    for name,m in models.items(): m.fit(Xtr,ytr); print(f"{name}: accuracy={accuracy_score(yte,m.predict(Xte)):.4f}")
    rows=[]
    for k in [1,13,101]:
        if k>len(Xtr): print(f"k={k} bo qua: tap hoc co {len(Xtr)} mau"); continue
        acc=[]
        for w in ["uniform","distance"]:
            m=KNeighborsClassifier(n_neighbors=k,weights=w).fit(Xtr,ytr); acc.append(accuracy_score(yte,m.predict(Xte)))
        rows.append((k,*acc)); print(f"k={k}: uniform={acc[0]:.4f}, distance={acc[1]:.4f}")
    if rows:
        x=np.arange(len(rows)); width=.36; plt.figure(figsize=(7,5)); plt.bar(x-width/2,[r[1] for r in rows],width,label="uniform"); plt.bar(x+width/2,[r[2] for r in rows],width,label="distance"); plt.xticks(x,[str(r[0]) for r in rows]); plt.xlabel("k"); plt.ylabel("Accuracy"); plt.title("Bai 4 - weights distance"); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"bai4_weights_distance.png",dpi=150); plt.close()

def bai5():
    title("BAI 5 - CHON K CHO KNN HOI QUY")
    d=load("gia_nha.csv",["dien_tich","gia"])
    if d is None: return
    X,y=d[["dien_tich"]],d.gia; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    max_k=min(20,len(Xtr)-1); cvmae={}
    for k in range(1,max_k+1): cvmae[k]=-cross_val_score(KNeighborsRegressor(n_neighbors=k),Xtr,ytr,cv=5,scoring="neg_mean_absolute_error").mean()
    best=min(cvmae,key=cvmae.get); knn=KNeighborsRegressor(n_neighbors=best).fit(Xtr,ytr); lr=LinearRegression().fit(Xtr,ytr)
    mk=mean_absolute_error(yte,knn.predict(Xte)); ml=mean_absolute_error(yte,lr.predict(Xte))
    print("MAE CV theo k:"); [print(f"k={k:2d}: {v:.4f}") for k,v in cvmae.items()]
    print(f"k tot nhat={best}; MAE CV={cvmae[best]:.4f}\nMAE KNN test={mk:.4f}\nMAE LinearRegression test={ml:.4f}")
    print("KNN tot hon." if mk<ml else "KNN khong vuot duoc hoi quy tuyen tinh.")
    plt.figure(figsize=(8,5)); plt.plot(list(cvmae),list(cvmae.values()),marker="o"); plt.xlabel("k"); plt.ylabel("MAE CV (thap hon tot hon)"); plt.title("Bai 5 - Chon k KNN Regressor"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG/"bai5_chon_k_regression.png",dpi=150); plt.close()

def bai6():
    title("BAI 6 - DO THOI GIAN FIT VA PREDICT")
    rng=np.random.default_rng(0); Xnew=rng.random((1000,2)); rows=[]
    for n in [1000,10000,100000]:
        X=rng.random((n,2)); y=rng.integers(0,2,size=n); model=KNeighborsClassifier(n_neighbors=5)
        t=time.perf_counter(); model.fit(X,y); tf=time.perf_counter()-t
        t=time.perf_counter(); model.predict(Xnew); tp=time.perf_counter()-t
        rows.append((n,tf,tp)); print(f"n={n:6d} | fit={tf:.6f}s | predict 1000 mau={tp:.6f}s")
    x=np.arange(len(rows)); w=.36; plt.figure(figsize=(8,5)); plt.bar(x-w/2,[r[1] for r in rows],w,label="fit"); plt.bar(x+w/2,[r[2] for r in rows],w,label="predict 1000 mau"); plt.xticks(x,[f"{r[0]:,}" for r in rows]); plt.xlabel("So mau huan luyen"); plt.ylabel("Thoi gian (giay)"); plt.title("Bai 6 - Thoi gian KNN"); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"bai6_thoi_gian.png",dpi=150); plt.close()
    print("KNN la hoc luoi: fit chu yeu luu du lieu, predict phai tim lang gieng gan nhat.")

def main():
    print("TONG HOP KNN BAI 1-6 | data:",DATA,"| figures:",FIG)
    for f in [bai1,bai2,bai3,bai4,bai5,bai6]:
        try: f()
        except Exception as e: print(f"[LOI] {f.__name__}: {e}")
    print("\nHoan tat. Bieu do (neu co du lieu) duoc luu trong figures/.")
if __name__=="__main__": main()
