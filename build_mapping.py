import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)

UNIT_RE = r"(kg|kilos?|gr|g|ml|cl|litres?|liters?|ltrs?|l)\b"


def category_key(v):
    return re.sub(r"\s+", "", str(v)).lower()


def unit_info(u):
    if u in ("kg", "kilo", "kilos"):
        return "w", 1000
    if u in ("g", "gr"):
        return "w", 1
    if u == "ml":
        return "v", 1
    if u == "cl":
        return "v", 10
    return "v", 1000  # l, litre, litres, liter, ltr


def pack_tokens(s):
    """Transforma pack_size intr-un text de tokeni: unitate, marime (log2), numar bucati."""
    s = str(s).lower().strip().replace(",", ".")
    toks, count, amount, kind = [], 1, None, None

    m = re.search(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*" + UNIT_RE, s)      # 2x300 gr, 12 x 1.5 litre
    if m:
        count = int(m.group(1))
        kind, mult = unit_info(m.group(3))
        amount = float(m.group(2)) * mult
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*" + UNIT_RE, s)              # 0.2kg, 500 g, 0.25 litre
        if m:
            kind, mult = unit_info(m.group(2))
            amount = float(m.group(1)) * mult
        m2 = (re.search(r"(\d+)\s*(?:pk|pack|packs|ct|pcs|pc|pieces)\b", s)
              or re.search(r"\bx\s*(\d+)\b", s))                       # 16pk, 24 pcs, x12
        if m2:
            count = int(m2.group(1))

    if amount and amount > 0:
        toks += [f"unit_{kind}", f"size_{kind}_{int(round(np.log2(amount)))}"]
    else:
        toks.append("nosize")
    toks.append(f"count_{min(count, 24)}")
    return " ".join(toks)


# ---------- Incarcare date ----------
prod = pd.read_csv("supplier_products.csv")
lab = pd.read_csv("labelled_products.csv")
sales = pd.read_csv("sales.csv")
cats = pd.read_csv("our_categories.csv").set_index("our_category_id")["path"]

df = prod.merge(sales, on="supplier_sku").merge(lab, on="supplier_sku", how="left")
df["cat_key"] = df["supplier_category"].map(category_key)
df["text"] = (df["product_name"] + " " + df["brand"]).str.lower()
df["pack_tok"] = df["pack_size"].map(pack_tokens)
df["w"] = df["revenue"] + 1.0  # +1 ca produsele cu venit 0 sa aiba totusi un vot mic
df["vote_w"] = 1.0  # vot pe numar de produse: fiecare produs un vot (testat: cel mai bun acc_categorie)


# ---------- Model ----------
def fit_model(d, y):
    wv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    pv = TfidfVectorizer(token_pattern=r"\S+")
    X = hstack([wv.fit_transform(d["text"]), cv.fit_transform(d["text"]),
                pv.fit_transform(d["pack_tok"])]).tocsr()
    clf = LogisticRegression(C=20, max_iter=2000).fit(X, y)
    return wv, cv, pv, clf


def predict_proba(model, d):
    wv, cv, pv, clf = model
    X = hstack([wv.transform(d["text"]), cv.transform(d["text"]),
                pv.transform(d["pack_tok"])]).tocsr()
    return clf.predict_proba(X), clf.classes_


# ---------- 1. Evaluare pe categorii nevazute (GroupKFold pe categoria furnizorului) ----------
L = df[df["our_category_id"].notna()].reset_index(drop=True)
L["our_category_id"] = L["our_category_id"].astype(int)

acc, wacc, cacc, cwacc = [], [], [], []
for tr, te in GroupKFold(n_splits=5).split(L, groups=L["cat_key"]):
    m = fit_model(L.loc[tr], L.loc[tr, "our_category_id"])
    T = L.loc[te].reset_index(drop=True)
    Pt, cl = predict_proba(m, T)

    # pe produs
    ok = cl[Pt.argmax(1)] == T["our_category_id"].values
    acc.append(ok.mean())
    wacc.append(np.average(ok, weights=T["w"]))

    # pe categorie: un singur vot pentru toata categoria, apoi comparam cu eticheta fiecarui produs
    sc = pd.DataFrame(Pt * T["vote_w"].values[:, None], columns=cl).groupby(T["cat_key"].values).sum()
    chosen = T["cat_key"].map(sc.idxmax(axis=1)).values
    okc = chosen == T["our_category_id"].values
    cacc.append(okc.mean())
    cwacc.append(np.average(okc, weights=T["w"]))

print(f"Pe produs:    {np.mean(acc):.3f}  (ponderat {np.mean(wacc):.3f})")
print(f"Pe categorie: {np.mean(cacc):.3f}  (ponderat {np.mean(cwacc):.3f})  <- metrica reala")

# ---------- 2. Model final + vot pe categorie ----------
model = fit_model(L, L["our_category_id"])
P, classes = predict_proba(model, df)
idx = {int(c): i for i, c in enumerate(classes)}

# produsele etichetate manual sunt adevar cunoscut: le fortam la one-hot
for r in np.where(df["our_category_id"].notna())[0]:
    P[r] = 0.0
    P[r, idx[int(df["our_category_id"].iat[r])]] = 1.0

scores = pd.DataFrame(P * df["vote_w"].values[:, None], columns=classes).groupby(df["cat_key"].values).sum()
best = scores.idxmax(axis=1).astype(int)
conf = scores.max(axis=1) / scores.sum(axis=1)

# ---------- 3. Export: cate o linie pentru fiecare valoare scrisa original ----------
os.makedirs("output", exist_ok=True)
out = prod[["supplier_category"]].drop_duplicates().copy()
out["our_category_id"] = out["supplier_category"].map(category_key).map(best)
assert len(out) == prod["supplier_category"].nunique()
assert out["our_category_id"].notna().all()
out.to_csv("output/mapping.csv", index=False)
print("\nScris output/mapping.csv:", len(out), "linii")

# predictii pe produs, pentru analiza de cost
df["pred_id"] = classes[P.argmax(1)]
df["dest"] = df["cat_key"].map(best)
df[["supplier_sku", "supplier_category", "cat_key", "product_name", "brand",
    "revenue", "our_category_id", "pred_id", "dest"]].to_csv(
    "output/product_predictions.csv", index=False)

# ---------- 4. Diagnostic: unde e riscul cel mai mare ----------
g = df.groupby("cat_key")
diag = pd.DataFrame({
    "n_products": g.size(),
    "n_labelled": g["our_category_id"].count(),
    "revenue": g["revenue"].sum(),
    "dest": best,
    "confidence": conf,
})
diag["at_risk_eur"] = diag["revenue"] * (1 - diag["confidence"])
diag["dest_path"] = diag["dest"].map(cats)
diag = diag.sort_values("at_risk_eur", ascending=False)
diag.to_csv("output/diagnostics.csv")

print("\nTop 15 categorii cu risc ridicat:")
print(diag[["n_products", "revenue", "confidence", "dest_path"]].head(15))
print()
print(diag.loc["herbs:drd.hrbs.spices", ["confidence", "dest_path"]])
print(diag.loc["bkng.in", ["confidence", "dest_path"]])


# ---------- 5. Inspectie: cum se imparte o categorie, produs cu produs ----------
def show(cat_key, n=40):
    sub = df[df["cat_key"] == cat_key].copy()
    pred = classes[P[sub.index].argmax(1)]
    sub["pred_path"] = pd.Series(pred, index=sub.index).map(cats)
    print(f"\n=== {cat_key} ===")
    print(sub[["product_name", "brand", "pack_size", "revenue", "pred_path"]].head(n).to_string())


show("citr.frt", 25)
show("tinnefrt", 25)
show("pr_ts", 25)