import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

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
    return "v", 1000


def pack_tokens(s):
    s = str(s).lower().strip().replace(",", ".")
    toks, count, amount, kind = [], 1, None, None
    m = re.search(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*" + UNIT_RE, s)
    if m:
        count = int(m.group(1))
        kind, mult = unit_info(m.group(3))
        amount = float(m.group(2)) * mult
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*" + UNIT_RE, s)
        if m:
            kind, mult = unit_info(m.group(2))
            amount = float(m.group(1)) * mult
        m2 = (re.search(r"(\d+)\s*(?:pk|pack|packs|ct|pcs|pc|pieces)\b", s)
              or re.search(r"\bx\s*(\d+)\b", s))
        if m2:
            count = int(m2.group(1))
    if amount and amount > 0:
        toks += [f"unit_{kind}", f"size_{kind}_{int(round(np.log2(amount)))}"]
    else:
        toks.append("nosize")
    toks.append(f"count_{min(count, 24)}")
    return " ".join(toks)


prod = pd.read_csv("supplier_products.csv")
lab = pd.read_csv("labelled_products.csv")
sales = pd.read_csv("sales.csv")
df = prod.merge(sales, on="supplier_sku").merge(lab, on="supplier_sku", how="left")
df["cat_key"] = df["supplier_category"].map(category_key)
df["text"] = (df["product_name"] + " " + df["brand"]).str.lower()
df["pack_tok"] = df["pack_size"].map(pack_tokens)
df["w"] = df["revenue"] + 1.0

L = df[df["our_category_id"].notna()].reset_index(drop=True)
L["our_category_id"] = L["our_category_id"].astype(int)

cap = df["w"].quantile(0.75)
WEIGHTS = {
    "venit": lambda w: w,
    "sqrt": lambda w: np.sqrt(w),
    "log": lambda w: np.log1p(w),
    "venit_plafonat": lambda w: np.minimum(w, cap),
    "numar_produse": lambda w: np.ones_like(w),
}

res = {k: ([], []) for k in WEIGHTS}
for tr, te in GroupKFold(n_splits=5).split(L, groups=L["cat_key"]):
    A, T = L.loc[tr], L.loc[te].reset_index(drop=True)
    wv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    pv = TfidfVectorizer(token_pattern=r"\S+")
    X = hstack([wv.fit_transform(A["text"]), cv.fit_transform(A["text"]), pv.fit_transform(A["pack_tok"])]).tocsr()
    clf = LogisticRegression(C=20, max_iter=2000).fit(X, A["our_category_id"])
    Pt = clf.predict_proba(hstack([wv.transform(T["text"]), cv.transform(T["text"]), pv.transform(T["pack_tok"])]).tocsr())
    for name, fn in WEIGHTS.items():
        sc = pd.DataFrame(Pt * fn(T["w"].values)[:, None], columns=clf.classes_).groupby(T["cat_key"].values).sum()
        ok = T["cat_key"].map(sc.idxmax(axis=1)).values == T["our_category_id"].values
        res[name][0].append(ok.mean())
        res[name][1].append(np.average(ok, weights=T["w"]))

for name, (a, w) in res.items():
    print(f"{name:16s} acc_categorie={np.mean(a):.3f}  acc_venit={np.mean(w):.3f}")