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


prod = pd.read_csv("supplier_products.csv")
lab = pd.read_csv("labelled_products.csv")
sales = pd.read_csv("sales.csv")

df = prod.merge(sales, on="supplier_sku").merge(lab, on="supplier_sku", how="left")
df["cat_key"] = df["supplier_category"].map(category_key)
df["text"] = (df["product_name"] + " " + df["brand"]).str.lower()
df["pack_tok"] = df["pack_size"].map(pack_tokens)
df["w"] = df["revenue"] + 1.0

# acoperire: cate pack_size nu au putut fi citite?
nosize = df["pack_tok"].str.contains("nosize")
nothing = nosize & df["pack_tok"].str.contains("count_1$")
print(f"fara marime: {nosize.sum()}, fara nimic (nici marime, nici bucati): {nothing.sum()} din {len(df)}")
print("Exemple fara nimic:", df.loc[nothing, "pack_size"].drop_duplicates().head(20).tolist())
print("Exemple recunoscute:")
print(df[["pack_size", "pack_tok"]].drop_duplicates("pack_size").sample(8, random_state=0).to_string())

L = df[df["our_category_id"].notna()].reset_index(drop=True)
L["our_category_id"] = L["our_category_id"].astype(int)


def run(use_pack):
    accs, waccs = [], []
    for tr, te in GroupKFold(n_splits=5).split(L, groups=L["cat_key"]):
        A, T = L.loc[tr], L.loc[te].reset_index(drop=True)
        wv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        mats_A = [wv.fit_transform(A["text"]), cv.fit_transform(A["text"])]
        mats_T = [wv.transform(T["text"]), cv.transform(T["text"])]
        if use_pack:
            pv = TfidfVectorizer(token_pattern=r"\S+")
            mats_A.append(pv.fit_transform(A["pack_tok"]))
            mats_T.append(pv.transform(T["pack_tok"]))
        clf = LogisticRegression(C=20, max_iter=2000).fit(hstack(mats_A).tocsr(), A["our_category_id"])
        Pt = clf.predict_proba(hstack(mats_T).tocsr())
        sc = pd.DataFrame(Pt * np.sqrt(T["w"].values)[:, None], columns=clf.classes_) \
               .groupby(T["cat_key"].values).sum()
        ok = T["cat_key"].map(sc.idxmax(axis=1)).values == T["our_category_id"].values
        accs.append(ok.mean())
        waccs.append(np.average(ok, weights=T["w"]))
    return np.mean(accs), np.mean(waccs)


for name, flag in [("fara pack_size", False), ("cu pack_size", True)]:
    a, w = run(flag)
    print(f"{name:16s} acc_categorie={a:.3f}  acc_venit={w:.3f}")