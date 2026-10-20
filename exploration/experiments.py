import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def category_key(v):
    return re.sub(r"\s+", "", str(v)).lower()


prod = pd.read_csv("../supplier_products.csv")
lab = pd.read_csv("../labelled_products.csv")
sales = pd.read_csv("../sales.csv")

df = prod.merge(sales, on="supplier_sku").merge(lab, on="supplier_sku", how="left")
df["cat_key"] = df["supplier_category"].map(category_key)
df["text"] = (df["product_name"] + " " + df["brand"]).str.lower()
df["name_only"] = df["product_name"].str.lower()
df["w"] = df["revenue"] + 1.0

L = df[df["our_category_id"].notna()].reset_index(drop=True)
L["our_category_id"] = L["our_category_id"].astype(int)

# configuratii de trasaturi: lista de (coloana, vectorizer)
def make_vecs(cfg):
    col = "name_only" if cfg == "fara_brand" else "text"
    vecs = [
        (col, TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        (col, TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ]
    if cfg == "cu_nume_categorie":
        vecs.append(("cat_key", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)))
    return vecs


def featurize(vecs, d, fit):
    mats = [(v.fit_transform(d[c]) if fit else v.transform(d[c])) for c, v in vecs]
    return hstack(mats).tocsr()


vote_weights = {
    "venit": lambda d: d["w"].values,
    "radacina_venit": lambda d: np.sqrt(d["w"].values),
    "numar_produse": lambda d: np.ones(len(d)),
}

results = []
for cfg in ["baseline", "fara_brand", "cu_nume_categorie"]:
    folds = {k: ([], []) for k in vote_weights}
    for tr, te in GroupKFold(n_splits=5).split(L, groups=L["cat_key"]):
        A, T = L.loc[tr], L.loc[te].reset_index(drop=True)
        vecs = make_vecs(cfg)
        X = featurize(vecs, A, fit=True)
        clf = LogisticRegression(C=20, max_iter=2000).fit(X, A["our_category_id"])
        Pt = clf.predict_proba(featurize(vecs, T, fit=False))
        for name, fn in vote_weights.items():
            sc = pd.DataFrame(Pt * fn(T)[:, None], columns=clf.classes_).groupby(T["cat_key"].values).sum()
            chosen = T["cat_key"].map(sc.idxmax(axis=1)).values
            ok = chosen == T["our_category_id"].values
            folds[name][0].append(ok.mean())
            folds[name][1].append(np.average(ok, weights=T["w"]))
    for name, (a, w) in folds.items():
        results.append({"trasaturi": cfg, "vot": name, "acc": np.mean(a), "acc_venit": np.mean(w)})
        print(f"{cfg:20s} vot={name:15s} acc={np.mean(a):.3f}  acc_venit={np.mean(w):.3f}")

pd.DataFrame(results).to_csv("../output/experiments.csv", index=False)