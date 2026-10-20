import re

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)


def category_key(v):
    return re.sub(r"\s+", "", str(v)).lower()


prod = pd.read_csv("supplier_products.csv")
sales = pd.read_csv("sales.csv")
cats = pd.read_csv("our_categories.csv").set_index("our_category_id")["path"]
mapping = pd.read_csv("output/mapping.csv")

df = prod.merge(sales, on="supplier_sku")
df["cat_key"] = df["supplier_category"].map(category_key)
mapping["key"] = mapping["supplier_category"].map(category_key)
key2dest = mapping.drop_duplicates("key").set_index("key")["our_category_id"]

KW = {
    7915: r"banana",
    1712: r"\bjam\b|marmalade|honey|curd|spread",
    4110: r"salmon|\bcod\b|haddock|trout|sea bass|fish fillet",
    5182: r"burger|nugget",
    7448: r"\bstock\b|gravy|bouillon",
    9498: r"lemonade|tonic|mixer|ginger ale|ginger beer",
}

for cid, pat in KW.items():
    hit = df[df["product_name"].str.contains(pat, case=False, regex=True)]
    print(f"\n##### {cid}  {cats[cid]}  ->  {len(hit)} produse potrivite")
    if hit.empty:
        continue
    g = hit.groupby("cat_key").agg(potrivite=("supplier_sku", "size"), venit=("revenue", "sum"),
                                   exemplu=("product_name", "first"))
    g["total_in_cat"] = df.groupby("cat_key").size().reindex(g.index)
    g["destinatie_actuala"] = g.index.map(key2dest).map(cats)
    print(g.sort_values("potrivite", ascending=False).head(6).to_string())