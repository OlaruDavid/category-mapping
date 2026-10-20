import pandas as pd

prod = pd.read_csv("supplier_products.csv")
lab = pd.read_csv("labelled_products.csv")
cats = pd.read_csv("our_categories.csv")
sales = pd.read_csv("sales.csv")

prod["cat_key"] = prod["supplier_category"].str.lower().str.replace(r"\s+", "", regex=True)

# 1. Duplicate / SKU-uri care nu se potrivesc
print("SKU duplicate in produse:", prod["supplier_sku"].duplicated().sum())
print("SKU etichetate care nu exista in produse:", (~lab["supplier_sku"].isin(prod["supplier_sku"])).sum())
print("SKU etichetate duplicate:", lab["supplier_sku"].duplicated().sum())
print("ID-uri etichete care nu exista in categoriile noastre:", (~lab["our_category_id"].isin(cats["our_category_id"])).sum())

# 2. Acoperire: cate categorii de furnizor au cel putin un produs etichetat
m = lab.merge(prod, on="supplier_sku")
covered = m["cat_key"].nunique()
total = prod["cat_key"].nunique()
print(f"\nCategorii (normalizate) cu cel putin o eticheta: {covered} din {total}")

# 3. Puritate: in fiecare categorie, cat de uniforma e destinatia produselor etichetate?
g = m.groupby("cat_key")["our_category_id"]
purity = g.agg(lambda s: s.value_counts().iloc[0] / len(s))
n = g.size()
summary = pd.DataFrame({"n_labelled": n, "purity": purity})
print("\nPuritate (1.0 = toate produsele etichetate merg in aceeasi categorie):")
print(summary["purity"].describe())
print("Categorii cu puritate < 1:", (summary["purity"] < 1).sum())
print("\nExemple de categorii amestecate:")
print(summary[summary["purity"] < 1].sort_values("n_labelled", ascending=False).head(10))

# 4. Cate categorii de-ale noastre apar in etichete
print("\nCategorii ale noastre folosite in etichete:", lab["our_category_id"].nunique(), "din", len(cats))

# 5. Vanzari
s = sales.merge(prod, on="supplier_sku")
print("\nProduse cu 0 vanzari:", (sales["revenue"] == 0).sum())
rev = s.groupby("cat_key")["revenue"].sum().sort_values(ascending=False)
print("Top 10 categorii dupa venit (procent din total):")
print((rev.head(10) / rev.sum() * 100).round(2))