import pandas as pd

cats = pd.read_csv("our_categories.csv")
prod = pd.read_csv("supplier_products.csv")
lab = pd.read_csv("labelled_products.csv")
sales = pd.read_csv("sales.csv")

print("=== Categoriile noastre ===")
print(cats.head(10))
print("Total:", len(cats))

print("\n=== Produse furnizor ===")
print(prod.head(10))
print("Total produse:", len(prod))

print("\n=== Categorii furnizor ===")
print("Distincte (ca scrise):", prod["supplier_category"].nunique())
norm = prod["supplier_category"].str.lower().str.replace(r"\s+", "", regex=True)
print("Distincte (dupa normalizare):", norm.nunique())

print("\n=== Etichete ===")
print("Produse etichetate:", len(lab))

print("\n=== Cele mai mari categorii furnizor ===")
print(prod["supplier_category"].value_counts().head(20))

print("\n=== Valori lipsa ===")
print(prod.isna().sum())

print("\n=== Vanzari ===")
print(sales.describe())