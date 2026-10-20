import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 55)

preds = pd.read_csv("output/product_predictions.csv")
diag = pd.read_csv("output/diagnostics.csv", index_col=0)
cats = pd.read_csv("our_categories.csv").set_index("our_category_id")["path"]
mapping = pd.read_csv("output/mapping.csv")

# "Mutat gresit" = produsul isi are, dupa model (sau dupa eticheta manuala),
# alta categorie decat cea aleasa pentru toata categoria furnizorului.
preds["misplaced"] = preds["pred_id"] != preds["dest"]
preds["misplaced_rev"] = preds["revenue"].where(preds["misplaced"], 0.0)

g = preds.groupby("cat_key")
cost = pd.DataFrame({
    "n_products": g.size(),
    "revenue": g["revenue"].sum(),
    "misplaced_products": g["misplaced"].sum(),
    "misplaced_rev": g["misplaced_rev"].sum(),
}).join(diag[["n_labelled", "confidence", "dest_path"]])
cost["misplaced_share"] = (cost["misplaced_rev"] / cost["revenue"]).fillna(0)

# unde ajunge, de fapt, venitul "mutat gresit"
alt = (preds[preds["misplaced"]]
       .groupby(["cat_key", "pred_id"])["revenue"].sum().reset_index()
       .sort_values("revenue", ascending=False)
       .drop_duplicates("cat_key").set_index("cat_key"))
cost["main_alternative"] = alt["pred_id"].map(cats)

cost = cost.sort_values("misplaced_rev", ascending=False)
cost.to_csv("output/cost_of_errors.csv")

total_rev = preds["revenue"].sum()
total_mis = preds["misplaced_rev"].sum()
print(f"Venit total: {total_rev:,.0f} EUR")
print(f"Venit estimat in categorie gresita: {total_mis:,.0f} EUR ({total_mis / total_rev:.1%})")
print(f"Primele 20 de categorii = {cost['misplaced_rev'].head(20).sum() / total_mis:.0%} din acest venit\n")

print("=== 1. Greselile cele mai costisitoare (top 20, EUR mutati gresit) ===")
print(cost[["n_products", "revenue", "misplaced_rev", "misplaced_share",
            "dest_path", "main_alternative"]].head(20).round(2))

# puncte slabe
mixed = cost[(cost["misplaced_share"] > 0.25) & (cost["revenue"] > 100_000)]
print(f"\n=== 2a. Categorii amestecate (>25% din venit mutat gresit, venit > 100k): {len(mixed)} ===")
print(mixed[["revenue", "misplaced_share", "dest_path", "main_alternative"]].head(15).round(2))

unlabelled = cost[(cost["n_labelled"] == 0) & (cost["misplaced_share"] > 0.25)]
print(f"\n=== 2b. Fara nicio eticheta manuala si amestecate: {len(unlabelled)} categorii, "
      f"{unlabelled['revenue'].sum():,.0f} EUR ===")

unused = sorted(set(cats.index) - set(mapping["our_category_id"]))
print(f"\n=== 2c. Categorii de-ale noastre neatinse de nicio mapare: {len(unused)} ===")
for u in unused:
    print(" ", u, cats[u])

cost[cost.index.isin(mixed.index) | cost.index.isin(unlabelled.index)].to_csv("output/weak_spots.csv")