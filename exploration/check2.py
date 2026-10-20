import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 50)

p = pd.read_csv("output/product_predictions.csv")
cats = pd.read_csv("our_categories.csv").set_index("our_category_id")["path"]
p["pred_path"] = p["pred_id"].map(cats)
p["dest_path"] = p["dest"].map(cats)

for k in ["energysprtsdrin", "smkdfish"]:
    sub = p[p["cat_key"] == k]
    print(f"\n=== {k}: {len(sub)} produse, destinatie {sub['dest_path'].iloc[0]} ===")
    print(sub["pred_path"].value_counts().to_string())
    print(sub[["product_name", "revenue", "pred_path"]].head(12).to_string())