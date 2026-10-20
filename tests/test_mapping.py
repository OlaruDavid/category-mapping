import re

import pandas as pd


def category_key(v):
    return re.sub(r"\s+", "", str(v)).lower()


def test_one_line_per_distinct_category():
    prod = pd.read_csv("supplier_products.csv")
    out = pd.read_csv("output/mapping.csv")
    assert set(out["supplier_category"]) == set(prod["supplier_category"])
    assert out["supplier_category"].is_unique


def test_same_key_same_answer():
    out = pd.read_csv("output/mapping.csv")
    out["key"] = out["supplier_category"].map(category_key)
    assert (out.groupby("key")["our_category_id"].nunique() == 1).all()


def test_ids_exist_in_our_tree():
    out = pd.read_csv("output/mapping.csv")
    cats = pd.read_csv("our_categories.csv")
    assert out["our_category_id"].isin(cats["our_category_id"]).all()