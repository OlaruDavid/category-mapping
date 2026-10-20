import pandas as pd

cats = pd.read_csv("our_categories.csv")
print(cats[cats["path"].str.contains("spice|herb|season|baking|sugar|rice", case=False)].to_string())