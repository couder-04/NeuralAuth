import pandas as pd

# Load the dataset
df = pd.read_csv("/Users/par_04/code_playground/projects/Transaction_engine/training/data/dataset.csv")

# Select the first 10 rows (all columns)
test_df = df.head(10)

# Save to a NEW file — never overwrite the source dataset in place.
test_df.to_csv("/Users/par_04/code_playground/projects/Transaction_engine/training/data/test_dataset.csv", index=False)

print("Created test_dataset.csv with the top 10 rows.")