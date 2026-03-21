import pandas as pd

# Read only first 50 rows
df = pd.read_csv("data/output_last_7_days.csv", nrows=50)

# Export to CSV
df.to_csv("output.csv", index=False)