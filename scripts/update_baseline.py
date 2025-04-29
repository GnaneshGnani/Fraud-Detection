import os
import pandas as pd

baseline_df = pd.read_csv("data/baseline.csv").drop(columns = ["id"])
curr_df = pd.read_csv("data/current_batch.csv").drop(columns = ["id"])

df = pd.concat([baseline_df, curr_df], ignore_index = True)
df.index.name = 'id'

df.to_csv("data/baseline.csv")

os.remove("data/current_batch.csv")
for file in os.listdir("data/incoming"):
    os.remove(f"data/incoming/{file}")
